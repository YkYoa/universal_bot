# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    OperationalSpaceControllerActionCfg,
)
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.utils.configclass import configclass

from .grasp_assist import apply_grasp_arm_assist
from .helpers import STAGE_GRASP, STAGE_PLACE, STAGE_REACH, finger_descended_for_close, finger_grasp_ready, finger_pad_asymmetric, finger_pad_severe_asymmetric, finger_ready_for_close, finger_symmetric_ready, place_release_ready, uses_grasp_lift, uses_place

_JOINT_POS_WARNED = False


def _joint_pos_mean_safe(asset, joint_ids, device):
    """Mean joint_pos qua joint_ids, chịu được cả ProxyArray lẫn wp.array thô.

    Ở apply_actions() (trước physics step), asset.data.joint_pos có lúc KHÔNG
    phải ProxyArray thường gặp ở nơi khác trong codebase (không có .torch),
    mà là wp.array — cần wp.to_torch() tường minh. Trả None nếu cả hai cách
    đều thất bại, để gọi nơi tự quyết định bỏ qua thay vì crash cả episode.
    """
    global _JOINT_POS_WARNED
    # self._joint_ids trong IsaacLab 3.0 là wp.array (in ra giống list Python
    # nhưng KHÔNG PHẢI list) — dùng thẳng làm index vào torch.Tensor khiến
    # warp tự cố indexing nội bộ ("Item indexing is not supported on
    # wp.array"). raw.torch tự nó trả về đúng torch.Tensor bình thường — chỉ
    # cần chuyển joint_ids sang list/tensor thường trước khi index.
    ids = joint_ids
    if hasattr(ids, "numpy"):
        ids = ids.numpy().tolist()
    elif not isinstance(ids, (list, tuple)):
        ids = list(ids)
    try:
        raw = asset.data.joint_pos
        joint_pos_t = getattr(raw, "torch", raw)  # ProxyArray (3.0) vs Tensor thường (2.3.2)
        return joint_pos_t[:, ids].mean(dim=1)
    except Exception as e:
        if not _JOINT_POS_WARNED:
            print(f"  ⚠️ [GripPressFreeze] Không đọc được joint_pos: {type(e).__name__}: {e} — bỏ qua dừng-theo-lực.", flush=True)
            _JOINT_POS_WARNED = True
        return None


class AssistedOperationalSpaceControllerAction(OperationalSpaceControllerAction):
    """OSC arm action with phase-2 descent/lift assist applied at process_actions."""

    def process_actions(self, actions: torch.Tensor):
        actions = apply_grasp_arm_assist(self._env, actions)
        super().process_actions(actions)


class AssistedBinaryGripperAction(BinaryJointPositionAction):
    """Binary gripper with optional auto-close in GRASP stage (Phase 2 bootstrap)."""

    def __init__(self, cfg: BinaryJointPositionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._grasp_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._descend_hold = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._sym_hold = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._reopen_cooldown = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._reopen_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._want_close = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._close_progress = torch.zeros(self.num_envs, device=self.device)
        self._prev_close_tilt = torch.zeros(self.num_envs, device=self.device)
        self._press_hold_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._grasp_latched[:] = False
            self._descend_hold[:] = 0
            self._sym_hold[:] = 0
            self._reopen_cooldown[:] = 0
            self._reopen_count[:] = 0
            self._want_close[:] = False
            self._close_progress[:] = 0.0
            self._prev_close_tilt[:] = 0.0
            self._press_hold_steps[:] = 0
        else:
            self._grasp_latched[env_ids] = False
            self._descend_hold[env_ids] = 0
            self._sym_hold[env_ids] = 0
            self._reopen_cooldown[env_ids] = 0
            self._reopen_count[env_ids] = 0
            self._want_close[env_ids] = False
            self._close_progress[env_ids] = 0.0
            self._prev_close_tilt[env_ids] = 0.0
            self._press_hold_steps[env_ids] = 0

    def process_actions(self, actions: torch.Tensor):
        env = self._env
        if (
            getattr(env.cfg, "grasp_auto_close", True)
            and uses_grasp_lift(getattr(env.cfg, "task_phase", 1))
            and hasattr(env, "_stage")
        ):
            actions = actions.clone()
            in_grasp = env._stage == STAGE_GRASP
            s = getattr(env, "_last_state", None)
            grip_thresh = getattr(env.cfg, "grasp_grip_threshold", 0.4)
            hold_closed = getattr(env.cfg, "grasp_hold_closed", True)

            if s is not None and hold_closed:
                lift_thresh = getattr(env.cfg, "grasp_lift_threshold", 0.03)
                reopen_z = float(getattr(env.cfg, "grasp_slip_reopen_z_finger", 0.0))
                if reopen_z > 0.0:
                    slipped = (
                        in_grasp
                        & self._grasp_latched
                        & (s["z_error_finger"] > reopen_z)
                        & (s["bottle_lift"] < lift_thresh)
                    )
                    if slipped.any():
                        self._grasp_latched[slipped] = False
                        self._close_progress[slipped] = 0.0
                        self._descend_hold[slipped] = 0

                # Mở lại chỉ sau khi khép xong + lệch nặng / tilt cao (không cắt giữa ramp).
                # KHÔNG BAO GIỜ mở khi đang nhấc: reopen là hành vi phục hồi TRƯỚC khi
                # nhấc; làm giữa không trung thì chai rơi. Lúc nhấc chai đung đưa nhẹ →
                # khoảng cách 2 pad lệch → severe_asym → mở kẹp → mất latch → hủy nhấc.
                _lp = getattr(env, "_lift_phase", None)
                lift_active = (
                    _lp != 0 if _lp is not None
                    else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                )
                reopen_asym = float(getattr(env.cfg, "grasp_reopen_asym_dist_delta", 0.0))
                reopen_tilt = float(getattr(env.cfg, "grasp_reopen_max_tilt_deg", 6.0))
                if reopen_asym > 0.0:
                    if in_grasp.any():
                        self._reopen_cooldown[in_grasp] = torch.clamp(
                            self._reopen_cooldown[in_grasp] - 1, min=0
                        )
                    reopen_cooldown = int(getattr(env.cfg, "grasp_reopen_cooldown_steps", 40))
                    reopen_max = int(getattr(env.cfg, "grasp_reopen_max_count", 2))
                    # Cap-aware: ramp không thể vượt grasp_close_freeze_at_progress,
                    # nên min_gc cố định 0.92 khiến cả nhánh reopen này thành dead code.
                    _freeze_at = float(getattr(env.cfg, "grasp_close_freeze_at_progress", 0.0))
                    _cap = _freeze_at if _freeze_at > 0.0 else 1.0
                    min_gc = min(float(getattr(env.cfg, "grasp_reopen_min_close_progress", 0.92)), _cap)
                    severe_delta = float(
                        getattr(env.cfg, "grasp_reopen_severe_asym_delta", max(reopen_asym * 1.5, 0.030))
                    )
                    gripped = s["gripper_state"] > grip_thresh
                    close_done = self._close_progress >= min_gc
                    if int(getattr(env.cfg, "grasp_close_ramp_steps", 0)) <= 0:
                        close_done = close_done | gripped
                    dist_delta = (s["dist_left_body"] - s["dist_right_body"]).abs()
                    severe_asym = dist_delta > severe_delta
                    tilt_bad = s["bottle_tilt_deg"] > reopen_tilt
                    abort_tilt = float(getattr(env.cfg, "grasp_reopen_abort_tilt_deg", 12.0))
                    # Không reopen khi chai đã ngã — mở grip chỉ làm tệ hơn
                    can_reopen_tilt = tilt_bad & (s["bottle_tilt_deg"] <= abort_tilt)
                    pause_tilt_gc = float(
                        getattr(env.cfg, "grasp_reopen_pause_tilt_min_gc", 0.70)
                    )
                    can_reopen_tilt = can_reopen_tilt & (self._close_progress < pause_tilt_gc)
                    bad_close = (
                        in_grasp
                        & ~lift_active
                        & gripped
                        & close_done
                        & (self._reopen_cooldown <= 0)
                        & (self._reopen_count < reopen_max)
                        & (s["bottle_lift"] < lift_thresh)
                        & (can_reopen_tilt | severe_asym)
                    )
                    # Ramp: reopen sớm — lệch pad trước khi tilt 6° (gc~79% hay lặp lại)
                    ramp_reopen_tilt = float(getattr(env.cfg, "grasp_reopen_ramp_tilt_deg", 7.0))
                    mid_tilt = float(getattr(env.cfg, "grasp_reopen_mid_tilt_deg", 4.0))
                    ramp_asym_tilt = float(
                        getattr(env.cfg, "grasp_reopen_ramp_asym_tilt_deg", 2.5)
                    )
                    ramp_asym_gc = float(
                        getattr(env.cfg, "grasp_reopen_ramp_asym_min_gc", 0.45)
                    )
                    pad_asym = finger_pad_asymmetric(env, s)
                    pause_tilt_gc = float(
                        getattr(env.cfg, "grasp_reopen_pause_tilt_min_gc", 0.70)
                    )
                    # gc cao: tilt do khép cuối — pause ramp, không reopen chỉ vì tilt
                    tilt_only_reopen = (s["bottle_tilt_deg"] > ramp_reopen_tilt) & (
                        self._close_progress < pause_tilt_gc
                    )
                    mid_ramp = (
                        in_grasp
                        & ~lift_active
                        & (self._close_progress > 0.25)
                        & (self._close_progress < min_gc)
                        & (self._want_close | self._grasp_latched)
                        & (self._reopen_cooldown <= 0)
                        & (self._reopen_count < reopen_max)
                        & (s["bottle_lift"] < lift_thresh)
                        & (
                            (severe_asym & (s["bottle_tilt_deg"] > mid_tilt))
                            | tilt_only_reopen
                            | (
                                pad_asym
                                & (self._close_progress > ramp_asym_gc)
                                & (s["bottle_tilt_deg"] > ramp_asym_tilt)
                            )
                        )
                    )
                    if mid_ramp.any():
                        bad_close = bad_close | mid_ramp
                    if bad_close.any():
                        preserve_gc = getattr(env.cfg, "grasp_reopen_preserve_gc_on_final", True)
                        final_reopen = bad_close & (self._reopen_count + 1 >= reopen_max)
                        reset_gc = bad_close & ~(final_reopen & preserve_gc)
                        self._reopen_count[bad_close] += 1
                        self._grasp_latched[bad_close] = False
                        self._want_close[bad_close] = False
                        self._close_progress[reset_gc] = 0.0
                        self._descend_hold[bad_close] = 0
                        self._sym_hold[bad_close] = 0
                        self._reopen_cooldown[bad_close] = reopen_cooldown
                        if env.num_envs <= 16 and getattr(env.cfg, "debug_success_log", False):
                            for idx in bad_close.nonzero(as_tuple=False).flatten().tolist():
                                if idx != 0:
                                    continue
                                dl = float(s["dist_left_body"][idx].item())
                                dr = float(s["dist_right_body"][idx].item())
                                dd = float(dist_delta[idx].item())
                                tilt = float(s["bottle_tilt_deg"][idx].item())
                                gc = float(self._close_progress[idx].item())
                                parts = []
                                if bool(tilt_bad[idx].item()) and float(s["bottle_tilt_deg"][idx]) <= abort_tilt:
                                    parts.append(f"tilt:{tilt:.1f}°>{reopen_tilt:.1f}°")
                                if bool(pad_asym[idx].item()):
                                    parts.append(f"Δpad:{dd:.3f}>{reopen_asym:.3f}")
                                elif bool(severe_asym[idx].item()):
                                    parts.append(f"Δpad:{dd:.3f}>{severe_delta:.3f}")
                                print(
                                    f"  [Grip] reopen ({', '.join(parts) or 'bad'})"
                                    f" | L/R:{dl:.3f}/{dr:.3f} gc:{gc:.0%}"
                                    f" → mở lại, căn XY, thử close"
                                )

                # Phase 27 lớp 2 (reclose sau phục hồi) ĐÃ THỬ và REVERT (Phase 29,
                # terminal_command.md): hoạt động đúng kỹ thuật (tạo retry thật
                # trong cùng episode) nhưng đo được kết quả hỗn hợp — grasp/latch
                # giảm ở kịch bản khó, thêm fail_mode `tilt` mới (lật chai lúc
                # đóng lại) — chi phí không bù được lợi ích rất nhỏ. Giữ lại lớp 1
                # (env._lift_recovery_active, grasp_assist.py) vì bản thân nó vô
                # hại (không đổi metric, không thêm rủi ro) dù chưa đủ để LIFT
                # thành công — nguyên nhân gốc là vật lý (lực nhấc/ma sát), không
                # phải thiếu cơ hội thử lại.

                latch_z = float(getattr(
                    env.cfg, "grasp_latch_max_z_finger",
                    getattr(env.cfg, "grasp_close_max_z_err", 0.04) * 0.85,
                ))
                low_enough = s["z_error_finger"] < latch_z
                align_ready = finger_grasp_ready(env, s)
                close_ready = finger_ready_for_close(env, s)
                hold_req = int(getattr(env.cfg, "grasp_descend_hold_steps", 5))
                sym_req = int(getattr(env.cfg, "grasp_sym_hold_steps", 0))
                # Đếm hold khi đã hạ + align cơ bản; đóng cần thêm sym (close_ready)
                if in_grasp.any():
                    hold_tick = low_enough & align_ready & in_grasp
                    self._descend_hold[hold_tick] += 1
                    self._descend_hold[~hold_tick & in_grasp] = 0
                    if sym_req > 0 and getattr(env.cfg, "grasp_symmetry_gate_enabled", False):
                        sym_tick = hold_tick & finger_symmetric_ready(env, s)
                        self._sym_hold[sym_tick] += 1
                        self._sym_hold[~sym_tick & in_grasp] = 0
                    else:
                        self._sym_hold[in_grasp] = sym_req
                else:
                    self._descend_hold[:] = 0
                    self._sym_hold[:] = 0
                hold_done = in_grasp & (self._descend_hold >= hold_req)
                if sym_req > 0 and getattr(env.cfg, "grasp_symmetry_gate_enabled", False):
                    sym_done = in_grasp & (self._sym_hold >= sym_req)
                else:
                    sym_done = in_grasp
                latch_ok = s["z_error_finger"] < latch_z
                self._grasp_latched |= in_grasp & close_ready & hold_done & sym_done & latch_ok
                latched = in_grasp & self._grasp_latched
                min_z_f = getattr(env.cfg, "grasp_close_min_z_finger", -0.01)
                may_close_z = s["z_error_finger"] >= min_z_f
                force_close = latched | (in_grasp & close_ready & hold_done & sym_done & may_close_z)
                # Sau latch: tiếp tục khép dù z_f hơi tăng do lift thử — không tắt close
                descended = finger_descended_for_close(env, s)
                term_gc = self._close_progress
                partial_gc = float(getattr(env.cfg, "grasp_lift_partial_min_gc", 0.52))
                keep_close = latched | (term_gc >= partial_gc)
                force_close = force_close & (descended | keep_close)
                # Không ép đóng tiếp ngay sau reopen lệch — chờ sym lại
                if reopen_asym > 0.0:
                    force_close = force_close & ~(
                        in_grasp
                        & finger_pad_asymmetric(env, s)
                        & ~finger_symmetric_ready(env, s)
                    )
                # GRASP: giữ gripper mở cho đến khi ngón hạ đủ thấp
                not_descended = ~finger_descended_for_close(env, s)
                actions[in_grasp & not_descended, 0] = 1.0
            elif s is not None:
                low_enough = finger_descended_for_close(env, s)
                hold_req = int(getattr(env.cfg, "grasp_descend_hold_steps", 5))
                if in_grasp.any():
                    self._descend_hold[low_enough & in_grasp] += 1
                    self._descend_hold[(~low_enough) & in_grasp] = 0
                else:
                    self._descend_hold[:] = 0
                ready = finger_ready_for_close(env, s) & in_grasp & (self._descend_hold >= hold_req)
                force_close = in_grasp & ready
            else:
                force_close = in_grasp

            if s is not None:
                too_far = s["dist_finger_body"] > getattr(env.cfg, "grasp_open_until_dist", 0.15)
                force_close = force_close & ~too_far
                # GRASP trước latch: ép mở — policy hay đóng sớm (dh chưa đủ)
                actions[in_grasp & ~force_close, 0] = 1.0
                actions[too_far & in_grasp, 0] = 1.0

            actions[force_close, 0] = -1.0
            actions[env._stage == STAGE_REACH, 0] = 1.0

            if uses_place(getattr(env.cfg, "task_phase", 1)) and s is not None:
                in_place = env._stage == STAGE_PLACE
                release_ready = place_release_ready(env, s)
                # Bootstrap tuỳ chọn (mirror grasp_hold_closed): ép giữ đóng
                # cứng suốt carry/descend để verify state machine + assist +
                # release trọn vẹn bằng scripted-assist trước khi có policy
                # nào học giữ kẹp qua reward (tắt khi train thật, S10).
                if getattr(env.cfg, "place_hold_closed", True):
                    actions[in_place & ~release_ready, 0] = -1.0
                actions[in_place & release_ready, 0] = 1.0

            self._want_close[:] = False
            if s is not None:
                self._want_close[:] = force_close
        super().process_actions(actions)

    def apply_actions(self):
        ramp_steps = int(getattr(self._env.cfg, "grasp_close_ramp_steps", 0))
        if ramp_steps > 0 and uses_grasp_lift(getattr(self._env.cfg, "task_phase", 1)):
            ramping = (self._want_close | self._grasp_latched) & (self._close_progress < 0.99)
            can_advance = ramping
            s = getattr(self._env, "_last_state", None)
            if s is not None and getattr(self._env.cfg, "grasp_close_pause_on_asym", True):
                ramp_tilt_max = float(getattr(self._env.cfg, "grasp_close_ramp_max_tilt_deg", 4.0))
                exhaust_tilt = float(
                    getattr(self._env.cfg, "grasp_close_exhaust_max_tilt_deg", 5.0)
                )
                pad_bad = finger_pad_severe_asymmetric(self._env, s)
                mild_asym = finger_pad_asymmetric(self._env, s)
                asym_pause_gc = float(
                    getattr(self._env.cfg, "grasp_close_asym_pause_max_gc", 0.55)
                )
                pad_bad = pad_bad | (mild_asym & (self._close_progress < asym_pause_gc))
                reopen_max = int(getattr(self._env.cfg, "grasp_reopen_max_count", 3))
                exhausted = self._reopen_count >= reopen_max
                freeze_at = float(getattr(self._env.cfg, "grasp_close_freeze_at_progress", 0.0))
                min_gc = float(getattr(self._env.cfg, "grasp_lift_partial_min_gc", 0.55))
                creep_cap = float(getattr(self._env.cfg, "grasp_close_slip_creep_progress", 0.88))
                slip_pause = getattr(self._env, "_lift_slip_pause", None)
                if slip_pause is None:
                    slip_pause = torch.zeros_like(self._close_progress, dtype=torch.bool)
                cap = torch.full_like(self._close_progress, freeze_at if freeze_at > 0.0 else 1.0)
                cap = torch.where(slip_pause, creep_cap, cap)
                sym_ok = (
                    finger_symmetric_ready(self._env, s)
                    if getattr(self._env.cfg, "grasp_symmetry_gate_enabled", False)
                    else torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                )
                # Sau hết reopen: chỉ khép thêm khi sym OK — lệch pad → freeze gc
                exhaust_sym_close = (
                    exhausted
                    & sym_ok
                    & ~pad_bad
                    & (self._close_progress < cap)
                    & (s["bottle_tilt_deg"] < exhaust_tilt)
                    & getattr(self._env.cfg, "grasp_close_exhaust_creep_enabled", False)
                )
                reopen_tilt = float(getattr(self._env.cfg, "grasp_reopen_max_tilt_deg", 4.5))
                mid_gc = float(getattr(self._env.cfg, "grasp_reopen_pause_tilt_min_gc", 0.70))
                tilt_limit = torch.where(
                    self._close_progress >= mid_gc,
                    torch.full_like(self._close_progress, ramp_tilt_max),
                    torch.full_like(self._close_progress, reopen_tilt),
                )
                can_normal = (
                    ramping
                    & ~exhausted
                    & ~pad_bad
                    & (s["bottle_tilt_deg"] < tilt_limit)
                )
                exhaust_restart = (
                    exhausted
                    & sym_ok
                    & ~pad_bad
                    & (s["bottle_tilt_deg"] < ramp_tilt_max)
                    & (self._close_progress < cap)
                    & (self._close_progress < min_gc)
                )
                can_advance = can_normal | exhaust_sym_close | exhaust_restart
                # Luôn dừng ramp tại freeze_at — tránh khép quá → tilt terminate
                if freeze_at > 0.0:
                    can_advance = can_advance & (self._close_progress < cap)
                if getattr(self._env.cfg, "grasp_close_pause_on_tilt_rise", True):
                    tilt_rise = (s["bottle_tilt_deg"] > self._prev_close_tilt + 0.05) & (
                        self._close_progress > 0.65
                    )
                    can_advance = can_advance & ~tilt_rise
                self._prev_close_tilt = s["bottle_tilt_deg"].clone()
                if getattr(self._env.cfg, "grasp_close_freeze_on_reopen_exhaust", True):
                    freeze_cap = exhausted & (self._close_progress >= cap)
                    # Lệch sau hết reopen: khóa ngay (không ép khép thêm → tilt)
                    freeze_asym = exhausted & ~sym_ok & (self._close_progress >= min_gc)
                    freeze = freeze_cap | freeze_asym
                    can_advance = can_advance & ~freeze

                # Dừng ramp theo LỰC CHẠM THẬT, không chỉ theo tỉ lệ % cố định.
                # Đo trực tiếp (DEBUG_STALL): ramp cũ siết tới grasp_close_freeze_at_progress
                # bất kể đã chạm hay chưa — tại gc≈0.71 (span≈43mm, đúng đường kính
                # chai) stall (chênh khớp-so-với-lệnh, tỉ lệ thuận lực ép) đạt đỉnh
                # ~3.4mm, nhưng ramp tiếp tục siết sâu hơn (gc→0.90+) và ĐẨY VĂNG
                # chai ra khỏi kẹp — lúc đó stall tụt hẳn về 0 dù ngón đã "khép"
                # theo lệnh. Baseline nhiễu actuator (chưa chạm gì) đo được ~2.4mm,
                # nên cần ngưỡng rõ ràng cao hơn VÀ đếm nhiều bước liên tiếp (như
                # slip_steps) để không dừng oan vì nhiễu tức thời.
                near_bottle = s["dist_finger_body"] < float(
                    getattr(self._env.cfg, "grasp_press_max_dist_f", 0.060)
                )
                # Cổng hình học: span_xy phải THỰC SỰ ở gần đường kính chai mới
                # cho stall được tính là "đã chạm" — chỉ riêng stall dễ bắt
                # nhầm nhiễu actuator-lag (tăng dần theo gc kể cả CHƯA chạm gì)
                # thành lực chạm thật, đóng băng SỚM hơn hẳn điểm chạm hình học
                # đã đo được (gc≈0.71 ↔ span≈44mm=đường kính chai, xem DEBUG_LIFT
                # cho thấy các env kẹt ở gc≈0.62-0.65 — trước cả điểm chạm thật).
                bottle_d = float(getattr(self._env.cfg, "bottle_diameter_m", 0.0433))
                span_margin = float(getattr(self._env.cfg, "grasp_press_span_margin_m", 0.004))
                span_ok = s["finger_span_xy"] < (bottle_d + span_margin)

                # Phase 32: dừng ramp khi CẢ HAI ngón có LỰC TIẾP XÚC THẬT
                # (ContactSensor), thay vì stall trung bình 2 ngón (proxy vị
                # trí). Đo trực tiếp (terminal_command.md Phase 32): khi tiếp
                # cận lệch tâm nhẹ, 2 khớp đóng đối xứng hoàn hảo về GÓC nên
                # stall trung bình đạt ngưỡng và dừng ramp SỚM ngay khi CHỈ
                # MỘT ngón thực sự chạm (F=0.079N) — ngón kia hoàn toàn không
                # tiếp xúc (F=0.000N) và không bao giờ được đóng thêm để bù.
                #
                # Phase 34: yêu cầu CẢ HAI làm sập lift_start_rate 0.43→0.03
                # (regression gate seed=0) vì lệch tâm khiến ngón xa gần như
                # KHÔNG BAO GIỜ đạt lực thật dù đóng hết cỡ (đã thử nới cap
                # lên 1.0 ở Phase 31 — không cải thiện, là hình học). Thêm
                # nhánh OR: 1 ngón vượt ngưỡng CAO HƠN (chắc chắn, không phải
                # nhiễu) + near_bottle + span_ok cũng coi là đang ép — khôi
                # phục hành vi trước Phase 32 (mirror grasp_assist._grip_pressing).
                if "left_finger_contact_force" in s and "right_finger_contact_force" in s:
                    min_force = float(getattr(self._env.cfg, "grasp_press_min_force_n", 0.15))
                    min_force_single = float(
                        getattr(self._env.cfg, "grasp_press_min_force_single_n", 0.30)
                    )
                    left_f = s["left_finger_contact_force"]
                    right_f = s["right_finger_contact_force"]
                    both_pressing_f = (left_f > min_force) & (right_f > min_force)
                    single_pressing_f = torch.maximum(left_f, right_f) > min_force_single
                    pressing_now = near_bottle & span_ok & (both_pressing_f | single_pressing_f)
                    self._press_hold_steps[pressing_now] += 1
                    self._press_hold_steps[~pressing_now] = 0
                    hold_req = int(getattr(self._env.cfg, "grasp_press_freeze_hold_steps", 3))
                    firm_contact = self._press_hold_steps >= hold_req
                    import os as _os
                    if _os.environ.get("DEBUG_FREEZE") == "1" and self.num_envs <= 16:
                        i = 1
                        if bool(self._want_close[i]) or bool(self._grasp_latched[i]):
                            print(
                                f"  [FreezeDbg] env{i} gc={float(self._close_progress[i]):.4f} "
                                f"F_left={float(left_f[i]):.3f}N F_right={float(right_f[i]):.3f}N "
                                f"span={float(s['finger_span_xy'][i])*1000:.2f}mm "
                                f"near={bool(near_bottle[i])} span_ok={bool(span_ok[i])} "
                                f"both={bool(both_pressing_f[i])} single={bool(single_pressing_f[i])} "
                                f"pressing={bool(pressing_now[i])} hold={int(self._press_hold_steps[i])} "
                                f"firm={bool(firm_contact[i])}",
                                flush=True,
                            )
                else:
                    firm_contact = torch.zeros_like(can_advance)
                can_advance = can_advance & ~firm_contact

                # Chậm lại khi đã gần chai — griplag.py đo được ramp chậm hơn
                # giữ actuator đồng bộ tốt hơn nhiều (7mm lệch ở 150 bước so
                # với 52mm ở 60 bước), tránh overshoot-rồi-bật-chai-ra trước cả
                # khi kịp đóng băng theo lực.
                slow_factor = float(getattr(self._env.cfg, "grasp_press_slow_factor", 1.0))
                step_size = torch.where(
                    near_bottle,
                    torch.full_like(self._close_progress, slow_factor / float(ramp_steps)),
                    torch.full_like(self._close_progress, 1.0 / float(ramp_steps)),
                )
            else:
                step_size = torch.full_like(self._close_progress, 1.0 / float(ramp_steps))
            self._close_progress[can_advance] += step_size[can_advance]
            self._close_progress.clamp_(0.0, 1.0)
            stale = ~self._want_close & ~self._grasp_latched
            if s is not None:
                reopen_max = int(getattr(self._env.cfg, "grasp_reopen_max_count", 3))
                exhausted = self._reopen_count >= reopen_max
                stale = stale & ~exhausted
            self._close_progress[stale] = 0.0

            prog = self._close_progress.unsqueeze(-1)
            open_t = self._open_command.unsqueeze(0)
            close_t = self._close_command.unsqueeze(0)
            squeeze = float(getattr(self._env.cfg, "grasp_close_squeeze_m", 0.0))
            squeeze_start = float(getattr(self._env.cfg, "grasp_close_squeeze_start", 0.80))
            if squeeze > 0.0:
                late = (self._close_progress >= squeeze_start).unsqueeze(-1)
                close_t = torch.where(late, close_t - squeeze, close_t)
            targets = open_t * (1.0 - prog) + close_t * prog
            self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)
            return
        super().apply_actions()


@configclass
class AssistedBinaryGripperActionCfg(BinaryJointPositionActionCfg):
    class_type: type = AssistedBinaryGripperAction


@configclass
class AssistedOperationalSpaceControllerActionCfg(OperationalSpaceControllerActionCfg):
    class_type: type = AssistedOperationalSpaceControllerAction


class OpenArmActionTerm(ActionTerm):
    """
    Custom 8-D action term for the OpenArm A1 robot arm.
    Controls 7 arm joint position deltas and 1 joint position target for the gripper.
    Conforms to the full abstract interface of Isaac Lab ActionTerm.
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._robot = env.scene["robot"]
        self._arm_joint_ids, _ = self._robot.find_joints([
            "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
            "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
            "openarm_left_joint7"
        ])
        self._gripper_joint_ids, _ = self._robot.find_joints([
            "openarm_left_finger_joint1", "openarm_left_finger_joint2"
        ])
        
        # Pre-initialize states
        self._raw_actions = torch.zeros(env.num_envs, 8, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 8, device=env.device)
        self.gripper_state = torch.zeros(env.num_envs, 1, device=env.device)

    @property
    def action_dim(self) -> int:
        return 8

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions.clone().clamp(-1.0, 1.0)
        self._processed_actions = self._raw_actions
        
        # Scale gripper command: gripper_cmd in [-1.0, 1.0] -> gripper_state in [0.0, 1.0] (0=open, 1=closed)
        gripper_cmd = self._raw_actions[:, 7:8]
        self.gripper_state = (gripper_cmd + 1.0) / 2.0

    def apply_actions(self):
        # Calculate and write arm target positions
        current_q = self._robot.data.joint_pos[:, self._arm_joint_ids]
        
        # Get control time step (decimation * sim_dt = 2 * (1/120) = 1/60s)
        dt = self._env.cfg.sim.dt * self._env.cfg.decimation
        
        # Velocity limits from joint_limits.yaml:
        # joint 1-4: 2.175 rad/s -> max step delta = 2.175 * dt
        # joint 5-7: 2.610 rad/s -> max step delta = 2.610 * dt
        max_vel = torch.tensor([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610], device=self._robot.device)
        max_delta = max_vel * dt

        # Phase 1 fine control: reduce arm step size when TCP is near the bottle
        arm_scale = torch.ones(self._env.num_envs, 1, device=self._robot.device)
        if hasattr(self._env, "_last_state") and self._env._last_state is not None:
            dist = self._env._last_state["dist_ee_bottle"]
            fine_dist = getattr(self._env.cfg, "fine_control_dist", 0.10)
            fine_scale = getattr(self._env.cfg, "fine_control_scale", 0.3)
            arm_scale[dist < fine_dist] = fine_scale

        delta_q = self._processed_actions[:, :7] * max_delta * arm_scale
        target_q = current_q + delta_q
        self._robot.set_joint_position_target(target_q, joint_ids=self._arm_joint_ids)

        # Calculate and write gripper target positions based on gripper_state
        # Open (gripper_state = 0) -> finger_pos = 0.044
        # Closed (gripper_state = 1) -> finger_pos = 0.0
        finger_pos = (1.0 - self.gripper_state) * 0.044
        finger_targets = finger_pos.repeat(1, 2)
        self._robot.set_joint_position_target(finger_targets, joint_ids=self._gripper_joint_ids)


@configclass
class OpenArmActionTermCfg(ActionTermCfg):
    """Configuration class for OpenArmActionTerm."""
    class_type: type = OpenArmActionTerm
    asset_name: str = "robot"
