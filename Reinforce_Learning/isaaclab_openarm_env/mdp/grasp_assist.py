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

"""Phase 2 action assists — optional bootstrap (training curriculum or demo)."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply

from .helpers import (
    STAGE_GRASP,
    STAGE_REACH,
    finger_descended_for_close,
    finger_grasp_ready,
    finger_ready_for_close,
    finger_symmetric_ready,
    finger_pad_asymmetric,
    grip_close_ramp_active,
    reach_align_ready,
    reach_descent_ready,
    uses_grasp_lift,
)


def _assist_scale(env: ManagerBasedRLEnv) -> float:
    return float(getattr(env, "_assist_blend_scale", 1.0))


def _grip_latched(env: ManagerBasedRLEnv) -> torch.Tensor:
    term = env.action_manager._terms.get("gripper_action")
    if term is not None and hasattr(term, "_grasp_latched"):
        return term._grasp_latched
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _grip_close_done(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Gripper đã khép xong — khi ramp: chờ progress, không tin lệnh binary."""
    term = env.action_manager._terms.get("gripper_action")
    ramp_steps = int(getattr(env.cfg, "grasp_close_ramp_steps", 0))
    if ramp_steps > 0 and term is not None and hasattr(term, "_close_progress"):
        done_gc = float(getattr(env.cfg, "grasp_close_done_progress", 0.96))
        return term._close_progress >= done_gc
    grip_thresh = getattr(env.cfg, "grasp_grip_threshold", 0.4)
    return s["gripper_state"] > grip_thresh


def _grip_partial_lift_ok(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Mimic kẹt gc~75% sau hết reopen — cho nhấc sớm nếu tilt còn thấp."""
    if not getattr(env.cfg, "grasp_lift_partial_enabled", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    term = env.action_manager._terms.get("gripper_action")
    if term is None or not hasattr(term, "_close_progress"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    min_gc = float(getattr(env.cfg, "grasp_lift_partial_min_gc", 0.68))
    max_tilt = float(getattr(env.cfg, "grasp_lift_partial_max_tilt_deg", 5.0))
    reopen_max = int(getattr(env.cfg, "grasp_reopen_max_count", 3))
    grip_thresh = getattr(env.cfg, "grasp_grip_threshold", 0.4)
    exhausted = term._reopen_count >= reopen_max
    return (
        exhausted
        & (term._close_progress >= min_gc)
        & (s["bottle_tilt_deg"] < max_tilt)
        & (s["gripper_state"] > grip_thresh)
        & _grip_latched(env)
    )


def _grip_committed(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Đã latch + gripper khép xong (ramp 99% — không tin gripper_state sớm)."""
    latched = _grip_latched(env)
    done = _grip_close_done(env, s)
    partial = _grip_partial_lift_ok(env, s)
    ramp_steps = int(getattr(env.cfg, "grasp_close_ramp_steps", 0))
    if ramp_steps > 0:
        return latched & (done | partial)
    grip_thresh = getattr(env.cfg, "grasp_grip_threshold", 0.4)
    return latched & (done | partial | (s["gripper_state"] > grip_thresh))


def _grasp_stable_for_lift(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Chai thẳng + ngón còn ôm sát thân trước khi nhấc."""
    max_tilt = getattr(env.cfg, "grasp_lift_start_max_tilt_deg", 8.0)
    # Dùng cùng biên grasp_close — lat_f ~0.077 khi kẹp ổn là bình thường
    max_lat = getattr(
        env.cfg, "grasp_lift_max_lat_f",
        getattr(env.cfg, "grasp_close_max_lat", 0.08) + 0.01,
    )
    max_dist_f = getattr(
        env.cfg, "grasp_lift_max_dist_f",
        getattr(env.cfg, "grasp_success_max_dist_finger", 0.10),
    )
    partial = _grip_partial_lift_ok(env, s)
    sym_ok = (
        finger_symmetric_ready(env, s)
        if getattr(env.cfg, "grasp_symmetry_gate_enabled", False)
        else torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    )
    sym_ok = sym_ok | partial
    max_z_f = float(getattr(env.cfg, "grasp_lift_max_z_finger", 0.030))
    max_dist_f_eff = max_dist_f
    max_tilt_eff = max_tilt
    if partial.any():
        max_z_f = float(getattr(env.cfg, "grasp_lift_partial_max_z_finger", 0.045))
        max_dist_f_eff = float(getattr(env.cfg, "grasp_lift_partial_max_dist_f", 0.075))
        max_tilt_eff = float(getattr(env.cfg, "grasp_lift_partial_max_tilt_deg", 8.0))
    return (
        (s["bottle_tilt_deg"] < max_tilt_eff)
        & (s["lateral_finger_xy"] < max_lat)
        & (s["dist_finger_body"] < max_dist_f_eff)
        & (s["z_error_finger"] < max_z_f)
        & sym_ok
    )


def _update_lift_settle(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Đếm bước ổn định sau khi khép xong — reset nếu chai nghiêng / trượt."""
    if not hasattr(env, "_lift_settle_steps"):
        env._lift_settle_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    in_grasp = env._stage == STAGE_GRASP
    ok = _grip_committed(env, s) & _grasp_stable_for_lift(env, s) & in_grasp
    env._lift_settle_steps[ok] += 1
    env._lift_settle_steps[~ok] = 0
    return env._lift_settle_steps


def _lift_ready(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Sẵn sàng nhấc: đã khép + ổn định đủ N bước."""
    partial = _grip_partial_lift_ok(env, s)
    settle_req = int(getattr(env.cfg, "grasp_lift_settle_steps", 8))
    if partial.any():
        settle_req = int(getattr(env.cfg, "grasp_lift_partial_settle_steps", 1))
    steps = _update_lift_settle(env, s)
    return _grip_committed(env, s) & _grasp_stable_for_lift(env, s) & (steps >= settle_req)


def _osc_tool_z_descend(
    env: ManagerBasedRLEnv, s: dict, mask: torch.Tensor, arm_actions: torch.Tensor, scale: float
) -> torch.Tensor:
    """OSC 6-D: hạ dọc tool-Z (chỉ khi top-down đủ thẳng)."""
    pos_scale = float(getattr(env.cfg, "osc_position_scale", 0.06))
    descend_m = float(getattr(env.cfg, "grasp_descent_world_m", 0.010))
    mag = descend_m / max(pos_scale, 1e-4) * scale
    tool_z = s["tool_z_w"]
    vert = tool_z[mask, 2].abs().clamp(min=0.35, max=1.0)
    z_cmd = (mag / vert).clamp(0.0, 1.0)
    # Hạ world-Z: nếu tool_z[2]>0 thì đi dọc -tool_z (action[2] âm)
    sign = torch.where(tool_z[mask, 2] >= 0.0, -1.0, 1.0)
    out = arm_actions.clone()
    out[mask] = 0.0
    out[mask, 2] = sign * z_cmd
    return out


def _osc_z_finger_descend(
    env: ManagerBasedRLEnv, s: dict, mask: torch.Tensor, arm_actions: torch.Tensor, scale: float
) -> torch.Tensor:
    """OSC: hạ theo z_error_finger (closed-loop), chỉ tool-Z — ít trượt ngang."""
    pos_scale = float(getattr(env.cfg, "osc_position_scale", 0.06))
    descend_m = float(getattr(env.cfg, "grasp_descent_world_m", 0.010))
    z_err = s["z_error_finger"][mask].clamp(min=0.0)
    step_m = torch.minimum(z_err, torch.full_like(z_err, descend_m * scale))
    tz = s["tool_z_w"][mask, 2]
    vert = tz.abs().clamp(min=0.25)
    az = torch.sign(tz) * (step_m / (vert * max(pos_scale, 1e-4))).clamp(0.0, 1.0)
    out = arm_actions.clone()
    out[mask] = 0.0
    out[mask, 2] = az
    return out


def _osc_world_down_descend(
    env: ManagerBasedRLEnv, s: dict, mask: torch.Tensor, arm_actions: torch.Tensor, scale: float
) -> torch.Tensor:
    """OSC pose_rel: hạ thẳng world -Z.

    Isaac Lab `pose_rel` cộng delta vị trí TRỰC TIẾP trong task frame (mặc định
    = base frame, thẳng đứng) — apply_delta_pose: target = source + delta.
    Bản cũ giải hệ theo trục tool (nghiêng) nên lệnh bị chiếu sai hướng.

    Nếu grasp_descent_xy_blend > 0: thêm thành phần XY kéo ngón về tâm grasp
    trong lúc hạ — tránh khép lệch tâm làm chai bị lật thay vì nhấc lên.
    """
    pos_scale = float(getattr(env.cfg, "osc_position_scale", 0.06))
    descend_m = float(getattr(env.cfg, "grasp_descent_world_m", 0.010))
    mag = min(descend_m / max(pos_scale, 1e-4) * scale, 1.0)
    out = arm_actions.clone()
    out[mask] = 0.0
    # stop sinking below the close height — keep only XY centering + align there
    z_stop = float(getattr(
        env.cfg, "grasp_descent_z_target",
        getattr(env.cfg, "grasp_close_max_z_err", 0.012),
    ))
    z_go = (s["z_error_finger"] > z_stop).float()
    out[mask, 2] = -mag * z_go[mask]
    xy_blend = float(getattr(env.cfg, "grasp_descent_xy_blend", 0.0))
    if xy_blend > 0.0:
        xy_err = s["grasp_pos_w"][:, :2] - s["finger_pos_w"][:, :2]
        xy_step = (xy_err / max(pos_scale, 1e-4)).clamp(-mag, mag) * xy_blend
        out[mask, 0:2] = xy_step[mask]
    align_blend = float(getattr(env.cfg, "grasp_descent_align_blend", 0.0))
    if align_blend > 0.0:
        # Tool tilt dominates finger lateral error (proxy tip = hand − tool_z·reach):
        # rotate tool_z toward vertical while descending so the pinch centers.
        out[mask, 3:6] = _align_vertical_rot_action(env, s)[mask] * align_blend
    return out


def _align_vertical_rot_action(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Axis-angle (base frame) đưa tool_z về thẳng đứng, đã chia orientation_scale."""
    ori_scale = float(getattr(env.cfg, "osc_orientation_scale", 0.20))
    tool_z = s["tool_z_w"]
    target = torch.zeros_like(tool_z)
    # keep the current hemisphere (URDF may point tool z up or down)
    target[:, 2] = torch.where(tool_z[:, 2] < 0, -torch.ones_like(tool_z[:, 2]), torch.ones_like(tool_z[:, 2]))
    axis = torch.cross(tool_z, target, dim=-1)
    sin_a = axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    cos_a = (tool_z * target).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.atan2(sin_a, cos_a)
    delta = axis / sin_a * angle
    return (delta / max(ori_scale, 1e-4)).clamp(-1.0, 1.0)


def _osc_world_up_lift(
    env: ManagerBasedRLEnv, s: dict, mask: torch.Tensor, arm_actions: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """OSC pose_rel: nhấc thẳng world +Z (delta vị trí trong base frame)."""
    pos_scale = float(getattr(env.cfg, "osc_position_scale", 0.06))
    lift_m = float(getattr(env.cfg, "grasp_lift_world_m", 0.012))
    lift_m_p = float(getattr(env.cfg, "grasp_lift_partial_world_m", lift_m))
    partial = _grip_partial_lift_ok(env, s)
    lift_vec = torch.full((env.num_envs,), lift_m, device=env.device)
    lift_vec = torch.where(partial, torch.full_like(lift_vec, lift_m_p), lift_vec)
    mag = (lift_vec / max(pos_scale, 1e-4) * scale).clamp(max=1.0)
    out = arm_actions.clone()
    out[mask] = 0.0
    out[mask, 2] = mag[mask]
    return out


def _pad_balance_xy_action(env: ManagerBasedRLEnv, s: dict, step_cap: float) -> torch.Tensor:
    """OSC XY nudge: kéo pad xa hơn về phía thân chai (cân L/R trước khi đóng)."""
    body = s["grasp_body_pos_w"][:, :2]
    left = s["left_finger_pos_w"][:, :2]
    right = s["right_finger_pos_w"][:, :2]
    dl = s["dist_left_body"]
    dr = s["dist_right_body"]
    pull_left = body - left
    pull_right = body - right
    pull_left = pull_left / (pull_left.norm(dim=-1, keepdim=True) + 1e-6)
    pull_right = pull_right / (pull_right.norm(dim=-1, keepdim=True) + 1e-6)
    dl_greater = (dl > dr).unsqueeze(-1)
    pull = torch.where(dl_greater, pull_left, pull_right)
    pos_scale = float(getattr(env.cfg, "osc_position_scale", 0.06))
    mag = min(step_cap, float(getattr(env.cfg, "grasp_descent_world_m", 0.010)) / max(pos_scale, 1e-4))
    return pull.clamp(-1.0, 1.0) * mag


def _assist_enabled(env: ManagerBasedRLEnv) -> bool:
    if not uses_grasp_lift(getattr(env.cfg, "task_phase", 1)):
        return False
    if not hasattr(env, "_stage"):
        return False
    assist_reach = getattr(env.cfg, "grasp_descent_assist_enabled", False)
    assist_lift = getattr(env.cfg, "grasp_lift_assist_enabled", False)
    assist_grasp = assist_reach and getattr(env.cfg, "grasp_descent_assist_grasp", False)
    return assist_reach or assist_grasp or assist_lift


def apply_grasp_arm_assist(env: ManagerBasedRLEnv, arm_actions: torch.Tensor) -> torch.Tensor:
    """OSC 6-D assist — gọi từ AssistedOSCActionTerm.process_actions."""
    env._assist_want_lift = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._assist_want_descend = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not _assist_enabled(env):
        return arm_actions

    s = getattr(env, "_last_state", None)
    if s is None:
        return arm_actions

    actions = arm_actions.clone()
    scale = _assist_scale(env)
    if scale <= 1e-6:
        return actions

    assist_reach = getattr(env.cfg, "grasp_descent_assist_enabled", False)
    assist_grasp = assist_reach and getattr(env.cfg, "grasp_descent_assist_grasp", False)
    assist_lift = getattr(env.cfg, "grasp_lift_assist_enabled", False)

    in_grasp = env._stage == STAGE_GRASP
    in_reach = env._stage == STAGE_REACH
    abort_tilt = float(getattr(env.cfg, "grasp_grasp_abort_tilt_deg", 18.0))
    grasp_aborted = in_grasp & (s["bottle_tilt_deg"] > abort_tilt)
    if grasp_aborted.any():
        # Chai đã ngã — dừng assist (tránh reopen/descend/lift làm tệ hơn)
        actions[grasp_aborted] = 0.0
        return actions
    aligned = finger_grasp_ready(env, s)
    sym_ok = (
        finger_symmetric_ready(env, s)
        if getattr(env.cfg, "grasp_symmetry_gate_enabled", False)
        else torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    )
    latched = _grip_latched(env)
    close_ramping = grip_close_ramp_active(env)
    partial_active = _grip_partial_lift_ok(env, s)
    term = env.action_manager._terms.get("gripper_action")
    reopen_max = int(getattr(env.cfg, "grasp_reopen_max_count", 3))
    partial_min_gc = float(getattr(env.cfg, "grasp_lift_partial_min_gc", 0.55))
    reopen_done = (
        term is not None
        and hasattr(term, "_reopen_count")
        and (term._reopen_count >= reopen_max)
        and (term._close_progress >= partial_min_gc)
    )
    lift_thresh = getattr(env.cfg, "grasp_lift_threshold", 0.03)
    lift_ready = _lift_ready(env, s)
    top_down = s["top_down_align"]

    lift_cmd = float(getattr(env.cfg, "grasp_lift_action", -0.45))
    descend_cmd = float(getattr(env.cfg, "grasp_descent_action", 0.3))

    z_stop = getattr(
        env.cfg,
        "grasp_descent_z_target",
        getattr(env.cfg, "grasp_close_max_z_err", 0.012),
    )
    z_high = s["z_error_finger"] > z_stop
    top_ok = top_down >= getattr(env.cfg, "grasp_descent_min_top_down", 0.50)

    need_down = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if assist_reach and getattr(env.cfg, "grasp_descent_in_reach", True):
        reach_hold = getattr(env.cfg, "reach_stage_hold_steps", 5)
        in_contact = getattr(
            env, "_in_contact_zone",
            torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        )
        hold_ok = env._steps_in_contact >= reach_hold
        descent_ready = reach_descent_ready(env, s) & in_contact & hold_ok
        need_down = need_down | (in_reach & descent_ready & z_high & top_ok)
        # Hạ sớm khi đã chạm nhưng top↓ chưa đủ 0.65 — tránh policy rung ~step 200
        min_top_adv = float(getattr(env.cfg, "reach_advance_min_top_down", 0.65))
        early_top = (top_down >= float(getattr(env.cfg, "grasp_descent_min_top_down", 0.50))) & (
            top_down < min_top_adv
        )
        early_z = s["z_error_finger"] > float(getattr(env.cfg, "reach_advance_max_z_finger", 0.072)) * 0.65
        need_down = need_down | (in_reach & in_contact & hold_ok & early_top & early_z)
        # Giảm dao động policy khi hover chờ gate (hình ảnh rung quanh chai)
        reach_hover = in_reach & in_contact & hold_ok & ~reach_align_ready(env, s, soft=False) & ~need_down
        if reach_hover.any():
            damp = float(getattr(env.cfg, "grasp_grip_arm_damp", 0.12))
            actions[reach_hover] *= damp

    if assist_grasp:
        descended = finger_descended_for_close(env, s)
        min_top_grasp = float(getattr(
            env.cfg, "grasp_descent_min_top_down_grasp",
            getattr(env.cfg, "grasp_descent_min_top_down", 0.50) - 0.05,
        ))
        top_ok_grasp = top_down >= min_top_grasp
        need_down = need_down | (in_grasp & ~descended & top_ok_grasp & ~lift_ready & ~close_ramping)
        max_tilt_descend = float(getattr(env.cfg, "grasp_descent_max_bottle_tilt_deg", 15.0))
        need_down = need_down & (s["bottle_tilt_deg"] < max_tilt_descend)
        # Sau hết reopen: không hạ thêm — tránh tilt tăng trước partial lift
        need_down = need_down & ~reopen_done & ~partial_active
        # Top hơi thấp nhưng vẫn cần hạ: closed-loop z_finger (ít trôi ngang hơn world-down)
        low_top = in_grasp & ~descended & ~top_ok_grasp & (top_down >= 0.40) & ~lift_ready
        if low_top.any():
            actions = _osc_z_finger_descend(env, s, low_top, actions, scale * 0.75)
            actions[low_top, 3:6] = 0.0
        # Đã hạ đủ thấp nhưng chưa align (lat/top gate chặn close): tiếp tục căn
        # giữa XY + dựng tool thẳng tại chỗ — không có nhánh này ngón kẹt lơ lửng
        # ở lat_f ~0.02 vĩnh viễn (descent tắt vì z đạt, close không bao giờ bắn).
        center_only = in_grasp & descended & ((~aligned) | (~sym_ok)) & ~latched & ~lift_ready
        # Đã khép lệch (latched hoặc grip cao): căn XY — reopen xử lý ở actions.py
        grip_thresh = getattr(env.cfg, "grasp_grip_threshold", 0.4)
        gripped = s["gripper_state"] > grip_thresh
        recover_asym = (
            in_grasp
            & (latched | gripped)
            & ~sym_ok
            & (s["bottle_lift"] < lift_thresh)
            & ~lift_ready
        )
        center_mask = center_only | recover_asym
        if center_mask.any():
            pos_scale = float(getattr(env.cfg, "osc_position_scale", 0.06))
            step_cap = float(getattr(env.cfg, "grasp_descent_world_m", 0.010)) / max(pos_scale, 1e-4)
            xy_err = s["grasp_pos_w"][:, :2] - s["finger_pos_w"][:, :2]
            xy_center = (xy_err / max(pos_scale, 1e-4)).clamp(-step_cap, step_cap)
            bal_blend = float(getattr(env.cfg, "grasp_sym_xy_balance_blend", 0.55))
            if getattr(env.cfg, "grasp_symmetry_gate_enabled", False) and (~sym_ok).any():
                xy_balance = _pad_balance_xy_action(env, s, step_cap)
                xy_step = xy_center * (1.0 - bal_blend) + xy_balance * bal_blend
            else:
                xy_step = xy_center
            actions[center_mask, 0:2] = xy_step[center_mask] * scale
            actions[center_mask, 2] = 0.0
            align_blend = float(getattr(env.cfg, "grasp_descent_align_blend", 0.0))
            if align_blend > 0.0:
                actions[center_mask, 3:6] = (
                    _align_vertical_rot_action(env, s)[center_mask] * align_blend * scale
                )
            else:
                actions[center_mask, 3:6] = 0.0

    # Freeze chỉ khi sym OK — nếu khép lệch, pad-balance branch phía trên vẫn chạy
    freeze_arm = getattr(env.cfg, "grasp_pregrasp_freeze_arm", True)
    not_sinkable = latched & ~finger_descended_for_close(env, s)
    sym_for_freeze = sym_ok if getattr(env.cfg, "grasp_symmetry_gate_enabled", False) else torch.ones(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    hold_pose = (
        in_grasp
        & (s["bottle_lift"] < lift_thresh)
        & ~lift_ready
        & freeze_arm
        & ~need_down
        & (aligned | latched)
        & sym_for_freeze
        & ~not_sinkable
    )
    # Sau latch: để lift assist điều khiển, không zero actions trước đó
    if getattr(env.cfg, "grasp_lift_assist_enabled", False):
        hold_pose = hold_pose & ~latched
    if hold_pose.any():
        actions[hold_pose] = 0.0

    if need_down.any():
        env._assist_want_descend = need_down.clone()
        force_reach = getattr(
            env.cfg, "grasp_descent_force_in_reach",
            getattr(env.cfg, "grasp_descent_force_override", False),
        )
        force_grasp = getattr(
            env.cfg, "grasp_descent_force_in_grasp",
            getattr(env.cfg, "grasp_descent_force_override", False),
        )
        in_reach = env._stage == STAGE_REACH
        reach_mask = need_down & in_reach
        grasp_mask = need_down & ~in_reach
        world_down = getattr(env.cfg, "grasp_descent_world_down", True)

        def _apply_force(mask: torch.Tensor, use_world_down: bool | None = None) -> None:
            nonlocal actions
            if not mask.any():
                return
            wd = world_down if use_world_down is None else use_world_down
            if wd:
                actions = _osc_world_down_descend(env, s, mask, actions, scale)
            else:
                actions = _osc_tool_z_descend(env, s, mask, actions, scale)

        def _apply_blend_world_z(mask: torch.Tensor, blend_val: float) -> None:
            """Giữ policy XY, blend chỉ trục hạ world-Z — tránh giật toàn pose."""
            nonlocal actions
            if not mask.any():
                return
            forced = actions.clone()
            forced = _osc_world_down_descend(env, s, mask, forced, scale)
            b = blend_val * scale
            actions[mask, 2] = (1.0 - b) * actions[mask, 2] + b * forced[mask, 2]

        def _apply_blend(mask: torch.Tensor, blend_val: float) -> None:
            if not mask.any():
                return
            b = blend_val * scale
            cur = actions[mask, 2]
            actions[mask, 2] = (1.0 - b) * cur + b * descend_cmd

        if reach_mask.any():
            z_loop = getattr(env.cfg, "grasp_descent_z_finger_loop", True)
            if z_loop:
                # Closed-loop z_f trong REACH — world-down + align làm top↑0.79 mà z_f vẫn ~0.06
                actions = _osc_z_finger_descend(env, s, reach_mask, actions, scale)
                pos_scale = float(getattr(env.cfg, "osc_position_scale", 0.06))
                descend_m = float(getattr(env.cfg, "grasp_descent_world_m", 0.010))
                mag = min(descend_m / max(pos_scale, 1e-4) * scale, 1.0)
                xy_blend = float(getattr(env.cfg, "grasp_descent_xy_blend", 0.0))
                if xy_blend > 0.0:
                    xy_err = s["grasp_pos_w"][:, :2] - s["finger_pos_w"][:, :2]
                    xy_step = (xy_err / max(pos_scale, 1e-4)).clamp(-mag, mag) * xy_blend
                    actions[reach_mask, 0:2] = xy_step[reach_mask]
                align_blend = float(getattr(env.cfg, "grasp_descent_align_blend", 0.0))
                if align_blend > 0.0:
                    # Nhẹ hơn GRASP — tránh xoay quá thẳng trước khi vào GRASP
                    actions[reach_mask, 3:6] = (
                        _align_vertical_rot_action(env, s)[reach_mask] * align_blend * 0.35 * scale
                    )
            elif force_reach:
                _apply_force(reach_mask)
            else:
                blend_reach = float(getattr(env.cfg, "grasp_descent_blend", 0.5))
                if getattr(env.cfg, "grasp_descent_world_down", True):
                    _apply_blend_world_z(reach_mask, blend_reach)
                else:
                    _apply_blend(reach_mask, blend_reach)
        if grasp_mask.any():
            if force_grasp:
                z_loop = getattr(env.cfg, "grasp_descent_z_finger_loop", True)
                if z_loop:
                    latched_m = grasp_mask & _grip_latched(env) & ~close_ramping
                    fresh_m = grasp_mask & ~_grip_latched(env) & ~close_ramping
                    if fresh_m.any():
                        actions = _osc_z_finger_descend(env, s, fresh_m, actions, scale)
                    if latched_m.any():
                        actions = _osc_z_finger_descend(env, s, latched_m, actions, scale * 0.45)
                    # GRASP z-loop trước đây chỉ hạ Z → pad phải lệch (R d~0.073, L d~0.028)
                    pos_scale = float(getattr(env.cfg, "osc_position_scale", 0.06))
                    descend_m = float(getattr(env.cfg, "grasp_descent_world_m", 0.010))
                    mag = min(descend_m / max(pos_scale, 1e-4) * scale, 1.0)
                    xy_blend = float(getattr(env.cfg, "grasp_descent_xy_blend", 0.0))
                    if xy_blend > 0.0:
                        xy_err = s["grasp_pos_w"][:, :2] - s["finger_pos_w"][:, :2]
                        xy_step = (xy_err / max(pos_scale, 1e-4)).clamp(-mag, mag) * xy_blend
                        actions[grasp_mask, 0:2] = xy_step[grasp_mask]
                    actions[grasp_mask, 3:6] = 0.0
                else:
                    _apply_force(grasp_mask, use_world_down=True)
                    # keep vertical-align rotation only BEFORE the gripper latches —
                    # wrenching the wrist while holding the bottle knocks it over
                    if float(getattr(env.cfg, "grasp_descent_align_blend", 0.0)) <= 0.0:
                        actions[grasp_mask, 3:6] = 0.0
                    else:
                        zero_rot = grasp_mask & _grip_latched(env)
                        actions[zero_rot, 3:6] = 0.0
                    # Gentle post-latch sink — full world-down step slips the pinch
                    latched_sink = grasp_mask & _grip_latched(env)
                    if latched_sink.any():
                        actions[latched_sink, 2] *= 0.35
            else:
                blend_grasp = float(getattr(env.cfg, "grasp_descent_grasp_blend", 0.35))
                _apply_blend(grasp_mask, blend_grasp)

    want_lift = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if assist_lift:
        grip_ok = _grip_committed(env, s)
        want_lift = lift_ready & in_grasp & grip_ok & (s["bottle_lift"] < lift_thresh)
        pre_hold = int(getattr(env.cfg, "grasp_pre_lift_hold_steps", 0))
        if not hasattr(env, "_pre_lift_hold_steps"):
            env._pre_lift_hold_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._pre_lift_hold_steps[want_lift] += 1
        env._pre_lift_hold_steps[~want_lift] = 0
        actually_lift = want_lift if pre_hold <= 0 else (want_lift & (env._pre_lift_hold_steps >= pre_hold))
        env._assist_want_lift = actually_lift.clone()
        if actually_lift.any():
            if getattr(env.cfg, "grasp_lift_world_up", True):
                actions = _osc_world_up_lift(env, s, actually_lift, actions, scale)
                # Gentle straightening only — full-rate align here wrenches the
                # held bottle against the table and knocks it over instantly.
                lift_align = float(getattr(env.cfg, "grasp_lift_align_blend", 0.0))
                if lift_align > 0.0:
                    rot = _align_vertical_rot_action(env, s)[actually_lift]
                    actions[actually_lift, 3:6] = rot.clamp(-0.25, 0.25) * lift_align
                else:
                    actions[actually_lift, 3:6] = 0.0
            else:
                tool_z_w = s["tool_z_w"]
                vert = tool_z_w[actually_lift, 2].abs().clamp(min=0.35, max=1.0)
                scaled_lift = (lift_cmd / vert).clamp(-1.0, 0.0)
                blend_lift = float(getattr(env.cfg, "grasp_lift_assist_blend", 1.0)) * scale
                if blend_lift >= 0.99:
                    actions[actually_lift, 2] = scaled_lift
                else:
                    cur = actions[actually_lift, 2]
                    actions[actually_lift, 2] = (1.0 - blend_lift) * cur + blend_lift * scaled_lift
        pre_only = want_lift & ~actually_lift
        if pre_only.any():
            actions[pre_only] = 0.0

    # Đã latch nhưng chưa lift: giữ pose — không freeze khi pad lệch (cần căn/recover)
    latched_hold = (
        latched & in_grasp & (s["bottle_lift"] < lift_thresh) & ~want_lift & ~need_down & sym_ok
    )
    if latched_hold.any():
        actions[latched_hold] = 0.0

    # GRASP chưa align: giảm drift policy (tránh trôi ngang khi không ép hạ)
    drift_damp = in_grasp & ~aligned & ~latched & ~lift_ready & ~need_down
    if drift_damp.any():
        damp = float(getattr(env.cfg, "grasp_grip_arm_damp", 0.05))
        actions[drift_damp] *= damp

    return actions


def apply_grasp_action_assist(env: ManagerBasedRLEnv, action: torch.Tensor) -> torch.Tensor:
    """Full 7-D wrapper — chỉnh slice arm trước process_action (backup path)."""
    if not _assist_enabled(env):
        return action
    if action.shape[-1] < 7:
        return action

    squeeze = False
    if not isinstance(action, torch.Tensor):
        action = torch.as_tensor(action, device=env.device, dtype=torch.float32)
    else:
        action = action.clone()
    if action.ndim == 1:
        action = action.unsqueeze(0)
        squeeze = True

    action[:, :6] = apply_grasp_arm_assist(env, action[:, :6])
    return action.squeeze(0) if squeeze else action
