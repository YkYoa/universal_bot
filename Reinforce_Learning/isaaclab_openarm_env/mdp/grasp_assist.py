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
import os
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


# Lift state machine (xem _update_lift_state). Lift là sub-mode của STAGE_GRASP,
# không phải một stage riêng — env._stage vẫn chỉ đi REACH → GRASP.
LIFT_IDLE = 0      # chưa đủ điều kiện, hoặc vừa bị hủy
LIFT_RISING = 1    # đang phát lệnh nhấc lên, KHÔNG đánh giá lại điều kiện khởi động
LIFT_HOLDING = 2   # đã vượt ngưỡng — giữ yên cho chai ổn định để tính thành công


def _assist_scale(env: ManagerBasedRLEnv) -> float:
    return float(getattr(env, "_assist_blend_scale", 1.0))


def _grip_latched(env: ManagerBasedRLEnv) -> torch.Tensor:
    term = env.action_manager._terms.get("gripper_action")
    if term is not None and hasattr(term, "_grasp_latched"):
        return term._grasp_latched
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _grip_reopen_exhausted(env: ManagerBasedRLEnv) -> torch.Tensor:
    term = env.action_manager._terms.get("gripper_action")
    if term is None or not hasattr(term, "_reopen_count"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reopen_max = int(getattr(env.cfg, "grasp_reopen_max_count", 3))
    return term._reopen_count >= reopen_max


def _grip_physically_closed(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Ngón thực sự khép — span/joint, không tin lệnh binary."""
    max_span = float(getattr(env.cfg, "grasp_phys_close_max_span", 0.045))
    span_ok = s["finger_span_xy"] < max_span
    if hasattr(env, "_gripper_joint_ids"):
        finger_pos = env._robot.data.joint_pos[:, env._gripper_joint_ids].mean(dim=1)
        joint_ok = finger_pos < 0.044 * float(getattr(env.cfg, "grasp_phys_close_max_joint_ratio", 0.35))
        return span_ok | joint_ok
    return span_ok


def _grip_pressing(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Ngón đang ÉP vào vật — bằng chứng thật sự của việc đang giữ chai.

    ``finger_span_xy`` được SUY RA từ góc khớp, nên khi khớp bị vật chặn lại thì
    span vẫn báo "đang mở" dù kẹp đang ép rất mạnh. Đo được: khớp kẹt ở 0.0231
    trong khi lệnh 0.0105 → chênh 12.6mm × stiffness 1500 = 18.9N ép vào chai,
    mà span lại báo 0.0583 (rộng hơn cả chai 0.0433).

    Độ chênh khớp-so-với-lệnh là tín hiệu đúng: kẹp không khí thì chênh ~1.5mm,
    kẹp chai thì ~12.6mm.
    """
    term = env.action_manager._terms.get("gripper_action")
    if term is None or not hasattr(term, "_close_progress") or not hasattr(env, "_gripper_joint_ids"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    joint = env._robot.data.joint_pos[:, env._gripper_joint_ids].mean(dim=1)
    open_m = float(getattr(env.cfg, "gripper_open_m", 0.044))
    target = open_m * (1.0 - term._close_progress)
    stall = joint - target
    min_stall = float(getattr(env.cfg, "grasp_press_min_stall_m", 0.005))
    near_bottle = s["dist_finger_body"] < float(getattr(env.cfg, "grasp_press_max_dist_f", 0.060))
    if os.environ.get("DEBUG_STALL") == "1" and env.num_envs <= 16 and int(env.step_counter) % 15 == 0:
        latched = _grip_latched(env)
        for i in latched.nonzero(as_tuple=False).flatten().tolist():
            print(
                f"  [Stall] env{i} step_ct={int(env.step_counter)} gc={float(term._close_progress[i]):.3f} "
                f"joint={float(joint[i])*1000:.2f}mm target={float(target[i])*1000:.2f}mm "
                f"stall={float(stall[i])*1000:.2f}mm near={bool(near_bottle[i])} "
                f"span={float(s['finger_span_xy'][i])*1000:.2f}mm",
                flush=True,
            )
    return (stall > min_stall) & near_bottle


def _grip_close_done(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Gripper đã khép xong — khi ramp: chờ progress, không tin lệnh binary.

    ``grasp_close_done_progress`` (0.96) đứng trên ``grasp_close_freeze_at_progress``
    (0.75), mà ramp bị chặn cứng ở cap đó trong actions.py → điều kiện cũ vĩnh viễn
    False, kéo theo cổng tilt lift tụt 6.0° → 4.0° và giết luôn đường reopen.
    Lấy min với cap để "khép xong" nghĩa là "đã chạm trần cho phép", tự nhất quán
    với mọi giá trị freeze_at về sau.
    """
    term = env.action_manager._terms.get("gripper_action")
    ramp_steps = int(getattr(env.cfg, "grasp_close_ramp_steps", 0))
    if ramp_steps > 0 and term is not None and hasattr(term, "_close_progress"):
        freeze_at = float(getattr(env.cfg, "grasp_close_freeze_at_progress", 0.0))
        cap = freeze_at if freeze_at > 0.0 else 1.0
        done_gc = min(float(getattr(env.cfg, "grasp_close_done_progress", 0.96)), cap)
        return term._close_progress >= done_gc
    grip_thresh = getattr(env.cfg, "grasp_grip_threshold", 0.4)
    return s["gripper_state"] > grip_thresh


def _grip_secure(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Kẹp đã latch VÀ thực sự khép chặt — điều kiện duy nhất để được phép nhấc.

    Thay cho _grip_committed cũ, vốn phụ thuộc _grip_partial_lift_ok (chỉ mở
    khoá sau khi grasp đã hỏng 3 lần) và _grip_close_done (bất khả thi trước
    khi sửa cap).

    BUG đã sửa (đo trực tiếp qua DEBUG_LIFT sau khi vá mimic gripper): 3 điều
    kiện cũ nối bằng OR, trong đó `_grip_physically_closed` CHỈ kiểm tra vị
    trí hình học (span/joint đã gần nhau) — KHÔNG kiểm tra có lực ép thật hay
    không. Lift được phép bắt đầu ngay khi stall (chênh khớp-so-với-lệnh, tỉ
    lệ thuận lực ép) mới chỉ 2.69mm — quá yếu so với mốc kẹp chắc thật sự đo
    được (~12.6mm). Kết quả: ngón trượt êm dọc thân chai ngay trong 2-3mm đầu
    tiên khi tay bắt đầu nhấc (stall tụt về 0 dần, không phải đột ngột — bộ
    phát hiện trượt vốn chỉ bắt delta-1-bước cũng không kịp phản ứng), tay
    bay lên khoảng không hàng trăm bước mà không hề biết đã gắp hụt.

    Fix: `_grip_pressing` (lực ép thật) giờ là điều kiện BẮT BUỘC, không còn
    tuỳ chọn — chỉ dùng `_grip_close_done`/`_grip_physically_closed` làm bằng
    chứng hình học BỔ SUNG, không thay thế được yêu cầu về lực.
    """
    return _grip_latched(env) & _grip_pressing(env, s) & (
        _grip_close_done(env, s) | _grip_physically_closed(env, s)
    )


def _lift_can_start(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Điều kiện ARM lệnh nhấc. Chỉ đánh giá khi đang IDLE."""
    in_grasp = env._stage == STAGE_GRASP
    max_tilt = float(getattr(env.cfg, "grasp_lift_start_max_tilt_deg", 6.0))
    contact_z = float(getattr(env.cfg, "grasp_lift_contact_z_finger", 0.022))
    max_span = float(getattr(env.cfg, "grasp_lift_max_span_xy", 0.065))
    return (
        in_grasp
        & _grip_secure(env, s)
        & (s["bottle_tilt_deg"] < max_tilt)
        & (s["z_error_finger"] < contact_z)
        & (s["finger_span_xy"] < max_span)
    )


def _lift_must_abort(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Điều kiện HỦY lệnh nhấc đang chạy — chỉ những lỗi thật sự.

    Cố tình KHÔNG bao gồm các ngưỡng tiếp xúc (z_f/dist/lat): khi cánh tay đi
    lên mà chai còn quán tính, chúng dao động và sẽ hủy nhấc oan.
    """
    in_grasp = env._stage == STAGE_GRASP
    abort_tilt = float(getattr(env.cfg, "grasp_grasp_abort_tilt_deg", 18.0))
    slip_max = int(getattr(env.cfg, "grasp_lift_abort_slip_steps", 8))
    slip_steps = getattr(env, "_lift_slip_steps", None)
    if slip_steps is None:
        slip_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return (
        ~in_grasp
        | ~_grip_latched(env)
        | (s["bottle_tilt_deg"] > abort_tilt)
        | (slip_steps >= slip_max)
    )


def _update_lift_state(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """State machine nhấc: IDLE → RISING → HOLDING.

    Tính chất quyết định: RISING KHÔNG đánh giá lại _lift_can_start. Logic cũ
    kiểm tra lại toàn bộ điều kiện mỗi bước rồi reset cứng bộ đếm khi lỗi 1
    bước, nên lệnh nhấc chỉ tồn tại được vài chục ms. Giờ đã armed thì chạy liên
    tục tới khi có lỗi thật.

    Trả về mask các env đang RISING (cần lệnh nhấc lên).
    """
    n = env.num_envs
    if not hasattr(env, "_lift_phase"):
        env._lift_phase = torch.zeros(n, dtype=torch.long, device=env.device)
    if not hasattr(env, "_lift_ready_steps"):
        env._lift_ready_steps = torch.zeros(n, dtype=torch.long, device=env.device)
    if not hasattr(env, "_lift_rising_steps"):
        env._lift_rising_steps = torch.zeros(n, dtype=torch.long, device=env.device)

    phase = env._lift_phase
    lift_thresh = float(getattr(env.cfg, "grasp_lift_threshold", 0.03))
    hyst = float(getattr(env.cfg, "grasp_lift_hold_hysteresis_m", 0.008))
    settle_req = int(getattr(env.cfg, "grasp_lift_settle_steps", 3))

    can_start = _lift_can_start(env, s)
    must_abort = _lift_must_abort(env, s)
    above = s["bottle_lift"] >= lift_thresh

    is_idle = phase == LIFT_IDLE
    is_rising = phase == LIFT_RISING
    is_holding = phase == LIFT_HOLDING

    # IDLE: tích luỹ số bước đủ điều kiện liên tiếp rồi arm
    env._lift_ready_steps[is_idle & can_start] += 1
    env._lift_ready_steps[is_idle & ~can_start] = 0
    to_rising = is_idle & (env._lift_ready_steps >= settle_req)

    # RISING: chỉ rời khi abort thật, hoặc khi đã vượt ngưỡng → HOLDING
    rise_abort = is_rising & must_abort
    rise_done = is_rising & above & ~must_abort

    # HOLDING: tụt dưới ngưỡng (trừ hysteresis) thì nhấc tiếp; abort thì về IDLE
    hold_abort = is_holding & must_abort
    hold_resume = is_holding & (s["bottle_lift"] < (lift_thresh - hyst)) & ~must_abort

    new_phase = phase.clone()
    new_phase[to_rising] = LIFT_RISING
    new_phase[rise_abort] = LIFT_IDLE
    new_phase[rise_done] = LIFT_HOLDING
    new_phase[hold_abort] = LIFT_IDLE
    new_phase[hold_resume] = LIFT_RISING
    env._lift_phase = new_phase
    env._lift_ready_steps[new_phase != LIFT_IDLE] = 0
    # Số bước liên tiếp đang RISING — dùng để ramp lực nhấc từ từ (xem
    # apply_grasp_arm_assist), tránh giật đột ngột ngay lúc bắt đầu nhấc.
    env._lift_rising_steps[new_phase == LIFT_RISING] += 1
    env._lift_rising_steps[new_phase != LIFT_RISING] = 0

    if os.environ.get("DEBUG_APPROACH") == "1" and env.num_envs <= 16:
        just_latched = _grip_latched(env) & (env._lift_ready_steps == 1) & (phase == LIFT_IDLE)
        for i in just_latched.nonzero(as_tuple=False).flatten().tolist():
            print(
                f"  [ApproachDbg] env{i} step_ct={int(env.step_counter)} "
                f"z_error_finger={float(s['z_error_finger'][i])*1000:.2f}mm "
                f"top_down_align={float(s['top_down_align'][i]):.3f} "
                f"lateral_finger_xy={float(s['lateral_finger_xy'][i])*1000:.2f}mm "
                f"finger_span_xy={float(s['finger_span_xy'][i])*1000:.2f}mm "
                f"dist_left_body={float(s['dist_left_body'][i])*1000:.2f}mm "
                f"dist_right_body={float(s['dist_right_body'][i])*1000:.2f}mm "
                f"bottle_tilt_deg={float(s['bottle_tilt_deg'][i]):.2f} "
                f"finger_level={float(s['finger_level'][i]):.3f}",
                flush=True,
            )

    if os.environ.get("DEBUG_LIFT") == "1" and env.num_envs <= 16:
        changed = to_rising | rise_abort | rise_done | hold_abort
        if changed.any():
            names = {0: "IDLE", 1: "RISING", 2: "HOLDING"}
            slip_steps = getattr(env, "_lift_slip_steps", None)
            term = env.action_manager._terms.get("gripper_action")
            latched_all = _grip_latched(env)
            for i in changed.nonzero(as_tuple=False).flatten().tolist():
                ss = int(slip_steps[i]) if slip_steps is not None else -1
                gc = float(term._close_progress[i]) if term is not None and hasattr(term, "_close_progress") else -1.0
                reason = "slip" if (slip_steps is not None and int(slip_steps[i]) >= int(getattr(env.cfg, "grasp_lift_abort_slip_steps", 8))) else (
                    "tilt" if float(s["bottle_tilt_deg"][i]) > float(getattr(env.cfg, "grasp_grasp_abort_tilt_deg", 18.0)) else (
                        "unlatched" if not bool(latched_all[i]) else "-"
                    )
                )
                print(
                    f"  [LiftDbg] env{i} step_ct={int(env.step_counter)} "
                    f"{names[int(phase[i])]}→{names[int(new_phase[i])]} "
                    f"| lift_m={float(s['bottle_lift'][i])*1000:.2f}mm "
                    f"tilt={float(s['bottle_tilt_deg'][i]):.1f} slip_steps={ss} "
                    f"gc={gc:.3f} latched={bool(latched_all[i])} reason={reason}",
                    flush=True,
                )

    # Lý do abort cho telemetry (đọc bởi eval/demo, không ảnh hưởng điều khiển)
    aborted = rise_abort | hold_abort
    if aborted.any():
        reasons: list[str | None] = [None] * n
        in_grasp = env._stage == STAGE_GRASP
        abort_tilt = float(getattr(env.cfg, "grasp_grasp_abort_tilt_deg", 18.0))
        latched_now = _grip_latched(env)
        slip_steps = getattr(env, "_lift_slip_steps", torch.zeros(n, dtype=torch.long, device=env.device))
        slip_max = int(getattr(env.cfg, "grasp_lift_abort_slip_steps", 8))
        for idx in aborted.nonzero(as_tuple=False).flatten().tolist():
            if not bool(in_grasp[idx]):
                reasons[idx] = "stage"
            elif not bool(latched_now[idx]):
                reasons[idx] = "unlatched"
            elif float(s["bottle_tilt_deg"][idx]) > abort_tilt:
                reasons[idx] = "tilt"
            elif int(slip_steps[idx]) >= slip_max:
                reasons[idx] = "slip"
            else:
                reasons[idx] = "other"
        env._lift_abort_reason = reasons
    else:
        env._lift_abort_reason = [None] * n

    return env._lift_phase == LIFT_RISING


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
    """OSC pose_rel: nhấc thẳng world +Z (delta vị trí trong base frame).

    Trước đây magnitude đi qua một bộ gain thích ứng ``follow`` đọc
    ``env._prev_lift_bottle``. Nhưng ``_update_lift_slip`` chạy TRƯỚC hàm này và
    ghi đè chính buffer đó bằng giá trị hiện tại → ``moved`` luôn bằng 0 → gain
    kẹt vĩnh viễn ở sàn 0.45. Không hành vi nào từng quan sát được phụ thuộc vào
    nó, nên bỏ hẳn: magnitude hằng số, suy ra trực tiếp từ config.
    """
    pos_scale = float(getattr(env.cfg, "osc_position_scale", 0.06))
    lift_m = float(getattr(env.cfg, "grasp_lift_world_m", 0.012))
    full_mag = min(lift_m / max(pos_scale, 1e-4) * scale, 1.0)

    # Ramp từ từ theo số bước đã RISING — grasp_lift_world_m bị đẩy lên 0.055
    # (17x tốc độ gốc) để nhấc đủ nhanh trong ngân sách episode, nhưng full
    # magnitude NGAY BƯỚC ĐẦU là một lệnh bước-nhảy (step input) tạo giật đột
    # ngột. Đo được (lift_too_low, DEBUG_STALL): chai theo tay đúng 1.5mm rồi
    # trượt hẳn — đúng thời điểm giật khởi động. Ma sát đo được rất cao (μ
    # 1.4-1.8), đủ giữ TRỌNG LƯỢNG tĩnh (0.93N) dễ dàng, nhưng có thể không đủ
    # cho GIA TỐC đột ngột (lực quán tính cộng thêm) ở bước đầu. Ramp tuyến
    # tính lên full_mag trong grasp_lift_onset_ramp_steps bước để lực ma sát
    # có thời gian "bắt kịp" thay vì bị vượt ngay tức thì.
    onset_steps = int(getattr(env.cfg, "grasp_lift_onset_ramp_steps", 15))
    rising_steps = getattr(env, "_lift_rising_steps", None)
    if rising_steps is not None and onset_steps > 0:
        ramp_frac = (rising_steps.float() / float(onset_steps)).clamp(0.0, 1.0)
    else:
        ramp_frac = torch.ones(env.num_envs, device=env.device)

    out = arm_actions.clone()
    out[mask] = 0.0
    out[mask, 2] = full_mag * ramp_frac[mask]
    return out


def _update_lift_slip(env: ManagerBasedRLEnv, s: dict, armed: torch.Tensor) -> torch.Tensor:
    """Dừng nhấc khi arm đi lên mà chai đứng yên (ngón trượt khỏi thân)."""
    n = env.num_envs
    if not hasattr(env, "_prev_lift_z_f"):
        env._prev_lift_z_f = s["z_error_finger"].clone()
    if not hasattr(env, "_prev_lift_bottle"):
        env._prev_lift_bottle = s["bottle_lift"].clone()
    dz_f = s["z_error_finger"] - env._prev_lift_z_f
    db = s["bottle_lift"] - env._prev_lift_bottle
    slip_step_z = float(getattr(env.cfg, "grasp_lift_slip_z_finger", 0.022)) * 0.45
    slip_b = float(getattr(env.cfg, "grasp_lift_slip_bottle_m", 0.004))
    slip = armed & (dz_f > slip_step_z) & (db < slip_b * 0.35)
    if not hasattr(env, "_lift_slip_warmup"):
        env._lift_slip_warmup = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._lift_slip_warmup[armed] += 1
    env._lift_slip_warmup[~armed] = 0
    warmup = int(getattr(env.cfg, "grasp_lift_slip_warmup_steps", 4))
    slip = slip & (env._lift_slip_warmup >= warmup)
    term = env.action_manager._terms.get("gripper_action")
    if term is not None and hasattr(term, "_close_progress"):
        min_gc = float(getattr(env.cfg, "grasp_lift_partial_min_gc", 0.52))
        firm_gc = term._close_progress >= min_gc
        firm_phys = _grip_physically_closed(env, s) | (s["gripper_state"] > 0.42)
        firm = firm_gc & firm_phys
        slip = slip & ~firm
    env._lift_slip_pause = slip
    # Hạ cấp thành bộ đếm: 1 bước trượt đơn lẻ không còn hủy nhấc ngay (logic cũ
    # làm vậy và đó là một nguồn nhấp nháy). State machine đòi N bước liên tiếp.
    if not hasattr(env, "_lift_slip_steps"):
        env._lift_slip_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._lift_slip_steps[slip] += 1
    env._lift_slip_steps[~slip] = 0

    if os.environ.get("DEBUG_LIFT") == "1" and env.num_envs <= 16 and int(env.step_counter) % 20 == 0:
        term2 = env.action_manager._terms.get("gripper_action")
        joint = env._robot.data.joint_pos[:, env._gripper_joint_ids].mean(dim=1) if hasattr(env, "_gripper_joint_ids") else None
        open_m = float(getattr(env.cfg, "gripper_open_m", 0.044))
        for i in armed.nonzero(as_tuple=False).flatten().tolist():
            tgt = open_m * (1.0 - float(term2._close_progress[i])) if term2 is not None and hasattr(term2, "_close_progress") else -1.0
            stall = float(joint[i]) - tgt if joint is not None else -1.0
            print(
                f"  [LiftSlip] env{i} step_ct={int(env.step_counter)} "
                f"lift_m={float(s['bottle_lift'][i])*1000:.2f}mm dz_f={float(dz_f[i])*1000:+.3f}mm "
                f"db={float(db[i])*1000:+.3f}mm slip={bool(slip[i])} slip_steps={int(env._lift_slip_steps[i])} "
                f"joint={float(joint[i])*1000:.2f}mm tgt={tgt*1000:.2f}mm stall={stall*1000:.2f}mm "
                f"z_f={float(s['z_error_finger'][i])*1000:.2f}mm tilt={float(s['bottle_tilt_deg'][i]):.1f}",
                flush=True,
            )

    env._prev_lift_z_f = s["z_error_finger"].clone()
    env._prev_lift_bottle = s["bottle_lift"].clone()
    if not armed.any():
        env._prev_lift_z_f = s["z_error_finger"].clone()
        env._prev_lift_bottle = s["bottle_lift"].clone()
    return slip


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

    # Giảm chấn hover REACH khi assist đã TẮT HẲN (scale<=1e-6) — nhánh
    # scale-gated phía dưới (giữ nguyên, không đổi) sẽ tự lo việc này khi
    # scale>0, nên khối này CHỈ chạy khi nhánh dưới sẽ return sớm và bỏ qua
    # hoàn toàn. Ban đầu viết khối này chạy VÔ ĐIỀU KIỆN (kể cả scale>0) —
    # gây damp ÁP HAI LẦN chồng lên nhánh gốc (0.12×0.12=0.0144), làm hồi quy
    # ngay cả ở scale=1.0 (top_down_align kẹt ~0.44 dù trước đó ready trong
    # 21-59 bước — xem eval_after_mimic_align.log). Chỉ nên chạy đúng lúc
    # code gốc sẽ bỏ qua.
    #
    # Phát hiện qua đo thật (assist_scale=0.0, --stage all): dist_ee_bottle
    # đạt mean=0.026m (rất gần) nhưng top_down_align kẹt ở 0.47-0.53 (cần
    # ≥0.58), CÒN GIẢM DẦN theo thời gian hold — mọi tiêu chí khác (lat_f,
    # z, z_f, finger_level) đều đạt. Đây KHÔNG phải "làm hộ" nhiệm vụ — chỉ
    # là bộ lọc nhiễu cho chính output thô của policy khi đang hover chờ
    # gate căn chỉnh. Nhiễu tự nhiên của các dim xoay cổ tay đủ lớn để phá
    # hỏng việc GIỮ ỔN ĐỊNH hướng úp xuống, dù giá trị TRUNG BÌNH của policy
    # đã đúng hướng.
    if (
        _assist_scale(env) <= 1e-6
        and getattr(env.cfg, "grasp_descent_assist_enabled", False)
        and getattr(env.cfg, "grasp_descent_in_reach", True)
    ):
        # Ở nhánh này scale<=1e-6 nên code gốc bên dưới sẽ return ngay sau khi
        # tính `scale` — nghĩa là `need_down` (lệnh hạ chủ động) sẽ KHÔNG bao
        # giờ được tính/áp dụng trong cùng bước này. Do đó không cần loại trừ
        # `need_down` ở đây: nó luôn False trong nhánh scale=0.
        _in_reach0 = env._stage == STAGE_REACH
        _in_contact0 = getattr(env, "_in_contact_zone", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        _hold_ok0 = env._steps_in_contact >= getattr(env.cfg, "reach_stage_hold_steps", 5)
        _reach_hover0 = _in_reach0 & _in_contact0 & _hold_ok0 & ~reach_align_ready(env, s, soft=False)
        if _reach_hover0.any():
            actions[_reach_hover0] *= float(getattr(env.cfg, "grasp_grip_arm_damp", 0.12))

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
    partial_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    term = env.action_manager._terms.get("gripper_action")
    reopen_max = int(getattr(env.cfg, "grasp_reopen_max_count", 3))
    partial_min_gc = float(getattr(env.cfg, "grasp_lift_partial_min_gc", 0.55))
    reopen_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if term is not None and hasattr(term, "_reopen_count") and hasattr(term, "_close_progress"):
        reopen_done = (term._reopen_count >= reopen_max) & (term._close_progress >= partial_min_gc)
    lift_thresh = getattr(env.cfg, "grasp_lift_threshold", 0.03)
    # Lift state machine chạy mỗi bước (kể cả khi assist tắt) để telemetry và
    # các freeze bên dưới luôn có _lift_phase nhất quán.
    rising = _update_lift_state(env, s)
    lift_ready = rising | (env._lift_phase == LIFT_HOLDING)
    top_down = s["top_down_align"]

    descend_cmd = float(getattr(env.cfg, "grasp_descent_action", 0.3))

    z_stop = getattr(
        env.cfg,
        "grasp_descent_z_target",
        getattr(env.cfg, "grasp_close_max_z_err", 0.012),
    )
    z_high = s["z_error_finger"] > z_stop
    top_ok = top_down >= getattr(env.cfg, "grasp_descent_min_top_down", 0.50)

    need_down = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Khai báo trước: chỉ được gán trong nhánh assist_grasp, nhưng idle_hold ở
    # cuối hàm đọc nó vô điều kiện.
    center_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

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
        need_down = need_down | (in_grasp & ~descended & top_ok_grasp & ~lift_ready & ~latched)
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
            & (s["bottle_tilt_deg"] < float(getattr(env.cfg, "grasp_grasp_abort_tilt_deg", 18.0)))
        )
        # Sau hết reopen + lệch: ưu tiên căn pad trước lift
        if reopen_done.any() and (~sym_ok).any():
            recover_asym = recover_asym | (in_grasp & reopen_done & ~sym_ok & (s["bottle_lift"] < lift_thresh))
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
    # Freeze TRƯỚC latch. Sau latch việc đóng băng do idle_hold ở cuối hàm lo,
    # dựa trên _lift_phase — nên không còn phụ thuộc grasp_lift_assist_enabled
    # để "mở van" (đó là lý do --stage grasp từng đóng băng cánh tay vĩnh viễn).
    pregrasp_hold = (
        in_grasp
        & ~latched
        & aligned
        & freeze_arm
        & ~need_down
        & sym_for_freeze
        & ~not_sinkable
    )
    if pregrasp_hold.any():
        actions[pregrasp_hold] = 0.0

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
        # RISING: phát lệnh nhấc thẳng world +Z. State machine đã lọc điều kiện,
        # ở đây không kiểm tra lại gì nữa (đó chính là nguồn nhấp nháy cũ).
        want_lift = rising
        _update_lift_slip(env, s, rising)
        env._assist_want_lift = rising.clone()
        if rising.any():
            # RESIDUAL, không ghi đè. PPO lưu log-prob của action đã SAMPLE rồi
            # gán advantage cho action đó; nếu env thực thi thứ khác thì gradient
            # cho việc nhấc bằng 0 (ghi đè) hoặc bị lệch (blend lồi). Cộng thêm
            # một lượng bias tất định vào trục Z giữ đúng tính chất on-policy.
            #   w=1 → hành vi hệt như ghi đè (Giai đoạn 1, đã kiểm chứng)
            #   w=0 → policy toàn quyền
            pos_scale = float(getattr(env.cfg, "osc_position_scale", 0.06))
            lift_m = float(getattr(env.cfg, "grasp_lift_world_m", 0.012))
            bias = min(lift_m / max(pos_scale, 1e-4), 1.0)
            w = max(min(scale, 1.0), 0.0)
            if w >= 1.0 - 1e-6:
                actions = _osc_world_up_lift(env, s, rising, actions, 1.0)
                actions[rising, 3:6] = 0.0
            else:
                actions[rising, 2] = (actions[rising, 2] + w * bias).clamp(-1.0, 1.0)
                # Các trục khác: giảm chấn theo w chứ không giết — policy vẫn
                # giữ được quyền điều chỉnh tư thế trong lúc nhấc.
                damp = 1.0 - 0.8 * w
                actions[rising, 0:2] = actions[rising, 0:2] * damp
                actions[rising, 3:6] = actions[rising, 3:6] * damp

        # HOLDING: đóng băng để chai giảm tốc. grasp_lift_success_ready đòi
        # ‖vel‖ < grasp_success_max_bottle_speed trong 5 bước liên tiếp; nếu vẫn
        # đẩy lên thì chai không bao giờ đứng yên đủ lâu để tính thành công.
        holding = env._lift_phase == LIFT_HOLDING
        if holding.any():
            actions[holding] = 0.0

    # Đã latch nhưng lift chưa arm: giữ pose, chặn policy drift (z_f tăng ảo).
    # Gộp latched_hold + gripped_hold cũ — cả hai đều mô tả cùng một trạng thái.
    idle_hold = (
        in_grasp
        & latched
        & (env._lift_phase == LIFT_IDLE)
        & ~need_down
        & ~center_mask
    )
    if idle_hold.any():
        actions[idle_hold] = 0.0

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
