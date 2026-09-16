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
    STAGE_PLACE,
    STAGE_REACH,
    PLACE_IDLE,
    PLACE_CARRY,
    PLACE_DESCEND,
    PLACE_HOLDING,
    finger_descended_for_close,
    finger_grasp_ready,
    finger_ready_for_close,
    finger_symmetric_ready,
    finger_pad_asymmetric,
    grip_close_ramp_active,
    place_release_ready,
    reach_align_ready,
    reach_descent_ready,
    uses_grasp_lift,
    uses_place,
    _t,
)


# Lift state machine (xem _update_lift_state). Lift là sub-mode của STAGE_GRASP,
# không phải một stage riêng — env._stage vẫn chỉ đi REACH → GRASP.
LIFT_IDLE = 0      # chưa đủ điều kiện, hoặc vừa bị hủy
LIFT_RISING = 1    # đang phát lệnh nhấc lên, KHÔNG đánh giá lại điều kiện khởi động
LIFT_HOLDING = 2   # đã vượt ngưỡng — giữ yên cho chai ổn định để tính thành công


def _assist_scale(env: ManagerBasedRLEnv) -> float:
    return float(getattr(env, "_assist_blend_scale", 1.0))


def _assist_scale_descent(env: ManagerBasedRLEnv) -> float:
    """Scale riêng cho assist REACH/GRASP-descent — KHÔNG bị anneal theo
    lịch LIFT/PLACE (_assist_blend_scale). Trước đây cả 2 dùng chung 1 scale
    (_assist_scale) và hàm apply_grasp_arm_assist còn return SỚM ngay khi
    scale<=1e-6 — nghĩa là khi lịch anneal LIFT/PLACE về 0 để "dạy" kỹ năng
    mới, REACH/GRASP-descent (đã hoạt động tốt từ trước, không liên quan gì
    tới LIFT/PLACE) cũng bị TẮT LUÔN theo. Xác nhận bằng thực nghiệm: fine-
    tune PLACE (task_phase=3, --assist-schedule) làm grasp_rate sập từ 0.83
    xuống 0.40 dù không đổi gì ở REACH/GRASP — đúng rủi ro đã cảnh báo từ
    Giai đoạn 2 cũ (S2.3 "tách 2 scale") nhưng chưa từng làm.

    Chỉ có `AssistScheduleCallback` (isaaclab_train.py, lúc training thật)
    set `_assist_blend_scale_descent` tường minh (cố định 1.0, không anneal).
    Eval/demo CHỈ set `_assist_blend_scale` (qua --assist-scale) — trong
    trường hợp đó hàm này PHẢI trả về CÙNG giá trị với _assist_scale(env),
    không phải hardcode 1.0, để không đổi hành vi của mọi lệnh
    `--assist-scale 0.0/0.5/...` đã dùng xuyên suốt investigation này.
    """
    descent = getattr(env, "_assist_blend_scale_descent", None)
    if descent is None:
        return _assist_scale(env)
    return float(descent)


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
    """Đang ép chai đủ chắc để nhấc — CẢ HAI ngón, HOẶC một ngón chắc + hình học đúng.

    Phase 32: thay proxy vị trí (joint-target stall trung bình 2 ngón) bằng
    LỰC TIẾP XÚC PHYSX THẬT của TỪNG ngón (ContactSensor, config.py). Đo trực
    tiếp bằng debug thật (terminal_command.md Phase 32) phát hiện: khi tay
    tiếp cận lệch tâm nhẹ (~7mm), 2 khớp vẫn đóng ĐỐI XỨNG HOÀN HẢO về góc
    (proxy vị trí trung bình `stall` không phân biệt được), nhưng CHỈ MỘT ngón
    thực sự chạm chai còn ngón kia hoàn toàn KHÔNG TIẾP XÚC.

    Phase 34: yêu cầu CẢ HAI vượt ngưỡng (Phase 32/33) ĐO ĐƯỢC làm sập
    lift_start_rate 0.4333→0.0333 (regression gate seed=0) — lệch tâm khiến
    ngón xa gần như KHÔNG BAO GIỜ đạt lực thật dù đóng hết cỡ (đã thử nới cap
    lên 1.0 ở Phase 31: joint≈0 mà lực vẫn không tăng — không phải vấn đề
    cap, là hình học không sửa được bằng cách đóng chặt hơn). Trước Phase 32,
    stall trung bình (không phân biệt 1-ngón-chạm) vẫn cho lift_start 43% —
    tức 1 ngón ép đủ mạnh + hình học đúng đường kính chai (span_ok) ĐÃ ĐỦ để
    giữ được trong thực tế. Khôi phục khả năng đó bằng nhánh OR: 1 ngón vượt
    ngưỡng CAO HƠN (single, chắc chắn không phải nhiễu) + near_bottle +
    span_ok cũng được tính là đang ép.
    """
    if "left_finger_contact_force" not in s or "right_finger_contact_force" not in s:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    min_force = float(getattr(env.cfg, "grasp_press_min_force_n", 0.15))
    min_force_single = float(getattr(env.cfg, "grasp_press_min_force_single_n", 0.30))
    near_bottle = s["dist_finger_body"] < float(getattr(env.cfg, "grasp_press_max_dist_f", 0.060))
    bottle_d = float(getattr(env.cfg, "bottle_diameter_m", 0.0433))
    span_margin = float(getattr(env.cfg, "grasp_press_span_margin_m", 0.004))
    span_ok = s["finger_span_xy"] < (bottle_d + span_margin)
    left_f = s["left_finger_contact_force"]
    right_f = s["right_finger_contact_force"]
    both_pressing = (left_f > min_force) & (right_f > min_force) & near_bottle
    single_pressing = (
        (torch.maximum(left_f, right_f) > min_force_single) & near_bottle & span_ok
    )
    pressing = both_pressing | single_pressing
    if os.environ.get("DEBUG_STALL") == "1" and env.num_envs <= 16 and int(env.step_counter) % 15 == 0:
        latched = _grip_latched(env)
        for i in latched.nonzero(as_tuple=False).flatten().tolist():
            print(
                f"  [Stall] env{i} step_ct={int(env.step_counter)} "
                f"F_left={float(left_f[i]):.3f}N F_right={float(right_f[i]):.3f}N "
                f"near={bool(near_bottle[i])} span_ok={bool(span_ok[i])} "
                f"both={bool(both_pressing[i])} single={bool(single_pressing[i])} "
                f"span={float(s['finger_span_xy'][i])*1000:.2f}mm",
                flush=True,
            )
    return pressing


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
    # Phase 26: trượt DẦN, đo thật bằng DEBUG_STALL/DEBUG_LIFT (terminal_command.md)
    # — z_error_finger tăng liên tục (~1mm/bước) sau khi chai đã tách khỏi kẹp,
    # không bị bắt bởi slip_steps (chỉ bắt bước-đơn-lẻ đột ngột) nên RISING chạy
    # vô ích tới hết episode. Bộ đếm riêng trong _update_lift_slip.
    zf_abort_steps_max = int(getattr(env.cfg, "grasp_lift_abort_z_finger_steps", 5))
    zf_abort_steps = getattr(env, "_lift_zf_abort_steps", None)
    if zf_abort_steps is None:
        zf_abort_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return (
        ~in_grasp
        | ~_grip_latched(env)
        | (s["bottle_tilt_deg"] > abort_tilt)
        | (slip_steps >= slip_max)
        | (zf_abort_steps >= zf_abort_steps_max)
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

    # Phase 27 (cơ chế phục hồi sau zf_drift-abort) ĐÃ THỬ VÀ REVERT (Phase 29,
    # terminal_command.md): đo được KHÔNG vô hại như tưởng — dù bản thân chỉ
    # đổi hành động lúc LIFT_IDLE (đứng yên → hạ tay), nó vẫn làm lệch quỹ đạo
    # episode đủ để regression gate không còn khớp CHÍNH XÁC baseline (success
    # 0.033→0.067, latch/lift_start giảm nhẹ) — dù chênh lệch nhỏ, không đạt
    # tiêu chuẩn "khớp chính xác" của project. Giữ nguyên Phase 26 (zf_drift
    # abort sớm hơn, đã xác nhận khớp TUYỆT ĐỐI baseline).

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
            zf_abort_steps = getattr(env, "_lift_zf_abort_steps", None)
            term = env.action_manager._terms.get("gripper_action")
            latched_all = _grip_latched(env)
            for i in changed.nonzero(as_tuple=False).flatten().tolist():
                ss = int(slip_steps[i]) if slip_steps is not None else -1
                gc = float(term._close_progress[i]) if term is not None and hasattr(term, "_close_progress") else -1.0
                reason = "slip" if (slip_steps is not None and int(slip_steps[i]) >= int(getattr(env.cfg, "grasp_lift_abort_slip_steps", 8))) else (
                    "zf_drift" if (zf_abort_steps is not None and int(zf_abort_steps[i]) >= int(getattr(env.cfg, "grasp_lift_abort_z_finger_steps", 5))) else (
                    "tilt" if float(s["bottle_tilt_deg"][i]) > float(getattr(env.cfg, "grasp_grasp_abort_tilt_deg", 18.0)) else (
                        "unlatched" if not bool(latched_all[i]) else "-"
                    )
                    )
                )
                dl = float(s["dist_left_body"][i]) * 1000 if "dist_left_body" in s else -1.0
                dr = float(s["dist_right_body"][i]) * 1000 if "dist_right_body" in s else -1.0
                print(
                    f"  [LiftDbg] env{i} step_ct={int(env.step_counter)} "
                    f"{names[int(phase[i])]}→{names[int(new_phase[i])]} "
                    f"| lift_m={float(s['bottle_lift'][i])*1000:.2f}mm "
                    f"tilt={float(s['bottle_tilt_deg'][i]):.1f} slip_steps={ss} "
                    f"gc={gc:.3f} latched={bool(latched_all[i])} reason={reason} "
                    f"dL={dl:.2f}mm dR={dr:.2f}mm dLR={dl-dr:+.2f}mm",
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


def _place_can_start(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Điều kiện vào PLACE_CARRY. Chỉ đánh giá khi đang PLACE_IDLE."""
    in_place = env._stage == STAGE_PLACE
    return in_place & _grip_secure(env, s)


def _place_must_abort(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Điều kiện HỦY carry/descend đang chạy — chỉ lỗi thật sự.

    Mirror _lift_must_abort: cố tình KHÔNG kiểm ngưỡng vị trí XY/độ cao tức
    thời (dao động quán tính lúc di chuyển sẽ hủy oan).
    """
    in_place = env._stage == STAGE_PLACE
    abort_tilt = float(getattr(env.cfg, "place_abort_tilt_deg", 25.0))
    abort_dist_ee = float(getattr(env.cfg, "place_abort_dist_ee_m", 0.15))
    return (
        ~in_place
        | ~_grip_latched(env)
        | (s["bottle_tilt_deg"] > abort_tilt)
        | (s["dist_ee_bottle"] > abort_dist_ee)
    )


def _update_place_state(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """State machine mang chai: IDLE → CARRY → DESCEND → HOLDING.

    Mirror kỷ luật của LIFT: một khi đã CARRY/DESCEND, KHÔNG đánh giá lại
    _place_can_start — chỉ _place_must_abort mới đẩy về IDLE. Không có đường
    quay lại CARRY như LIFT resume (chai lệch nặng khi đang mang gần như
    không cứu được — dựa vào bottle_misplaced_termination, chưa cài ở bước
    này, thay vì cố hồi phục).

    Trả về mask các env đang CARRY hoặc DESCEND (cần lệnh assist arm).
    """
    n = env.num_envs
    if not hasattr(env, "_place_phase"):
        env._place_phase = torch.zeros(n, dtype=torch.long, device=env.device)
    if not hasattr(env, "_place_carry_steps"):
        env._place_carry_steps = torch.zeros(n, dtype=torch.long, device=env.device)
    if not hasattr(env, "_place_arrival_steps"):
        env._place_arrival_steps = torch.zeros(n, dtype=torch.long, device=env.device)
    if not hasattr(env, "_place_release_steps"):
        env._place_release_steps = torch.zeros(n, dtype=torch.long, device=env.device)
    if not hasattr(env, "_steps_place_holding"):
        env._steps_place_holding = torch.zeros(n, dtype=torch.long, device=env.device)

    phase = env._place_phase
    xy_radius = float(getattr(env.cfg, "place_xy_arrival_radius_m", 0.03))
    arrival_settle = int(getattr(env.cfg, "place_arrival_settle_steps", 5))
    # S9: bát chỉ sâu 5.2cm thật (đo bbox) — 0.05 (gần bằng cả độ sâu) quá lỏng,
    # siết còn 0.02 (2cm trên đáy trong thật, đã sửa ở height_above_bowl_floor).
    release_height = float(getattr(env.cfg, "place_release_height_m", 0.02))
    release_settle = int(getattr(env.cfg, "place_release_hold_steps", 5))

    can_start = _place_can_start(env, s)
    must_abort = _place_must_abort(env, s)

    is_idle = phase == PLACE_IDLE
    is_carry = phase == PLACE_CARRY
    is_descend = phase == PLACE_DESCEND
    is_holding = phase == PLACE_HOLDING

    # IDLE → CARRY: _place_can_start đã bao gồm _grip_secure — vào ngay,
    # không cần bộ đệm settle như LIFT (LIFT cần lọc dao động tiếp xúc lúc
    # vừa latch; PLACE chỉ cần grip đã ổn định, đã đảm bảo từ GRASP).
    to_carry = is_idle & can_start

    # CARRY → DESCEND: đã hội tụ XY đủ N bước liên tiếp
    xy_ok = s["dist_bottle_bowl_xy"] < xy_radius
    env._place_arrival_steps[is_carry & xy_ok] += 1
    env._place_arrival_steps[is_carry & ~xy_ok] = 0
    carry_abort = is_carry & must_abort
    carry_done = is_carry & (env._place_arrival_steps >= arrival_settle) & ~must_abort

    # DESCEND → HOLDING: đã hạ đủ thấp trên bát đủ N bước liên tiếp
    height_ok = s["height_above_bowl_floor"] < release_height
    env._place_release_steps[is_descend & height_ok] += 1
    env._place_release_steps[is_descend & ~height_ok] = 0
    descend_abort = is_descend & must_abort
    descend_done = is_descend & (env._place_release_steps >= release_settle) & ~must_abort

    hold_abort = is_holding & must_abort

    new_phase = phase.clone()
    new_phase[to_carry] = PLACE_CARRY
    new_phase[carry_abort] = PLACE_IDLE
    new_phase[carry_done] = PLACE_DESCEND
    new_phase[descend_abort] = PLACE_IDLE
    new_phase[descend_done] = PLACE_HOLDING
    new_phase[hold_abort] = PLACE_IDLE
    env._place_phase = new_phase

    env._place_arrival_steps[new_phase != PLACE_CARRY] = 0
    env._place_release_steps[new_phase != PLACE_DESCEND] = 0
    env._steps_place_holding[new_phase == PLACE_HOLDING] += 1
    env._steps_place_holding[new_phase != PLACE_HOLDING] = 0
    # Số bước liên tiếp đang CARRY/DESCEND — ramp assist onset (mirror
    # _lift_rising_steps), tránh giật lúc vừa bắt đầu mang.
    moving = (new_phase == PLACE_CARRY) | (new_phase == PLACE_DESCEND)
    env._place_carry_steps[moving] += 1
    env._place_carry_steps[~moving] = 0

    if os.environ.get("DEBUG_PLACE") == "1" and env.num_envs <= 16:
        names = {0: "IDLE", 1: "CARRY", 2: "DESCEND", 3: "HOLDING"}
        changed = to_carry | carry_abort | carry_done | descend_abort | descend_done | hold_abort
        for i in changed.nonzero(as_tuple=False).flatten().tolist():
            print(
                f"  [PlaceDbg] env{i} step_ct={int(env.step_counter)} "
                f"{names[int(phase[i])]}→{names[int(new_phase[i])]} "
                f"xy={float(s['dist_bottle_bowl_xy'][i])*1000:.1f}mm "
                f"h_bowl={float(s['height_above_bowl_floor'][i])*1000:.1f}mm "
                f"tilt={float(s['bottle_tilt_deg'][i]):.1f}",
                flush=True,
            )

    return (env._place_phase == PLACE_CARRY) | (env._place_phase == PLACE_DESCEND)


def _osc_carry_to_bowl(
    env: ManagerBasedRLEnv, s: dict, mask: torch.Tensor, arm_actions: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """OSC pose_rel: tịnh tiến world XY về phía bát, giữ độ cao carry.

    Ramp tuyến tính theo _place_carry_steps để tránh giật lúc vừa bắt đầu
    mang (mirror _osc_world_up_lift).

    BUG đã sửa (đo trực tiếp qua demo thật, DEBUG_PLACE=1): công thức
    err/pos_scale bão hoà ở ±1.0 (max speed) suốt hàng chục bước liên tục khi
    lỗi XY/Z còn lớn (176mm+ XY, carry_height 150mm), KHÁC với
    _osc_world_up_lift (magnitude hằng số nhỏ, không tỉ lệ lỗi) — di chuyển 3
    trục cùng lúc ở tốc độ tối đa liên tục làm chai đung đưa, tilt leo dần
    vượt 15° (ngưỡng bottle_tipped_termination_deg đã bị siết cho GRASP qua
    phase2_overrides.py, giờ áp dụng chung cho PLACE) → episode bị coi là
    lật chai, không bao giờ tới được DESCEND. Fix: nhân thêm
    place_carry_speed_scale (mặc định 0.35 — chậm hơn hẳn max speed) VÀ chủ
    động giữ hướng thẳng đứng bằng _align_vertical_rot_action (đã có sẵn cho
    descend) để chống lại xu hướng nghiêng khi di chuyển ngang.
    """
    pos_scale = float(getattr(env.cfg, "osc_position_scale", 0.06))
    carry_height = float(getattr(env.cfg, "place_carry_height_m", 0.15))
    onset_steps = int(getattr(env.cfg, "place_carry_onset_ramp_steps", 15))
    speed_scale = float(getattr(env.cfg, "place_carry_speed_scale", 0.35))
    align_blend = float(getattr(env.cfg, "place_carry_align_blend", 0.5))
    carry_steps = getattr(env, "_place_carry_steps", None)
    if carry_steps is not None and onset_steps > 0:
        ramp_frac = (carry_steps.float() / float(onset_steps)).clamp(0.0, 1.0)
    else:
        ramp_frac = torch.ones(env.num_envs, device=env.device)

    # S9: dùng bowl_center_pos (đã sửa lệch tâm ~8cm so với root) và bowl_rim_z
    # (miệng bát thật, không phải root) làm mục tiêu — xem comment đo đạc ở
    # helpers.py::compute_state.
    xy_err = s["bowl_center_pos"][:, :2] - s["ee_pos"][:, :2]
    xy_step = (xy_err / max(pos_scale, 1e-4)).clamp(-1.0, 1.0) * scale * speed_scale

    z_target = s["bowl_rim_z"] + carry_height
    z_err = z_target - s["ee_pos"][:, 2]
    z_step = (z_err / max(pos_scale, 1e-4)).clamp(-1.0, 1.0) * scale * speed_scale

    out = arm_actions.clone()
    out[mask] = 0.0
    out[mask, 0:2] = xy_step[mask] * ramp_frac[mask].unsqueeze(-1)
    out[mask, 2] = z_step[mask] * ramp_frac[mask]
    if align_blend > 0.0:
        out[mask, 3:6] = _align_vertical_rot_action(env, s)[mask] * align_blend
    return out


def _osc_descend_to_bowl(
    env: ManagerBasedRLEnv, s: dict, mask: torch.Tensor, arm_actions: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """OSC pose_rel: hạ thẳng world -Z về độ cao thả trên bát, giữ XY.

    Mirror _osc_world_down_descend nhưng target là bowl thay vì bottle body.
    """
    pos_scale = float(getattr(env.cfg, "osc_position_scale", 0.06))
    descend_m = float(getattr(env.cfg, "place_descend_world_m", 0.02))
    mag = min(descend_m / max(pos_scale, 1e-4) * scale, 1.0)

    xy_err = s["bowl_center_pos"][:, :2] - s["ee_pos"][:, :2]
    xy_step = (xy_err / max(pos_scale, 1e-4)).clamp(-mag, mag)

    out = arm_actions.clone()
    out[mask] = 0.0
    out[mask, 0:2] = xy_step[mask]
    out[mask, 2] = -mag
    return out


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

    # Phase 26: trượt DẦN — z_error_finger tăng liên tục nhưng mỗi bước dưới hẳn
    # slip_step_z (đo thật: 0.7-2.2mm/bước, dưới xa ngưỡng 9.9mm/bước ở trên) nên
    # không bao giờ bị bắt bởi bộ đếm delta-1-bước. Dùng NGƯỠNG TUYỆT ĐỐI thay vì
    # delta, đếm N bước liên tiếp vượt ngưỡng (giống pattern slip ở trên) để không
    # huỷ oan dao động thoáng qua.
    zf_abort_thresh = float(getattr(env.cfg, "grasp_lift_abort_z_finger", 0.05))
    zf_over = armed & (s["z_error_finger"] > zf_abort_thresh)
    if not hasattr(env, "_lift_zf_abort_steps"):
        env._lift_zf_abort_steps = torch.zeros(n, dtype=torch.long, device=env.device)
    env._lift_zf_abort_steps[zf_over] += 1
    env._lift_zf_abort_steps[~zf_over] = 0

    if os.environ.get("DEBUG_LIFT") == "1" and env.num_envs <= 16 and int(env.step_counter) % 20 == 0:
        term2 = env.action_manager._terms.get("gripper_action")
        joint = env._robot.data.joint_pos[:, env._gripper_joint_ids].mean(dim=1) if hasattr(env, "_gripper_joint_ids") else None
        # Phase 32: joint1/joint2 RIÊNG (không average) — nghi vấn averaging
        # đang che giấu 1 khớp kẹt mở trong khi khớp kia đóng bình thường.
        joint1 = env._robot.data.joint_pos[:, env._gripper_joint_ids[0]] if hasattr(env, "_gripper_joint_ids") else None
        joint2 = env._robot.data.joint_pos[:, env._gripper_joint_ids[1]] if hasattr(env, "_gripper_joint_ids") else None
        open_m = float(getattr(env.cfg, "gripper_open_m", 0.044))
        # Phase 32: lực tiếp xúc PhysX THẬT (ContactSensor, config.py) — thay
        # cho proxy vị trí (stall) đã chứng minh không tin cậy được (Phase 31).
        lf_sensor = env.scene.sensors.get("left_finger_contact") if hasattr(env, "scene") else None
        rf_sensor = env.scene.sensors.get("right_finger_contact") if hasattr(env, "scene") else None
        lf_force = _t(lf_sensor.data.force_matrix_w) if lf_sensor is not None else None
        rf_force = _t(rf_sensor.data.force_matrix_w) if rf_sensor is not None else None
        for i in armed.nonzero(as_tuple=False).flatten().tolist():
            tgt = open_m * (1.0 - float(term2._close_progress[i])) if term2 is not None and hasattr(term2, "_close_progress") else -1.0
            stall = float(joint[i]) - tgt if joint is not None else -1.0
            dl = float(s["dist_left_body"][i]) * 1000 if "dist_left_body" in s else -1.0
            dr = float(s["dist_right_body"][i]) * 1000 if "dist_right_body" in s else -1.0
            lf_n = float(lf_force[i, 0].norm()) if lf_force is not None else -1.0
            rf_n = float(rf_force[i, 0].norm()) if rf_force is not None else -1.0
            print(
                f"  [LiftSlip] env{i} step_ct={int(env.step_counter)} "
                f"lift_m={float(s['bottle_lift'][i])*1000:.2f}mm dz_f={float(dz_f[i])*1000:+.3f}mm "
                f"db={float(db[i])*1000:+.3f}mm slip={bool(slip[i])} slip_steps={int(env._lift_slip_steps[i])} "
                f"joint={float(joint[i])*1000:.2f}mm tgt={tgt*1000:.2f}mm stall={stall*1000:.2f}mm "
                f"z_f={float(s['z_error_finger'][i])*1000:.2f}mm tilt={float(s['bottle_tilt_deg'][i]):.1f} "
                f"dL={dl:.2f}mm dR={dr:.2f}mm dLR={dl-dr:+.2f}mm "
                f"F_left={lf_n:.3f}N F_right={rf_n:.3f}N "
                f"j1={float(joint1[i])*1000:.2f}mm j2={float(joint2[i])*1000:.2f}mm",
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
        _assist_scale_descent(env) <= 1e-6
        and getattr(env.cfg, "grasp_descent_assist_enabled", False)
        and getattr(env.cfg, "grasp_descent_in_reach", True)
    ):
        # Điều kiện đã đổi từ _assist_scale(env) sang _assist_scale_descent(env)
        # (xem docstring hàm đó) — chỉ khi assist REACH/GRASP-descent THẬT SỰ
        # tắt (không phải khi lịch LIFT/PLACE anneal về 0) mới cần bọc damping
        # dự phòng này, vì code chính bên dưới giờ KHÔNG còn return sớm chỉ vì
        # scale (LIFT/PLACE) = 0 nữa.
        _in_reach0 = env._stage == STAGE_REACH
        _in_contact0 = getattr(env, "_in_contact_zone", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        _hold_ok0 = env._steps_in_contact >= getattr(env.cfg, "reach_stage_hold_steps", 5)
        _reach_hover0 = _in_reach0 & _in_contact0 & _hold_ok0 & ~reach_align_ready(env, s, soft=False)
        if _reach_hover0.any():
            actions[_reach_hover0] *= float(getattr(env.cfg, "grasp_grip_arm_damp", 0.12))

    scale = _assist_scale(env)
    descent_scale = _assist_scale_descent(env)
    if scale <= 1e-6 and descent_scale <= 1e-6:
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
            actions = _osc_z_finger_descend(env, s, low_top, actions, descent_scale * 0.75)
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
            actions[center_mask, 0:2] = xy_step[center_mask] * descent_scale
            actions[center_mask, 2] = 0.0
            align_blend = float(getattr(env.cfg, "grasp_descent_align_blend", 0.0))
            if align_blend > 0.0:
                actions[center_mask, 3:6] = (
                    _align_vertical_rot_action(env, s)[center_mask] * align_blend * descent_scale
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
                actions = _osc_world_down_descend(env, s, mask, actions, descent_scale)
            else:
                actions = _osc_tool_z_descend(env, s, mask, actions, descent_scale)

        def _apply_blend_world_z(mask: torch.Tensor, blend_val: float) -> None:
            """Giữ policy XY, blend chỉ trục hạ world-Z — tránh giật toàn pose."""
            nonlocal actions
            if not mask.any():
                return
            forced = actions.clone()
            forced = _osc_world_down_descend(env, s, mask, forced, descent_scale)
            b = blend_val * descent_scale
            actions[mask, 2] = (1.0 - b) * actions[mask, 2] + b * forced[mask, 2]

        def _apply_blend(mask: torch.Tensor, blend_val: float) -> None:
            if not mask.any():
                return
            b = blend_val * descent_scale
            cur = actions[mask, 2]
            actions[mask, 2] = (1.0 - b) * cur + b * descend_cmd

        if reach_mask.any():
            z_loop = getattr(env.cfg, "grasp_descent_z_finger_loop", True)
            if z_loop:
                # Closed-loop z_f trong REACH — world-down + align làm top↑0.79 mà z_f vẫn ~0.06
                actions = _osc_z_finger_descend(env, s, reach_mask, actions, descent_scale)
                pos_scale = float(getattr(env.cfg, "osc_position_scale", 0.06))
                descend_m = float(getattr(env.cfg, "grasp_descent_world_m", 0.010))
                mag = min(descend_m / max(pos_scale, 1e-4) * descent_scale, 1.0)
                xy_blend = float(getattr(env.cfg, "grasp_descent_xy_blend", 0.0))
                if xy_blend > 0.0:
                    xy_err = s["grasp_pos_w"][:, :2] - s["finger_pos_w"][:, :2]
                    xy_step = (xy_err / max(pos_scale, 1e-4)).clamp(-mag, mag) * xy_blend
                    actions[reach_mask, 0:2] = xy_step[reach_mask]
                align_blend = float(getattr(env.cfg, "grasp_descent_align_blend", 0.0))
                if align_blend > 0.0:
                    # Nhẹ hơn GRASP — tránh xoay quá thẳng trước khi vào GRASP
                    actions[reach_mask, 3:6] = (
                        _align_vertical_rot_action(env, s)[reach_mask] * align_blend * 0.35 * descent_scale
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
                        actions = _osc_z_finger_descend(env, s, fresh_m, actions, descent_scale)
                    if latched_m.any():
                        actions = _osc_z_finger_descend(env, s, latched_m, actions, descent_scale * 0.45)
                    # GRASP z-loop trước đây chỉ hạ Z → pad phải lệch (R d~0.073, L d~0.028)
                    pos_scale = float(getattr(env.cfg, "osc_position_scale", 0.06))
                    descend_m = float(getattr(env.cfg, "grasp_descent_world_m", 0.010))
                    mag = min(descend_m / max(pos_scale, 1e-4) * descent_scale, 1.0)
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

    assist_place = getattr(env.cfg, "assist_place", False)
    if assist_place and uses_place(getattr(env.cfg, "task_phase", 1)):
        # State machine PLACE chạy mỗi bước (kể cả khi assist tắt) để
        # telemetry/gripper release luôn có _place_phase nhất quán — mirror
        # cách _update_lift_state chạy vô điều kiện ở trên.
        moving = _update_place_state(env, s)
        w = max(min(scale, 1.0), 0.0)
        carrying = moving & (env._place_phase == PLACE_CARRY)
        descending = moving & (env._place_phase == PLACE_DESCEND)
        if w >= 1.0 - 1e-6:
            if carrying.any():
                actions = _osc_carry_to_bowl(env, s, carrying, actions, 1.0)
            if descending.any():
                actions = _osc_descend_to_bowl(env, s, descending, actions, 1.0)
        else:
            if carrying.any():
                bias = _osc_carry_to_bowl(env, s, carrying, torch.zeros_like(actions), 1.0)
                actions[carrying] = (actions[carrying] + w * bias[carrying]).clamp(-1.0, 1.0)
            if descending.any():
                bias = _osc_descend_to_bowl(env, s, descending, torch.zeros_like(actions), 1.0)
                actions[descending] = (actions[descending] + w * bias[descending]).clamp(-1.0, 1.0)
        holding_place = env._place_phase == PLACE_HOLDING
        if holding_place.any():
            actions[holding_place] = 0.0

    # Phase 27 (cơ chế phục hồi active-descent sau zf_drift-abort) ĐÃ THỬ VÀ
    # REVERT (Phase 29, terminal_command.md) — đo được không khớp chính xác
    # baseline dù chỉ đổi hành động lúc LIFT_IDLE. Giữ nguyên idle_hold gốc.

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
