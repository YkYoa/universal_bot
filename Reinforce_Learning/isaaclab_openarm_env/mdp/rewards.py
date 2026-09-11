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

import os
import torch
from isaaclab.envs import ManagerBasedRLEnv
from .helpers import (
    check_init_buffers,
    compute_state,
    STAGE_REACH,
    STAGE_GRASP,
    STAGE_PLACE,
    PLACE_IDLE,
    PLACE_CARRY,
    PLACE_DESCEND,
    PLACE_HOLDING,
    finger_grasp_ready,
    finger_symmetric_ready,
    finger_ready_for_close,
    grasp_lift_success_ready,
    place_in_bowl_success,
    place_release_ready,
    reach_align_ready,
    uses_grasp_lift,
    uses_place,
    format_bottle_debug,
    format_finger_debug,
)


def _log_reach_align_status(env: ManagerBasedRLEnv, s: dict, hold_steps: int) -> None:
    """Print alignment gate status for env 0 (demo / few-env debugging)."""
    if env.num_envs > 16:
        return
    reach_hold = getattr(env.cfg, "reach_stage_hold_steps", hold_steps)
    tol = getattr(env.cfg, "reach_advance_tolerance", 1.10)
    min_top = getattr(env.cfg, "reach_advance_min_top_down", 0.92)
    max_lat = getattr(env.cfg, "reach_advance_max_lateral", 0.04)
    max_lat_f = getattr(
        env.cfg, "reach_advance_max_lateral_f",
        getattr(env.cfg, "grasp_close_max_lat", 0.08),
    )
    use_finger_lat = getattr(env.cfg, "reach_advance_use_finger_lateral", False)
    max_z = getattr(env.cfg, "reach_advance_max_z_error", 0.065)
    max_z_descent = getattr(
        env.cfg, "reach_descent_max_z_error",
        getattr(env.cfg, "reach_advance_max_z_error", 0.08) * 1.5,
    )
    max_z_f = getattr(env.cfg, "reach_advance_max_z_finger", 0.10)
    min_lvl = getattr(env.cfg, "reach_advance_min_finger_level", 0.85)
    hold_ok = int(env._steps_in_contact[0].item()) >= reach_hold
    top_ok = float(s["top_down_align"][0].item()) > min_top
    lat_val = float(s["lateral_finger_xy" if use_finger_lat else "lateral_dist_xy"][0].item())
    lat_lim = max_lat_f if use_finger_lat else max_lat
    lat_ok = lat_val < lat_lim
    lat_tag = "lat_f" if use_finger_lat else "lat"
    lat_cmp = "<" if lat_ok else ">"
    z_val = float(s["z_error"][0].item())
    z_ok = (z_val < max_z) and (z_val > -0.02)
    z_descent_ok = (z_val < max_z_descent) and (z_val > -0.02)
    z_f_val = float(s["z_error_finger"][0].item())
    z_f_ok = (z_f_val < max_z_f) and (z_f_val > -0.03)
    z_f_soft = z_f_val < max_z_f * tol
    lvl_ok = float(s["finger_level"][0].item()) > min_lvl
    ready = hold_ok and top_ok and lat_ok and z_ok and z_f_ok and lvl_ok
    descent_ready = hold_ok and top_ok and lat_ok and z_descent_ok and lvl_ok
    soft_descent = hold_ok and top_ok and lat_ok and z_ok and z_f_soft and lvl_ok
    print(
        f"  [Align] hold:{int(env._steps_in_contact[0])}/{reach_hold}"
        f" top↓:{float(s['top_down_align'][0]):.2f}>{min_top}"
        f" {lat_tag}:{lat_val:.3f}{lat_cmp}{lat_lim}"
        f" z:{z_val:+.3f}<{max_z}"
        f" z_f:{z_f_val:+.3f}<{max_z_f} (soft<{max_z_f * tol:.3f})"
        f" lvl:{float(s['finger_level'][0]):.2f}>{min_lvl}"
        f" → {'READY' if ready else ('DESCEND' if descent_ready and z_f_val > max_z_f * tol else 'waiting')}"
    )


def _log_success_status(env: ManagerBasedRLEnv, s: dict) -> None:
    """Print phase-2 success gate progress for env 0 (lift + grip hold)."""
    if env.num_envs > 16 or not getattr(env.cfg, "debug_success_log", False):
        return
    if env._stage[0] != STAGE_GRASP:
        return

    grip_thresh = getattr(env.cfg, "grasp_grip_threshold", 0.4)
    lift_thresh = getattr(env.cfg, "grasp_lift_threshold", 0.03)
    lift_hold = getattr(env.cfg, "grasp_lift_hold_steps", 5)
    max_lat = getattr(env.cfg, "grasp_close_max_lat", 0.040)
    max_z = getattr(env.cfg, "grasp_close_max_z_err", 0.04)
    min_top = getattr(env.cfg, "grasp_close_min_top_down", 0.88)

    grip = float(s["gripper_state"][0].item())
    lift_m = float(s["bottle_lift"][0].item())
    lift_steps = int(env._steps_bottle_lifted[0].item())
    lat_f = float(s["lateral_finger_xy"][0].item())
    z_f = float(s["z_error_finger"][0].item())
    top_down = float(s["top_down_align"][0].item())
    dist_f = float(s["dist_finger_body"][0].item())
    dist_ee = float(s["dist_ee_bottle"][0].item())
    finger_ok = bool(finger_grasp_ready(env, s)[0].item())
    sym_ok = (
        bool(finger_symmetric_ready(env, s)[0].item())
        if getattr(env.cfg, "grasp_symmetry_gate_enabled", False)
        else True
    )
    close_ok = bool(finger_ready_for_close(env, s)[0].item())

    gripped = grip > grip_thresh
    lifted = lift_m > lift_thresh
    lift_ok = bool(grasp_lift_success_ready(env, s)[0].item())
    success = lift_steps >= lift_hold
    max_dist_f = getattr(env.cfg, "grasp_success_max_dist_finger", 0.10)
    max_dist_ee = getattr(env.cfg, "grasp_success_max_dist_ee", 0.12)
    min_top_succ = getattr(env.cfg, "grasp_success_min_top_down", 0.50)
    max_tilt = getattr(env.cfg, "grasp_success_max_tilt_deg", 12.0)
    tilt_deg = float(s["bottle_tilt_deg"][0].item())

    if success:
        status = "✅ SUCCESS (lift hold met)"
    elif lift_ok:
        status = f"counting hold {lift_steps}/{lift_hold}"
    elif gripped and lifted and not lift_ok:
        status = "lift/contact invalid — not counting"
    elif gripped and lifted and tilt_deg >= max_tilt:
        status = f"tilt too high ({tilt_deg:.1f}°>={max_tilt}°) — not success"
    elif gripped and not lifted:
        status = "gripped — need lift"
    elif finger_ok and not sym_ok and not gripped:
        status = "align symmetric pads (L/R)"
    elif close_ok and not gripped:
        status = "finger OK — close grip"
    elif finger_ok and not gripped:
        status = "finger OK — close grip"
    else:
        status = "align fingers / descend"

    print(
        f"  [Success] {status}"
        f" | Δlift:{lift_m:.3f}m>{lift_thresh}"
        f" grip:{grip:.2f}>{grip_thresh}"
        f" hold:{lift_steps}/{lift_hold}"
        f" | dist_f:{dist_f:.3f}<{max_dist_f}"
        f" dist:{dist_ee:.3f}<{max_dist_ee}"
        f" lat_f:{lat_f:.3f} top↓:{top_down:.2f}>{min_top_succ}"
        f" tilt:{tilt_deg:.1f}°<{max_tilt}"
        f" finger_ready:{finger_ok} sym:{sym_ok}"
        f" L/R:{float(s['dist_left_body'][0]):.3f}/{float(s['dist_right_body'][0]):.3f}"
    )
    print(f"  [Bottle] {format_bottle_debug(env, s, env_id=0)}")
    print(f"  [Fingers] {format_finger_debug(s, env_id=0, env=env)}")


def _update_contact_and_stages(env: ManagerBasedRLEnv, s: dict, hold_steps: int) -> torch.Tensor:
    """Hysteresis contact zone, hold counter, and reach→grasp stage transitions."""
    dist_ee_bottle = s["dist_ee_bottle"]
    contact_enter = getattr(env.cfg, "success_dist_threshold", 0.07)
    contact_exit = getattr(env.cfg, "success_contact_exit_threshold", 0.10)

    if not hasattr(env, "_in_contact_zone"):
        env._in_contact_zone = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._in_contact_zone |= dist_ee_bottle < contact_enter
    env._in_contact_zone &= dist_ee_bottle <= contact_exit

    in_contact = env._in_contact_zone
    env._steps_in_contact[in_contact] += 1
    env._steps_in_contact[~in_contact] = 0

    task_phase = getattr(env.cfg, "task_phase", 1)
    if uses_grasp_lift(task_phase):
        reach_hold = getattr(env.cfg, "reach_stage_hold_steps", hold_steps)
        aligned = reach_align_ready(env, s, soft=False)
        advance = (env._stage == STAGE_REACH) & (env._steps_in_contact >= reach_hold) & aligned
        log_iv = getattr(env.cfg, "align_log_interval", 50)
        if getattr(env.cfg, "suppress_align_log", False):
            pass
        elif env.num_envs <= 16 and in_contact[0] and (env._stage[0] == STAGE_REACH):
            step_ct = int(env._steps_in_contact[0].item())
            if advance[0] or (step_ct % log_iv == 0):
                _log_reach_align_status(env, s, hold_steps)
        if env.num_envs <= 16 and not getattr(env.cfg, "suppress_align_log", False) and advance[0]:
            print(
                f"  [Stage] REACH → GRASP @ step {env.step_counter}"
                f" ep_step:{int(env.episode_length_buf[0])}"
                f" | z_f:{float(s['z_error_finger'][0]):+.3f}"
                f" lat_f:{float(s['lateral_finger_xy'][0]):.3f}"
                f" top↓:{float(s['top_down_align'][0]):.2f}"
            )
        env._stage[advance] = STAGE_GRASP
        env._steps_in_contact[advance] = 0

        if getattr(env.cfg, "debug_success_log", False) and env.num_envs <= 16 and advance.any():
            for idx in advance.nonzero(as_tuple=False).flatten().tolist():
                if idx == 0:
                    print(f"  [Success] GRASP stage started — need lift>{getattr(env.cfg, 'grasp_lift_threshold', 0.03)}m"
                          f" + grip>{getattr(env.cfg, 'grasp_grip_threshold', 0.4)}"
                          f" for {getattr(env.cfg, 'grasp_lift_hold_steps', 5)} steps")

    if uses_place(task_phase):
        # Tái dùng ĐÚNG tín hiệu "lift-hold đã đạt" mà trước đây dùng để
        # TERMINATE episode (xem success_termination) — giờ chỉ dùng để
        # chuyển sang STAGE_PLACE, không kết thúc episode nữa. An toàn với
        # task_phase==2: uses_place trả False nên khối này là dead code,
        # env._stage không bao giờ chạm STAGE_PLACE.
        lift_hold = getattr(env.cfg, "grasp_lift_hold_steps", 5)
        advance_place = (env._stage == STAGE_GRASP) & (env._steps_bottle_lifted >= lift_hold)
        if getattr(env.cfg, "debug_success_log", False) and env.num_envs <= 16 and advance_place.any():
            for idx in advance_place.nonzero(as_tuple=False).flatten().tolist():
                if idx == 0:
                    print(f"  [Success] PLACE stage started — mang chai tới bát")
        env._stage[advance_place] = STAGE_PLACE
        env._steps_bottle_lifted[advance_place] = 0  # PLACE dùng counter settle riêng

    env._steps_in_grasp[env._stage == STAGE_GRASP] += 1
    env._steps_in_grasp[env._stage != STAGE_GRASP] = 0

    # Số bước kể từ lúc ĐÃ LATCH (không phải từ lúc vào GRASP) — dùng để suy
    # giảm camp reward CHỈ cho thời gian đứng yên SAU KHI đã kẹp xong, không
    # phạt nhầm thời gian tiếp cận/đóng kẹp cần thiết trước đó (xem bug ở
    # _compute_grasp_reward: đo thật bằng DEBUG_DECAY cho thấy latch chỉ xảy
    # ra ở steps_in_grasp=136-371, tức decay cũ (neo vào steps_in_grasp,
    # decay_n=200) đã gần hết hạn NGAY TẠI thời điểm latch — giết tín hiệu
    # thưởng vị trí/hướng trong suốt cả quá trình nhấc sau đó, đúng lúc cần
    # nhất). `_grasp_latched` là bộ tích luỹ OR (chỉ False khi reopen thật sự
    # xảy ra, có cooldown/max-count riêng) nên không dao động nhanh như
    # `latched` tức thời — an toàn để neo trực tiếp mà không tái tạo lỗ hổng
    # dao động biên đã gặp ở v1/v2.
    gripper_term = env.action_manager._terms.get("gripper_action")
    latched_now = (
        gripper_term._grasp_latched
        if gripper_term is not None and hasattr(gripper_term, "_grasp_latched")
        else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    env._steps_since_latch[latched_now] += 1
    env._steps_since_latch[~latched_now] = 0

    # Tổng thời gian ở STAGE_REACH cả episode — dùng cho suy giảm reward REACH
    # (xem _compute_reach_reward). Đơn điệu, không phụ thuộc khoảng cách/trạng
    # thái tức thời nào nên không có "vùng an toàn" nào để né.
    env._steps_in_reach[env._stage == STAGE_REACH] += 1
    env._steps_in_reach[env._stage != STAGE_REACH] = 0

    # Số bước kể từ lúc THỰC SỰ bắt đầu PLACE (_place_phase != IDLE) — mirror
    # đúng cách _steps_since_latch neo vào mốc thật thay vì thời gian vào
    # stage, tránh tái tạo bug decay đã sửa ở GRASP (xem Phase 8). Đơn điệu:
    # reset về 0 khi state machine PLACE rơi về IDLE (must_abort thật).
    place_phase = getattr(env, "_place_phase", None)
    if place_phase is not None:
        if not hasattr(env, "_steps_since_place_start"):
            env._steps_since_place_start = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        moving_or_holding = place_phase != PLACE_IDLE
        env._steps_since_place_start[moving_or_holding] += 1
        env._steps_since_place_start[~moving_or_holding] = 0

        # S5 — số bước liên tiếp chai đã "nằm yên trong bát" SAU KHI đã thả
        # tay (gripper mở) — dùng làm điều kiện settle cho cả thành công thật
        # (success_termination) lẫn thất bại thật (bottle_misplaced_termination).
        # Neo vào place_in_bowl_success (vị trí/tốc độ/nghiêng thật) VÀ gripper
        # đã mở — không neo vào thời gian hay vào stage, tránh lặp lại bug
        # decay đã sửa ở GRASP (Phase 8) và lỗi lệch tâm bát vừa sửa (Phase 11).
        if not hasattr(env, "_steps_bottle_settled"):
            env._steps_bottle_settled = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        if not hasattr(env, "_steps_bottle_misplaced"):
            env._steps_bottle_misplaced = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        # BUG đã sửa (phát hiện qua demo thật task_phase=3, mọi episode chết ở
        # 13 bước): thiếu gate theo stage khiến "gripper mở + chai đứng yên"
        # ở REACH (TRƯỚC KHI từng chạm chai) cũng thoả at_rest & ~in_bowl —
        # bottle_misplaced_termination coi "chưa từng gắp" là "đã làm rơi".
        # Chỉ tính settle/misplaced khi ĐANG ở STAGE_PLACE (đã từng CARRY ít
        # nhất một lần, không phải chưa từng bắt đầu).
        in_place_stage = env._stage == STAGE_PLACE
        grip_threshold = getattr(env.cfg, "grasp_grip_threshold", 0.4)
        released = s["gripper_state"] < grip_threshold
        max_settle_speed = float(getattr(env.cfg, "place_success_max_speed", 0.15))
        at_rest = in_place_stage & released & (s["bottle_lin_speed"] < max_settle_speed)
        in_bowl = place_in_bowl_success(env, s)
        settled_now = at_rest & in_bowl
        misplaced_now = at_rest & ~in_bowl
        env._steps_bottle_settled[settled_now] += 1
        env._steps_bottle_settled[~settled_now] = 0
        env._steps_bottle_misplaced[misplaced_now] += 1
        env._steps_bottle_misplaced[~misplaced_now] = 0

    return in_contact


def _compute_reach_reward(
    env: ManagerBasedRLEnv,
    s: dict,
    in_contact: torch.Tensor,
    hold_steps: int,
) -> torch.Tensor:
    """Phase 1 — approach bottle top with top-down orientation."""
    dist_ee_bottle = s["dist_ee_bottle"]
    lateral_dist_xy = s["lateral_dist_xy"]
    z_error = s["z_error"]
    table_clearance = s["table_clearance"]
    top_down_align = s["top_down_align"]
    finger_level = s["finger_level"]
    descent_align = s["descent_align"]
    top_down_min = getattr(env.cfg, "top_down_min_for_close_reward", 0.65)

    orient_quality = top_down_align.clamp(0.0, 1.0) * finger_level

    joint_vel = env._robot.data.joint_vel[:, env._arm_joint_ids]
    vel_penalty = -0.005 * torch.sum(joint_vel ** 2, dim=-1)
    vel_penalty = vel_penalty + torch.where(
        in_contact, -0.015 * torch.sum(joint_vel ** 2, dim=-1), 0.0
    )

    low_sweep_depth = torch.clamp(-table_clearance, min=0.0, max=0.08)
    table_penalty = -5.0 * (low_sweep_depth / 0.08) ** 2

    near_bottle = dist_ee_bottle < 0.12
    orient_gate = torch.where(near_bottle, orient_quality.clamp(min=0.05), torch.ones_like(orient_quality))

    r_reach = ((1.0 - dist_ee_bottle.clamp(max=1.0)) * 3.0 + torch.exp(-8.0 * dist_ee_bottle) * 3.0) * orient_gate
    r_lateral = (
        torch.exp(-40.0 * lateral_dist_xy) * 12.0
        + (1.0 - lateral_dist_xy.clamp(max=0.15)) * 4.0
    ) * orient_gate

    above_target = torch.clamp(z_error, min=0.0)
    r_z_descent = torch.exp(-20.0 * above_target) * 4.0
    final_descent = (dist_ee_bottle < 0.08) & (top_down_align > 0.9)
    r_z_descent = r_z_descent + torch.where(
        final_descent, torch.exp(-60.0 * above_target) * 10.0, 0.0
    )
    below_target = torch.clamp(-z_error, min=0.0)
    r_z_floor = -2.0 * (below_target / 0.05) ** 2

    r_close = torch.exp(-80.0 * dist_ee_bottle) * 20.0 * orient_gate
    dist_improvement = env._prev_dist_ee_bottle - dist_ee_bottle
    r_progress = torch.clamp(dist_improvement * 40.0, min=-2.0, max=5.0)

    milestones = (
        (dist_ee_bottle < 0.10).float() * 3.0 +
        (dist_ee_bottle < 0.08).float() * 6.0 +
        (dist_ee_bottle < 0.07).float() * 10.0 +
        (dist_ee_bottle < 0.06).float() * 15.0 +
        (dist_ee_bottle < 0.04).float() * 25.0 +
        (dist_ee_bottle < 0.03).float() * 40.0
    ) * orient_gate

    r_stall = -5.0 * torch.clamp(dist_ee_bottle - 0.05, min=0.0)
    r_hold = (env._steps_in_contact.float() / hold_steps).clamp(max=1.0) * 10.0

    close_mask = (dist_ee_bottle < 0.15).float()
    r_top_down = (
        top_down_align.clamp(min=0.0) * 4.0
        + top_down_align.clamp(min=0.0) * close_mask * 25.0
        + (top_down_align > top_down_min).float() * close_mask * 10.0
    )
    r_finger_level = finger_level * close_mask * 8.0
    r_descent = descent_align * close_mask * 6.0

    tilt_shortfall = (top_down_min - top_down_align).clamp(min=0.0)
    bad_tilt = near_bottle & (lateral_dist_xy < 0.10)
    r_tilt_penalty = torch.zeros_like(dist_ee_bottle)
    r_tilt_penalty[bad_tilt] = -25.0 * tilt_shortfall[bad_tilt]

    env._prev_dist_ee_bottle = dist_ee_bottle.detach()

    # Họ exploit REACH-camping (3 biến thể xác nhận qua các run 5M bước thật):
    #   #5  — đứng yên đúng ở ngưỡng advance (đủ hold nhưng cố tình lệch align)
    #   #5b — "bơm": ra/vào vùng hysteresis contact liên tục để reset counter
    #   #5c — đứng ngoài vùng hysteresis (0.07 < dist < 0.10) — không bao giờ
    #         trigger `_in_contact_zone` nên né được MỌI cơ chế khoá theo vị
    #         trí/trạng thái tức thời (grasp_rate=0 suốt ~1 triệu bước trong
    #         khi ep_rew_mean tăng đều lên hơn 1000).
    # Mỗi lần vá theo kiểu "surgical" (chỉ suy giảm khi ở một vùng/trạng thái
    # cụ thể) đều bị lách bằng một trạng thái khác không bị khoá. Fix: suy
    # giảm TOÀN BỘ reward REACH theo TỔNG thời gian ở STAGE_REACH cả episode.
    #
    # BUG HIỆU CHỈNH LẦN 1 (xác nhận qua run 5M local thật): đặt onset=1000,
    # decay=200 → về 0 đúng tại step 1200 — TRÙNG KHÍT `env.max_episode_length`
    # (episode_length_s=20s / step_dt=1/60 = 1200 bước), dựa trên comment cũ
    # CHƯA TỪNG ĐO THẬT ("REACH cần ~700-1000 bước"). Exploit #5c vẫn sống
    # (grasp_rate sụp về 0.3, ep_rew_mean lên 810 ở step 3M).
    #
    # ĐÃ ĐO THẬT bằng log `ep_step` (episode_length_buf lúc advance, chạy với
    # assist đầy đủ scale=1.0 — script luôn làm đúng): REACH thật chỉ mất
    # **21-59 bước**, không phải 700-1000 — sai lệch 15-30 LẦN, đó là lý do
    # camping vẫn cực lời dù có "suy giảm". Thời gian REACH do động lực học
    # cánh tay quyết định, KHÔNG tỉ lệ với episode_length_s, nên dùng số bước
    # tuyệt đối (biên độ an toàn ~3-5 lần so với 59 đo được) thay vì tỉ lệ
    # theo max_episode_length.
    reach_steps = env._steps_in_reach.float()
    onset = float(getattr(env.cfg, "reach_reward_decay_onset_steps", 200))
    decay_n = float(getattr(env.cfg, "reach_reward_decay_steps", 100))
    reach_decay = (1.0 - (reach_steps - onset).clamp(min=0.0) / max(decay_n, 1.0)).clamp(0.0, 1.0)

    total = (
        r_reach + r_lateral + r_z_descent + r_z_floor
        + r_progress + milestones + r_close + r_top_down + r_finger_level + r_descent + r_hold
        + r_stall + r_tilt_penalty + table_penalty + vel_penalty
    )
    return total * reach_decay


def _compute_grasp_reward(env: ManagerBasedRLEnv, s: dict, in_contact: torch.Tensor) -> torch.Tensor:
    """Phase 2 — align fingertips with bottle body, close, and lift."""
    dist_finger_body = s["dist_finger_body"]
    lateral_finger_xy = s["lateral_finger_xy"]
    z_error_finger = s["z_error_finger"]
    gripper_state = s["gripper_state"]
    bottle_lift = s["bottle_lift"]
    table_clearance = s["table_clearance"]
    top_down_align = s["top_down_align"]
    in_grasp = env._stage == STAGE_GRASP
    finger_ready = finger_grasp_ready(env, s)

    grip_threshold = getattr(env.cfg, "grasp_grip_threshold", 0.4)
    lift_threshold = getattr(env.cfg, "grasp_lift_threshold", 0.03)
    lift_hold_steps = getattr(env.cfg, "grasp_lift_hold_steps", 5)

    gripped = gripper_state > grip_threshold
    lifted = bottle_lift > lift_threshold
    lift_ok = grasp_lift_success_ready(env, s)
    env._steps_bottle_lifted[lift_ok] += 1
    env._steps_bottle_lifted[~lift_ok] = 0

    if getattr(env.cfg, "debug_success_log", False) and env.num_envs <= 16 and in_grasp[0]:
        lift_hold = getattr(env.cfg, "grasp_lift_hold_steps", 5)
        lift_steps = int(env._steps_bottle_lifted[0].item())
        prev = int(getattr(env, "_dbg_prev_lift_steps", 0))
        log_iv = getattr(env.cfg, "success_log_interval", 25)
        in_grasp_steps = int(env._steps_in_grasp[0].item()) if hasattr(env, "_steps_in_grasp") else 0
        milestone = (
            lift_steps != prev
            or lift_steps >= lift_hold
            or (gripped[0] and not getattr(env, "_dbg_logged_grip", False))
            or (lifted[0] and not getattr(env, "_dbg_logged_lift", False))
            or (in_grasp_steps % log_iv == 0)
        )
        if milestone:
            _log_success_status(env, s)
            if gripped[0]:
                env._dbg_logged_grip = True
            if lifted[0]:
                env._dbg_logged_lift = True
        env._dbg_prev_lift_steps = lift_steps
        if lift_steps >= lift_hold and prev < lift_hold:
            print(
                f"  [Success] 🎉 TERMINATION READY — bottle lifted {lift_steps}/{lift_hold} steps"
                f" | Δlift:{float(bottle_lift[0]):.3f}m grip:{float(gripper_state[0]):.2f}"
            )

    # Reward policy gripper command (dim 7): negative = close
    r_grip_cmd = torch.zeros_like(dist_finger_body)
    if "gripper_action" in env.action_manager._terms:
        grip_cmd = env.action_manager._terms["gripper_action"].raw_actions[:, 0]
        r_grip_cmd = torch.where(
            in_grasp & finger_ready,
            (-grip_cmd).clamp(-1.0, 1.0) * 20.0,
            0.0,
        )

    joint_vel = env._robot.data.joint_vel[:, env._arm_joint_ids]
    vel_penalty = -0.008 * torch.sum(joint_vel ** 2, dim=-1)

    low_sweep_depth = torch.clamp(-table_clearance, min=0.0, max=0.08)
    table_penalty = -8.0 * (low_sweep_depth / 0.08) ** 2

    r_align = torch.exp(-40.0 * lateral_finger_xy) * 15.0
    r_align = r_align + torch.exp(-50.0 * dist_finger_body) * 20.0
    above_body = torch.clamp(z_error_finger, min=0.0)
    r_descend = torch.exp(-35.0 * above_body) * 25.0
    r_descend = r_descend + torch.where(above_body < 0.02, 15.0, 0.0)
    r_hover = -25.0 * torch.clamp(z_error_finger - 0.02, min=0.0)
    below_body = torch.clamp(-z_error_finger, min=0.0)
    r_z_floor = -3.0 * (below_body / 0.03) ** 2

    r_grip_close = gripper_state * 15.0
    r_grip_close = r_grip_close + torch.where(finger_ready, gripper_state * 25.0, 0.0)
    r_grip_close = r_grip_close + torch.where(gripped & finger_ready, 30.0, 0.0)
    # Penalise closing on empty air
    r_air_grasp = torch.where((gripper_state > 0.5) & ~finger_ready, -30.0, 0.0)

    # min=0.0: the gripper pressing down produces small negative lifts, and an
    # unclamped negative here pays a standing penalty for a bottle at rest.
    r_lift = torch.clamp(bottle_lift / lift_threshold, min=0.0, max=1.5) * 30.0
    r_lift = r_lift + (bottle_lift > lift_threshold).float() * 25.0
    lift_scale = float(getattr(env.cfg, "grasp_reward_lift_scale", 1.0))
    r_lift = r_lift * lift_scale

    r_track = torch.exp(-15.0 * dist_finger_body) * 8.0 * gripped.float()

    hold_scale = float(getattr(env.cfg, "grasp_reward_lift_hold_scale", 1.0))
    r_lift_hold = (
        (env._steps_bottle_lifted.float() / lift_hold_steps).clamp(max=1.0) * 40.0 * hold_scale
    )

    r_bad_grip = torch.where(
        (gripper_state < 0.2) & finger_ready & in_grasp & (bottle_lift > 0.005),
        -20.0,
        0.0,
    )
    r_lift_no_grip = torch.where(lifted & ~gripped, -20.0, 0.0)

    r_orient = torch.where(z_error_finger < 0.03, top_down_align.clamp(min=0.0) * 2.0, 0.0)

    ee_vel_z = env._robot.data.body_lin_vel_w[:, env._ee_body_id, 2]
    r_lift_motion = torch.where(gripped, ee_vel_z.clamp(min=0.0) * 15.0, 0.0)
    r_up_before_grip = torch.where(in_grasp & ~gripped, ee_vel_z.clamp(min=0.0) * (-10.0), 0.0)

    # ── Camping annuity decay (v3 — bỏ ngoại lệ lifted) ─────────────────────
    # Trước đây các số hạng shaping dưới đây được trả MÃI MÃI sau khi đã kẹp
    # xong, tổng ~+121 raw/bước cho việc không làm gì. Cộng với việc thành công
    # kết thúc episode (V=0, không bootstrap, không bonus), đứng yên có giá trị
    # gấp 5.8 lần hoàn thành nhiệm vụ — agent tối ưu đúng theo tín hiệu đó.
    #
    # v1 neo vào `latched`, v2 neo vào `finger_ready|latched` — CẢ HAI bị lợi
    # dụng qua dao động biên (policy noise flick qua lại điều kiện tức thời,
    # reset bộ đếm về 0 mỗi lần). v2b đổi sang `_steps_in_grasp` (đơn điệu, chỉ
    # reset khi episode reset thật — miễn nhiễm dao động, ĐÚNG hướng) nhưng vẫn
    # sụp (latch 0.73→0.27 trong 280k bước) vì lỗ hổng THẬT SỰ nằm chỗ khác:
    # `camp_factor` có ngoại lệ "trừ khi lifted", mà `lifted = bottle_lift >
    # 0.03m` chỉ là NGƯỠNG VỊ TRÍ THÔ, không kiểm tra có đang kẹp không — chai
    # 95g chỉ cần bị cánh tay hất nảy lên >3cm (không cần kẹp) là đủ bật "miễn
    # suy giảm" trở lại VĨNH VIỄN. Không phải lỗi ở anchor `_steps_in_grasp`
    # (nó vẫn đúng và miễn nhiễm dao động) — lỗi ở việc THÊM một lối thoát mới
    # (giả lift) sau khi đã bịt lối thoát cũ (dao động biên).
    #
    # v3: BỎ hẳn ngoại lệ lifted. Phần thưởng nhấc thật đã có r_lift/
    # r_lift_hold/terminal_success_bonus riêng (không nằm trong static_shaping)
    # nên không cần "miễn suy giảm" ở đây nữa — suy giảm áp dụng đều cho MỌI
    # trạng thái trong GRASP, không phân biệt lifted giả hay thật.
    # v4: BỎ HẲN r_camp (phạt tích luỹ theo bước). Tính bằng số cho thấy phạt
    # lật chai một-lần (-30) LUÔN rẻ hơn phạt camp dồn tới hết episode (−600
    # đến −4200 tuỳ lúc vào GRASP sớm/muộn) — agent học cách CỐ TÌNH lật chai
    # để kết thúc episode sớm, né phạt camp. Đây là lỗi cấu trúc: bất kỳ phạt
    # một-lần nào rẻ hơn phạt-tích-luỹ đang chạy đều biến thành lối thoát.
    #
    # Sửa gốc: suy giảm static_shaping về 0 là ĐỦ để triệt tiêu động cơ camping
    # (không còn gì để farm) — không cần thêm phạt escalating tạo ra "chi phí
    # cần trốn". Động lực tiến tới lift giờ đến từ CHI PHÍ CƠ HỘI (bỏ lỡ
    # r_lift/r_lift_hold/+60 bonus), không phải từ áp lực né một hình phạt.
    # tipped_penalty (-30, terminal, xem terminal_tipped_penalty) vẫn giữ
    # nguyên vai trò răn đe — giờ nó bị thống trị hoàn toàn bởi "không làm gì"
    # (0 phạt) vì không còn phạt camp dồn để so sánh rẻ hơn nữa.
    # v5: neo vào _steps_since_latch (bước kể từ lúc ĐÃ LATCH), không phải
    # _steps_in_grasp (bước kể từ lúc VÀO GRASP). Đo thật bằng DEBUG_DECAY
    # (2026-09-06, sau khi sửa gripper actuator đối xứng — grasp/lift lần đầu
    # hoạt động đủ tốt để lộ ra bug này): latch thật xảy ra ở steps_in_grasp=
    # 136-371 — với anchor cũ, decay (decay_n=200, KHÔNG có onset) đã gần/
    # bằng 0 NGAY TẠI thời điểm latch, giết static_shaping (align/descend/
    # grip_close/track/orient) trong suốt toàn bộ quá trình nhấc sau đó
    # (lift là sub-mode của STAGE_GRASP nên _steps_in_grasp tiếp tục tăng
    # xuyên suốt). Xác nhận bằng training thật 5M bước: latch_rate giảm ĐỀU
    # 0.28→0.00 trong ~500k bước trong khi ep_rew_mean TĂNG 202→228 — chữ ký
    # kinh điển của "bỏ dở giữa chừng lời hơn đi trọn" khi phần thưởng cho
    # đi trọn đã bị decay giết trước khi kịp tính. `_steps_since_latch` reset
    # về 0 khi CHƯA latch (full reward suốt thời gian tiếp cận/đóng kẹp cần
    # thiết) và chỉ bắt đầu đếm SAU KHI latch — đúng ý định gốc trong comment
    # bên dưới ("trả MÃI MÃI SAU KHI ĐÃ KẸP XONG").
    grasp_steps = env._steps_since_latch.float()
    decay_n = float(getattr(env.cfg, "grasp_camp_decay_steps", 200))
    decay = (1.0 - grasp_steps / max(decay_n, 1.0)).clamp(0.0, 1.0)

    if os.environ.get("DEBUG_DECAY") == "1" and env.num_envs <= 16:
        term = env.action_manager._terms.get("gripper_action")
        latched_now = term._grasp_latched if term is not None and hasattr(term, "_grasp_latched") else torch.zeros_like(in_grasp)
        just_latched = latched_now & ~getattr(env, "_dbg_prev_latched", torch.zeros_like(latched_now))
        env._dbg_prev_latched = latched_now.clone()
        lift_phase = getattr(env, "_lift_phase", None)
        for i in just_latched.nonzero(as_tuple=False).flatten().tolist():
            print(
                f"  [DecayDbg] env{i} LATCH tại steps_since_latch={int(grasp_steps[i])} "
                f"decay={float(decay[i]):.3f} (decay_n={decay_n:.0f})",
                flush=True,
            )
        just_holding = (lift_phase == 2) & (getattr(env, "_dbg_prev_lift_phase", torch.full_like(lift_phase, -1)) != 2) if lift_phase is not None else torch.zeros_like(in_grasp)
        if lift_phase is not None:
            for i in just_holding.nonzero(as_tuple=False).flatten().tolist():
                print(
                    f"  [DecayDbg] env{i} HOLDING (lift ổn định) tại steps_since_latch={int(grasp_steps[i])} "
                    f"decay={float(decay[i]):.3f}",
                    flush=True,
                )
            env._dbg_prev_lift_phase = lift_phase.clone()

    static_shaping = (
        r_align + r_descend + r_hover + r_grip_close + r_grip_cmd + r_track + r_orient
    ) * decay

    return (
        static_shaping
        + r_lift + r_lift_hold + r_lift_motion
        + r_z_floor + r_bad_grip + r_lift_no_grip + r_air_grasp + r_up_before_grip
        + table_penalty + vel_penalty
    )


def _compute_place_reward(env: ManagerBasedRLEnv, s: dict, in_contact: torch.Tensor) -> torch.Tensor:
    """Phase 3 — mang chai đã kẹp qua bát, thả xuống, giữ ổn định.

    S4 (Phase 10 trong plan): các thành phần 1-4 dùng tín hiệu đã đo/có sẵn
    (dist_bottle_bowl_xy, height_above_bowl_floor, gripper_state) nên đáng
    tin ngay. Thành phần release-quality/settle (5-6) dùng ngưỡng CHƯA ĐO
    thật (place_success_* — xem bảng "đã đo hay đoán" trong plan) nên giữ
    biên độ nhỏ, không phải nguồn động lực chính — tiêu chí thành công thật
    và terminal bonus/penalty để dành cho S5 sau khi đo hình học bát (S9).
    """
    dist_bottle_bowl = s["dist_bottle_bowl"]
    dist_bottle_bowl_xy = s["dist_bottle_bowl_xy"]
    height_above_bowl = s["height_above_bowl_floor"]
    gripper_state = s["gripper_state"]
    grip_threshold = getattr(env.cfg, "grasp_grip_threshold", 0.4)
    gripped = gripper_state > grip_threshold

    place_phase = getattr(env, "_place_phase", None)
    if place_phase is None:
        place_phase = torch.full((env.num_envs,), PLACE_IDLE, dtype=torch.long, device=env.device)
    carrying = place_phase == PLACE_CARRY
    descending = place_phase == PLACE_DESCEND
    holding_place = place_phase == PLACE_HOLDING
    moving = carrying | descending

    # 1. Tiến triển XY về bát (mirror r_progress của REACH: thưởng ĐỘ GIẢM
    # khoảng cách mỗi bước, không phải khoảng cách tuyệt đối — tránh thưởng
    # đứng yên gần bát ngay từ đầu nếu random spawn tình cờ gần).
    dist_improvement = env._prev_dist_bottle_bowl - dist_bottle_bowl
    r_carry_progress = torch.clamp(dist_improvement * 40.0, min=-2.0, max=5.0)
    env._prev_dist_bottle_bowl = dist_bottle_bowl.detach()

    # 2. Hội tụ XY (giá trị tuyệt đối, để có gradient ổn định gần đích chứ
    # không chỉ lúc đang tiến gần) — chỉ tính khi đang carry/descend.
    r_xy_converge = torch.where(moving, torch.exp(-30.0 * dist_bottle_bowl_xy) * 15.0, 0.0)

    # 3. Hạ xuống đúng trên bát — chỉ kích hoạt khi đã DESCEND (đã hội tụ XY
    # đủ để bắt đầu hạ theo state machine, xem _update_place_state).
    r_descend_over_bowl = torch.where(
        descending, torch.exp(-20.0 * height_above_bowl.clamp(min=0.0)) * 20.0, 0.0
    )

    # 4. Giữ kẹp đóng trong lúc mang — suy giảm theo mốc THẬT (bước kể từ lúc
    # PLACE thực sự bắt đầu, không phải từ lúc vào stage) để không lặp lại
    # bug decay đã sửa ở GRASP (Phase 8): camp "mang mãi không thả" sẽ hết
    # thưởng sau place_camp_decay_steps, nhưng thời gian carry/descend cần
    # thiết (đo thật ở S9) không bị phạt nhầm.
    place_steps = getattr(env, "_steps_since_place_start", torch.zeros(env.num_envs, device=env.device)).float()
    decay_n = float(getattr(env.cfg, "place_camp_decay_steps", 150))
    decay = (1.0 - place_steps / max(decay_n, 1.0)).clamp(0.0, 1.0)
    r_hold_grip = torch.where(moving & gripped, 15.0, 0.0) * decay

    # 5. Chất lượng thả — one-time khi vừa release (gripper vừa mở lúc đang
    # HOLDING và place_release_ready). Thưởng theo độ chính xác XY/độ cao tại
    # thời điểm mở — milestone tiers mirror REACH, ngưỡng khoảng cách dùng
    # place_success_xy_radius_m (CHƯA đo thật, xem S9).
    if not hasattr(env, "_dbg_prev_gripped_place"):
        env._dbg_prev_gripped_place = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    just_released = holding_place & env._dbg_prev_gripped_place & ~gripped
    env._dbg_prev_gripped_place = gripped.clone()
    xy_radius = float(getattr(env.cfg, "place_success_xy_radius_m", 0.10))
    r_release_quality = torch.where(
        just_released,
        (
            (dist_bottle_bowl_xy < xy_radius * 2.0).float() * 10.0
            + (dist_bottle_bowl_xy < xy_radius).float() * 20.0
            + (dist_bottle_bowl_xy < xy_radius * 0.5).float() * 30.0
        ),
        0.0,
    )

    # 6. Settle sau khi thả — nhỏ, capped, chỉ khi đã ở gần bát và tốc độ
    # chai đang giảm. Không phải nguồn thưởng chính (đó là terminal bonus
    # thật ở S5) — chỉ giúp policy không vội bỏ đi ngay sau khi mở kẹp.
    bottle_speed = s["bottle_lin_speed"]
    max_settle_speed = float(getattr(env.cfg, "place_success_max_speed", 0.15))
    settling = ~gripped & (dist_bottle_bowl_xy < xy_radius) & (bottle_speed < max_settle_speed)
    r_settle = torch.where(settling, 8.0, 0.0)

    return r_carry_progress + r_xy_converge + r_descend_over_bowl + r_hold_grip + r_release_quality + r_settle


def _update_bottle_rest_baseline(env: ManagerBasedRLEnv) -> None:
    """Track the bottle's settled resting height during the post-reset drop.

    ``_bottle_rest_z`` seeds from the config spawn z, which sits ~12mm above the
    table surface — the reset writes the *commanded* teleport target, before
    physics has run, so the bottle then free-falls onto the table and every
    subsequent ``bottle_lift`` reads a constant -0.012m. That silently inflates
    the success threshold by 40% and makes ``r_lift`` pay a standing fine for a
    stationary bottle.

    Re-measuring while the episode is young (still in REACH, hundreds of steps
    before any grasp) pins the datum to where physics actually put the bottle.
    Tracking continuously across the window — rather than snapshotting once —
    keeps ``bottle_lift`` near zero throughout the drop, so no transient leaks
    into the gates.
    """
    settle_n = int(getattr(env.cfg, "bottle_rest_settle_steps", 15))
    if settle_n <= 0:
        return
    warm = (env.episode_length_buf <= settle_n) & (env._stage == STAGE_REACH)
    if not warm.any():
        return
    root_pos = env._bottle.data.root_pos_w
    root_pos = getattr(root_pos, "torch", root_pos)  # IsaacLab 3.0 ProxyArray
    env._bottle_rest_z[warm] = root_pos[warm, 2] - env.scene.env_origins[warm, 2]

    if not getattr(env, "_dbg_logged_rest_z", False) and env.num_envs <= 16:
        if int(env.episode_length_buf[0].item()) == settle_n:
            env._dbg_logged_rest_z = True
            spawn_z = float(env._bottle_nominal_pos[2].item())
            rest_z = float(env._bottle_rest_z[0].item())
            print(
                f"  [RestZ] bottle settled at z={rest_z:.4f} (spawn cfg {spawn_z:.4f},"
                f" table {getattr(env, '_table_z', 0.626):.4f}, drop {spawn_z - rest_z:+.4f}m)"
            )


def compute_curriculum_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Curriculum reward: Phase 1 reach, Phase 2 grasp+lift (same episode)."""
    check_init_buffers(env)
    # Must run before compute_state: bottle_lift is derived from this datum, and
    # the observation path recomputes state from the same buffer later this step.
    _update_bottle_rest_baseline(env)
    s = compute_state(env)

    hold_steps = getattr(env.cfg, "success_hold_steps", 5)
    task_phase = getattr(env.cfg, "task_phase", 1)

    env.step_counter += 1
    in_contact = _update_contact_and_stages(env, s, hold_steps)

    reach_reward = _compute_reach_reward(env, s, in_contact, hold_steps)

    if uses_place(task_phase):
        # Cạm bẫy #2 đã sửa (xem plan Phase 10 / S4): trước đây chỉ có 2
        # nhánh torch.where(in_grasp, grasp_reward, reach_reward) — ở
        # STAGE_PLACE, in_grasp=False nên reward rơi nhầm về reach_reward
        # (thưởng full-strength việc TCP gần thân chai, dương cao dù tay
        # đứng yên không mang đi đâu). Giờ dispatch đủ 3 nhánh.
        grasp_reward = _compute_grasp_reward(env, s, in_contact)
        place_reward = _compute_place_reward(env, s, in_contact)
        in_grasp = env._stage == STAGE_GRASP
        in_place = env._stage == STAGE_PLACE
        total_reward = torch.where(
            in_place, place_reward, torch.where(in_grasp, grasp_reward, reach_reward)
        )
    elif uses_grasp_lift(task_phase):
        grasp_reward = _compute_grasp_reward(env, s, in_contact)
        in_grasp = env._stage == STAGE_GRASP
        total_reward = torch.where(in_grasp, grasp_reward, reach_reward)
    else:
        total_reward = reach_reward

    if env.num_envs > 1 and env.step_counter % 2000 == 0:
        dist = s["dist_ee_bottle"]
        print(f"\n[Curriculum Debug @ step {env.step_counter}] task_phase={task_phase}")
        print(f"  stage REACH/GRASP : {(env._stage == STAGE_REACH).sum()}/{(env._stage == STAGE_GRASP).sum()}")
        print(f"  dist_ee_bottle    : min={dist.min():.3f} mean={dist.mean():.3f}")
        print(f"  bottle_lift       : min={s['bottle_lift'].min():.3f} mean={s['bottle_lift'].mean():.3f}")
        print(f"  gripper           : mean={s['gripper_state'].mean():.3f}")
        print(f"  total_reward      : mean={total_reward.mean():.4f}")

    return total_reward


def terminal_success_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Thưởng một lần khi episode kết thúc vì THÀNH CÔNG.

    ``success`` được đăng ký KHÔNG có ``time_out=True`` nên SB3 coi là termination
    thật → ``V(s_cuối) = 0``, không bootstrap. Trước đây không có bonus nào, nên
    hoàn thành nhiệm vụ tự xoá toàn bộ giá trị tương lai: đứng yên ôm chai đáng
    giá ~162, nhấc thành công chỉ ~28. Agent tối ưu đúng theo tín hiệu đó.

    Chia ``step_dt`` để RewardManager (nhân lại ``dt``) cho ra đúng con số cấu
    hình, tức ``grasp_success_bonus`` đọc thẳng là "cộng bấy nhiêu vào return".
    """
    if getattr(env.cfg, "task_phase", 1) < 2:
        return torch.zeros(env.num_envs, device=env.device)
    try:
        done = env.termination_manager.get_term("success")
    except (KeyError, ValueError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)
    bonus = float(getattr(env.cfg, "grasp_success_bonus", 60.0))
    return done.float() * (bonus / env.step_dt)


def terminal_tipped_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Phạt một lần khi episode kết thúc vì LẬT CHAI.

    KHÔNG phải tuỳ chọn: khi camping đã có giá trị âm, ``V(lật) = 0`` sẽ biến
    việc cố tình gạt đổ chai thành hành động TỐT NHẤT — lỗi tự huỷ kinh điển,
    và nó sẽ trông y hệt triệu chứng hiện tại nhưng ở tầng sâu hơn.
    """
    if getattr(env.cfg, "task_phase", 1) < 2:
        return torch.zeros(env.num_envs, device=env.device)
    try:
        done = env.termination_manager.get_term("tipped_bottle")
    except (KeyError, ValueError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)
    pen = float(getattr(env.cfg, "grasp_tipped_penalty", 30.0))
    return -done.float() * (pen / env.step_dt)


def terminal_place_success_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Thưởng một lần khi episode kết thúc vì ĐÃ ĐẶT chai đúng vào bát (S5).

    Mirror ``terminal_success_bonus`` — ``success`` ở phase>=3 KHÔNG có
    ``time_out=True`` nên là termination thật, ``V(s_cuối)=0``, không bootstrap.
    Không có bonus này, hoàn thành PLACE sẽ tự xoá giá trị tương lai y hệt bug
    gốc của GRASP/LIFT (xem LIFT_BUG_THEORY.md RC2).
    """
    if getattr(env.cfg, "task_phase", 1) < 3:
        return torch.zeros(env.num_envs, device=env.device)
    try:
        done = env.termination_manager.get_term("success")
    except (KeyError, ValueError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)
    bonus = float(getattr(env.cfg, "place_success_bonus", 90.0))
    return done.float() * (bonus / env.step_dt)


def terminal_place_drop_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Phạt một lần khi episode kết thúc vì LÀM RƠI/ĐẶT SAI chai ngoài bát (S5).

    KHÔNG tuỳ chọn — cùng lý do với ``terminal_tipped_penalty``: nếu không có
    phạt tường minh này, cố tình thả chai ra ngoài bát (kết thúc episode sớm,
    tránh phạt-tích-luỹ camping) sẽ trở thành lối thoát rẻ hơn hoàn thành đúng.
    """
    if getattr(env.cfg, "task_phase", 1) < 3:
        return torch.zeros(env.num_envs, device=env.device)
    try:
        done = env.termination_manager.get_term("bottle_misplaced")
    except (KeyError, ValueError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)
    pen = float(getattr(env.cfg, "place_drop_penalty", 30.0))
    return -done.float() * (pen / env.step_dt)
