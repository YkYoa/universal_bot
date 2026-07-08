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
from isaaclab.envs import ManagerBasedRLEnv
from .helpers import (
    check_init_buffers,
    compute_state,
    STAGE_REACH,
    STAGE_GRASP,
    finger_grasp_ready,
    finger_symmetric_ready,
    finger_ready_for_close,
    grasp_lift_success_ready,
    reach_align_ready,
    uses_grasp_lift,
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

    env._steps_in_grasp[env._stage == STAGE_GRASP] += 1
    env._steps_in_grasp[env._stage != STAGE_GRASP] = 0

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

    return (
        r_reach + r_lateral + r_z_descent + r_z_floor
        + r_close + r_progress + milestones + r_stall + r_hold
        + r_top_down + r_finger_level + r_descent + r_tilt_penalty
        + table_penalty + vel_penalty
    )


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

    r_lift = torch.clamp(bottle_lift / lift_threshold, max=1.5) * 30.0
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

    return (
        r_align + r_descend + r_hover + r_z_floor + r_grip_close + r_grip_cmd + r_lift + r_track + r_lift_hold
        + r_lift_motion + r_orient + r_bad_grip + r_lift_no_grip + r_air_grasp + r_up_before_grip
        + table_penalty + vel_penalty
    )


def compute_curriculum_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Curriculum reward: Phase 1 reach, Phase 2 grasp+lift (same episode)."""
    check_init_buffers(env)
    s = compute_state(env)

    hold_steps = getattr(env.cfg, "success_hold_steps", 5)
    task_phase = getattr(env.cfg, "task_phase", 1)

    env.step_counter += 1
    in_contact = _update_contact_and_stages(env, s, hold_steps)

    reach_reward = _compute_reach_reward(env, s, in_contact, hold_steps)

    if uses_grasp_lift(task_phase):
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
