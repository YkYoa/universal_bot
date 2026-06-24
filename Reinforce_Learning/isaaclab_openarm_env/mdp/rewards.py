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
from isaaclab.utils.math import quat_apply
from .helpers import check_init_buffers, compute_state, STAGE_REACH, STAGE_GRASP, STAGE_PLACE


def compute_curriculum_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    3-Stage curriculum reward term function for ManagerBasedRLEnv.
    Integrates all phase conditions, milestone bonuses, and logging callbacks.
    """
    check_init_buffers(env)
    s = compute_state(env)

    stage = env._stage
    is_stage0 = (stage == STAGE_REACH)
    is_stage1 = (stage == STAGE_GRASP)
    is_stage2 = (stage == STAGE_PLACE)

    gripper_state = s["gripper_state"]
    dist_ee_grasp = s["dist_ee_grasp"]
    dist_ee_bottle = s["dist_ee_bottle"]
    dist_bottle_bowl = s["dist_bottle_bowl"]
    table_clearance = s["table_clearance"]
    bottle_lift = s["bottle_lift"]
    is_lifted = s["is_lifted"]
    ee_pos = s["ee_pos"]
    grasp_target_pos = s["grasp_target_pos"]
    grasp_pos = s["grasp_pos"]
    hand_quat_w = s["hand_quat_w"]       # needed for inline alignment computation
    bottle_quat_w = s["bottle_quat_w"]   # needed for inline alignment computation

    # Increment environment step counter
    env.step_counter += 1

    # ─── SHARED PENALTIES ─────────────────────────────────────────────────
    joint_vel = env._robot.data.joint_vel[:, env._arm_joint_ids]
    vel_penalty = -0.01 * torch.sum(joint_vel ** 2, dim=-1)

    low_sweep_depth = torch.clamp(-table_clearance, min=0.0, max=0.08)
    table_penalty = -5.0 * (low_sweep_depth / 0.08) ** 2

    # ─── STAGE 0: REACH & AVOID TABLE ─────────────────────────────────────
    r_reach = torch.zeros(env.num_envs, device=env.device)
    r_align = torch.zeros(env.num_envs, device=env.device)
    r_progress = torch.zeros(env.num_envs, device=env.device)
    r_z_align = torch.zeros(env.num_envs, device=env.device)
    r_open_grip = torch.zeros(env.num_envs, device=env.device)

    if is_stage0.any():
        # Combination of linear and exponential reaching reward to guarantee dense gradients at all distances
        r_reach[is_stage0] = (1.0 - dist_ee_grasp[is_stage0]) * 3.0 + torch.exp(-5.0 * dist_ee_grasp[is_stage0]) * 2.0

        milestones = (
            (dist_ee_grasp < 0.20).float() * 1.0 +
            (dist_ee_grasp < 0.15).float() * 3.0 +
            (dist_ee_grasp < 0.10).float() * 6.0 +
            (dist_ee_grasp < 0.06).float() * 12.0
        )
        r_reach[is_stage0] += milestones[is_stage0]

        dist_improvement = env._prev_dist_ee_bottle - dist_ee_grasp
        r_progress[is_stage0] = torch.clamp(
            dist_improvement[is_stage0] * 30.0,
            min=-1.5, max=3.0
        )

        # Dense vertical alignment gradient (less aggressive exponential decay + linear component)
        z_error = torch.abs(ee_pos[:, 2] - grasp_target_pos[:, 2])
        r_z_align[is_stage0] = (1.0 - z_error[is_stage0]) * 2.0 + torch.exp(-5.0 * z_error[is_stage0]) * 2.0

        # Encourage vertical gripper alignment during the entire Reach stage (Stage 0) - DISABLED
        # r_align[is_stage0] = alignment[is_stage0] * 5.0

        # Encourage keeping the gripper open during the entire Reach stage (Stage 0)
        r_open_grip[is_stage0] = (1.0 - gripper_state[is_stage0]) * 2.0

        # Curriculum transition 0 -> 1 (require being within 6cm of hover target)
        near_grasp = (dist_ee_grasp < 0.06)
        above_table = (table_clearance > -0.02)
        ready_to_grasp = near_grasp & above_table & is_stage0
        env._steps_near_grasp[ready_to_grasp] += 1
        env._steps_near_grasp[~ready_to_grasp & is_stage0] = 0
        
        advance_to_grasp = is_stage0 & (env._steps_near_grasp >= 5)
        env._stage[advance_to_grasp] = STAGE_GRASP
        if advance_to_grasp.any():
            print(f"[Curriculum] {advance_to_grasp.sum().item()} envs advanced: Stage 0→1 (Reach→Grasp)")

    # ─── STAGE 1: GRASP ───────────────────────────────────────────────────────
    # In Stage 1 the arm must DESCEND from the hover point down to the bottle
    # body and close the gripper AROUND the bottle before lifting.
    z_unit = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    y_unit = torch.tensor([0.0, 1.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    hand_y_w = quat_apply(hand_quat_w, y_unit)
    bottle_z_w = quat_apply(bottle_quat_w, z_unit)

    # Approach-perpendicular alignment: fingers (hand Y-axis) should be perpendicular
    # to the EE→bottle approach direction. Maximized (=1) when hand_y is perpendicular
    # to approach vector → fingers wrap around the bottle correctly.
    # Old metric |hand_y · bottle_z|² rewarded fingers pointing VERTICAL (wrong for grasping).
    approach_vec = grasp_pos - ee_pos                                     # EE → grasp point
    approach_norm = torch.norm(approach_vec, dim=-1, keepdim=True).clamp(min=1e-6)
    approach_dir = approach_vec / approach_norm                            # unit vector
    hand_y_dot_approach = torch.abs(torch.sum(hand_y_w * approach_dir, dim=-1))
    alignment = 1.0 - hand_y_dot_approach ** 2  # 1=fingers perpendicular to approach (correct)

    r_squeeze = torch.zeros(env.num_envs, device=env.device)
    r_lift = torch.zeros(env.num_envs, device=env.device)
    r_grasp_bonus = torch.zeros(env.num_envs, device=env.device)
    r_descent = torch.zeros(env.num_envs, device=env.device)

    if is_stage1.any():
        # 1. Guide TCP to the bottle body in 3D — stronger than before so the
        #    robot must align XY AND z, not just z alone.
        r_reach[is_stage1] = (
            (1.0 - dist_ee_bottle[is_stage1]) * 8.0
            + torch.exp(-10.0 * dist_ee_bottle[is_stage1]) * 8.0
        )

        # 2. Lateral-gated descent reward:
        #    - When the robot is already XY-aligned (< 10cm lateral offset), apply
        #      a strong z-descent push so it commits to the final downward approach.
        #    - When the robot is still far in XY, only a mild z hint is given so
        #      the 3D r_reach above drives XY alignment first.
        #    This prevents the robot from descending at the wrong XY position.
        lateral_dist = torch.norm(ee_pos[:, :2] - grasp_pos[:, :2], dim=-1)
        xy_close = (lateral_dist < 0.12)  # 12cm gate: fires z-descent at current ~8cm lateral
        above_grasp_z = torch.clamp(ee_pos[:, 2] - grasp_pos[:, 2], min=0.0)
        z_descent_error = torch.abs(ee_pos[:, 2] - grasp_pos[:, 2])

        # XY-aligned envs: strong z-descent penalty + exponential peak at grasp height
        r_descent[is_stage1 & xy_close] = (
            -above_grasp_z[is_stage1 & xy_close] * 25.0
            + torch.exp(-10.0 * z_descent_error[is_stage1 & xy_close]) * 5.0
        )
        # XY-misaligned envs: mild z hint only, let r_reach dominate
        r_descent[is_stage1 & ~xy_close] = (
            torch.exp(-10.0 * z_descent_error[is_stage1 & ~xy_close]) * 1.0
        )

        # 3. Squeeze reward fires only when TCP is physically at the bottle body.
        #    Boosted 8 → 30 to overcome the strong "open gripper" prior from prior training.
        at_bottle = (dist_ee_bottle < 0.06)
        squeeze_envs = is_stage1 & at_bottle
        r_squeeze[squeeze_envs] = gripper_state[squeeze_envs] * 30.0

        # Gripper orientation: reward fingers (hand Y-axis) being perpendicular to the
        # approach direction. Fires in Stage 1 only (not Stage 0 to avoid disturbing reach).
        r_align[is_stage1] = alignment[is_stage1] * 4.0

        # NOTE: r_open_grip for Stage 1 is intentionally removed.
        # Keeping it (even gated to ~at_bottle) created a 50M+ step "always open" prior
        # so strong that r_squeeze=8 could never overcome it in brief contact windows.

        # 4. Lift reward: exponential + linear for dense gradient
        target_lift = 0.05
        lift_frac = bottle_lift / target_lift
        lift_frac = torch.clamp(lift_frac, min=-1.0, max=1.0)
        r_lift[is_stage1] = lift_frac[is_stage1] * 8.0

        r_grasp_bonus[is_stage1 & is_lifted] = 15.0

        # Curriculum transition 1 -> 2
        bottle_stably_lifted = is_lifted & (gripper_state > 0.5)
        ready_for_place = bottle_stably_lifted & is_stage1
        env._steps_bottle_lifted[ready_for_place] += 1
        env._steps_bottle_lifted[~ready_for_place & is_stage1] = 0
        
        advance_to_place = is_stage1 & (env._steps_bottle_lifted >= 5)
        env._stage[advance_to_place] = STAGE_PLACE
        if advance_to_place.any():
            print(f"[Curriculum] {advance_to_place.sum().item()} envs advanced: Stage 1→2 (Grasp→Place)")

        # ── Stage 1 Hover-Plateau Demotion ────────────────────────────────────
        # If the robot stays above grasp_pos z-height + 5cm for too many steps,
        # it is stuck hovering. Demote back to Stage 0 to reset and retry.
        # This replaces the old dist_ee_grasp > 0.12 demotion which fired
        # erroneously during legitimate descent attempts.
        still_hovering = is_stage1 & (above_grasp_z > 0.05)
        env._steps_hovering_in_grasp[still_hovering] += 1
        env._steps_hovering_in_grasp[~still_hovering & is_stage1] = 0

        demote_to_reach = is_stage1 & (env._steps_hovering_in_grasp >= 80)  # faster cycling → more exploration
        env._stage[demote_to_reach] = STAGE_REACH
        env._steps_near_grasp[demote_to_reach] = 0
        env._steps_hovering_in_grasp[demote_to_reach] = 0
        if demote_to_reach.any():
            print(f"[Curriculum] {demote_to_reach.sum().item()} envs demoted: Stage 1→0 (hover timeout)")

    # ─── STAGE 2: TRANSPORT & PLACE ───────────────────────────────────────
    r_transport = torch.zeros(env.num_envs, device=env.device)
    r_grip_hold = torch.zeros(env.num_envs, device=env.device)
    r_lift_hold = torch.zeros(env.num_envs, device=env.device)
    r_place = torch.zeros(env.num_envs, device=env.device)
    r_release = torch.zeros(env.num_envs, device=env.device)

    if is_stage2.any():
        r_grip_hold[is_stage2] = gripper_state[is_stage2] * 2.0

        target_height = 0.15
        lift_error = torch.abs(bottle_lift - target_height)
        r_lift_hold[is_stage2] = torch.exp(-3.0 * lift_error[is_stage2]) * 2.0

        r_transport[is_stage2] = torch.exp(-5.0 * dist_bottle_bowl[is_stage2]) * 10.0

        # Use thresholds from env.cfg if available, else default to 0.10
        place_dist_threshold = getattr(env.cfg, "place_dist_threshold", 0.10)
        placed_in_bowl = (dist_bottle_bowl < place_dist_threshold)
        gripper_open = (gripper_state < 0.4)
        r_place[is_stage2 & placed_in_bowl] = 15.0
        r_release[is_stage2 & placed_in_bowl & gripper_open] = 5.0

        # Demotion from Stage 2 to Stage 1 (Grasp) if bottle is dropped or gripper opens
        demote_to_grasp = is_stage2 & (~is_lifted | (gripper_state < 0.3))
        env._stage[demote_to_grasp] = STAGE_GRASP
        env._steps_bottle_lifted[demote_to_grasp] = 0
        if demote_to_grasp.any():
            print(f"[Curriculum] {demote_to_grasp.sum().item()} envs demoted: Stage 2→1 (Place→Grasp)")

    # ─── STAGE PROGRESSION BONUSES ────────────────────────────────────────
    # Offset bonuses to prevent reward drops (cliffs) when transitioning stages
    r_stage_bonus = torch.zeros(env.num_envs, device=env.device)
    r_stage_bonus[is_stage1] = 20.0
    r_stage_bonus[is_stage2] = 50.0

    # ─── COMBINE ALL REWARDS ──────────────────────────────────────────────
    total_reward = (
        r_reach +
        r_align +
        r_progress +
        r_z_align +
        r_open_grip +
        r_squeeze +
        r_lift +
        r_grasp_bonus +
        r_descent +
        r_transport +
        r_grip_hold +
        r_lift_hold +
        r_place +
        r_release +
        r_stage_bonus +
        table_penalty +
        vel_penalty
    )

    # Update state history trackers
    env._prev_dist_ee_bottle = dist_ee_grasp.detach()
    env._prev_dist_bottle_bowl = dist_bottle_bowl.detach()

    # Debug Log Terminal metrics every 500 steps
    if env.step_counter % 500 == 0:
        s0_count = is_stage0.sum().item()
        s1_count = is_stage1.sum().item()
        s2_count = is_stage2.sum().item()
        print(f"\n[Reward Debug @ step {env.step_counter}]")
        print(f"  Stage distribution : S0(Reach)={s0_count} | S1(Grasp)={s1_count} | S2(Place)={s2_count}")
        print(f"  dist_ee_grasp      : min={dist_ee_grasp.min():.3f} mean={dist_ee_grasp.mean():.3f}  (hover target)")
        print(f"  dist_ee_bottle     : min={dist_ee_bottle.min():.3f} mean={dist_ee_bottle.mean():.3f}  (grasp point)")
        _above_gz = torch.clamp(ee_pos[:, 2] - grasp_pos[:, 2], min=0.0)
        _lat_dist = torch.norm(ee_pos[:, :2] - grasp_pos[:, :2], dim=-1)
        print(f"  above_grasp_z      : min={_above_gz.min():.3f} mean={_above_gz.mean():.3f}  (0=at grasp height)")
        print(f"  lateral_dist_xy    : min={_lat_dist.min():.3f} mean={_lat_dist.mean():.3f}  (0=over bottle)")
        if is_stage1.any():
            _xy_close_n   = (is_stage1 & (_lat_dist < 0.10)).sum().item()
            _at_bottle_n  = (is_stage1 & (dist_ee_bottle < 0.06)).sum().item()
            _hovering_n   = env._steps_hovering_in_grasp[is_stage1].max().item()
            print(f"  [S1] xy_close(<10cm): {_xy_close_n}/{s1_count} envs | at_bottle(<6cm): {_at_bottle_n}/{s1_count} envs | max_hover_steps: {_hovering_n}")
        print(f"  gripper_state      : mean={gripper_state.mean():.3f} max={gripper_state.max():.3f}")
        print(f"  table_clearance    : min={table_clearance.min():.3f} mean={table_clearance.mean():.3f}")
        print(f"  table_penalty      : mean={table_penalty.mean():.4f}")
        print(f"  bottle_lift        : min={bottle_lift.min():.3f} mean={bottle_lift.mean():.3f} max={bottle_lift.max():.3f}")
        print(f"  is_lifted          : {is_lifted.sum().item()}/{env.num_envs} envs")
        print(f"  r_reach            : mean={r_reach.mean():.4f}")
        print(f"  r_descent          : mean={r_descent.mean():.4f}")
        print(f"  r_squeeze          : mean={r_squeeze.mean():.4f}")
        print(f"  r_progress         : mean={r_progress.mean():.4f}")
        print(f"  total_reward       : mean={total_reward.mean():.4f}")

    # Milestone: first real bottle contact in Stage 1 (dist_ee_bottle < 6cm)
    if not getattr(env, "_milestone_bottle_contact_logged", False):
        real_contact = is_stage1 & (dist_ee_bottle < 0.06)
        if real_contact.any():
            env._milestone_bottle_contact_logged = True
            print(f"\n🏆 [MILESTONE @ step {env.step_counter}] FIRST REAL BOTTLE CONTACT in Stage 1!")
            print(f"   dist_ee_bottle={dist_ee_bottle[real_contact].min():.4f}m | {real_contact.sum().item()} envs | lateral_dist={torch.norm(ee_pos[:, :2] - grasp_pos[:, :2], dim=-1)[real_contact].min():.4f}m")

    # Watch low TCP safety floor breaches
    if env.step_counter % 100 == 0:
        heavy_hits = (table_clearance <= -0.02).sum().item()
        if heavy_hits > 0:
            print(f"[Low TCP] {heavy_hits}/{env.num_envs} envs below safety floor! min_clearance={table_clearance.min():.3f}m")

    return total_reward
