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
    dist_bottle_bowl = s["dist_bottle_bowl"]
    table_clearance = s["table_clearance"]
    bottle_lift = s["bottle_lift"]
    is_lifted = s["is_lifted"]
    alignment = s["alignment"]
    ee_pos = s["ee_pos"]
    grasp_target_pos = s["grasp_target_pos"]

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
        r_reach[is_stage0] = torch.exp(-5.0 * dist_ee_grasp[is_stage0]) * 5.0

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

        z_error = torch.abs(ee_pos[:, 2] - grasp_target_pos[:, 2])
        r_z_align[is_stage0] = torch.exp(-12.0 * z_error[is_stage0]) * 4.0

        near_bottle = (dist_ee_grasp < 0.25)
        aligned_envs = is_stage0 & near_bottle
        r_align[aligned_envs] = alignment[aligned_envs] * 1.0

        physically_touching = (dist_ee_grasp < 0.15)
        early_squeeze_envs = is_stage0 & physically_touching
        r_open_grip[early_squeeze_envs] = (1.0 - gripper_state[early_squeeze_envs]) * 4.0

        # Curriculum transition 0 -> 1
        near_grasp = (dist_ee_grasp < 0.15)
        above_table = (table_clearance > -0.02)
        ready_to_grasp = near_grasp & above_table & is_stage0
        env._steps_near_grasp[ready_to_grasp] += 1
        env._steps_near_grasp[~ready_to_grasp & is_stage0] = 0
        
        advance_to_grasp = is_stage0 & (env._steps_near_grasp >= 5)
        env._stage[advance_to_grasp] = STAGE_GRASP
        if advance_to_grasp.any():
            print(f"[Curriculum] {advance_to_grasp.sum().item()} envs advanced: Stage 0→1 (Reach→Grasp)")

    # ─── STAGE 1: GRASP ───────────────────────────────────────────────────
    r_squeeze = torch.zeros(env.num_envs, device=env.device)
    r_lift = torch.zeros(env.num_envs, device=env.device)
    r_grasp_bonus = torch.zeros(env.num_envs, device=env.device)

    if is_stage1.any():
        r_reach[is_stage1] = torch.exp(-5.0 * dist_ee_grasp[is_stage1]) * 1.0

        in_contact = (dist_ee_grasp < 0.15)
        squeeze_envs = is_stage1 & in_contact
        r_squeeze[squeeze_envs] = gripper_state[squeeze_envs] * 5.0

        target_lift = 0.05
        lift_frac = bottle_lift / target_lift
        lift_frac = torch.clamp(lift_frac, min=-1.0, max=1.0)
        r_lift[is_stage1] = lift_frac[is_stage1] * 5.0

        r_grasp_bonus[is_stage1 & is_lifted] = 10.0

        # Curriculum transition 1 -> 2
        bottle_stably_lifted = is_lifted & (gripper_state > 0.5)
        ready_for_place = bottle_stably_lifted & is_stage1
        env._steps_bottle_lifted[ready_for_place] += 1
        env._steps_bottle_lifted[~ready_for_place & is_stage1] = 0
        
        advance_to_place = is_stage1 & (env._steps_bottle_lifted >= 5)
        env._stage[advance_to_place] = STAGE_PLACE
        if advance_to_place.any():
            print(f"[Curriculum] {advance_to_place.sum().item()} envs advanced: Stage 1→2 (Grasp→Place)")

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
        r_transport +
        r_grip_hold +
        r_lift_hold +
        r_place +
        r_release +
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
        print(f"  dist_ee_grasp      : min={dist_ee_grasp.min():.3f} mean={dist_ee_grasp.mean():.3f}")
        print(f"  alignment          : mean={alignment.mean():.3f} max={alignment.max():.3f}")
        print(f"  gripper_state      : mean={gripper_state.mean():.3f} max={gripper_state.max():.3f}")
        print(f"  table_clearance    : min={table_clearance.min():.3f} mean={table_clearance.mean():.3f}")
        print(f"  table_penalty      : mean={table_penalty.mean():.4f}")
        print(f"  bottle_lift        : min={bottle_lift.min():.3f} mean={bottle_lift.mean():.3f}")
        print(f"  is_lifted          : {is_lifted.sum().item()}/{env.num_envs} envs")
        print(f"  r_reach            : mean={r_reach.mean():.4f}")
        print(f"  r_progress         : mean={r_progress.mean():.4f}")
        print(f"  total_reward       : mean={total_reward.mean():.4f}")

    # Watch low TCP safety floor breaches
    if env.step_counter % 100 == 0:
        heavy_hits = (table_clearance <= -0.02).sum().item()
        if heavy_hits > 0:
            print(f"[Low TCP] {heavy_hits}/{env.num_envs} envs below safety floor! min_clearance={table_clearance.min():.3f}m")

    return total_reward
