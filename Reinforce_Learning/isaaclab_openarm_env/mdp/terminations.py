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
from isaaclab.utils.math import sample_uniform
from .helpers import check_init_buffers, reset_action_terms, STAGE_REACH, STAGE_GRASP


def success_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Phase 1: TCP hold at bottle. Phase 2+: bottle lifted with gripper closed."""
    check_init_buffers(env)
    task_phase = getattr(env.cfg, "task_phase", 1)
    if task_phase <= 1:
        hold_steps = getattr(env.cfg, "success_hold_steps", 5)
        return env._steps_in_contact >= hold_steps
    lift_hold = getattr(env.cfg, "grasp_lift_hold_steps", 5)
    return env._steps_bottle_lifted >= lift_hold


def tipped_bottle_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Phase 2+: terminate early when bottle is knocked over beyond recovery.

    Saves the ~700 wasted steps observed when the arm tips the bottle to 90° and
    the episode has no way to recover. Only fires in GRASP stage so that an upright
    bottle that hasn't been grasped yet is not incorrectly terminated.
    """
    check_init_buffers(env)
    task_phase = getattr(env.cfg, "task_phase", 1)
    if task_phase < 2:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_last_state") or env._last_state is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    max_tilt = getattr(env.cfg, "bottle_tipped_termination_deg", 60.0)
    tilt = env._last_state["bottle_tilt_deg"]
    in_grasp = env._stage == STAGE_GRASP

    # Track consecutive tipped steps to avoid premature termination from transient tilts
    if not hasattr(env, "_steps_bottle_tipped"):
        env._steps_bottle_tipped = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    tipped_now = in_grasp & (tilt > max_tilt)
    env._steps_bottle_tipped[tipped_now] += 1
    env._steps_bottle_tipped[~tipped_now] = 0

    min_steps = getattr(env.cfg, "bottle_tipped_min_steps", 5)
    return env._steps_bottle_tipped >= min_steps


def reset_robot(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Event callback: Resets robot joint targets, dynamics, actions, and curriculum flags."""
    check_init_buffers(env)
    if env_ids is None or len(env_ids) == 0:
        return

    default_joint_pos = env._robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = torch.zeros_like(default_joint_pos)
    env._robot.set_joint_position_target(default_joint_pos, env_ids=env_ids)
    env._robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
    env._robot.reset(env_ids)

    reset_action_terms(env, env_ids)

    env._stage[env_ids] = STAGE_REACH
    env._steps_near_grasp[env_ids] = 0
    env._steps_bottle_lifted[env_ids] = 0
    env._steps_hovering_in_grasp[env_ids] = 0
    env._steps_in_contact[env_ids] = 0
    env._steps_in_grasp[env_ids] = 0
    if hasattr(env, "_grasp_descent_ref_dist"):
        env._grasp_descent_ref_dist[env_ids] = 1.0
    if hasattr(env, "_pre_lift_hold_steps"):
        env._pre_lift_hold_steps[env_ids] = 0
    if hasattr(env, "_steps_bottle_tipped"):
        env._steps_bottle_tipped[env_ids] = 0
    env._dbg_prev_lift_steps = 0
    env._dbg_logged_grip = False
    env._dbg_logged_lift = False
    if hasattr(env, "_in_contact_zone"):
        env._in_contact_zone[env_ids] = False
    env._prev_dist_ee_bottle[env_ids] = 1.0
    env._prev_dist_bottle_bowl[env_ids] = 1.0


def reset_bottle(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Event callback: Randomizes the bottle's horizontal position upon environment reset."""
    check_init_buffers(env)
    if env_ids is None or len(env_ids) == 0:
        return
    num_reset = len(env_ids)

    bottle_pos = env._bottle_nominal_pos.unsqueeze(0).expand(num_reset, -1).clone()
    bottle_pos_noise = getattr(env.cfg, "bottle_pos_noise", 0.05)
    
    bottle_pos[:, 0] += sample_uniform(-bottle_pos_noise, bottle_pos_noise, (num_reset,), device=env.device)
    bottle_pos[:, 1] += sample_uniform(-bottle_pos_noise, bottle_pos_noise, (num_reset,), device=env.device)

    bottle_pos_world = bottle_pos + env.scene.env_origins[env_ids]
    bottle_root_state = env._bottle.data.default_root_state[env_ids].clone()
    bottle_root_state[:, :3] = bottle_pos_world
    bottle_root_state[:, 7:] = 0.0  # zero out linear/angular velocities
    env._bottle.write_root_state_to_sim(bottle_root_state, env_ids=env_ids)
    env._bottle.reset(env_ids)

    if not hasattr(env, "_bottle_spawn_env_pos"):
        env._bottle_spawn_env_pos = torch.zeros(env.num_envs, 3, device=env.device)
    env._bottle_spawn_env_pos[env_ids] = bottle_pos
    if not hasattr(env, "_bottle_rest_z"):
        env._bottle_rest_z = env._bottle_nominal_pos[2].expand(env.num_envs).clone()
    env._bottle_rest_z[env_ids] = bottle_pos[:, 2]


def reset_bowl(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Event callback: Randomizes the bowl's horizontal position upon environment reset."""
    check_init_buffers(env)
    if env_ids is None or len(env_ids) == 0:
        return
    num_reset = len(env_ids)

    bowl_pos = env._bowl_nominal_pos.unsqueeze(0).expand(num_reset, -1).clone()
    bowl_pos_noise = getattr(env.cfg, "bowl_pos_noise", 0.05)
    
    bowl_pos[:, 0] += sample_uniform(-bowl_pos_noise, bowl_pos_noise, (num_reset,), device=env.device)
    bowl_pos[:, 1] += sample_uniform(-bowl_pos_noise, bowl_pos_noise, (num_reset,), device=env.device)

    bowl_pos_world = bowl_pos + env.scene.env_origins[env_ids]
    bowl_root_state = env._bowl.data.default_root_state[env_ids].clone()
    bowl_root_state[:, :3] = bowl_pos_world
    bowl_root_state[:, 7:] = 0.0  # zero velocities
    env._bowl.write_root_state_to_sim(bowl_root_state, env_ids=env_ids)
    env._bowl.reset(env_ids)
