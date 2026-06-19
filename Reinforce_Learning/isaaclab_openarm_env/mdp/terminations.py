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
from .helpers import check_init_buffers, STAGE_REACH, STAGE_PLACE


def success_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns True if the success criteria are met: stage 2 reached, bottle inside bowl, gripper released."""
    check_init_buffers(env)
    origins = env.scene.env_origins
    bottle_pos = env._bottle.data.root_pos_w + env._bottle_offset - origins
    bowl_pos = env._bowl.data.root_pos_w + env._bowl_offset - origins

    dist_bottle_bowl = torch.norm(bottle_pos - bowl_pos, dim=-1)
    
    # Fetch gripper state via private _terms dictionary
    gripper_state = env.action_manager._terms["openarm_action"].gripper_state[:, 0]
    gripper_open = (gripper_state < 0.4)
    is_lifted = (bottle_pos[:, 2] > 0.656)

    place_dist_threshold = getattr(env.cfg, "place_dist_threshold", 0.10)
    
    # Success condition
    terminated = (env._stage == STAGE_PLACE) & (dist_bottle_bowl < place_dist_threshold) & gripper_open & is_lifted
    return terminated


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

    # Reset action tracking states via private _terms dictionary
    env.action_manager._terms["openarm_action"]._raw_actions[env_ids] = 0.0
    env.action_manager._terms["openarm_action"].gripper_state[env_ids] = 0.0

    # Reset stages and history distance indicators
    env._stage[env_ids] = STAGE_REACH
    env._steps_near_grasp[env_ids] = 0
    env._steps_bottle_lifted[env_ids] = 0
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
