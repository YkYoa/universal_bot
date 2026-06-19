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
from .helpers import check_init_buffers, compute_state


def get_apple_pick_place_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Returns the standard 26-D observation vector for the policy:
    [0:7]   Left arm joint positions (rad)
    [7:14]  Left arm joint velocities (rad/s)
    [14:17] End-effector position (relative to env origin)
    [17:20] EE -> grasp target vector
    [20:23] Bottle -> bowl vector
    [23]    Gripper state (0.0=open, 1.0=closed)
    [24]    Table clearance (m)
    [25]    Curriculum stage (0.0, 0.5, 1.0 representing Stage 0, 1, 2)
    """
    check_init_buffers(env)
    s = compute_state(env)

    # Fetch joint states
    q = env._robot.data.joint_pos[:, env._arm_joint_ids]
    dq = env._robot.data.joint_vel[:, env._arm_joint_ids]

    # Coordinate visualizer updates in Isaac Sim
    wrist_pos_w = env._robot.data.body_pos_w[:, env._ee_body_id, :]
    wrist_quat_w = env._robot.data.body_quat_w[:, env._ee_body_id, :]

    if hasattr(env, "_ee_markers") and env._ee_markers is not None:
        env._ee_markers.visualize(wrist_pos_w, wrist_quat_w)
    if hasattr(env, "_tcp_markers") and env._tcp_markers is not None:
        env._tcp_markers.visualize(s["ee_world"], s["hand_quat_w"])
    if hasattr(env, "_bottle_markers") and env._bottle_markers is not None:
        env._bottle_markers.visualize(s["bottle_world"], s["bottle_quat_w"])

    # Curriculum Stage representation as a normalized float
    stage_obs = (env._stage.float() / 2.0).unsqueeze(-1)

    # Clamped Table Clearance observation
    clearance_obs = s["table_clearance"].clamp(-0.1, 0.3).unsqueeze(-1)

    # Gripper State
    gripper_state = s["gripper_state"].unsqueeze(-1)

    # Assemble full 26-D observation tensor
    obs = torch.cat([
        q,                     # [0:7]
        dq,                    # [7:14]
        s["ee_pos"],           # [14:17]
        s["ee_to_grasp"],      # [17:20]
        s["bottle_to_bowl"],   # [20:23]
        gripper_state,         # [23]
        clearance_obs,         # [24]
        stage_obs,             # [25]
    ], dim=-1)

    if torch.isnan(obs).any():
        print("[WARNING] Observation contains NaNs!")

    return obs
