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

# ── Curriculum Stage Constants ─────────────────────────────────────────────
STAGE_REACH = 0   # Move EE above bottle, avoid table
STAGE_GRASP = 1   # Close gripper and lift bottle
STAGE_PLACE = 2   # Transport bottle to bowl and release


def check_init_buffers(env: ManagerBasedRLEnv):
    """Dynamically initialize tracking, visualization, and reference buffers on the env instance."""
    if hasattr(env, "_arm_joint_ids"):
        return

    # Cache robot, bottle, and bowl references
    env._robot = env.scene["robot"]
    env._bottle = env.scene["bottle"]
    env._bowl = env.scene["bowl"]

    # Joint index arrays
    env._arm_joint_ids, _ = env._robot.find_joints([
        "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
        "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
        "openarm_left_joint7"
    ])
    env._gripper_joint_ids, _ = env._robot.find_joints([
        "openarm_left_finger_joint1", "openarm_left_finger_joint2"
    ])

    # End-effector body index
    try:
        env._ee_body_ids, _ = env._robot.find_bodies(["openarm_left_hand_tcp"])
        env._ee_body_id = env._ee_body_ids[0]
    except Exception:
        env._ee_body_ids, _ = env._robot.find_bodies(["openarm_left_link7"])
        env._ee_body_id = env._ee_body_ids[0]

    try:
        env._hand_body_ids, _ = env._robot.find_bodies(["openarm_left_hand"])
        env._hand_body_id = env._hand_body_ids[0]
    except Exception:
        env._hand_body_id = env._ee_body_id

    # Nominal positions and offsets
    env._bottle_nominal_pos = torch.tensor([0.53, 0.40, 0.64], device=env.device)
    env._bowl_nominal_pos = torch.tensor([0.58, 0.22, 0.67], device=env.device)
    env._bottle_offset = torch.tensor([-0.08157, -0.02198, 0.0], device=env.device)
    env._bowl_offset = torch.tensor([-0.10619, 0.0, -0.01851], device=env.device)

    # TCP safety floor and curriculum variables
    env._table_z = 0.58
    env.step_counter = 0
    env._stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._steps_near_grasp = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._steps_bottle_lifted = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    env._prev_dist_ee_bottle = torch.full((env.num_envs,), 1.0, device=env.device)
    env._prev_dist_bottle_bowl = torch.full((env.num_envs,), 1.0, device=env.device)

    # Initialize Coordinate Frame Visualizers (markers)
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG
    
    ee_marker_cfg = FRAME_MARKER_CFG.copy()
    ee_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    env._ee_markers = VisualizationMarkers(ee_marker_cfg.replace(prim_path="/Visuals/ee_marker"))

    tcp_marker_cfg = FRAME_MARKER_CFG.copy()
    tcp_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    env._tcp_markers = VisualizationMarkers(tcp_marker_cfg.replace(prim_path="/Visuals/tcp_marker"))

    bottle_marker_cfg = FRAME_MARKER_CFG.copy()
    bottle_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    env._bottle_markers = VisualizationMarkers(bottle_marker_cfg.replace(prim_path="/Visuals/bottle_marker"))


def compute_state(env: ManagerBasedRLEnv) -> dict:
    """Compute and cache all coordinate transformations and task state vectors."""
    origins = env.scene.env_origins

    hand_pos_w = env._robot.data.body_pos_w[:, env._hand_body_id, :]
    hand_quat_w = env._robot.data.body_quat_w[:, env._hand_body_id, :]
    tcp_offset = torch.tensor([0.0, 0.0, 0.08], device=env.device).repeat(env.num_envs, 1)
    ee_world = hand_pos_w + quat_apply(hand_quat_w, tcp_offset)
    ee_pos = ee_world - origins

    bottle_world = env._bottle.data.root_pos_w + env._bottle_offset
    bottle_quat_w = env._bottle.data.root_quat_w
    bottle_pos = bottle_world - origins

    bowl_world = env._bowl.data.root_pos_w + env._bowl_offset
    bowl_pos = bowl_world - origins

    bottle_resting_z = 0.626

    bottle_pos_for_target = bottle_pos.clone()
    bottle_pos_for_target[:, 2] = torch.clamp(bottle_pos_for_target[:, 2], min=bottle_resting_z)
    grasp_target_pos = bottle_pos_for_target
    grasp_target_pos[:, 2] += 0.101

    ee_to_grasp = grasp_target_pos - ee_pos
    bottle_to_bowl = bowl_pos - bottle_pos

    dist_ee_grasp = torch.norm(ee_to_grasp, dim=-1)
    dist_bottle_bowl = torch.norm(bottle_to_bowl, dim=-1)

    table_clearance = ee_pos[:, 2] - env._table_z
    bottle_lift = bottle_pos[:, 2] - bottle_resting_z
    is_lifted = (bottle_pos[:, 2] > 0.656)

    # Fetch gripper state from action term via private _terms dictionary
    gripper_state = env.action_manager._terms["openarm_action"].gripper_state[:, 0]

    z_unit = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    y_unit = torch.tensor([0.0, 1.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    hand_z_w = quat_apply(hand_quat_w, z_unit)
    bottle_y_w = quat_apply(bottle_quat_w, y_unit)
    alignment = torch.abs(torch.sum(hand_z_w * bottle_y_w, dim=-1)) ** 2

    return {
        "origins": origins,
        "ee_world": ee_world,
        "hand_quat_w": hand_quat_w,
        "ee_pos": ee_pos,
        "bottle_world": bottle_world,
        "bottle_quat_w": bottle_quat_w,
        "bottle_pos": bottle_pos,
        "bowl_pos": bowl_pos,
        "grasp_target_pos": grasp_target_pos,
        "ee_to_grasp": ee_to_grasp,
        "bottle_to_bowl": bottle_to_bowl,
        "dist_ee_grasp": dist_ee_grasp,
        "dist_bottle_bowl": dist_bottle_bowl,
        "table_clearance": table_clearance,
        "bottle_lift": bottle_lift,
        "is_lifted": is_lifted,
        "gripper_state": gripper_state,
        "alignment": alignment,
    }
