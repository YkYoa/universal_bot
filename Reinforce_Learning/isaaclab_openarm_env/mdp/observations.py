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
from .helpers import check_init_buffers, compute_state, uses_grasp_lift


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
    env._last_state = s

    # Fetch joint states
    q = env._robot.data.joint_pos[:, env._arm_joint_ids]
    dq = env._robot.data.joint_vel[:, env._arm_joint_ids]

    # Coordinate visualizer updates in Isaac Sim
    wrist_pos_w = env._robot.data.body_pos_w[:, env._ee_body_id, :]
    wrist_quat_w = env._robot.data.body_quat_w[:, env._ee_body_id, :]

    # Demo/GUI: luôn vẽ 2 marker — TCP (ee) + nắp chai (bottle)
    if env.num_envs <= 16:
        if hasattr(env, "_ee_markers") and env._ee_markers is not None:
            env._ee_markers.visualize(
                translations=s["ee_world"],
                orientations=s["ee_quat_w"],
                marker_indices=torch.zeros(env.num_envs, dtype=torch.int, device=env.device),
            )

    # Target depends on curriculum stage
    task_phase = getattr(env.cfg, "task_phase", 1)
    if uses_grasp_lift(task_phase):
        lift_height = getattr(env.cfg, "grasp_lift_threshold", 0.03) + 0.05
        lift_target = s["bottle_pos"].clone()
        lift_target[:, 2] = s["bottle_pos"][:, 2] + lift_height
        grip_thresh = getattr(env.cfg, "grasp_grip_threshold", 0.4)
        gripped = s["gripper_state"] > grip_thresh
        in_grasp = (env._stage >= 1).unsqueeze(-1)
        in_lift = in_grasp & gripped.unsqueeze(-1)
        # Policy v3 train với target = nắp (grasp_pos), không hover — giữ tương thích demo
        reach_top = s["grasp_pos"] - s["ee_pos"]
        descend_body = s["grasp_ee_descend_pos"] - s["ee_pos"]
        lift_delta = lift_target - s["ee_pos"]
        descend_z = getattr(env.cfg, "grasp_descend_z_threshold", 0.03)
        finger_high = (s["z_error_finger"] > descend_z).unsqueeze(-1)
        in_wrap = in_grasp & (~finger_high) & (~gripped.unsqueeze(-1))
        ee_to_target = torch.where(in_lift, lift_delta, torch.where(in_wrap, descend_body, reach_top))
    else:
        ee_to_target = s["grasp_pos"] - s["ee_pos"]

    # Bottle marker at cap/top — target gắp (nắp chai) [marker LỚN]
    if env.num_envs <= 16:
        marker_pos = s["grasp_pos_w"]
        if hasattr(env, "_bottle_markers") and env._bottle_markers is not None:
            env._bottle_markers.visualize(
                translations=marker_pos,
                orientations=s["bottle_quat_w"],
                marker_indices=torch.zeros(env.num_envs, dtype=torch.int, device=env.device),
            )
        # Grasp BODY marker — điểm ngón tay thực sự đặt (body_z + z_offset) [marker NHỎ]
        if hasattr(env, "_grasp_body_markers") and env._grasp_body_markers is not None:
            env._grasp_body_markers.visualize(
                translations=s["grasp_body_pos_w"],
                orientations=s["bottle_quat_w"],
                marker_indices=torch.zeros(env.num_envs, dtype=torch.int, device=env.device),
            )

    # Curriculum Stage representation as a normalized float (always 0.0 for this simple reach task)
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
        ee_to_target,          # [17:20]
        s["bottle_to_bowl"],   # [20:23]
        gripper_state,         # [23]
        clearance_obs,         # [24]
        stage_obs,             # [25]
    ], dim=-1)

    if torch.isnan(obs).any():
        print("[WARNING] Observation contains NaNs!")
        if torch.isnan(q).any(): print(f"  -> NaN in q (joint pos): {q}")
        if torch.isnan(dq).any(): print(f"  -> NaN in dq (joint vel): {dq}")
        if torch.isnan(s["ee_pos"]).any(): print(f"  -> NaN in ee_pos: {s['ee_pos']}")
        if torch.isnan(s["ee_to_grasp"]).any(): print(f"  -> NaN in ee_to_grasp: {s['ee_to_grasp']}")
        if torch.isnan(s["bottle_to_bowl"]).any(): print(f"  -> NaN in bottle_to_bowl: {s['bottle_to_bowl']}")

    return obs
