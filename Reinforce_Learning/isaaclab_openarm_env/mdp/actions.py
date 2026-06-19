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
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass


class OpenArmActionTerm(ActionTerm):
    """
    Custom 8-D action term for the OpenArm A1 robot arm.
    Controls 7 arm joint position deltas and 1 joint position target for the gripper.
    Conforms to the full abstract interface of Isaac Lab ActionTerm.
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._robot = env.scene["robot"]
        self._arm_joint_ids, _ = self._robot.find_joints([
            "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
            "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
            "openarm_left_joint7"
        ])
        self._gripper_joint_ids, _ = self._robot.find_joints([
            "openarm_left_finger_joint1", "openarm_left_finger_joint2"
        ])
        
        # Pre-initialize states
        self._raw_actions = torch.zeros(env.num_envs, 8, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 8, device=env.device)
        self.gripper_state = torch.zeros(env.num_envs, 1, device=env.device)

    @property
    def action_dim(self) -> int:
        return 8

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions.clone().clamp(-1.0, 1.0)
        self._processed_actions = self._raw_actions
        
        # Scale gripper command: gripper_cmd in [-1.0, 1.0] -> gripper_state in [0.0, 1.0] (0=open, 1=closed)
        gripper_cmd = self._raw_actions[:, 7:8]
        self.gripper_state = (gripper_cmd + 1.0) / 2.0

    def apply_actions(self):
        # Calculate and write arm target positions
        current_q = self._robot.data.joint_pos[:, self._arm_joint_ids]
        delta_q = self._processed_actions[:, :7] * 0.08
        target_q = current_q + delta_q
        self._robot.set_joint_position_target(target_q, joint_ids=self._arm_joint_ids)

        # Calculate and write gripper target positions (finger_pos in [0.0, 0.044])
        finger_pos = (1.0 - self.gripper_state) * 0.044
        finger_targets = finger_pos.expand(-1, 2)
        self._robot.set_joint_position_target(finger_targets, joint_ids=self._gripper_joint_ids)


@configclass
class OpenArmActionTermCfg(ActionTermCfg):
    """Configuration class for OpenArmActionTerm."""
    class_type: type = OpenArmActionTerm
    asset_name: str = "robot"
