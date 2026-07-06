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
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    OperationalSpaceControllerActionCfg,
)
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.utils import configclass

from .grasp_assist import apply_grasp_arm_assist
from .helpers import STAGE_GRASP, STAGE_REACH, finger_grasp_ready, finger_descended_for_close, uses_grasp_lift


class AssistedOperationalSpaceControllerAction(OperationalSpaceControllerAction):
    """OSC arm action with phase-2 descent/lift assist applied at process_actions."""

    def process_actions(self, actions: torch.Tensor):
        actions = apply_grasp_arm_assist(self._env, actions)
        super().process_actions(actions)


class AssistedBinaryGripperAction(BinaryJointPositionAction):
    """Binary gripper with optional auto-close in GRASP stage (Phase 2 bootstrap)."""

    def __init__(self, cfg: BinaryJointPositionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._grasp_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._descend_hold = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._want_close = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._close_progress = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._grasp_latched[:] = False
            self._descend_hold[:] = 0
            self._want_close[:] = False
            self._close_progress[:] = 0.0
        else:
            self._grasp_latched[env_ids] = False
            self._descend_hold[env_ids] = 0
            self._want_close[env_ids] = False
            self._close_progress[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        env = self._env
        if (
            getattr(env.cfg, "grasp_auto_close", True)
            and uses_grasp_lift(getattr(env.cfg, "task_phase", 1))
            and hasattr(env, "_stage")
        ):
            actions = actions.clone()
            in_grasp = env._stage == STAGE_GRASP
            s = getattr(env, "_last_state", None)
            grip_thresh = getattr(env.cfg, "grasp_grip_threshold", 0.4)
            hold_closed = getattr(env.cfg, "grasp_hold_closed", True)

            if s is not None and hold_closed:
                reopen_z = float(getattr(env.cfg, "grasp_slip_reopen_z_finger", 0.0))
                if reopen_z > 0.0:
                    lift_thresh = getattr(env.cfg, "grasp_lift_threshold", 0.03)
                    slipped = (
                        in_grasp
                        & self._grasp_latched
                        & (s["z_error_finger"] > reopen_z)
                        & (s["bottle_lift"] < lift_thresh)
                    )
                    if slipped.any():
                        self._grasp_latched[slipped] = False
                        self._close_progress[slipped] = 0.0
                        self._descend_hold[slipped] = 0

                ready = finger_grasp_ready(env, s)
                latch_z = float(getattr(
                    env.cfg, "grasp_latch_max_z_finger",
                    getattr(env.cfg, "grasp_close_max_z_err", 0.04) * 0.85,
                ))
                low_enough = s["z_error_finger"] < latch_z
                hold_req = int(getattr(env.cfg, "grasp_descend_hold_steps", 5))
                # Chỉ đếm hold trong GRASP — tránh REACH tích 100+ bước rồi đóng ngay khi vào GRASP
                if in_grasp.any():
                    self._descend_hold[low_enough & in_grasp] += 1
                    self._descend_hold[(~low_enough) & in_grasp] = 0
                else:
                    self._descend_hold[:] = 0
                hold_done = in_grasp & (self._descend_hold >= hold_req)
                latch_ok = s["z_error_finger"] < latch_z
                self._grasp_latched |= in_grasp & ready & hold_done & latch_ok
                latched = in_grasp & self._grasp_latched
                min_z_f = getattr(env.cfg, "grasp_close_min_z_finger", -0.01)
                may_close_z = s["z_error_finger"] >= min_z_f
                # Sau latch: luôn giữ đóng; trước latch: cần ready + dh + z_f
                force_close = latched | (in_grasp & ready & hold_done & may_close_z)
                # GRASP: giữ gripper mở cho đến khi ngón hạ đủ thấp
                not_descended = ~finger_descended_for_close(env, s)
                actions[in_grasp & not_descended, 0] = 1.0
            elif s is not None:
                low_enough = finger_descended_for_close(env, s)
                hold_req = int(getattr(env.cfg, "grasp_descend_hold_steps", 5))
                if in_grasp.any():
                    self._descend_hold[low_enough & in_grasp] += 1
                    self._descend_hold[(~low_enough) & in_grasp] = 0
                else:
                    self._descend_hold[:] = 0
                ready = finger_grasp_ready(env, s) & in_grasp & (self._descend_hold >= hold_req)
                force_close = in_grasp & ready
            else:
                force_close = in_grasp

            if s is not None:
                too_far = s["dist_finger_body"] > getattr(env.cfg, "grasp_open_until_dist", 0.15)
                force_close = force_close & ~too_far
                # GRASP trước latch: ép mở — policy hay đóng sớm (dh chưa đủ)
                actions[in_grasp & ~force_close, 0] = 1.0
                actions[too_far & in_grasp, 0] = 1.0

            actions[force_close, 0] = -1.0
            actions[env._stage == STAGE_REACH, 0] = 1.0
            self._want_close[:] = False
            if s is not None:
                self._want_close[:] = force_close
        super().process_actions(actions)

    def apply_actions(self):
        ramp_steps = int(getattr(self._env.cfg, "grasp_close_ramp_steps", 0))
        if ramp_steps > 0 and uses_grasp_lift(getattr(self._env.cfg, "task_phase", 1)):
            # Tăng dần progress — KHÔNG snap 1.0 khi latch (trước đây bỏ qua ramp)
            ramping = self._want_close & (self._close_progress < 0.99)
            self._close_progress[ramping] += 1.0 / float(ramp_steps)
            self._close_progress.clamp_(0.0, 1.0)
            if not self._want_close.any() and not self._grasp_latched.any():
                self._close_progress[:] = 0.0

            prog = self._close_progress.unsqueeze(-1)
            open_t = self._open_command.unsqueeze(0)
            close_t = self._close_command.unsqueeze(0)
            squeeze = float(getattr(self._env.cfg, "grasp_close_squeeze_m", 0.0))
            squeeze_start = float(getattr(self._env.cfg, "grasp_close_squeeze_start", 0.80))
            if squeeze > 0.0:
                late = (self._close_progress >= squeeze_start).unsqueeze(-1)
                close_t = torch.where(late, close_t - squeeze, close_t)
            targets = open_t * (1.0 - prog) + close_t * prog
            self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)
            return
        super().apply_actions()


@configclass
class AssistedBinaryGripperActionCfg(BinaryJointPositionActionCfg):
    class_type: type = AssistedBinaryGripperAction


@configclass
class AssistedOperationalSpaceControllerActionCfg(OperationalSpaceControllerActionCfg):
    class_type: type = AssistedOperationalSpaceControllerAction


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
        
        # Get control time step (decimation * sim_dt = 2 * (1/120) = 1/60s)
        dt = self._env.cfg.sim.dt * self._env.cfg.decimation
        
        # Velocity limits from joint_limits.yaml:
        # joint 1-4: 2.175 rad/s -> max step delta = 2.175 * dt
        # joint 5-7: 2.610 rad/s -> max step delta = 2.610 * dt
        max_vel = torch.tensor([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610], device=self._robot.device)
        max_delta = max_vel * dt

        # Phase 1 fine control: reduce arm step size when TCP is near the bottle
        arm_scale = torch.ones(self._env.num_envs, 1, device=self._robot.device)
        if hasattr(self._env, "_last_state") and self._env._last_state is not None:
            dist = self._env._last_state["dist_ee_bottle"]
            fine_dist = getattr(self._env.cfg, "fine_control_dist", 0.10)
            fine_scale = getattr(self._env.cfg, "fine_control_scale", 0.3)
            arm_scale[dist < fine_dist] = fine_scale

        delta_q = self._processed_actions[:, :7] * max_delta * arm_scale
        target_q = current_q + delta_q
        self._robot.set_joint_position_target(target_q, joint_ids=self._arm_joint_ids)

        # Calculate and write gripper target positions based on gripper_state
        # Open (gripper_state = 0) -> finger_pos = 0.044
        # Closed (gripper_state = 1) -> finger_pos = 0.0
        finger_pos = (1.0 - self.gripper_state) * 0.044
        finger_targets = finger_pos.repeat(1, 2)
        self._robot.set_joint_position_target(finger_targets, joint_ids=self._gripper_joint_ids)


@configclass
class OpenArmActionTermCfg(ActionTermCfg):
    """Configuration class for OpenArmActionTerm."""
    class_type: type = OpenArmActionTerm
    asset_name: str = "robot"
