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

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import (
    ArticulationCfg,
    RigidObjectCfg,
    AssetBaseCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
    EventTermCfg,
)

from . import mdp
from .mdp.helpers import STAGE_REACH, STAGE_GRASP, STAGE_PLACE
from .scene import spawn_qvic_with_physics

# ── Asset paths ───────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Reinforce_Learning/ package root directory
_RL_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))

# Your pre-built Isaac Sim scene
QVIC_USD_PATH = os.path.join(_THIS_DIR, "qvic.usd")

# OpenArm A1 v10 robot USD (using symlink or directory under Reinforce_Learning)
V10_USD_PATH = os.path.join(
    _RL_DIR,
    "openarm_description", "urdf", "robot", "v10", "v10.usd"
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Scene Configuration
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class OpenArmSceneCfg(InteractiveSceneCfg):
    """
    Defines all assets in the simulation scene.
    Clones everything that has {ENV_REGEX_NS} in its prim_path per parallel environment.
    """

    # Enable high-speed physics replication for instant cloning/spawning
    replicate_physics: bool = True

    # Static scene backdrop - Loads via our custom physics-injecting spawn function
    scene_env: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(
            func=spawn_qvic_with_physics,
            usd_path=QVIC_USD_PATH,
            rigid_props=None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # OpenArm A1 v10 robot
    robot_spawn: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/openarm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=V10_USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                # Disable rigid body on all non-root links coming from the USD
                # so that nested prims (e.g. Realsense RSD455) do not form an
                # illegal rigid-body-inside-rigid-body hierarchy.
                rigid_body_enabled=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                fix_root_link=True,
            ),
        ),
    )

    # Articulation entity
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/openarm/root_joint",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "openarm_left_joint1":          0.0,
                "openarm_left_joint2":          0.0,
                "openarm_left_joint3":          0.0,
                "openarm_left_joint4":          1.57,
                "openarm_left_joint5":          0.0,
                "openarm_left_joint6":          0.0,
                "openarm_left_joint7":          0.0,
                "openarm_left_finger_joint1":   0.044,
                "openarm_left_finger_joint2":   0.044,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_joint[1-7]"],
                stiffness=80.0,
                damping=8.0,
            ),
            # NOTE: Use explicit joint names (not character class regex) to
            # avoid silent mismatch on finger joints ending in 1 or 2.
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_finger_joint1", "openarm_left_finger_joint2"],
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )

    # Yellow Bottle
    bottle: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Bottle",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.53, 0.40, 0.64),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Bowl
    bowl: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Bowl",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.58, 0.22, 0.67),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.95, 0.95, 0.95)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Manager Configurations
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class ActionsCfg:
    """Action manager configuration."""
    openarm_action = mdp.OpenArmActionTermCfg()


@configclass
class ObservationsCfg:
    """Observation manager configuration."""
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        obs_term = ObservationTermCfg(
            func=mdp.get_apple_pick_place_obs,
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward manager configuration."""
    curriculum_reward = RewardTermCfg(
        func=mdp.compute_curriculum_reward,
        weight=1.0,
    )


@configclass
class TerminationsCfg:
    """Termination manager configuration."""
    success = TerminationTermCfg(
        func=mdp.success_termination,
    )


@configclass
class EventsCfg:
    """Event manager configuration."""
    # Reset callbacks (prestartup physics applied natively by spawn_qvic_with_physics)
    reset_robot = EventTermCfg(
        func=mdp.reset_robot,
        mode="reset",
    )
    reset_bottle = EventTermCfg(
        func=mdp.reset_bottle,
        mode="reset",
    )
    reset_bowl = EventTermCfg(
        func=mdp.reset_bowl,
        mode="reset",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Environment Configuration
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class ApplePickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """
    Top-level Manager-based RL environment configuration for the Apple Pick-and-Place task.
    """

    # Simulation setup
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,      # 120 Hz physics
        render_interval=2,   # render every 2 physics steps
    )

    # Scene setup
    scene: OpenArmSceneCfg = OpenArmSceneCfg(
        num_envs=1024,
        env_spacing=2.5,     # meters between parallel environment origins
    )

    # RL Decimation & Length
    decimation: int = 2                   # RL step = 2 physics steps -> 60 Hz
    episode_length_s: float = 15.0        # 15 second episodes -> 900 steps

    # ── Manager Declarations ──
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # Task-specific thresholds
    grasp_dist_threshold: float = 0.10
    grasp_grip_threshold: float = 0.4
    place_dist_threshold: float = 0.10

    # Domain randomization
    bottle_pos_noise: float = 0.05
    bowl_pos_noise: float = 0.05
