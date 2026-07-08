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
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import (
    ArticulationCfg,
    RigidObjectCfg,
    AssetBaseCfg,
)
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg

from . import mdp
import isaaclab.envs.mdp as isaaclab_mdp
from .mdp.actions import AssistedBinaryGripperActionCfg, AssistedOperationalSpaceControllerActionCfg
from isaaclab.managers import ActionTermCfg as ActionTerm, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
    EventTermCfg,
)

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
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
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
            # v10.usd gripper is a MIMIC mechanism: only finger_joint1 has a drive;
            # finger_joint2 is a PhysxMimicJoint follower (gearing -1). Driving j2
            # with its own implicit actuator FIGHTS the mimic constraint → j1 slams
            # shut instantly while j2 lags, shoving the bottle out of the pinch.
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_finger_joint1"],
                stiffness=1500.0,
                damping=60.0,
            ),
        },
    )

    # Yellow Bottle
    bottle: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Bottle",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.53, 0.40, 0.638),
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
class JointSpaceActionsCfg:
    """Legacy 8-D joint-space action (7 joint deltas + gripper)."""
    openarm_action = mdp.OpenArmActionTermCfg()


@configclass
class OSCActionsCfg:
    """OSC 6-DOF task-space arm + binary gripper (7-D total for RL)."""
    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm = MISSING


_LEFT_ARM_JOINTS = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
]


def _make_osc_actions_cfg(
    position_scale: float = 0.04,
    orientation_scale: float = 0.20,
    stiffness: float = 80.0,
    damping_ratio: float = 1.5,
) -> OSCActionsCfg:
    """Build OSC pose_rel (6-D) + binary gripper action terms."""
    return OSCActionsCfg(
        arm_action=AssistedOperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=_LEFT_ARM_JOINTS,
            # v10.usd exposes openarm_left_hand but not the fixed TCP link from URDF
            body_name="openarm_left_hand",
            body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.08)),
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                impedance_mode="fixed",
                motion_stiffness_task=(stiffness,) * 6,
                motion_damping_ratio_task=damping_ratio,
                inertial_dynamics_decoupling=True,
                gravity_compensation=True,
                nullspace_control="position",
            ),
            nullspace_joint_pos_target="default",
            position_scale=position_scale,
            orientation_scale=orientation_scale,
        ),
        # Mimic gripper: chỉ drive joint1 — joint2 là PhysxMimic follower (gearing -1).
        # Drive cả hai joint → tranh chấp mimic → một ngón đóng trước, đẩy chai lệch.
        gripper_action=AssistedBinaryGripperActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_finger_joint1"],
            open_command_expr={"openarm_left_finger_joint1": 0.044},
            close_command_expr={"openarm_left_finger_joint1": 0.0},
        ),
    )


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


from isaaclab.envs.mdp import time_out

@configclass
class TerminationsCfg:
    """Termination manager configuration."""
    success = TerminationTermCfg(
        func=mdp.success_termination,
    )
    tipped_bottle = TerminationTermCfg(
        func=mdp.tipped_bottle_termination,
    )
    time_out = TerminationTermCfg(
        func=time_out,
        time_out=True,
    )


@configclass
class EventsCfg:
    """Event manager configuration."""
    # Finger-pad & bottle friction (startup, via PhysX) — default ~0.5 lets the
    # bottle squirt out of the pinch; grippers need ≥1.0 to hold a cylinder.
    finger_friction = EventTermCfg(
        func=isaaclab_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["openarm_left_left_finger", "openarm_left_right_finger"],
            ),
            "static_friction_range": (3.0, 3.0),
            "dynamic_friction_range": (2.4, 2.4),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )
    bottle_friction = EventTermCfg(
        func=isaaclab_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("bottle"),
            "static_friction_range": (1.8, 1.8),
            "dynamic_friction_range": (1.4, 1.4),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )
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
    actions: JointSpaceActionsCfg | OSCActionsCfg = _make_osc_actions_cfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # Action control: OSC 6-DOF (default) or legacy joint-space (incompatible checkpoints)
    use_joint_space_actions: bool = False

    # Curriculum: 1=reach (default), 2=reach+grasp+lift — train/demo phải truyền --task_phase 2
    task_phase: int = 1

    # Task-specific thresholds
    grasp_dist_threshold: float = 0.10
    grasp_grip_threshold: float = 0.4
    place_dist_threshold: float = 0.10

    # Phase 1 reach
    success_dist_threshold: float = 0.07
    success_contact_exit_threshold: float = 0.10
    success_hold_steps: int = 5
    reach_stage_hold_steps: int = 5
    reach_advance_tolerance: float = 1.10   # +10% slack for soft descent / close
    reach_advance_min_top_down: float = 0.4
    reach_advance_max_lateral: float = 0.100
    reach_advance_use_finger_lateral: bool = True   # gate REACH→GRASP on finger XY (not EE TCP offset)
    reach_advance_max_lateral_f: float = 0.100
    reach_advance_max_z_error: float = 0.08
    reach_advance_max_z_finger: float = 0.090
    reach_advance_min_finger_level: float = 0.80
    align_log_interval: int = 50              # demo align log every N contact steps
    debug_success_log: bool = False           # demo phase 2 bật tự động trong isaaclab_demo.py
    suppress_align_log: bool = False            # headless eval: tắt [Align]/[Stage] prints
    success_log_interval: int = 25            # log every N steps while in GRASP

    # Bottle geometry (qvic.usd) — measured from USD bbox 2026-07-02:
    #   size (0.0433, 0.0433, 0.0768), mesh center − root = (−0.0816, −0.0220), bottom = root z
    bottle_height_m: float = 0.0768        # chiều cao toàn chai (đáy → nắp), mét — đo từ USD
    bottle_root_at_base: bool = True       # root USD ở đáy (bbox min z == root z)
    bottle_cap_height_ratio: float = 1.0   # 0=đáy, 1=nắp — chỉnh nếu marker lệch nắp (thử 0.92–1.0)
    bottle_use_usd_mesh_xy: bool = False   # True = đo USD lúc setup (không gọi trong step)
    bottle_grasp_local_xy_x: float = -0.082  # trục chai vs root (local frame), đo từ USD bbox
    bottle_grasp_local_xy_y: float = -0.022  # USD đo −0.022 (config cũ +0.022 sai dấu)
    bottle_grasp_xy_offset_x: float = 0.0     # fine-tune — reset sau khi đo USD ground truth
    bottle_grasp_xy_offset_y: float = 0.0
    bottle_grasp_z_offset: float = 0.0        # cũ −0.05 đặt điểm kẹp DƯỚI đáy chai
    grasp_body_height_ratio: float = 0.52  # điểm kẹp ~giữa-thân (38% quá thấp → z_f dương khi kẹp miệng)
    # v10.usd zero-pose measurements (2026-07-02): hand origin z 0.6099, finger link
    # origins 1.5cm below hand, finger mesh extends to z 0.5144 → tip ≈ 9.55cm below
    # hand origin. Pads at CLOSED are y ±0.006 from hand center; joint travel 0.044
    # moves them OUT to ±0.05 (old code had the sign inverted: open↔closed swapped).
    finger_reach_m: float = 0.0955             # proxy: real fingertip depth below hand
    finger_tip_mode: str = "hand_frame"        # hand_frame = pads from hand kinematics + joint travel
    finger_tip_hand_y_closed_m: float = 0.006    # pad Y from hand center at CLOSED (j=0)
    finger_tip_hand_y_open_m: float = 0.05       # pad Y at fully open (j=0.044)
    finger_tip_hand_z_below_m: float = 0.085     # grip-pad center below hand origin (tip 0.0955)
    finger_tip_local_left: tuple = (0.0, -0.05, -0.673001)
    finger_tip_local_right: tuple = (0.0, 0.05, -0.673001)
    grasp_ee_descend_offset_m: float = 0.07   # TCP below body when wrapping fingers
    grasp_tcp_above_body_m: float = 0.10   # metric only (z_error_tcp_grasp)
    grasp_descent_assist_enabled: bool = False
    grasp_descent_assist_grasp: bool = False  # chỉ bật với --assist / training bootstrap
    grasp_descent_grasp_blend: float = 0.45
    grasp_descent_in_reach: bool = True
    grasp_descent_blend: float = 0.5
    grasp_descent_force_override: bool = False  # legacy; dùng force_in_reach / force_in_grasp
    grasp_descent_force_in_reach: bool = False  # REACH: blend mượt (tránh giật)
    grasp_descent_force_in_grasp: bool = True   # GRASP: ép hạ world-down trước khi đóng
    grasp_descent_world_down: bool = True
    grasp_descent_world_down_min_top: float = 0.62  # top↓ thấp → tool-Z (tránh trượt ngang)
    grasp_descent_world_m: float = 0.010       # mét/bước env (trước osc_position_scale)
    reach_descent_max_z_error: float = 0.15     # rộng hơn advance — đang hạ nên z_err tạm cao OK
    grasp_reward_lift_scale: float = 1.0
    grasp_reward_lift_hold_scale: float = 1.0
    grasp_lift_hold_steps: int = 5
    grasp_success_max_dist_finger: float = 0.10   # ngón vẫn gần thân chai
    grasp_success_max_dist_ee: float = 0.12        # TCP vẫn gần chai
    grasp_success_max_lateral: float = 0.10
    grasp_success_min_top_down: float = 0.50      # gripper vẫn hướng xuống (không hất ngang)
    grasp_success_max_bottle_speed: float = 0.35  # loại chai bay văng
    grasp_success_max_tilt_deg: float = 12.0    # SUCCESS chỉ khi chai đứng (<12°)
    grasp_lift_max_tilt_deg: float = 10.0       # không nhấc khi chai đã nghiêng
    grasp_descend_hold_steps: int = 8            # giữ z_f thấp N bước trước khi đóng
    grasp_latch_max_z_finger: float = 0.010      # latch chỉ khi z_f dưới ngưỡng này (chặt hơn close)
    grasp_slip_reopen_z_finger: float = 0.0      # >0: mở lại grip nếu z_f tăng sau latch (anti-slip)
    grasp_descent_z_finger_loop: bool = True     # GRASP: hạ closed-loop theo z_f (không world-down)
    # Tipped-bottle early termination (Phase 2) — eliminates 700+ wasted steps
    bottle_tipped_termination_deg: float = 60.0  # terminate if GRASP tilt > this
    bottle_tipped_min_steps: int = 5             # require N consecutive tipped steps
    # Sensor-grip threshold for fallback latch and lift assist (see actions.py / grasp_assist.py)
    grasp_sensor_grip_thresh: float = 0.70
    grasp_auto_close: bool = True
    grasp_hold_closed: bool = True   # giữ gripper đóng sau khi kẹp (tránh bung khi nhấc)
    grasp_close_max_lat: float = 0.08   # relaxed: arm achieves lat_f ~0.05-0.10
    grasp_close_max_z_err: float = 0.06  # relaxed: arm achieves z_f ~0.04-0.06
    grasp_close_min_top_down: float = 0.55  # relaxed: arm achieves top_down ~0.59-0.68
    grasp_descent_z_threshold: float = 0.02
    grasp_descent_z_target: float = 0.010   # dừng ép hạ khi z_f dưới ngưỡng này
    grasp_descent_min_top_down: float = 0.50  # assist fires at arm's actual top_down ~0.59
    grasp_descent_action: float = 0.3
    grasp_lift_assist_enabled: bool = False   # demo phase 2 bật tự động
    grasp_lift_action: float = -0.45        # OSC Z âm = nhấc (nhẹ hơn -0.85 để không hất chai)
    grasp_lift_assist_blend: float = 1.0    # 1.0 = ép nhấc (policy v3 không học lift)
    grasp_lift_min_top_down: float = 0.72   # chờ tay thẳng hơn trước khi nhấc
    grasp_lift_max_bottle_tilt_deg: float = 8.0  # không nhấc khi chai đã nghiêng
    grasp_lift_freeze_arm: bool = True       # đứng yên XY/xoay khi đã kẹp, chỉ nhấc Z
    grasp_pregrasp_freeze_arm: bool = True   # GRASP đã align: đứng yên, chỉ đóng gripper
    grasp_lift_world_up: bool = True         # nhấc dọc world +Z (không theo trục EE nghiêng)
    grasp_lift_world_m: float = 0.012        # mét/bước env (trước osc_position_scale)
    grasp_grip_arm_damp: float = 0.05        # khi freeze_arm=False: giảm drift sau khi kẹp
    grasp_close_min_z_finger: float = -0.01  # chặn auto-close khi ngón dưới thân
    grasp_close_ramp_steps: int = 0          # 0=đóng tức; >0=khép từ từ (demo ~50)
    grasp_close_squeeze_m: float = 0.0       # ép nhẹ cuối ramp (m)
    grasp_close_squeeze_start: float = 0.80  # chỉ squeeze khi gc >= 80%
    grasp_lift_settle_steps: int = 5          # bước ổn định sau khép trước khi nhấc
    grasp_lift_start_max_tilt_deg: float = 8.0
    grasp_lift_max_lat_f: float = 0.09
    grasp_lift_max_dist_f: float = 0.10
    grasp_open_until_dist: float = 0.15      # giữ gripper mở khi ngón xa thân chai
    # Mimic gripper: require both pads balanced before close / lift (phase2_overrides)
    grasp_symmetry_gate_enabled: bool = False
    grasp_sym_max_dist_each: float = 0.058
    grasp_sym_max_dist_delta: float = 0.012
    grasp_sym_max_z_delta: float = 0.012
    grasp_sym_xy_balance_blend: float = 0.55
    grasp_reopen_asym_dist_delta: float = 0.018   # pause ramp khi L/R lệch nhẹ
    grasp_reopen_severe_asym_delta: float = 0.030  # reopen sau khép: lệch nặng
    grasp_reopen_max_count: int = 3               # tối đa N lần reopen / episode
    grasp_close_done_progress: float = 0.96       # coi là khép xong (lift settle)
    grasp_reopen_max_tilt_deg: float = 5.0
    grasp_reopen_min_close_progress: float = 0.92  # không cắt giữa ramp (tránh loop)
    grasp_reopen_cooldown_steps: int = 45
    grasp_reopen_abort_tilt_deg: float = 12.0   # trên ngưỡng: không reopen (chai đã ngã)
    grasp_reopen_mid_tilt_deg: float = 4.0    # mid-ramp reopen khi lệch nặng + tilt nhẹ
    grasp_reopen_ramp_tilt_deg: float = 5.0   # mid-ramp reopen chỉ vì nghiêng
    grasp_reopen_ramp_asym_tilt_deg: float = 2.5  # reopen khi pad lệch + tilt nhẹ
    grasp_reopen_ramp_asym_min_gc: float = 0.45   # gc tối thiểu trước asym-reopen
    grasp_close_ramp_max_tilt_deg: float = 3.0  # pause ramp khi tilt vượt ngưỡng
    grasp_grasp_abort_tilt_deg: float = 18.0    # dừng mọi assist GRASP
    grasp_close_pause_on_asym: bool = True
    grasp_lift_partial_enabled: bool = False
    grasp_lift_partial_min_gc: float = 0.55
    grasp_lift_partial_max_tilt_deg: float = 6.5
    grasp_lift_partial_settle_steps: int = 1
    grasp_lift_partial_max_z_finger: float = 0.045
    grasp_lift_partial_max_dist_f: float = 0.075
    grasp_lift_partial_world_m: float = 0.024
    grasp_close_freeze_on_reopen_exhaust: bool = True
    grasp_sym_hold_steps: int = 0          # giữ sym ổn định N bước trước auto-close

    # Option C — assisted training curriculum (blend 1.0 → 0.0 over training)
    grasp_assist_schedule_enabled: bool = False
    grasp_assist_blend_start: float = 1.0
    grasp_assist_blend_end: float = 0.0

    # OSC speed / responsiveness (demo tuning — retrain if changing a lot)
    osc_position_scale: float = 0.06      # ↑ faster linear motion (try 0.06–0.08)
    osc_orientation_scale: float = 0.20
    osc_stiffness: float = 90.0           # ↑ snappier (try 100–120)
    osc_damping_ratio: float = 1.5        # ↓ faster but may oscillate (try 1.2)

    fine_control_dist: float = 0.10
    fine_control_scale: float = 0.3
    top_down_min_for_close_reward: float = 0.65

    # Domain randomization
    bottle_pos_noise: float = 0.05
    bowl_pos_noise: float = 0.05

    def __post_init__(self):
        """Wire OSC torque control or legacy joint-space actions."""
        if self.use_joint_space_actions:
            self.actions = JointSpaceActionsCfg()
        else:
            self.actions = _make_osc_actions_cfg(
                position_scale=self.osc_position_scale,
                orientation_scale=self.osc_orientation_scale,
                stiffness=self.osc_stiffness,
                damping_ratio=self.osc_damping_ratio,
            )
            self.scene.robot.actuators["arm"].stiffness = 0.0
            self.scene.robot.actuators["arm"].damping = 4.0
        if self.task_phase >= 2:
            self.episode_length_s = 20.0
