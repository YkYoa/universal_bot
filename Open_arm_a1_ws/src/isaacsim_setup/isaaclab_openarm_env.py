#!/usr/bin/env python3
"""
isaaclab_openarm_env.py
========================
Apple Pick-and-Place RL Environment for OpenArm A1 v10 — Isaac Lab Edition.

Follows the Isaac Lab DirectRLEnv API as described in:
  https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/technical_env_design.html
  https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/api_env_design.html

Design Pattern (DirectRLEnv workflow):
  1. @configclass  ApplePickPlaceEnvCfg(DirectRLEnvCfg)
       - sim:     SimulationCfg    — physics timestep, substeps
       - scene:   OpenArmSceneCfg  — robot + objects + environment
  2. class ApplePickPlaceEnv(DirectRLEnv)
       - _pre_physics_step()  — store actions
       - _apply_action()      — send joint position targets
       - _get_observations()  — build 24-D observation vector (GPU tensors)
       - _get_rewards()       — dense reward computation
       - _get_dones()         — success / timeout termination
       - _reset_idx()         — per-env reset

Scene Assets:
  • Scene backdrop  : qvic.usd      — table, walls, cameras (static)
  • Robot           : v10.usd       — OpenArm A1 v10 bimanual (ArticulationCfg)
  • Bottle          : RigidObject   — yellow bottle
  • Bowl            : RigidObject   — red bowl

Joint naming convention (from v10.urdf):
  Left arm  : openarm_left_joint1 … openarm_left_joint7
  L gripper : openarm_left_finger_joint1, openarm_left_finger_joint2

Usage (headless, no rendering):
  /path/to/isaac-sim-standalone/python.sh isaaclab_openarm_env.py --num_envs 4 --headless

Usage (with viewport):
  /path/to/isaac-sim-standalone/python.sh isaaclab_openarm_env.py --num_envs 1
"""

from __future__ import annotations

import os
import math
import torch
import gymnasium as gym

# ── Isaac Lab imports ──────────────────────────────────────────────────────────
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    RigidObject,
    RigidObjectCfg,
    AssetBaseCfg,
)
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform, quat_apply
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

# ── Asset paths ───────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WS_DIR   = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

# Your pre-built Isaac Sim scene (table, environment backdrop, cameras)
QVIC_USD_PATH = os.path.join(_THIS_DIR, "qvic.usd")

# OpenArm A1 v10 robot USD (from openarm_description package)
V10_USD_PATH  = os.path.join(
    _WS_DIR,
    "src", "openarm_description", "urdf", "robot", "v10", "v10.usd"
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Scene Configuration
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class OpenArmSceneCfg(InteractiveSceneCfg):
    """
    Defines all assets in the simulation scene.

    Isaac Lab clones everything that has {ENV_REGEX_NS} in its prim_path
    across all parallel environments automatically.
    Assets without {ENV_REGEX_NS} are global (shared backdrop).
    """

    # ── 1. Static scene backdrop (your existing qvic.usd) ─────────────────────
    # Loaded once at /World/Scene — table, backdrop walls, etc.
    # The robot that was "baked in" to qvic.usd is IGNORED here;
    # we spawn a fresh ArticulationCfg robot below so Isaac Lab can manage it.
    scene_env: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path=QVIC_USD_PATH,
            rigid_props=None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # ── 2. OpenArm A1 v10 robot — cloned per environment ─────────────────────
    # {ENV_REGEX_NS} expands to /World/envs/env_0, env_1, … per env.
    # We first spawn the USD file at the base path `{ENV_REGEX_NS}/openarm`.
    robot_spawn: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/openarm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=V10_USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                fix_root_link=True,
            ),
        ),
    )

    # Then we define the Articulation entity pointing to the child prim `/root_joint`
    # (where the ArticulationRootAPI actually resides), without triggering another spawn.
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/openarm/root_joint",
        spawn=None,
        # Safe home configuration within OpenArm A1 joint limits:
        #   joint1 ∈ [-1.40, 3.49]    → 0.0  ✓
        #   joint2 ∈ [-1.75, 1.75]    → -0.4 ✓
        #   joint3 ∈ [-1.57, 1.57]    → 0.0  ✓
        #   joint4 ∈ [ 0.00, 2.44]    → 1.2  ✓  (was -1.5 in old env → INVALID)
        #   joint5 ∈ [-1.57, 1.57]    → 0.0  ✓
        #   joint6 ∈ [-0.79, 0.79]    → 0.5  ✓  (was 1.1 in old env → INVALID)
        #   joint7 ∈ [-1.57, 1.57]    → 0.0  ✓
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "openarm_left_joint1":          0.0,
                "openarm_left_joint2":          0.0,
                "openarm_left_joint3":          0.0,
                "openarm_left_joint4":          0.0,
                "openarm_left_joint5":          0.0,
                "openarm_left_joint6":          0.0,
                "openarm_left_joint7":          0.0,
                # Fingers open (0.044 m = fully open)
                "openarm_left_finger_joint1":   0.044,
                "openarm_left_finger_joint2":   0.044,
            },
        ),
        # Implicit actuators — PD control, driven by position targets
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_joint[1-7]"],
                stiffness=80.0,
                damping=4.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_finger_joint[12]"],
                stiffness=200.0,
                damping=10.0,
            ),
        },
    )

    # ── 3. Bottle — yellow bottle, spawned per environment ────────────────────────
    bottle: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Bottle",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.53, 0.40, 0.64),  # on the table top from qvic.usd
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ── 4. Bowl — shallow cylinder, spawned per environment ───────────────────
    bowl: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Bowl",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.58, 0.22, 0.67),  # on the table top from qvic.usd
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ── Ground plane ──────────────────────────────────────────────────────────
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
    )

    # ── Lighting ──────────────────────────────────────────────────────────────
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.95, 0.95, 0.95)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Environment Configuration
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class ApplePickPlaceEnvCfg(DirectRLEnvCfg):
    """
    Top-level configuration for the Apple Pick-and-Place task.
    Passed to ApplePickPlaceEnv.__init__().
    """

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,      # 120 Hz physics
        render_interval=2,   # render every 2 physics steps
    )

    # Scene
    scene: OpenArmSceneCfg = OpenArmSceneCfg(
        num_envs=1024,
        env_spacing=2.5,     # meters between parallel environment origins
    )

    # RL hyperparameters
    decimation: int = 2                   # RL step = 2 physics steps → 60 Hz effective
    episode_length_s: float = 15.0        # 15 second episodes = 900 RL steps (extended for full pick+place)
    action_space: int = 8                 # [joint_delta×7, gripper_cmd×1]
    observation_space: int = 24           # [q×7, dq×7, ee_pos×3, bottle×3, bowl×3, grip×1]
    state_space: int = 0                  # no asymmetric states

    # Task-specific thresholds
    grasp_dist_threshold: float = 0.10   # EE–bottle distance to trigger grasp (loosened from 0.06)
    grasp_grip_threshold: float = 0.4    # gripper closedness fraction to grasp (loosened from 0.6)
    place_dist_threshold: float = 0.10   # bottle–bowl distance to trigger success (loosened from 0.08)

    # Domain randomization
    bottle_pos_noise: float = 0.05        # ±m in x/y around nominal position
    bowl_pos_noise: float = 0.05         # ±m in x/y around nominal position


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DirectRLEnv Implementation
# ─────────────────────────────────────────────────────────────────────────────

class ApplePickPlaceEnv(DirectRLEnv):
    """
    Apple Pick-and-Place environment using Isaac Lab DirectRLEnv workflow.

    Observation (24-D):
        [0:7]   Left arm joint positions  (rad)
        [7:14]  Left arm joint velocities (rad/s)
        [14:17] End-effector position     (m, relative to env origin)
        [17:20] Bottle position           (m, relative to env origin)
        [20:23] Bowl position             (m, relative to env origin)
        [23]    Gripper state             (0.0=open, 1.0=closed)

    Action (8-D):
        [0:7]   Joint position deltas     (scaled ×0.05 rad internally)
        [7]     Gripper command           (-1.0=open, +1.0=close)
    """

    cfg: ApplePickPlaceEnvCfg

    # ── Lifecycle: __init__ ───────────────────────────────────────────────────

    def __init__(self, cfg: ApplePickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        # super().__init__ triggers _setup_scene() internally
        super().__init__(cfg, render_mode, **kwargs)

        # ── Asset references (set after super().__init__) ─────────────────
        self._robot: Articulation = self.scene["robot"]
        self._bottle: RigidObject  = self.scene["bottle"]
        self._bowl:  RigidObject  = self.scene["bowl"]

        # ── Joint index arrays ────────────────────────────────────────────
        # find_joints returns (joint_ids, joint_names)
        self._arm_joint_ids, _   = self._robot.find_joints(
            ["openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
             "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
             "openarm_left_joint7"]
        )
        self._gripper_joint_ids, _ = self._robot.find_joints(
            ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
        )

        # ── End-effector body index ───────────────────────────────────────
        # Try to find the left hand TCP link; fall back to last arm link
        try:
            self._ee_body_ids, _ = self._robot.find_bodies(["openarm_left_hand_tcp"])
            self._ee_body_id = self._ee_body_ids[0]
        except Exception:
            # Fall back: use link7 as end-effector reference
            self._ee_body_ids, _ = self._robot.find_bodies(["openarm_left_link7"])
            self._ee_body_id = self._ee_body_ids[0]

        # Try to find the hand link specifically for TCP tracking
        try:
            self._hand_body_ids, _ = self._robot.find_bodies(["openarm_left_hand"])
            self._hand_body_id = self._hand_body_ids[0]
        except Exception:
            self._hand_body_id = self._ee_body_id

        # ── Pre-allocated action buffer ───────────────────────────────────
        self._actions     = torch.zeros(self.num_envs, 8, device=self.device)
        self._gripper_state = torch.zeros(self.num_envs, 1, device=self.device)

        # ── Nominal object positions (for domain-randomized resets) ───────
        self._bottle_nominal_pos = torch.tensor([0.53, 0.40, 0.64], device=self.device)
        self._bowl_nominal_pos  = torch.tensor([0.58, 0.22, 0.67], device=self.device)

        # ── Offsets to correct visual/collision geometries relative to parent roots ──
        self._bottle_offset = torch.tensor([-0.08157, -0.02198, 0.0], device=self.device)
        self._bowl_offset   = torch.tensor([-0.10619, 0.0, -0.01851], device=self.device)

        # ── Progress buffers: previous distances for delta-based rewards ───
        self._prev_dist_ee_bottle   = torch.full((self.num_envs,), 1.0, device=self.device)
        self._prev_dist_bottle_bowl = torch.full((self.num_envs,), 1.0, device=self.device)
        # Deprecated name kept for compatibility with reset code
        self._prev_bottle_bowl_dist = self._prev_dist_bottle_bowl

        print(f"[ApplePickPlaceEnv] Initialized with {self.num_envs} envs on device={self.device}")
        print(f"[ApplePickPlaceEnv] Arm joint IDs   : {self._arm_joint_ids}")
        print(f"[ApplePickPlaceEnv] Gripper joint IDs: {self._gripper_joint_ids}")
        print(f"[ApplePickPlaceEnv] EE body ID       : {self._ee_body_id}")
        print(f"[ApplePickPlaceEnv] Hand body ID     : {self._hand_body_id}")

        # ── Coordinate Frame Visualizers ──────────────────────────────────
        ee_marker_cfg = FRAME_MARKER_CFG.copy()
        ee_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self._ee_markers = VisualizationMarkers(ee_marker_cfg.replace(prim_path="/Visuals/ee_marker"))

        tcp_marker_cfg = FRAME_MARKER_CFG.copy()
        tcp_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self._tcp_markers = VisualizationMarkers(tcp_marker_cfg.replace(prim_path="/Visuals/tcp_marker"))

    # ── Lifecycle: _setup_scene ────────────────────────────────────────────────
    def _setup_scene(self):
        """
        Isaac Lab calls this during super().__init__().
        InteractiveScene automatically parses OpenArmSceneCfg and spawns all
        assets — we just need to register the cloning pattern.
        """
        # Clone environments: use copy_from_source=True for small env counts (e.g. demo viewport)
        # to ensure static visual meshes are duplicated and visible, but use copy_from_source=False
        # for large env counts (e.g. training) to save startup time and memory.
        copy_from_source = True if self.num_envs <= 32 else False
        self.scene.clone_environments(copy_from_source=copy_from_source)
        # Filter collisions across environments
        self.scene.filter_collisions(global_prim_paths=["/World/GroundPlane"])

        # Deactivate duplicate and unused dynamic assets loaded from the qvic.usd backdrop
        # to prevent them from colliding with our environment-cloned active assets.
        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        if stage:
            # Prevent the table from shaking by removing RigidBodyAPI from it and its children in all env clones.
            # We perform a single traversal of the USD stage here to avoid O(N^2) complexity with large env counts.
            for prim in stage.Traverse():
                prim_path_str = str(prim.GetPath())
                if "/Scene/Table" in prim_path_str:
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                        if "/envs/env_0/" in prim_path_str:
                            print(f"[ApplePickPlaceEnv] Removed UsdPhysics.RigidBodyAPI from table prim: {prim_path_str}")

            from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf

            def enable_collisions(prim, approximation_type="convexHull"):
                if not prim.IsValid():
                    return
                if prim.IsA(UsdGeom.Mesh):
                    if not prim.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI.Apply(prim)
                    # Apply PhysxCollisionAPI
                    PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    # Set the approximation attribute directly via Sdf to avoid bindings version mismatch
                    prim.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set(approximation_type)
                for child in prim.GetChildren():
                    enable_collisions(child, approximation_type)

            for i in range(self.num_envs):
                # Deactivate duplicate static robot, ActionGraph, and PushGraph in each env clone
                for prim_name in ["openarm", "ActionGraph", "PushGraph"]:
                    prim_path = f"/World/envs/env_{i}/Scene/{prim_name}"
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim.IsValid():
                        prim.SetActive(False)
                        # Only print for env_0 to avoid flooding the console
                        if i == 0:
                            print(f"[ApplePickPlaceEnv] Deactivated duplicate static prim at: {prim_path}")

                # Apply UsdPhysics.RigidBodyAPI to the Bowl if missing and make it kinematic
                bowl_path = f"/World/envs/env_{i}/Scene/Bowl"
                bowl_prim = stage.GetPrimAtPath(bowl_path)
                if bowl_prim.IsValid():
                    if not bowl_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        rb = UsdPhysics.RigidBodyAPI.Apply(bowl_prim)
                        rb.CreateKinematicEnabledAttr(True)
                        if i == 0:
                            print(f"[ApplePickPlaceEnv] Applied UsdPhysics.RigidBodyAPI (kinematic) to Bowl at: {bowl_path}")
                    # Enable concave collision decomposition so the bottle can fit inside the bowl
                    enable_collisions(bowl_prim, "convexDecomposition")
                    if i == 0:
                        print(f"[ApplePickPlaceEnv] Applied UsdPhysics.CollisionAPI (convexDecomposition) to Bowl at: {bowl_path}")
 
                # Apply UsdPhysics.RigidBodyAPI to the Bottle if missing and make it dynamic
                bottle_path = f"/World/envs/env_{i}/Scene/Bottle"
                bottle_prim = stage.GetPrimAtPath(bottle_path)
                if bottle_prim.IsValid():
                    if not bottle_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        rb = UsdPhysics.RigidBodyAPI.Apply(bottle_prim)
                        rb.CreateKinematicEnabledAttr(False)
                        if i == 0:
                            print(f"[ApplePickPlaceEnv] Applied UsdPhysics.RigidBodyAPI (dynamic) to Bottle at: {bottle_path}")
                    # Enable convex hull collisions for the bottle
                    enable_collisions(bottle_prim, "convexHull")
                    if i == 0:
                        print(f"[ApplePickPlaceEnv] Applied UsdPhysics.CollisionAPI (convexHull) to Bottle at: {bottle_path}")

                # Deactivate Realsense cameras on the spawned robot to silence camera warnings
                for arm in ["left", "right"]:
                    realsense_path = f"/World/envs/env_{i}/openarm/openarm_{arm}_hand/openarm_{arm}_hand_tcp/Realsense"
                    realsense_prim = stage.GetPrimAtPath(realsense_path)
                    if realsense_prim.IsValid():
                        realsense_prim.SetActive(False)
                        if i == 0:
                            print(f"[ApplePickPlaceEnv] Deactivated Realsense camera at: {realsense_path}")

    # ── Step 1: Cache actions before physics steps ─────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Called once per RL step BEFORE the physics sub-steps."""
        self._actions = actions.clone().clamp(-1.0, 1.0)

    # ── Step 2: Send commands to actuators ────────────────────────────────────
    def _apply_action(self) -> None:
        """Called once per PHYSICS sub-step to push targets to the robot."""
        # ── Arm joints: current position + delta * 0.05 rad ──────────────
        current_q = self._robot.data.joint_pos[:, self._arm_joint_ids]
        delta_q   = self._actions[:, :7] * 0.05                        # scale delta
        target_q  = current_q + delta_q
        self._robot.set_joint_position_target(target_q, joint_ids=self._arm_joint_ids)

        # ── Gripper: map [-1, +1] → finger position [0.044 m, 0.0 m] ────
        # gripper_cmd=+1 → fully closed (0.0 m), gripper_cmd=-1 → open (0.044 m)
        gripper_cmd    = self._actions[:, 7:8]                           # (N, 1)
        finger_pos     = (1.0 - (gripper_cmd + 1.0) / 2.0) * 0.044     # (N, 1)
        finger_targets = finger_pos.expand(-1, 2)                        # (N, 2)
        self._robot.set_joint_position_target(finger_targets, joint_ids=self._gripper_joint_ids)

        # Update gripper state scalar (0=open, 1=closed)
        self._gripper_state = (gripper_cmd + 1.0) / 2.0

    # ── Step 3: Build observation ──────────────────────────────────────────────
    def _get_observations(self) -> dict:
        """
        Returns a dict with key 'policy' mapping to a (num_envs, 24) tensor.
        All positions are expressed relative to each env's origin for
        translation-invariant observations across parallel envs.
        """
        origins = self.scene.env_origins   # (N, 3)

        # Joint states
        q  = self._robot.data.joint_pos[:, self._arm_joint_ids]   # (N, 7)
        dq = self._robot.data.joint_vel[:, self._arm_joint_ids]   # (N, 7)

        # TCP position (world frame): offset 8cm along local Z-axis of openarm_left_hand
        hand_pos_w = self._robot.data.body_pos_w[:, self._hand_body_id, :] # (N, 3)
        hand_quat_w = self._robot.data.body_quat_w[:, self._hand_body_id, :] # (N, 4)
        tcp_offset = torch.tensor([0.0, 0.0, 0.08], device=self.device).repeat(self.num_envs, 1)
        
        # Use actual TCP for the end-effector relative coordinates
        ee_world = hand_pos_w + quat_apply(hand_quat_w, tcp_offset)
        ee_rel   = ee_world - origins                                     # (N, 3)
        ee_quat_w = hand_quat_w.clone()

        # Update visualizer coordinate axes in Isaac Sim
        # (ee_markers shows the wrist link, tcp_markers shows the actual TCP)
        wrist_pos_w = self._robot.data.body_pos_w[:, self._ee_body_id, :]
        wrist_quat_w = self._robot.data.body_quat_w[:, self._ee_body_id, :]
        self._ee_markers.visualize(wrist_pos_w, wrist_quat_w)
        self._tcp_markers.visualize(ee_world, ee_quat_w)

        # Bottle position relative to env origin (with offset correction)
        bottle_world = self._bottle.data.root_pos_w + self._bottle_offset   # (N, 3)
        bottle_rel   = bottle_world - origins                               # (N, 3)

        # Bowl position relative to env origin (with offset correction)
        bowl_world  = self._bowl.data.root_pos_w + self._bowl_offset       # (N, 3)
        bowl_rel    = bowl_world - origins                                # (N, 3)

        # Translate absolute coordinates into direct steering vectors (relative views)
        ee_to_bottle = bottle_rel - ee_rel                                 # (N, 3)
        bottle_to_bowl = bowl_rel - bottle_rel                             # (N, 3)

        # Total is still 7 + 7 + 3 + 3 + 3 + 1 = 24 dimensions (100% backward compatible)
        obs = torch.cat([q, dq, ee_rel, ee_to_bottle, bottle_to_bowl, self._gripper_state], dim=-1)
        
        if torch.isnan(obs).any():
            print("=== OBS CONTAINS NAN ===")
            print("NaN in q:", torch.isnan(q).any())
            print("NaN in dq:", torch.isnan(dq).any())
            print("NaN in ee_rel:", torch.isnan(ee_rel).any())
            print("NaN in ee_to_bottle:", torch.isnan(ee_to_bottle).any())
            print("NaN in bottle_to_bowl:", torch.isnan(bottle_to_bowl).any())
            print("NaN in gripper_state:", torch.isnan(self._gripper_state).any())
        return {"policy": obs}

    # ── Step 4: Compute rewards ────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        """
        Dense multi-stage picking and placing reward function.
        Designed to avoid reward hacking while maintaining continuous learning signals.
        """
        origins = self.scene.env_origins  # (N, 3)

        # TCP position (world frame): offset 8cm along local Z-axis of openarm_left_hand
        hand_pos_w = self._robot.data.body_pos_w[:, self._hand_body_id, :] # (N, 3)
        hand_quat_w = self._robot.data.body_quat_w[:, self._hand_body_id, :] # (N, 4)
        tcp_offset = torch.tensor([0.0, 0.0, 0.08], device=self.device).repeat(self.num_envs, 1)
        ee_world   = hand_pos_w + quat_apply(hand_quat_w, tcp_offset)
        
        ee_pos     = ee_world - origins
        bottle_pos = self._bottle.data.root_pos_w + self._bottle_offset - origins
        bowl_pos   = self._bowl.data.root_pos_w + self._bowl_offset - origins

        dist_ee_bottle   = torch.norm(ee_pos - bottle_pos, dim=-1)    # (N,)
        dist_bottle_bowl = torch.norm(bottle_pos - bowl_pos, dim=-1)  # (N,)

        # 1. Approach bottle (Dense continuous exponentially shaped reward)
        # Scaled up to encourage reaching immediately
        r_reach = torch.exp(-5.0 * dist_ee_bottle) * 2.5

        # Gripper wrapping incentive: Open gripper when approaching, close when very close
        gripper_state = self._gripper_state[:, 0]
        r_gripper = torch.where(
            dist_ee_bottle < 0.05,
            gripper_state * 1.5,                 # Close kẹp to wrap and squeeze
            (1.0 - gripper_state) * 0.5          # Keep kẹp open to avoid colliding early
        )

        # 2. Squeeze/Grasp detection (Physical check: bottle must be raised above table level 0.64m)
        bottle_height = bottle_pos[:, 2] - 0.64
        is_lifted = (bottle_pos[:, 2] > 0.67)    # Real physical lift > 3cm

        r_lift = torch.zeros_like(r_reach)
        r_transport = torch.zeros_like(r_reach)
        r_grasp_bonus = torch.zeros_like(r_reach)

        # 3. Lift and Transport rewards (Only enabled when the bottle is PHYSICALLY lifted)
        if is_lifted.any():
            # Constant grasp reward for holding the bottle off the table
            r_grasp_bonus = is_lifted.float() * 5.0
            
            # Target height is 15cm above table (Z = 0.79m)
            target_height = 0.15
            lift_error = torch.abs(bottle_height - target_height)
            r_lift = is_lifted.float() * torch.exp(-3.0 * lift_error) * 2.0
            
            # Strong transport pull towards the bowl
            r_transport = is_lifted.float() * torch.exp(-5.0 * dist_bottle_bowl) * 10.0

        # 4. Place and Release reward
        placed_in_bowl = (dist_bottle_bowl < self.cfg.place_dist_threshold)
        gripper_open   = (gripper_state < 0.4)
        
        # Large placing bonus + opening gripper to let go
        r_place = (placed_in_bowl & is_lifted).float() * 15.0
        r_release = (placed_in_bowl & gripper_open).float() * 5.0

        # 5. Penalties for joint velocities and table boundaries to ensure smooth motions
        joint_vel   = self._robot.data.joint_vel[:, self._arm_joint_ids]
        vel_penalty = -0.001 * torch.sum(joint_vel ** 2, dim=-1)

        # Table penalty (EE or bottle falls below 0.58m to prevent aggressive table smashing)
        ee_off_table  = (ee_pos[:, 2] < 0.58)
        table_penalty = ee_off_table.float() * -2.0

        # Update previous distance buffers for compatibility
        self._prev_dist_ee_bottle   = dist_ee_bottle.detach()
        self._prev_dist_bottle_bowl = dist_bottle_bowl.detach()

        total_reward = (
            r_reach + r_gripper + r_grasp_bonus + r_lift + 
            r_transport + r_place + r_release + vel_penalty + table_penalty
        )
        return total_reward

    # ── Step 5: Termination conditions ────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            terminated  (N,) bool — success condition (bottle placed in bowl and gripper opened)
            truncated   (N,) bool — timeout condition (max episode length)
        """
        origins = self.scene.env_origins
        bottle_pos = self._bottle.data.root_pos_w + self._bottle_offset - origins
        bowl_pos   = self._bowl.data.root_pos_w + self._bowl_offset - origins

        dist_bottle_bowl = torch.norm(bottle_pos - bowl_pos, dim=-1)
        gripper_open     = (self._gripper_state[:, 0] < 0.4)

        # Success: bottle inside bowl, gripper released, and bottle was physically lifted
        is_lifted = (bottle_pos[:, 2] > 0.65)
        terminated = (dist_bottle_bowl < self.cfg.place_dist_threshold) & gripper_open & is_lifted

        # Timeout: episode exceeded max length (set by DirectRLEnv base class)
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    # ── Step 6: Reset individual environments ─────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """
        Reset joint positions + object poses for specified environment indices.
        Called by DirectRLEnv at episode boundaries.
        """
        if env_ids is None or len(env_ids) == 0:
            return

        # Reset base class buffers (like episode_length_buf and noise models)
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)

        # ── Reset robot to home pose ──────────────────────────────────────
        # joint_pos shape: (num_envs, num_joints)
        default_joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = torch.zeros_like(default_joint_pos)
        self._robot.set_joint_position_target(default_joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        self._robot.reset(env_ids)

        # ── Reset bottle with domain randomization ─────────────────────────
        bottle_pos = self._bottle_nominal_pos.unsqueeze(0).expand(num_reset, -1).clone()
        # Random ±5 cm in x and y
        bottle_pos[:, 0] += sample_uniform(-self.cfg.bottle_pos_noise, self.cfg.bottle_pos_noise, (num_reset,), device=self.device)
        bottle_pos[:, 1] += sample_uniform(-self.cfg.bottle_pos_noise, self.cfg.bottle_pos_noise, (num_reset,), device=self.device)
        # Translate to world frame
        bottle_pos_world = bottle_pos + self.scene.env_origins[env_ids]
        bottle_root_state = self._bottle.data.default_root_state[env_ids].clone()
        bottle_root_state[:, :3] = bottle_pos_world
        bottle_root_state[:, 7:] = 0.0  # zero velocities
        self._bottle.write_root_state_to_sim(bottle_root_state, env_ids=env_ids)
        self._bottle.reset(env_ids)

        # ── Reset bowl with domain randomization ──────────────────────────
        bowl_pos = self._bowl_nominal_pos.unsqueeze(0).expand(num_reset, -1).clone()
        bowl_pos[:, 0] += sample_uniform(-self.cfg.bowl_pos_noise, self.cfg.bowl_pos_noise, (num_reset,), device=self.device)
        bowl_pos[:, 1] += sample_uniform(-self.cfg.bowl_pos_noise, self.cfg.bowl_pos_noise, (num_reset,), device=self.device)
        bowl_pos_world = bowl_pos + self.scene.env_origins[env_ids]
        bowl_root_state = self._bowl.data.default_root_state[env_ids].clone()
        bowl_root_state[:, :3] = bowl_pos_world
        bowl_root_state[:, 7:] = 0.0
        self._bowl.write_root_state_to_sim(bowl_root_state, env_ids=env_ids)
        self._bowl.reset(env_ids)

        # ── Reset gripper state and action buffers ────────────────────────
        self._actions[env_ids]               = 0.0
        self._gripper_state[env_ids]         = 0.0
        # Reset progress buffers to large initial values so first step gets no false reward
        self._prev_dist_ee_bottle[env_ids]   = 1.0
        self._prev_dist_bottle_bowl[env_ids] = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Gymnasium Registration
# ─────────────────────────────────────────────────────────────────────────────

gym.register(
    id="Isaac-OpenArm-Apple-Pick-Place-v0",
    entry_point="isaaclab_openarm_env:ApplePickPlaceEnv",
    kwargs={
        "env_cfg_entry_point": "isaaclab_openarm_env:ApplePickPlaceEnvCfg",
    },
    disable_env_checker=True,  # Isaac Lab envs don't conform to standard gym checker
)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Standalone entry point (for quick testing)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run with Isaac Sim's bundled python:
        /path/to/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \\
            src/isaacsim_setup/isaaclab_openarm_env.py --num_envs 4

    Optional flags:
        --headless       run without GUI viewport
        --num_envs N     number of parallel environments
        --num_steps N    number of RL steps to simulate (default 200)
    """
    import argparse
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="OpenArm Apple Pick-and-Place — Isaac Lab")
    parser.add_argument("--num_envs",  type=int, default=4,   help="Number of parallel environments")
    parser.add_argument("--num_steps", type=int, default=200, help="Number of RL steps to simulate")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch Isaac Sim application
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Build environment configuration
    env_cfg = ApplePickPlaceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # Instantiate the environment
    env = ApplePickPlaceEnv(cfg=env_cfg)

    print(f"\n{'='*60}")
    print(f"  Isaac Lab: OpenArm Apple Pick-and-Place")
    print(f"  Num envs  : {env.num_envs}")
    print(f"  Device    : {env.device}")
    print(f"  Obs space : {env.observation_space}")
    print(f"  Act space : {env.action_space}")
    print(f"{'='*60}\n")

    # Run random policy for num_steps
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs['policy'].shape}")

    for step in range(args.num_steps):
        # Random actions in [-1, 1]
        actions = torch.rand(env.num_envs, 8, device=env.device) * 2.0 - 1.0
        obs, rewards, terminated, truncated, infos = env.step(actions)

        if (step + 1) % 50 == 0:
            mean_reward = rewards.mean().item()
            num_done    = (terminated | truncated).sum().item()
            print(f"  Step {step+1:4d} | mean_reward={mean_reward:+.3f} | envs_done={num_done}/{env.num_envs}")

    print("\n✅  Test complete — environment ran successfully.")
    env.close()
    simulation_app.close()
