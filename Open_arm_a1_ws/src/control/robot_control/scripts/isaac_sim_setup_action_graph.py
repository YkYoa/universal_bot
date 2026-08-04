"""
Isaac Sim Action Graph Setup Script — OpenArm Bimanual ROS2 Bridge
===================================================================
Compatible with Isaac Sim 4.5.0+ using the new isaacsim.ros2.bridge extension.

Run in Isaac Sim Script Editor (Window > Script Editor).
ISAACSIM VERSION = 5.1.0

Robot URDF: v10.urdf (openarm bimanual robot)
  Left arm joints:  openarm_left_joint1..7
  Right arm joints: openarm_right_joint1..7
  Left gripper:     openarm_left_finger_joint1, openarm_left_finger_joint2
  Right gripper:    openarm_right_finger_joint1, openarm_right_finger_joint2

Published topics:
  /clock, /tf
  /isaac_left_arm/joint_states, /isaac_right_arm/joint_states
  /isaac_left_gripper/joint_states, /isaac_right_gripper/joint_states
  /camera_front/image_raw, /camera_left/image_raw
  /camera_front/camera_info, /camera_left/camera_info

Subscribed topics:
  /isaac_left_arm/joint_commands, /isaac_right_arm/joint_commands
  /isaac_left_gripper/joint_commands, /isaac_right_gripper/joint_commands

ROS_DOMAIN_ID = 42
"""

import carb
import omni
import omni.usd
import omni.graph.core as og
import omni.kit.commands

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GRAPH_PATH = "/World/ActionGraph"

ROBOT_PRIM         = "/World/openarm"
BASE_LINK_PRIM     = "/World/openarm/openarm_base_link"
CAMERA_FRONT_PRIM  = "/World/Camera_front"
CAMERA_LEFT_PRIM   = "/World/openarm/openarm_left_hand/openarm_left_hand_tcp/Realsense/RSD455/Camera_OmniVision_OV9782_Color"
CAMERA_WIDTH       = 1280
CAMERA_HEIGHT      = 720

# Joint names from v10.urdf
LEFT_ARM_JOINTS = [
    "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
    "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
    "openarm_left_joint7",
]
RIGHT_ARM_JOINTS = [
    "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
    "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6",
    "openarm_right_joint7",
]
LEFT_GRIPPER_JOINTS = [
    "openarm_left_finger_joint1", "openarm_left_finger_joint2",
]
RIGHT_GRIPPER_JOINTS = [
    "openarm_right_finger_joint1", "openarm_right_finger_joint2",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_extensions_loaded():
    """Enable required Isaac Sim 4.5+ extensions and ensure CycloneDDS config is valid."""
    import omni.kit.app
    import os
    import os.path
    
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    
    # Check and fix CYCLONEDDS_URI if it points to a non-existent local file (common in SSH environments)
    uri = os.environ.get("CYCLONEDDS_URI", "")
    bad_uri = False
    if uri.startswith("file://"):
        filepath = uri[7:]
        if not os.path.exists(filepath):
            bad_uri = True
            
    if bad_uri or not uri:
        # Look for valid server-side candidates
        candidates = [
            "/home/naiscorp/openarm_ws/docker/cyclonedds_server.xml",
            "/home/naiscorp/openarm_ws/cyclonedds.xml",
        ]
        found_config = None
        for candidate in candidates:
            if os.path.exists(candidate):
                found_config = candidate
                break
                
        if found_config:
            new_uri = f"file://{found_config}"
            carb.log_warn(f"[OpenArm] CYCLONEDDS_URI '{uri}' is invalid/missing on this machine. Correcting to '{new_uri}'")
            os.environ["CYCLONEDDS_URI"] = new_uri
            os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
            
            # Force reload the extension so the new environment variable takes effect
            ext_name = "isaacsim.ros2.bridge"
            if ext_manager.is_extension_enabled(ext_name):
                carb.log_warn(f"[OpenArm] Reloading {ext_name} to apply correct CYCLONEDDS_URI...")
                try:
                    ext_manager.set_extension_enabled_immediate(ext_name, False)
                    ext_manager.set_extension_enabled_immediate(ext_name, True)
                except Exception as e:
                    carb.log_error(f"[OpenArm] Failed to reload {ext_name}: {e}")

    # Isaac Sim 4.5.0+ uses isaacsim.ros2.bridge (not omni.isaac.ros2_bridge)
    required = [
        "isaacsim.ros2.bridge",
        "isaacsim.core.nodes",
    ]
    for ext in required:
        if not ext_manager.is_extension_enabled(ext):
            carb.log_warn(f"[OpenArm] Enabling extension: {ext}")
            ext_manager.set_extension_enabled_immediate(ext, True)
        else:
            carb.log_info(f"[OpenArm] Extension already loaded: {ext}")


def _delete_graph_if_exists(graph_path: str):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(graph_path)
    if prim.IsValid():
        carb.log_warn(f"[OpenArm] Deleting existing graph at {graph_path}")
        omni.kit.commands.execute("DeletePrims", paths=[graph_path])


def _verify_stage_assets():
    """Print Camera, Articulation, and Top-level prims in stage to help debugging."""
    from pxr import UsdPhysics, PhysxSchema
    stage = omni.usd.get_context().get_stage()

    carb.log_warn("[OpenArm] === Top-level prims under /World ===")
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim.IsValid():
        for child in world_prim.GetChildren():
            carb.log_warn(f"[OpenArm]   {child.GetPath()} ({child.GetTypeName()})")

    carb.log_warn("[OpenArm] === Camera prims in stage ===")
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Camera":
            carb.log_warn(f"[OpenArm]   {prim.GetPath()}")

    carb.log_warn("[OpenArm] === Articulation Root prims in stage ===")
    found_any = False
    for prim in stage.Traverse():
        if (prim.HasAPI(UsdPhysics.ArticulationRootAPI) or 
            prim.HasAPI(PhysxSchema.PhysxArticulationAPI) or 
            prim.GetTypeName() == "PhysicsArticulationRoot"):
            carb.log_warn(f"[OpenArm]   {prim.GetPath()} ({prim.GetTypeName()})")
            found_any = True
    if not found_any:
        carb.log_error("[OpenArm]   ❌ NO Articulation Roots found in the stage! Make sure your robot has 'Articulation Root' API applied.")


def _resolve_left_camera_prim():
    """Find the left RealSense camera prim even if the mount path changed in USD."""
    global CAMERA_LEFT_PRIM
    stage = omni.usd.get_context().get_stage()

    configured = stage.GetPrimAtPath(CAMERA_LEFT_PRIM)
    if configured.IsValid() and configured.GetTypeName() == "Camera":
        return CAMERA_LEFT_PRIM

    candidates = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Camera":
            continue
        path = str(prim.GetPath())
        lowered = path.lower()
        if not path.startswith("/World/openarm/"):
            continue
        if "realsense" in lowered or "rsd455" in lowered or "omnivision" in lowered:
            candidates.append(path)

    if candidates:
        # Prefer the hand-mounted camera if more than one matching RealSense camera exists.
        candidates.sort(key=lambda p: ("openarm_left_hand" not in p, len(p)))
        CAMERA_LEFT_PRIM = candidates[0]
        carb.log_warn(f"[OpenArm] Updated left camera prim to existing stage camera: {CAMERA_LEFT_PRIM}")
    else:
        carb.log_error(f"[OpenArm] Left camera prim does not exist: {CAMERA_LEFT_PRIM}")

    return CAMERA_LEFT_PRIM


def _fix_camera_physics():
    """Remove RigidBodyAPI and CollisionAPI from the camera mounting prims
    so they behave as static children attached to the TCP link, avoiding nested rigid body errors."""
    from pxr import Usd, UsdPhysics, PhysxSchema
    stage = omni.usd.get_context().get_stage()
    
    _resolve_left_camera_prim()

    # Prims that represent the camera mount and may be nested under the hand/TCP.
    paths_to_clean = [
        "/World/openarm/openarm_left_hand/Realsense",
        "/World/openarm/openarm_left_hand/Realsense/RSD455",
        "/World/openarm/openarm_left_hand/openarm_left_hand_tcp/Realsense",
        "/World/openarm/openarm_left_hand/openarm_left_hand_tcp/Realsense/RSD455",
        "/World/openarm/openarm_left_hand/openarm_left_hand_tcp/rsd455",
        "/World/openarm/openarm_left_hand/openarm_left_hand_tcp/rsd455/RSD455",
    ]

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        lowered = path.lower()
        if (
            path.startswith("/World/openarm/openarm_left_hand/")
            and ("realsense" in lowered or "rsd455" in lowered)
            and path not in paths_to_clean
        ):
            paths_to_clean.append(path)

    # Also dynamically trace upwards from CAMERA_LEFT_PRIM up to the mount parent
    if CAMERA_LEFT_PRIM:
        curr_path = CAMERA_LEFT_PRIM
        for _ in range(6):
            parent_path = curr_path.rsplit('/', 1)[0]
            parent_lower = parent_path.lower()
            # Only trace camera mount components; do not modify actual robot links.
            if "rsd455" in parent_lower or "realsense" in parent_lower:
                if parent_path not in paths_to_clean:
                    paths_to_clean.append(parent_path)
                curr_path = parent_path
            else:
                break

    carb.log_warn(f"[OpenArm] Cleaning physics APIs from camera hierarchy: {paths_to_clean}")

    # Joint type names that can appear from URDF importers
    JOINT_TYPE_NAMES = {
        "PhysicsFixedJoint", "PhysicsRevoluteJoint", "PhysicsPrismaticJoint",
        "PhysicsSphericalJoint", "PhysicsDistanceJoint", "PhysicsJoint",
    }

    # 1. Delete any joint prims that reference these camera paths
    #    Use BOTH IsA(UsdPhysics.Joint) AND typename check to catch all variants.
    joints_to_delete = []
    for prim in stage.Traverse():
        is_joint = prim.IsA(UsdPhysics.Joint) or (prim.GetTypeName() in JOINT_TYPE_NAMES)
        if is_joint:
            joint = UsdPhysics.Joint(prim)
            b0_targets = [str(t) for t in joint.GetBody0Rel().GetTargets()]
            b1_targets = [str(t) for t in joint.GetBody1Rel().GetTargets()]
            all_targets = b0_targets + b1_targets
            # Match exact paths AND any path that is a child of a camera path
            if any(
                t == cam_path or t.startswith(cam_path + "/")
                for t in all_targets
                for cam_path in paths_to_clean
            ):
                joints_to_delete.append(str(prim.GetPath()))

    # Also scan below the hand for joints pointing to RealSense/RSD455 paths.
    hand_prim = stage.GetPrimAtPath("/World/openarm/openarm_left_hand")
    if hand_prim.IsValid():
        for child in Usd.PrimRange(hand_prim):
            if child.GetTypeName() in JOINT_TYPE_NAMES:
                path_str = str(child.GetPath())
                if path_str not in joints_to_delete:
                    joint = UsdPhysics.Joint(child)
                    b0 = [str(t) for t in joint.GetBody0Rel().GetTargets()]
                    b1 = [str(t) for t in joint.GetBody1Rel().GetTargets()]
                    if any(("rsd455" in t.lower() or "realsense" in t.lower()) for t in (b0 + b1)):
                        joints_to_delete.append(path_str)

    if joints_to_delete:
        carb.log_warn(f"[OpenArm] Deleting joints referencing camera mount: {joints_to_delete}")
        omni.kit.commands.execute("DeletePrims", paths=joints_to_delete)

    # 2. Clean up APIs on the target paths and their children.
    cleaned_paths = []
    for path_str in paths_to_clean:
        root = stage.GetPrimAtPath(path_str)
        if not root.IsValid():
            continue

        for prim in Usd.PrimRange(root):
            prim_path = str(prim.GetPath())
            # Remove RigidBodyAPI
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                carb.log_warn(f"[OpenArm] Removing UsdPhysics.RigidBodyAPI from {prim_path} to fix nested rigid body conflict.")
                try:
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    cleaned_paths.append(prim_path)
                except Exception as e:
                    carb.log_error(f"[OpenArm] Failed to remove UsdPhysics.RigidBodyAPI: {e}")

            # Remove PhysxSchema.PhysxRigidBodyAPI
            if hasattr(PhysxSchema, "PhysxRigidBodyAPI") and prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                try:
                    prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
                except Exception as e:
                    pass

            # Remove general Collision API
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                try:
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)
                    cleaned_paths.append(prim_path)
                except Exception as e:
                    pass

            # Remove Physx Collision API
            if hasattr(PhysxSchema, "PhysxCollisionAPI") and prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                try:
                    prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
                except Exception as e:
                    pass

    if cleaned_paths:
        carb.log_warn(f"[OpenArm] Camera physics cleanup touched {len(set(cleaned_paths))} prim(s).")


def _delete_stale_render_products():
    """Remove old Replicator/Hydra render products before rebuilding camera graph nodes."""
    stage = omni.usd.get_context().get_stage()
    stale_paths = []
    render_root = stage.GetPrimAtPath("/Render")

    if render_root.IsValid():
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if (
                path == "/Render/OmniverseKit/HydraTextures/Replicator"
                or path.startswith("/Render/OmniverseKit/HydraTextures/Replicator")
            ):
                stale_paths.append(path)

    # Delete deepest prims first so children do not outlive their parents in the request.
    stale_paths = sorted(set(stale_paths), key=len, reverse=True)
    if stale_paths:
        carb.log_warn(f"[OpenArm] Deleting stale render products: {stale_paths}")
        try:
            omni.kit.commands.execute("DeletePrims", paths=stale_paths)
        except Exception as e:
            carb.log_warn(f"[OpenArm] Could not delete stale render products: {e}")


def _fix_dynamic_mesh_collisions():
    """Use convex hull collision for robot mesh colliders so dynamic links are valid in PhysX."""
    from pxr import Usd, UsdPhysics
    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(ROBOT_PRIM)
    if not robot_prim.IsValid():
        carb.log_warn(f"[OpenArm] Cannot fix mesh collisions; robot prim not found: {ROBOT_PRIM}")
        return

    updated = []
    for prim in Usd.PrimRange(robot_prim):
        if "Realsense" in str(prim.GetPath()) or "rsd455" in str(prim.GetPath()).lower():
            continue
        if not (prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.MeshCollisionAPI)):
            continue

        try:
            mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)

            approx_attr = mesh_collision.GetApproximationAttr()
            if approx_attr.Get() != "convexHull":
                approx_attr.Set("convexHull")
                updated.append(str(prim.GetPath()))
        except Exception as e:
            carb.log_warn(f"[OpenArm] Could not set convexHull collision on {prim.GetPath()}: {e}")

    if updated:
        carb.log_warn(f"[OpenArm] Set convexHull collision approximation on {len(updated)} robot collider prim(s).")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SETUP
# ─────────────────────────────────────────────────────────────────────────────

# Node type prefix for Isaac Sim 4.5+ (changed from omni.isaac.ros2_bridge)
ROS2  = "isaacsim.ros2.bridge"
CORE  = "isaacsim.core.nodes"
GRAPH = "omni.graph.action"

def setup_ros2_action_graph():
    global ROBOT_PRIM, BASE_LINK_PRIM
    carb.log_info("[OpenArm] === Setting up ROS2 Action Graph (Isaac Sim 4.5+) ===")

    _ensure_extensions_loaded()
    _verify_stage_assets()
    _delete_stale_render_products()
    _fix_camera_physics()
    _fix_dynamic_mesh_collisions()

    # Find any articulation root dynamically if the default path is invalid/missing
    from pxr import Usd, UsdPhysics, PhysxSchema
    stage = omni.usd.get_context().get_stage()
    default_prim = stage.GetPrimAtPath(ROBOT_PRIM)
    is_valid_articulation = False

    def _prim_is_articulation(p):
        return (p.HasAPI(UsdPhysics.ArticulationRootAPI) or
                p.HasAPI(PhysxSchema.PhysxArticulationAPI) or
                p.GetTypeName() == "PhysicsArticulationRoot")

    if default_prim.IsValid():
        if _prim_is_articulation(default_prim):
            # ROBOT_PRIM itself is the articulation root
            is_valid_articulation = True
            carb.log_info(f"[OpenArm] ROBOT_PRIM '{ROBOT_PRIM}' is a valid Articulation Root.")
        else:
            # Check if any DESCENDANT of ROBOT_PRIM already has ArticulationRootAPI
            # (e.g. openarm_base_link). If so, do NOT apply a second one — that
            # would conflict with the existing root and break physics / make arms vanish.
            descendant_root = None
            for desc in Usd.PrimRange(default_prim):
                if desc == default_prim:
                    continue
                if _prim_is_articulation(desc):
                    descendant_root = str(desc.GetPath())
                    break

            if descendant_root:
                carb.log_warn(
                    f"[OpenArm] ROBOT_PRIM '{ROBOT_PRIM}' is not an Articulation Root, "
                    f"but descendant '{descendant_root}' already is. "
                    f"Updating ROBOT_PRIM to '{descendant_root}' (physics requires exact articulation root path)."
                )
                ROBOT_PRIM = descendant_root
                is_valid_articulation = True
            else:
                # No articulation root anywhere in the robot subtree — safe to apply
                carb.log_warn(
                    f"[OpenArm] No Articulation Root found in '{ROBOT_PRIM}' subtree. "
                    f"Applying ArticulationRootAPI to '{ROBOT_PRIM}'..."
                )
                UsdPhysics.ArticulationRootAPI.Apply(default_prim)
                PhysxSchema.PhysxArticulationAPI.Apply(default_prim)
                is_valid_articulation = True

    if not is_valid_articulation:
        # ROBOT_PRIM doesn't exist — search the entire stage for any articulation root
        carb.log_warn(f"[OpenArm] Target ROBOT_PRIM '{ROBOT_PRIM}' is invalid or missing. Searching stage for any existing articulation root...")
        for prim in stage.Traverse():
            if _prim_is_articulation(prim):
                new_path = str(prim.GetPath())
                carb.log_warn(f"[OpenArm] Found articulation root at: {new_path}. Automatically updating ROBOT_PRIM!")
                ROBOT_PRIM = new_path
                BASE_LINK_PRIM = f"{ROBOT_PRIM}/openarm_base_link"
                is_valid_articulation = True
                break

    if not is_valid_articulation:
        carb.log_error(f"[OpenArm] ❌ No articulation roots found anywhere in the stage and ROBOT_PRIM '{ROBOT_PRIM}' does not exist!")

    _delete_graph_if_exists(GRAPH_PATH)

    keys = og.Controller.Keys

    try:
        og.Controller.edit(
            {
                "graph_path": GRAPH_PATH,
                "evaluator_name": "execution",
            },
            {
                # ── Nodes ────────────────────────────────────────────────────
                # Node names match the exported graph from the scene
                keys.CREATE_NODES: [
                    ("on_playback_tick",                          f"{GRAPH}.OnPlaybackTick"),
                    ("ros2_context",                              f"{ROS2}.ROS2Context"),
                    ("sim_time",                                  f"{CORE}.IsaacReadSimulationTime"),

                    # Clock & TF
                    ("clock_pub",                                 f"{ROS2}.ROS2PublishClock"),
                    ("tf_pub",                                    f"{ROS2}.ROS2PublishTransformTree"),

                    # Joint State Publishers (publish current robot state → ROS2)
                    ("ros2_publish_joint_state_left",              f"{ROS2}.ROS2PublishJointState"),
                    ("ros2_publish_joint_state_right",             f"{ROS2}.ROS2PublishJointState"),
                    ("ros2_publish_joint_state_gripper_left",      f"{ROS2}.ROS2PublishJointState"),
                    ("ros2_publish_joint_state_gripper_right",     f"{ROS2}.ROS2PublishJointState"),

                    # Joint Command Subscribers (receive commands from ROS2)
                    ("ros2_subscribe_joint_state_left",            f"{ROS2}.ROS2SubscribeJointState"),
                    ("articulation_controller_left",              f"{CORE}.IsaacArticulationController"),
                    ("ros2_subscribe_joint_state_right",           f"{ROS2}.ROS2SubscribeJointState"),
                    ("articulation_controller_right",             f"{CORE}.IsaacArticulationController"),
                    ("ros2_subscribe_joint_state_gripper_left",    f"{ROS2}.ROS2SubscribeJointState"),
                    ("articulation_controller_gripper_left",      f"{CORE}.IsaacArticulationController"),
                    ("ros2_subscribe_joint_state_gripper_right",   f"{ROS2}.ROS2SubscribeJointState"),
                    ("articulation_controller_gripper_right",     f"{CORE}.IsaacArticulationController"),

                    # Cameras
                    ("isaac_create_render_product_front",          f"{CORE}.IsaacCreateRenderProduct"),
                    ("ros2_camera_helper_front",                  f"{ROS2}.ROS2CameraHelper"),
                    ("ros2_camera_info_front",                    f"{ROS2}.ROS2CameraInfoHelper"),
                    ("isaac_create_render_product_left",           f"{CORE}.IsaacCreateRenderProduct"),
                    ("ros2_camera_helper_left",                   f"{ROS2}.ROS2CameraHelper"),
                    ("ros2_camera_info_left",                     f"{ROS2}.ROS2CameraInfoHelper"),
                ],

                # ── Values ───────────────────────────────────────────────────
                keys.SET_VALUES: [
                    # ROS2 Context — explicit domain ID
                    ("ros2_context.inputs:domain_id",                          42),
                    ("ros2_context.inputs:useDomainIDEnvVar",                  False),

                    # Clock & TF
                    ("clock_pub.inputs:topicName",                             "/clock"),
                    ("tf_pub.inputs:topicName",                                "/tf"),
                    ("tf_pub.inputs:targetPrims",                              [ROBOT_PRIM]),

                    # ── Joint State Publishers ──
                    # LEFT ARM
                    ("ros2_publish_joint_state_left.inputs:topicName",          "/isaac_left_arm/joint_states"),
                    ("ros2_publish_joint_state_left.inputs:targetPrim",         [ROBOT_PRIM]),
                    # RIGHT ARM
                    ("ros2_publish_joint_state_right.inputs:topicName",         "/isaac_right_arm/joint_states"),
                    ("ros2_publish_joint_state_right.inputs:targetPrim",        [ROBOT_PRIM]),
                    # LEFT GRIPPER
                    ("ros2_publish_joint_state_gripper_left.inputs:topicName",  "/isaac_left_gripper/joint_states"),
                    ("ros2_publish_joint_state_gripper_left.inputs:targetPrim", [ROBOT_PRIM]),
                    # RIGHT GRIPPER
                    ("ros2_publish_joint_state_gripper_right.inputs:topicName", "/isaac_right_gripper/joint_states"),
                    ("ros2_publish_joint_state_gripper_right.inputs:targetPrim",[ROBOT_PRIM]),

                    # ── Joint Command Subscribers + Articulation Controllers ──
                    # LEFT ARM
                    ("ros2_subscribe_joint_state_left.inputs:topicName",        "/isaac_left_arm/joint_commands"),
                    ("articulation_controller_left.inputs:targetPrim",          [ROBOT_PRIM]),
                    ("articulation_controller_left.inputs:jointNames",          LEFT_ARM_JOINTS),

                    # RIGHT ARM
                    ("ros2_subscribe_joint_state_right.inputs:topicName",       "/isaac_right_arm/joint_commands"),
                    ("articulation_controller_right.inputs:targetPrim",         [ROBOT_PRIM]),
                    ("articulation_controller_right.inputs:jointNames",         RIGHT_ARM_JOINTS),

                    # LEFT GRIPPER
                    ("ros2_subscribe_joint_state_gripper_left.inputs:topicName","/isaac_left_gripper/joint_commands"),
                    ("articulation_controller_gripper_left.inputs:targetPrim",  [ROBOT_PRIM]),
                    ("articulation_controller_gripper_left.inputs:jointNames",  LEFT_GRIPPER_JOINTS),

                    # RIGHT GRIPPER
                    ("ros2_subscribe_joint_state_gripper_right.inputs:topicName","/isaac_right_gripper/joint_commands"),
                    ("articulation_controller_gripper_right.inputs:targetPrim", [ROBOT_PRIM]),
                    ("articulation_controller_gripper_right.inputs:jointNames", RIGHT_GRIPPER_JOINTS),

                    # ── Cameras ──
                    # FRONT
                    ("isaac_create_render_product_front.inputs:cameraPrim",     [CAMERA_FRONT_PRIM]),
                    ("isaac_create_render_product_front.inputs:enabled",        True),
                    ("isaac_create_render_product_front.inputs:width",          CAMERA_WIDTH),
                    ("isaac_create_render_product_front.inputs:height",         CAMERA_HEIGHT),
                    ("ros2_camera_helper_front.inputs:topicName",              "/camera_front/image_raw"),
                    ("ros2_camera_helper_front.inputs:type",                   "rgb"),
                    ("ros2_camera_helper_front.inputs:frameId",                "camera_front_link"),
                    ("ros2_camera_info_front.inputs:topicName",                "/camera_front/camera_info"),
                    ("ros2_camera_info_front.inputs:frameId",                  "camera_front_link"),

                    # LEFT
                    ("isaac_create_render_product_left.inputs:cameraPrim",      [CAMERA_LEFT_PRIM]),
                    ("isaac_create_render_product_left.inputs:enabled",         True),
                    ("isaac_create_render_product_left.inputs:width",           CAMERA_WIDTH),
                    ("isaac_create_render_product_left.inputs:height",          CAMERA_HEIGHT),
                    ("ros2_camera_helper_left.inputs:topicName",               "/camera_left/image_raw"),
                    ("ros2_camera_helper_left.inputs:type",                    "rgb"),
                    ("ros2_camera_helper_left.inputs:frameId",                 "camera_left_link"),
                    ("ros2_camera_info_left.inputs:topicName",                 "/camera_left/camera_info"),
                    ("ros2_camera_info_left.inputs:frameId",                   "camera_left_link"),
                ],

                # ── Connections ───────────────────────────────────────────────
                keys.CONNECT: [
                    # ── Clock ──
                    ("on_playback_tick.outputs:tick",                           "clock_pub.inputs:execIn"),
                    ("ros2_context.outputs:context",                           "clock_pub.inputs:context"),
                    ("sim_time.outputs:simulationTime",                        "clock_pub.inputs:timeStamp"),

                    # ── TF ──
                    ("on_playback_tick.outputs:tick",                           "tf_pub.inputs:execIn"),
                    ("ros2_context.outputs:context",                           "tf_pub.inputs:context"),
                    ("sim_time.outputs:simulationTime",                        "tf_pub.inputs:timeStamp"),

                    # ── Joint State Publishers ──
                    # Left Arm
                    ("on_playback_tick.outputs:tick",                           "ros2_publish_joint_state_left.inputs:execIn"),
                    ("ros2_context.outputs:context",                           "ros2_publish_joint_state_left.inputs:context"),
                    ("sim_time.outputs:simulationTime",                        "ros2_publish_joint_state_left.inputs:timeStamp"),
                    # Right Arm
                    ("on_playback_tick.outputs:tick",                           "ros2_publish_joint_state_right.inputs:execIn"),
                    ("ros2_context.outputs:context",                           "ros2_publish_joint_state_right.inputs:context"),
                    ("sim_time.outputs:simulationTime",                        "ros2_publish_joint_state_right.inputs:timeStamp"),
                    # Left Gripper
                    ("on_playback_tick.outputs:tick",                           "ros2_publish_joint_state_gripper_left.inputs:execIn"),
                    ("ros2_context.outputs:context",                           "ros2_publish_joint_state_gripper_left.inputs:context"),
                    ("sim_time.outputs:simulationTime",                        "ros2_publish_joint_state_gripper_left.inputs:timeStamp"),
                    # Right Gripper
                    ("on_playback_tick.outputs:tick",                           "ros2_publish_joint_state_gripper_right.inputs:execIn"),
                    ("ros2_context.outputs:context",                           "ros2_publish_joint_state_gripper_right.inputs:context"),
                    ("sim_time.outputs:simulationTime",                        "ros2_publish_joint_state_gripper_right.inputs:timeStamp"),

                    # ── Joint Command Subscribers → Articulation Controllers ──
                    # Left Arm
                    ("on_playback_tick.outputs:tick",                           "ros2_subscribe_joint_state_left.inputs:execIn"),
                    ("ros2_context.outputs:context",                           "ros2_subscribe_joint_state_left.inputs:context"),
                    ("ros2_subscribe_joint_state_left.outputs:execOut",         "articulation_controller_left.inputs:execIn"),
                    ("ros2_subscribe_joint_state_left.outputs:positionCommand", "articulation_controller_left.inputs:positionCommand"),
                    ("ros2_subscribe_joint_state_left.outputs:velocityCommand", "articulation_controller_left.inputs:velocityCommand"),
                    ("ros2_subscribe_joint_state_left.outputs:effortCommand",   "articulation_controller_left.inputs:effortCommand"),
                    # Right Arm
                    ("on_playback_tick.outputs:tick",                           "ros2_subscribe_joint_state_right.inputs:execIn"),
                    ("ros2_context.outputs:context",                           "ros2_subscribe_joint_state_right.inputs:context"),
                    ("ros2_subscribe_joint_state_right.outputs:execOut",        "articulation_controller_right.inputs:execIn"),
                    ("ros2_subscribe_joint_state_right.outputs:positionCommand","articulation_controller_right.inputs:positionCommand"),
                    ("ros2_subscribe_joint_state_right.outputs:velocityCommand","articulation_controller_right.inputs:velocityCommand"),
                    ("ros2_subscribe_joint_state_right.outputs:effortCommand",  "articulation_controller_right.inputs:effortCommand"),
                    # Left Gripper
                    ("on_playback_tick.outputs:tick",                           "ros2_subscribe_joint_state_gripper_left.inputs:execIn"),
                    ("ros2_context.outputs:context",                           "ros2_subscribe_joint_state_gripper_left.inputs:context"),
                    ("ros2_subscribe_joint_state_gripper_left.outputs:execOut",           "articulation_controller_gripper_left.inputs:execIn"),
                    ("ros2_subscribe_joint_state_gripper_left.outputs:positionCommand",   "articulation_controller_gripper_left.inputs:positionCommand"),
                    ("ros2_subscribe_joint_state_gripper_left.outputs:velocityCommand",   "articulation_controller_gripper_left.inputs:velocityCommand"),
                    ("ros2_subscribe_joint_state_gripper_left.outputs:effortCommand",     "articulation_controller_gripper_left.inputs:effortCommand"),
                    # Right Gripper
                    ("on_playback_tick.outputs:tick",                           "ros2_subscribe_joint_state_gripper_right.inputs:execIn"),
                    ("ros2_context.outputs:context",                           "ros2_subscribe_joint_state_gripper_right.inputs:context"),
                    ("ros2_subscribe_joint_state_gripper_right.outputs:execOut",          "articulation_controller_gripper_right.inputs:execIn"),
                    ("ros2_subscribe_joint_state_gripper_right.outputs:positionCommand",  "articulation_controller_gripper_right.inputs:positionCommand"),
                    ("ros2_subscribe_joint_state_gripper_right.outputs:velocityCommand",  "articulation_controller_gripper_right.inputs:velocityCommand"),
                    ("ros2_subscribe_joint_state_gripper_right.outputs:effortCommand",    "articulation_controller_gripper_right.inputs:effortCommand"),

                    # ── Camera Front ──
                    ("on_playback_tick.outputs:tick",                           "isaac_create_render_product_front.inputs:execIn"),
                    ("isaac_create_render_product_front.outputs:execOut",       "ros2_camera_helper_front.inputs:execIn"),
                    ("ros2_context.outputs:context",                           "ros2_camera_helper_front.inputs:context"),
                    ("isaac_create_render_product_front.outputs:renderProductPath", "ros2_camera_helper_front.inputs:renderProductPath"),
                    ("isaac_create_render_product_front.outputs:execOut",       "ros2_camera_info_front.inputs:execIn"),
                    ("ros2_context.outputs:context",                           "ros2_camera_info_front.inputs:context"),
                    ("isaac_create_render_product_front.outputs:renderProductPath", "ros2_camera_info_front.inputs:renderProductPath"),

                    # ── Camera Left ──
                    ("on_playback_tick.outputs:tick",                           "isaac_create_render_product_left.inputs:execIn"),
                    ("isaac_create_render_product_left.outputs:execOut",        "ros2_camera_helper_left.inputs:execIn"),
                    ("ros2_context.outputs:context",                           "ros2_camera_helper_left.inputs:context"),
                    ("isaac_create_render_product_left.outputs:renderProductPath", "ros2_camera_helper_left.inputs:renderProductPath"),
                    ("isaac_create_render_product_left.outputs:execOut",        "ros2_camera_info_left.inputs:execIn"),
                    ("ros2_context.outputs:context",                           "ros2_camera_info_left.inputs:context"),
                    ("isaac_create_render_product_left.outputs:renderProductPath", "ros2_camera_info_left.inputs:renderProductPath"),
                ],
            }
        )
        carb.log_info("[OpenArm] ✅ ROS2 Action Graph setup complete!")
        carb.log_info(f"[OpenArm]   Graph: {GRAPH_PATH}")
        carb.log_info(f"[OpenArm]   Robot: {ROBOT_PRIM}")
        carb.log_info("[OpenArm]   ROS2 extension: isaacsim.ros2.bridge (Isaac Sim 4.5+)")
        carb.log_info("[OpenArm]   ROS_DOMAIN_ID: 42")

    except Exception as e:
        carb.log_error(f"[OpenArm] ❌ Error: {e}")
        import traceback
        carb.log_error(traceback.format_exc())


if __name__ == "__main__":
    setup_ros2_action_graph()
elif __name__ == "__builtin__":
    setup_ros2_action_graph()
