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

from __future__ import annotations

import os
import torch
import isaaclab.sim as sim_utils

# Absolute path to the qvic.usd backdrop file (lives next to this module)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_QVIC_USD_PATH = os.path.join(_THIS_DIR, "qvic.usd")
_PATCH_SENTINEL = _QVIC_USD_PATH + ".patched"


def patch_qvic_usd_once():
    """
    Permanently patch qvic.usd to remove the duplicate robot/graph prims
    (openarm, ActionGraph, PushGraph) and strip RigidBodyAPI from any nested
    Realsense/RSD455 prims.

    This function uses the pxr USD runtime (available after AppLauncher has
    started Isaac Sim) and writes a sentinel file so patching only happens once.
    Idempotent: safe to call on every startup.
    """
    if os.path.exists(_PATCH_SENTINEL):
        return  # Already patched

    print(f"\n[ApplePickPlaceEnv] Patching {_QVIC_USD_PATH} (one-time operation)...")

    try:
        from pxr import Usd, UsdPhysics, Sdf

        stage = Usd.Stage.Open(_QVIC_USD_PATH)
        if not stage:
            print("[ApplePickPlaceEnv] WARNING: Could not open qvic.usd for patching. "
                  "Skipping permanent patch — runtime fix will handle it.")
            return

        changed = False

        # ── 1. Deactivate duplicate static robot/graph prims ─────────────
        # In the raw USD these may be under /World/, /Scene/, or at top level.
        DUPLICATE_NAMES = {"openarm", "ActionGraph", "PushGraph"}
        prims_to_deactivate = []
        for prim in stage.TraverseAll():
            path_str = str(prim.GetPath())
            parts = path_str.strip("/").split("/")
            # Match /Scene/openarm, /World/openarm, etc.
            if len(parts) == 2 and parts[0] in ("Scene", "World") and parts[1] in DUPLICATE_NAMES:
                prims_to_deactivate.append(Sdf.Path(path_str))
            # Also top-level /openarm etc. if present
            elif len(parts) == 1 and parts[0] in DUPLICATE_NAMES:
                prims_to_deactivate.append(Sdf.Path(path_str))

        for path in prims_to_deactivate:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid() and prim.IsActive():
                prim.SetActive(False)
                print(f"[ApplePickPlaceEnv]   Deactivated prim: {path}")
                changed = True

        # ── 2. Strip RigidBodyAPI from Realsense/RSD455 camera prims ─────
        for prim in stage.Traverse():
            path_str = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                name = prim.GetName()
                if name in ("RSD455", "Realsense") or "camera" in name.lower():
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    print(f"[ApplePickPlaceEnv]   Stripped RigidBodyAPI from: {path_str}")
                    changed = True

        if changed:
            stage.GetRootLayer().Save()
            print(f"[ApplePickPlaceEnv] ✅ qvic.usd patched and saved.")
        else:
            print(f"[ApplePickPlaceEnv] ℹ️  No changes needed in qvic.usd "
                  f"(duplicate prims not found at top-level paths).")

        # Write sentinel regardless so we don't try again
        with open(_PATCH_SENTINEL, "w") as f:
            f.write("patched\n")

    except Exception as e:
        print(f"[ApplePickPlaceEnv] WARNING: USD patch failed: {e}. "
              f"Runtime scene.py traversal will still handle the issue.")



def spawn_qvic_with_physics(prim_path: str, cfg: sim_utils.UsdFileCfg, translation=None, orientation=None):
    """
    Custom spawn function for scene_env (qvic.usd).
    Loads the USD file and immediately configures the USD rigid body and collision APIs
    on the Bowl and Bottle relative to the active concrete prim_path.
    Optimized to avoid traversing all environments in the USD stage, preventing OOM.
    """
    # 1. Load the scene backdrop USD using standard loader
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation)

    # 2. Apply custom physics and API modifications
    import omni.usd
    from pxr import UsdPhysics, UsdGeom, PhysxSchema, Sdf

    stage = omni.usd.get_context().get_stage()
    if not stage:
        return prim

    # Resolve regex patterns (e.g. env_.*) to the concrete source template environment (env_0)
    # so that we can query and modify real, well-formed SdfPaths on the stage.
    concrete_prim_path = prim_path.replace("env_.*", "env_0")

    # ── Strip duplicate robot physics under this concrete_prim_path ───────────
    duplicate_prim_names = ["openarm", "ActionGraph", "PushGraph"]
    for prim_name in duplicate_prim_names:
        dup_path = f"{concrete_prim_path}/{prim_name}"
        dup_prim = stage.GetPrimAtPath(dup_path)
        if dup_prim.IsValid() and dup_prim.IsActive():
            dup_prim.SetActive(False)
            print(f"[ApplePickPlaceEnv] Deactivated duplicate static robot/graph prim: {dup_path}")

    # ── Strip physics from the Table prim under this concrete_prim_path ───────
    table_path = f"{concrete_prim_path}/Table"
    table_prim = stage.GetPrimAtPath(table_path)
    if table_prim.IsValid():
        if table_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            table_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            print(f"[ApplePickPlaceEnv] Stripped Table rigid body physics: {table_path}")
        # Disable collision on the table to prevent static-intersection explosions with the robot base
        table_prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        print(f"[ApplePickPlaceEnv] Disabled Table collision: {table_path}")

    def enable_collisions(p, approximation_type="convexHull"):
        if not p.IsValid():
            return
        if p.IsA(UsdGeom.Mesh):
            if not p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(p)
            PhysxSchema.PhysxCollisionAPI.Apply(p)
            p.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set(approximation_type)
        for c in p.GetChildren():
            enable_collisions(c, approximation_type)

    # Apply Kinematic physics and convexDecomposition to Bowl under this concrete_prim_path
    bowl_path = f"{concrete_prim_path}/Bowl"
    bowl_prim = stage.GetPrimAtPath(bowl_path)
    if bowl_prim.IsValid():
        if not bowl_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI.Apply(bowl_prim)
            rb.CreateKinematicEnabledAttr(True)
            print(f"[ApplePickPlaceEnv] Applied Kinematic RigidBodyAPI to Bowl: {bowl_path}")
        enable_collisions(bowl_prim, "convexDecomposition")
        print(f"[ApplePickPlaceEnv] Configured Bowl collision & physics: {bowl_path}")

    # Apply Dynamic physics and convexHull to Bottle under this concrete_prim_path
    bottle_path = f"{concrete_prim_path}/Bottle"
    bottle_prim = stage.GetPrimAtPath(bottle_path)
    if bottle_prim.IsValid():
        if not bottle_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI.Apply(bottle_prim)
            rb.CreateKinematicEnabledAttr(False)
            print(f"[ApplePickPlaceEnv] Applied Dynamic RigidBodyAPI to Bottle: {bottle_path}")
        enable_collisions(bottle_prim, "convexHull")
        print(f"[ApplePickPlaceEnv] Configured Bottle collision & physics: {bottle_path}")

    return prim
