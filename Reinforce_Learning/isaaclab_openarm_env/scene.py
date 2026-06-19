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



def _strip_duplicate_scene_robot_physics(stage, UsdPhysics, Sdf):
    """
    Helper: deactivates ALL duplicate robot/graph prims that live under
    /World/envs/env_N/Scene/openarm, ActionGraph, PushGraph without traversing the full stage.
    Deactivating them ensures they (and all their descendants/joints) are ignored by PhysX
    and Isaac Lab, avoiding joint validation warnings and articulation errors.
    """
    duplicate_prim_names = ["openarm", "ActionGraph", "PushGraph"]
    deactivated_paths = []
    
    envs_prim = stage.GetPrimAtPath("/World/envs")
    if not envs_prim.IsValid():
        return

    for env_child in envs_prim.GetChildren():
        env_name = env_child.GetName()
        if env_name.startswith("env_"):
            for prim_name in duplicate_prim_names:
                path_str = f"/World/envs/{env_name}/Scene/{prim_name}"
                prim = stage.GetPrimAtPath(path_str)
                if prim.IsValid() and prim.IsActive():
                    prim.SetActive(False)
                    deactivated_paths.append(path_str)

    if deactivated_paths:
        n_envs = len({p.rsplit("/", 2)[0] for p in deactivated_paths})
        print(f"[ApplePickPlaceEnv] Deactivated duplicate static robot/graph prims "
              f"({len(deactivated_paths)} prims) from {n_envs} env Scene backdrops.")


def spawn_qvic_with_physics(prim_path: str, cfg: sim_utils.UsdFileCfg, translation=None, orientation=None):
    """
    Custom spawn function for scene_env (qvic.usd).
    Loads the USD file and immediately configures the USD rigid body and collision APIs
    on the Bowl and Bottle before other scene entities initialize.
    This guarantees clean loading with replicate_physics=False.
    """
    # 1. Load the scene backdrop USD using standard loader
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation)

    # 2. Apply custom physics and API modifications
    import omni.usd
    from pxr import UsdPhysics, UsdGeom, PhysxSchema, Sdf

    stage = omni.usd.get_context().get_stage()
    if not stage:
        return prim

    # ── Strip duplicate robot physics (immediate pass) ────────────────────
    _strip_duplicate_scene_robot_physics(stage, UsdPhysics, Sdf)

    # ── Register a deferred pre-physics callback as a belt-and-suspenders ─
    # PhysX validates the stage hierarchy when the physics scene is initialized.
    # We register a one-shot subscription that fires just before that happens
    # to guarantee the duplicate /Scene/openarm physics is gone.
    try:
        import omni.physx
        _sub_holder = []  # list used to keep the subscription alive until it fires

        def _on_physics_ready(step_size, _sub_ref=_sub_holder, _stage=stage,
                              _UsdPhysics=UsdPhysics, _Sdf=Sdf):
            _strip_duplicate_scene_robot_physics(_stage, _UsdPhysics, _Sdf)
            # Unsubscribe after first call
            if _sub_ref:
                _sub_ref.clear()

        sub = omni.physx.get_physx_interface().subscribe_physics_step_events(_on_physics_ready)
        _sub_holder.append(sub)
    except Exception as e:
        # Non-fatal: the immediate pass above may be sufficient
        print(f"[ApplePickPlaceEnv] Could not register deferred physics callback: {e}")

    # Gather all spawned environments
    envs_prim = stage.GetPrimAtPath("/World/envs")
    if envs_prim.IsValid():
        env_children = [c.GetName() for c in envs_prim.GetChildren() if c.GetName().startswith("env_")]
    else:
        env_children = ["env_0"]

    # ── Strip physics from the Table prim ─────────────────────────────────
    for env_name in env_children:
        table_path = f"/World/envs/{env_name}/Scene/Table"
        table_prim = stage.GetPrimAtPath(table_path)
        if table_prim.IsValid():
            if table_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                table_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                print(f"[ApplePickPlaceEnv] Stripped Table rigid body physics: {table_path}")

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

    # Apply Kinematic physics and convexDecomposition to Bowl on ALL environments
    bowl_count = 0
    for env_name in env_children:
        bowl_path = f"/World/envs/{env_name}/Scene/Bowl"
        bowl_prim = stage.GetPrimAtPath(bowl_path)
        if bowl_prim.IsValid():
            if not bowl_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rb = UsdPhysics.RigidBodyAPI.Apply(bowl_prim)
                rb.CreateKinematicEnabledAttr(True)
                print(f"[ApplePickPlaceEnv] Applied Kinematic RigidBodyAPI to Bowl: {bowl_path}")
            enable_collisions(bowl_prim, "convexDecomposition")
            bowl_count += 1

    if bowl_count:
        print(f"[ApplePickPlaceEnv] Configured Bowl collision & physics across {bowl_count} environments.")

    # Apply Dynamic physics and convexHull to Bottle on ALL environments
    bottle_count = 0
    for env_name in env_children:
        bottle_path = f"/World/envs/{env_name}/Scene/Bottle"
        bottle_prim = stage.GetPrimAtPath(bottle_path)
        if bottle_prim.IsValid():
            if not bottle_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rb = UsdPhysics.RigidBodyAPI.Apply(bottle_prim)
                rb.CreateKinematicEnabledAttr(False)
                print(f"[ApplePickPlaceEnv] Applied Dynamic RigidBodyAPI to Bottle: {bottle_path}")
            enable_collisions(bottle_prim, "convexHull")
            bottle_count += 1

    if bottle_count:
        print(f"[ApplePickPlaceEnv] Configured Bottle collision & physics across {bottle_count} environments.")

    return prim
