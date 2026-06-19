#!/usr/bin/env python3
"""
patch_qvic_usd.py
=================
One-shot script to permanently patch qvic.usd:
  1. Removes the duplicate static robot/graph prims (openarm, ActionGraph, PushGraph)
  2. Strips RigidBodyAPI from any remaining nested physics prims

Run with Isaac Sim's Python runtime:
  /home/hans/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh patch_qvic_usd.py

IMPORTANT: Run this from the Reinforce_Learning/ directory, OUTSIDE of conda env.
"""

import sys
import os

# Must import AppLauncher first for any omni/pxr usage
try:
    from isaaclab.app import AppLauncher
    import argparse
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--headless"])
    launcher = AppLauncher(args)
    simulation_app = launcher.app
except Exception as e:
    print(f"Failed to launch AppLauncher: {e}")
    sys.exit(1)

from pxr import Usd, UsdPhysics, Sdf

USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "isaaclab_openarm_env", "qvic.usd")

print(f"\n=== Patching {USD_PATH} ===\n")

stage = Usd.Stage.Open(USD_PATH)
if not stage:
    print("ERROR: Could not open USD stage. Exiting.")
    simulation_app.close()
    sys.exit(1)

# ── 1. Show current structure ──────────────────────────────────────────────
print("Current top-level prims:")
for p in stage.GetPseudoRoot().GetChildren():
    print(f"  {p.GetPath()}")
    for c in p.GetChildren():
        print(f"    {c.GetPath()}")
        for cc in c.GetChildren():
            print(f"      {cc.GetPath()}")

print("\nPrims with RigidBodyAPI:")
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"  {prim.GetPath()}")

# ── 2. Deactivate duplicate static prims ──────────────────────────────────
DUPLICATE_PRIMS = ["openarm", "ActionGraph", "PushGraph"]
deactivated = []
for prim_name in DUPLICATE_PRIMS:
    # In the raw USD these live directly under /Scene/, /World/, or at root
    for candidate_path in [f"/World/{prim_name}", f"/Scene/{prim_name}", f"/{prim_name}"]:
        candidate_sdf = Sdf.Path(candidate_path)
        prim = stage.GetPrimAtPath(candidate_sdf)
        if prim.IsValid() and prim.IsActive():
            prim.SetActive(False)
            deactivated.append(candidate_path)
            print(f"[PATCH] Deactivated prim: {candidate_path}")

# ── 3. Strip RigidBodyAPI from any Realsense/RSD455 nested prims ──────────
stripped = []
for prim in stage.Traverse():
    path_str = str(prim.GetPath())
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        # Strip from any camera/sensor prims that are children of arm links
        if any(name in path_str for name in ["Realsense", "RSD455", "Camera", "camera"]):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            stripped.append(path_str)
            print(f"[PATCH] Stripped RigidBodyAPI from: {path_str}")

# ── 4. Save ────────────────────────────────────────────────────────────────
if deactivated or stripped:
    stage.GetRootLayer().Save()
    print(f"\n✅ Saved patched qvic.usd")
    print(f"   Deactivated prims: {deactivated}")
    print(f"   Stripped APIs    : {stripped}")
else:
    print("\n⚠️  No changes made. The duplicate prims may already be absent or deactivated.")

print("\nFinal RigidBodyAPI prims after patch:")
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"  {prim.GetPath()}")

simulation_app.close()
