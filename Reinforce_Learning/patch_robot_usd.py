import os
import sys

# Must import AppLauncher first for omni/pxr usage
from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
launcher = AppLauncher(args)
simulation_app = launcher.app

from pxr import Usd, UsdPhysics, Sdf

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V10_USD_PATH = os.path.abspath(os.path.join(
    _THIS_DIR, "openarm_description", "urdf", "robot", "v10", "v10.usd"
))

print(f"\n=== Inspecting and Patching Robot USD: {V10_USD_PATH} ===\n")

stage = Usd.Stage.Open(V10_USD_PATH)
if not stage:
    print("ERROR: Could not open robot USD stage. Exiting.")
    simulation_app.close()
    sys.exit(1)

# Traverse and find any RigidBodyAPI on non-root links
changed = False
for prim in stage.Traverse():
    path_str = str(prim.GetPath())
    # The root link of the robot articulation is usually at /robot/base_link or similar.
    # We want to keep RigidBodyAPI on the root and revolute link joints, but remove it from
    # any nested camera, Realsense, RSD455, or sensor prims that are children of arm links.
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        name = prim.GetName()
        # Check if it's a camera or sensor prim
        if any(keyword in name.lower() for keyword in ["realsense", "rsd455", "camera", "sensor"]):
            print(f"Found RigidBodyAPI on nested sensor prim: {path_str}")
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            print(f"  -> Successfully STRIPPED RigidBodyAPI from: {path_str}")
            changed = True

if changed:
    stage.GetRootLayer().Save()
    print("\n✅ Successfully patched and saved robot v10.usd!")
else:
    print("\nℹ️ No nested RigidBodyAPIs on sensor links found to strip in v10.usd.")

simulation_app.close()
