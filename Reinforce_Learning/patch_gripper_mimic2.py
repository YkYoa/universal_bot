"""Vá lần 2: thêm drive lực (maxForce) cho finger_joint2.

Phát hiện qua bảng "Simulation Joint Information" lúc load env: finger_joint2
có Effort Limit = 0.000 (finger_joint1 = 333.000) — khớp này KHÔNG có khối
<drive> riêng trong USD (chỉ có ràng buộc PhysxMimicJoint), nên PhysX coi lực
tối đa khớp này được phép tạo ra là 0, bất kể naturalFrequency/dampingRatio
của mimic được chỉnh thế nào.

Đo trực tiếp (loadtest.py) xác nhận hậu quả: đóng theo ramp thật (không phải
bước nhảy tức thời) khiến 2 ngón lệch tới 17mm và KHÔNG BAO GIỜ hội tụ lại,
kể cả giữ nguyên target 120 bước sau đó — ngón thứ hai gần như không tạo được
lực gì, "trôi theo" chứ không thực sự bám.

Thêm PhysicsDriveAPI cho finger_joint2 với stiffness=damping=0 (không cạnh
tranh lực với chính ràng buộc mimic) nhưng maxForce=333 (bằng finger_joint1)
— chỉ để nâng trần lực cho phép, không thêm nguồn lực riêng.
"""
import os
import sys

from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--visualizer", "none"])
launcher = AppLauncher(args)
simulation_app = launcher.app

from pxr import Usd, UsdPhysics  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V10_USD_PATH = os.path.abspath(os.path.join(
    _THIS_DIR, "openarm_description", "assets", "robot", "openarm_v1.0", "urdf", "v10", "v10.usd"
))

MAX_FORCE = 333.0

print(f"\n=== Vá lần 2 — thêm drive lực cho finger_joint2: {V10_USD_PATH} ===\n")

stage = Usd.Stage.Open(V10_USD_PATH)
if not stage:
    print("ERROR: Khong the mo v10.usd. Thoat.")
    simulation_app.close()
    sys.exit(1)

changed = False
for prim in stage.Traverse():
    name = prim.GetName()
    if "finger_joint2" not in name:
        continue
    existing = prim.GetAttribute("drive:linear:physics:maxForce")
    before = existing.Get() if existing else None
    drive_api = UsdPhysics.DriveAPI.Apply(prim, "linear")
    drive_api.CreateTypeAttr("force")
    drive_api.CreateMaxForceAttr(MAX_FORCE)
    drive_api.CreateStiffnessAttr(0.0)
    drive_api.CreateDampingAttr(0.0)
    after = prim.GetAttribute("drive:linear:physics:maxForce").Get()
    print(f"  {prim.GetPath()}")
    print(f"    drive:linear:physics:maxForce: {before} -> {after}")
    changed = True

if changed:
    stage.GetRootLayer().Save()
    print("\n✅ Da va va luu v10.usd!")
else:
    print("\nℹ️ Khong tim thay prim finger_joint2 nao de va.")

simulation_app.close()
