"""Vá vĩnh viễn ràng buộc mimic giữa 2 ngón kẹp trong v10.usd.

v10.usd chỉ drive `finger_joint1`; `finger_joint2` bám theo bằng
PhysxMimicJoint với naturalFrequency=25, dampingRatio=0.005 — một lò xo RẤT
MỀM và gần như KHÔNG giảm chấn, trong khi khớp chủ động có stiffness=1500.
Đo thực tế (griplag.py) cho thấy hai ngón lệch nhau tới 52mm trong pha đóng
— lớn hơn cả đường kính chai (43.3mm): một ngón đóng hẳn còn ngón kia kẹt ở
giới hạn mở. Không hề có lực kẹp thật, chỉ có một ngón hất chai sang bên.
Đây là lý do mean_max_lift_m luôn ~0.8mm và success_rate chưa bao giờ khác 0
suốt cả tuần, bất kể mọi thay đổi về reward — nhiệm vụ bất khả thi vật lý.

Chạy MỘT LẦN, giống patch_robot_usd.py, rồi commit v10.usd đã vá.
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

from pxr import Usd  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V10_USD_PATH = os.path.abspath(os.path.join(
    _THIS_DIR, "openarm_description", "urdf", "robot", "v10", "v10.usd"
))

NEW_FREQ = 200.0
NEW_DAMPING = 1.0

print(f"\n=== Vá gripper mimic joint: {V10_USD_PATH} ===\n")

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
    freq_attr = prim.GetAttribute("physxMimicJoint:rotY:naturalFrequency")
    damp_attr = prim.GetAttribute("physxMimicJoint:rotY:dampingRatio")
    if not freq_attr or not damp_attr:
        print(f"  ⚠️ {prim.GetPath()}: khong tim thay attribute mimic")
        continue
    old_freq = freq_attr.Get()
    old_damp = damp_attr.Get()
    freq_attr.Set(NEW_FREQ)
    damp_attr.Set(NEW_DAMPING)
    print(f"  {prim.GetPath()}")
    print(f"    naturalFrequency: {old_freq} -> {NEW_FREQ}")
    print(f"    dampingRatio    : {old_damp} -> {NEW_DAMPING}")
    changed = True

if changed:
    stage.GetRootLayer().Save()
    print("\n✅ Da va va luu v10.usd!")
else:
    print("\nℹ️ Khong tim thay prim finger_joint2 nao de va.")

simulation_app.close()
