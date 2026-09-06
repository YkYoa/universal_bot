#!/usr/bin/env python3
"""Teleop JOINT-SPACE bang ban phim qua HTTP API cua MoveIt
(robot_api_server.py, da co san tren IQ-9075, co kiem tra va cham + gioi
han khop), dong thoi ghi du lieu demo (camera that + joint state + action)
truc tiep theo dinh dang LeRobotDataset - dung de fine-tune pi0.5 sau nay,
khong can qua buoc convert rieng cua openarm_dataset.

QUAN TRONG - lich su quyet dinh: ban dau viet Cartesian XYZ (WASD di
chuyen end-effector), nhung /api/move/pose can /compute_fk + frame TF
'world' - phat hien 2 bug ha tang tren robot nay khi test that (2026-09-04):
  1. Khong co static_transform_publisher nao publish 'world' -> openarm_base_link
     du SRDF co khai bao virtual joint nay (openarm_bimanual.srdf:412).
  2. Ngay ca sau khi vá tam bang static_transform_publisher, move_group's
     compute_fk capability van TREO VO THOI HAN (uncaught exception khi TF
     lookup that bai -> khong bao gio tra response ve client).
Ca 2 bug nay chua ai gap truoc do vi QVIC sequence/waypoint_recorder chi
dung joint-space control (JointConstraint, khong can TF/world). Vi vay
script nay CHUYEN SANG joint-space (/api/move/joint) - dung duong da
duoc chung minh on dinh trong production, khong dung lai bug tren.
Xem VLA/command.md muc "Teleop WASD" de biet chi tiet dieu tra.

KHONG dung lerobot's bi_openarm_follower (dieu khien CAN bus truc tiep) -
se bo qua kiem tra va cham cua MoveIt dang chay production.

Chay tren laptop (co lerobot da cai o .venv_local):
  .venv_local/bin/python teleop_record_openarm.py \\
      --api-host 192.168.1.226 --camera-host 192.168.1.226 \\
      --task "pick up the object" --dataset-root VLA/datasets/openarm_teleop_v1

Phim dieu khien:
  1-7       : chon khop dang dieu khien cua tay active (joint1..joint7)
  W / S     : tang / giam gia tri khop dang chon (mac dinh 2 do/lan)
  Tab       : chuyen tay active (left <-> right)
  Space     : dong/mo gripper cua tay active
  [ / ]     : giam / tang buoc di chuyen (do)
  N         : bat dau episode moi (bat dau ghi)
  M         : luu episode hien tai
  X         : huy episode hien tai (khong luu)
  Q         : thoat (tu dong luu episode dang ghi neu co)
"""
import argparse
import io
import math
import select
import sys
import termios
import tty

import numpy as np
import requests
from PIL import Image as PILImage

GRIPPER_JOINT_NAME = "openarm_{side}_finger_joint1"
ACTION_STATE_NAMES = (
    [f"right_joint{i}.pos" for i in range(1, 8)] + ["right_gripper.pos"]
    + [f"left_joint{i}.pos" for i in range(1, 8)] + ["left_gripper.pos"]
)


class RawTerminal:
    """Doc tung phim khong can Enter, khong can pynput/X server - chay
    duoc qua SSH thuan text. Restore terminal settings khi thoat."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read_key(self, timeout=0.05):
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            return sys.stdin.read(1)
        return None


class RobotAPI:
    def __init__(self, api_host, api_port=5050):
        self.base = f"http://{api_host}:{api_port}"

    def move_joints(self, group, positions_rad, velocity_scaling=0.2):
        # Dung /api/move/joints (ca nhom, 1 lan) thay vi /api/move/joint (1
        # khop): /api/move/joint gui lai CA 7 khop moi lan goi, voi 6 khop
        # "giu nguyen" dung GIA TRI DO DUOC luc goi (dung sai long ~2.9 do)
        # - qua nhieu lan goi, nhieu cam bien/do tre co khi cong don thanh
        # troi that. /api/move/joints dung dung sai chat (0.057 do) cho CA
        # 7 khop, va gia tri "giu nguyen" o day la gia tri MINH TU THEO DOI
        # (khong doc lai tu robot) nen khong the troi duoc.
        body = {
            "group": group,
            "positions": list(positions_rad),
            "unit": "rad",
            "velocity_scaling": velocity_scaling,
        }
        r = requests.post(f"{self.base}/api/move/joints", json=body, timeout=10)
        return r.json()

    def gripper(self, side, action):
        r = requests.post(f"{self.base}/api/gripper", json={"side": side, "action": action}, timeout=5)
        return r.json()

    def status(self):
        r = requests.get(f"{self.base}/api/status", timeout=5)
        return r.json()


def fetch_camera_frame(camera_host, camera_port, slot):
    r = requests.get(f"http://{camera_host}:{camera_port}/frame/{slot}/latest", timeout=3)
    r.raise_for_status()
    img = PILImage.open(io.BytesIO(r.content)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)  # HWC, uint8, RGB


def build_state_vector(joints: dict) -> np.ndarray:
    vals = []
    for side in ("right", "left"):
        for i in range(1, 8):
            name = f"openarm_{side}_joint{i}"
            vals.append(float(joints.get(name, 0.0)))
        vals.append(float(joints.get(f"openarm_{side}_finger_joint1", 0.0)))
    return np.asarray(vals, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-host", required=True, help="IP cua may chay robot_api_server.py (IQ-9075)")
    ap.add_argument("--api-port", type=int, default=5050)
    ap.add_argument("--camera-host", required=True, help="IP cua may chay camera_bridge_node.py")
    ap.add_argument("--camera-port", type=int, default=8090)
    ap.add_argument("--dataset-root", required=True, help="Thu muc luu dataset (se tao moi neu chua co)")
    ap.add_argument("--repo-id", default="local/openarm_teleop_v1")
    ap.add_argument("--task", required=True, help="Mo ta task (vd 'pick up the red block')")
    ap.add_argument("--step-deg", type=float, default=2.0, help="Buoc di chuyen ban dau (do)")
    args = ap.parse_args()

    from pathlib import Path

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    api = RobotAPI(args.api_host, args.api_port)
    step_deg = args.step_deg
    active_side = "left"
    active_joint = 1  # 1-7

    print("[init] Dang lay joint state hien tai...")
    st = api.status()
    if not st.get("success") or not st.get("joints"):
        print(f"[loi] Khong lay duoc joint state: {st}. Kiem tra robot_api_server co dang chay khong.")
        sys.exit(1)
    # Theo doi vi tri 7 khop/tay NOI BO (khong doc lai tu robot moi lan move)
    # - xem ghi chu trong RobotAPI.move_joints() ve ly do (tranh troi cong don).
    live_joints = st["joints"]
    joint_positions = {
        side: [live_joints.get(f"openarm_{side}_joint{i}", 0.0) for i in range(1, 8)]
        for side in ("left", "right")
    }

    print("[init] Dang lay 1 frame camera de xac dinh do phan giai...")
    try:
        sample_front = fetch_camera_frame(args.camera_host, args.camera_port, "front")
        sample_left = fetch_camera_frame(args.camera_host, args.camera_port, "left")
    except Exception as e:
        print(f"[loi] Khong lay duoc camera frame: {e}. Kiem tra camera_bridge_node co dang chay khong.")
        sys.exit(1)

    features = {
        "action": {"dtype": "float32", "shape": (16,), "names": ACTION_STATE_NAMES},
        "observation.state": {"dtype": "float32", "shape": (16,), "names": ACTION_STATE_NAMES},
        "observation.images.front": {
            "dtype": "video", "shape": sample_front.shape, "names": ["height", "width", "channels"],
        },
        "observation.images.left": {
            "dtype": "video", "shape": sample_left.shape, "names": ["height", "width", "channels"],
        },
    }

    # tasks.parquet chi duoc tao SAU lan save_episode() dau tien - neu chi
    # co info.json (vd script crash truoc khi bam N/M lan nao) thi resume()
    # se that bai (LeRobotDataset co gang fallback sang tra cuu HuggingFace
    # Hub cho repo_id "local/..." khong ton tai that, loi 401). Coi la
    # "chua ton tai" trong truong hop do de tao lai sach.
    dataset_exists = (Path(args.dataset_root) / "meta" / "tasks.parquet").exists()
    if dataset_exists:
        print(f"[init] Dataset da ton tai tai {args.dataset_root}, resume de them episode moi...")
        dataset = LeRobotDataset.resume(repo_id=args.repo_id, root=args.dataset_root)
    else:
        root_path = Path(args.dataset_root)
        if root_path.exists():
            # Thu muc dang do (vd info.json da tao nhung chua save_episode()
            # lan nao - crash/thoat som lan truoc) - create() se tu choi
            # neu thu muc da ton tai, nen don sach truoc (khong mat gi vi
            # chua co episode nao).
            print(f"[init] Xoa thu muc dataset do dang (chua co episode nao) tai {root_path}...")
            import shutil
            shutil.rmtree(root_path)
        print(f"[init] Tao dataset moi tai {args.dataset_root} ...")
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=10,
            features=features,
            root=args.dataset_root,
            robot_type="openarm_bimanual",
            use_videos=True,
        )

    recording = False
    print(__doc__)
    print(f"[san sang] Tay active: {active_side} | khop: {active_joint} | "
          f"buoc: {step_deg:.1f}deg | Q de thoat")

    with RawTerminal() as term:
        while True:
            key = term.read_key(timeout=0.05)
            if key is None:
                continue

            if key in ("q", "Q"):
                if recording and dataset.has_pending_frames():
                    print("[thoat] Dang luu episode dang ghi truoc khi thoat...")
                    dataset.save_episode()
                dataset.finalize()
                print("[thoat] Tam biet.")
                break

            elif key == "\t":
                active_side = "right" if active_side == "left" else "left"
                print(f"[chuyen tay] active: {active_side}, khop: {active_joint}")

            elif key in ("1", "2", "3", "4", "5", "6", "7"):
                active_joint = int(key)
                print(f"[chon khop] {active_side}_joint{active_joint}")

            elif key == "[":
                step_deg = max(0.5, step_deg - 0.5)
                print(f"[buoc] {step_deg:.1f}deg")
            elif key == "]":
                step_deg = min(15.0, step_deg + 0.5)
                print(f"[buoc] {step_deg:.1f}deg")

            elif key == "n":
                if recording:
                    print("[canh bao] Da dang ghi episode, bam M de luu hoac X de huy truoc.")
                else:
                    recording = True
                    print("[ghi] Bat dau episode moi.")

            elif key == "m":
                if not recording or not dataset.has_pending_frames():
                    print("[canh bao] Khong co gi de luu.")
                else:
                    dataset.save_episode()
                    recording = False
                    print("[ghi] Da luu episode.")

            elif key == "x":
                if not recording:
                    print("[canh bao] Khong dang ghi episode nao.")
                else:
                    dataset.clear_episode_buffer()
                    recording = False
                    print("[ghi] Da huy episode (khong luu).")

            elif key == " ":
                js = api.status()
                joints = js.get("joints", {})
                cur = joints.get(GRIPPER_JOINT_NAME.format(side=active_side), 0.0)
                new_action = "close" if cur > 0.022 else "open"
                obs_state, obs_images = None, None
                if recording:
                    obs_state = build_state_vector(joints)
                    obs_images = {
                        "front": fetch_camera_frame(args.camera_host, args.camera_port, "front"),
                        "left": fetch_camera_frame(args.camera_host, args.camera_port, "left"),
                    }
                res = api.gripper(active_side, new_action)
                if not res.get("success"):
                    print(f"[loi] gripper: {res.get('message')}")
                    continue
                print(f"[gripper] {active_side} -> {new_action}")
                if recording:
                    js2 = api.status()
                    action_vec = build_state_vector(js2.get("joints", {}))
                    dataset.add_frame({
                        "observation.state": obs_state,
                        "observation.images.front": obs_images["front"],
                        "observation.images.left": obs_images["left"],
                        "action": action_vec,
                        "task": args.task,
                    })

            elif key in ("w", "s"):
                joint_name = f"openarm_{active_side}_joint{active_joint}"
                delta_rad = math.radians(step_deg) * (1 if key == "w" else -1)
                new_positions = list(joint_positions[active_side])
                new_positions[active_joint - 1] += delta_rad

                obs_state, obs_images = None, None
                if recording:
                    js = api.status()
                    obs_state = build_state_vector(js.get("joints", {}))
                    obs_images = {
                        "front": fetch_camera_frame(args.camera_host, args.camera_port, "front"),
                        "left": fetch_camera_frame(args.camera_host, args.camera_port, "left"),
                    }

                res = api.move_joints(f"{active_side}_arm", new_positions)
                if not res.get("success"):
                    print(f"[loi di chuyen] {res.get('message')}")
                    continue
                joint_positions[active_side] = new_positions
                new_val = new_positions[active_joint - 1]

                print(f"[di chuyen] {joint_name} -> {math.degrees(new_val):.1f}deg")

                if recording:
                    js2 = api.status()
                    action_vec = build_state_vector(js2.get("joints", {}))
                    dataset.add_frame({
                        "observation.state": obs_state,
                        "observation.images.front": obs_images["front"],
                        "observation.images.left": obs_images["left"],
                        "action": action_vec,
                        "task": args.task,
                    })


if __name__ == "__main__":
    main()
