#!/usr/bin/env python3
"""Chay TREN Jetson NX (192.168.10.3) - khong can ROS2.

Doc 2 camera that (video0 = camera thuong, video6 = Orbbec Gemini RGB -
da xac nhan hoat dong qua VLA/camera_test/), encode JPEG, POST lien tuc
sang camera_bridge_node dang chay tren IQ-9075 (192.168.10.1:8090) qua
duong Ethernet truc tiep end0.

Chay:
  python3 jetson_camera_streamer.py

Dung:
  video6 (Orbbec, ro net) -> slot "front"
  video0 (camera thuong)  -> slot "left"
"""
import time

import cv2
import requests

BRIDGE_URL = "http://192.168.10.1:8090"
FPS = 5  # 5Hz du cho VLA replanning (1-10Hz theo DEPLOY_GUIDE.md), tiet kiem bang thong
JPEG_QUALITY = 80

CAMERAS = {
    "front": 6,  # Orbbec Gemini RGB
    "left": 0,   # camera thuong
}


def open_camera(idx, warmup=20):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    # can >10 frame de sensor on dinh (da xac nhan thuc te truoc do)
    for _ in range(warmup):
        cap.read()
    return cap


def main():
    caps = {}
    for slot, idx in CAMERAS.items():
        cap = open_camera(idx)
        if cap is None:
            print(f"[warn] khong mo duoc video{idx} cho slot '{slot}'")
        else:
            caps[slot] = cap
            print(f"[ok] video{idx} -> slot '{slot}'")

    if not caps:
        print("[error] khong co camera nao mo duoc, dung.")
        return

    period = 1.0 / FPS
    while True:
        t0 = time.time()
        for slot, cap in caps.items():
            ok, frame = cap.read()
            if not ok:
                continue
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok2:
                continue
            try:
                resp = requests.post(
                    f"{BRIDGE_URL}/frame/{slot}", data=buf.tobytes(),
                    headers={"Content-Type": "application/octet-stream"}, timeout=2,
                )
                if resp.status_code != 200:
                    print(f"[warn] {slot}: HTTP {resp.status_code} {resp.text}")
            except requests.RequestException as e:
                print(f"[warn] {slot}: gui that bai - {e}")

        elapsed = time.time() - t0
        time.sleep(max(0.0, period - elapsed))


if __name__ == "__main__":
    main()
