#!/usr/bin/env python3
"""Node TEST (khong phai production) - subscribe /camera_front/image_raw
(da publish that tu Jetson qua camera_bridge_node), POST sang server pi0.5
tren huyhoang-4090 (port 9091), CHI LOG action ra console - KHONG actuate
tay that. Muc tieu: kiem tra vong lap camera-that -> model -> action voi
du lieu camera OpenArm that (thay vi anh mau).

Chay tren IQ-9075:
  source /opt/ros/jazzy/setup.bash
  source ~/arm_ws/install/setup.bash
  export ROS_DOMAIN_ID=42
  python3 -m camera_bridge.pi05_camera_test_node
  # hoac: ros2 run camera_bridge pi05_camera_test (sau khi rebuild)

Chi goi model moi 1/N frame (RATE_LIMIT_SEC) vi inference CPU tren
huyhoang-4090 mat ~6s/lan - goi lien tuc theo 2.7Hz cua camera se lam
nghen hang doi.
"""
import base64
import time

import cv2
import requests
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

PREDICT_URL = "http://192.168.1.122:9091/predict"
RATE_LIMIT_SEC = 8.0  # inference CPU ~6s/lan, cho du dem an toan
TASK_INSTRUCTION = "pick up the object"


class Pi05CameraTestNode(Node):
    """Node TEST: subscribe camera that, goi server pi0.5 tren huyhoang-4090 va
    chi log ket qua action - khong actuate tay. Xem module docstring de biet
    ly do gioi han toc do goi (RATE_LIMIT_SEC)."""

    def __init__(self):
        """Subscribe /camera_front/image_raw va khoi tao rate limiter cho on_frame()."""
        super().__init__("pi05_camera_test_node")
        self.bridge = CvBridge()
        self.last_call = 0.0
        self.sub = self.create_subscription(
            Image, "/camera_front/image_raw", self.on_frame, 5
        )
        self.get_logger().info(
            f"pi05_camera_test_node san sang - se goi {PREDICT_URL} toi da 1 lan/{RATE_LIMIT_SEC}s"
        )

    def on_frame(self, msg: Image):
        """Callback moi frame /camera_front/image_raw: bo qua neu chua du
        RATE_LIMIT_SEC ke tu lan goi truoc, nguoc lai encode JPEG va POST
        sang pi0.5 server, chi log action tra ve."""
        now = time.time()
        if now - self.last_call < RATE_LIMIT_SEC:
            return
        self.last_call = now

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            self.get_logger().warn("encode JPEG that bai, bo qua frame")
            return

        b64 = base64.b64encode(buf.tobytes()).decode()
        payload = {"front_b64": b64, "task": TASK_INSTRUCTION}

        self.get_logger().info("Goi pi0.5 server voi frame that tu camera_front...")
        t0 = time.time()
        try:
            resp = requests.post(PREDICT_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - t0
            action = data.get("action", [])
            self.get_logger().info(
                f"[KET QUA - CHI LOG, KHONG ACTUATE] "
                f"inference_time={data.get('inference_time_sec')}s "
                f"(HTTP round-trip={elapsed:.2f}s) "
                f"action[0:6]={[round(a, 4) for a in action[:6]]}..."
            )
        except requests.RequestException as e:
            self.get_logger().error(f"Goi pi0.5 server that bai: {e}")


def main():
    """Diem vao: spin Pi05CameraTestNode toi khi Ctrl+C."""
    rclpy.init()
    node = Pi05CameraTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
