#!/usr/bin/env python3
"""Nhan frame JPEG qua HTTP tu Jetson NX va republish thanh sensor_msgs/Image.

Boi canh: 2 cong Type-C camera cua IQ-9075 da chay, nen camera thuc te
(Orbbec Gemini + camera thuong) cam vao 1 Jetson NX rieng, noi qua Ethernet
truc tiep (end0, 192.168.10.0/24). Jetson khong can cai ROS2 - chi can 1
script Python nho (xem jetson_streamer.py) POST JPEG lien tuc sang day.

Endpoint:
  POST /frame/front  (body = raw JPEG bytes)  -> publish /camera_front/image_raw
  POST /frame/left   (body = raw JPEG bytes)  -> publish /camera_left/image_raw
  GET  /frame/<slot>/latest  -> raw JPEG bytes cua frame moi nhat da nhan
    (dung boi teleop_record_openarm.py de lay anh khi ghi 1 frame dataset,
    khong can node do la ROS2 node - chi can HTTP client don gian)

Topic name khop dung convention da dung trong Isaac Sim action graph
(control/robot_control/scripts/isaac_sim_setup_action_graph.py), de
vla_bridge sau nay doc duoc ca 2 nguon (sim/that) ma khong doi code.
"""
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from flask import Flask, request
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

app = Flask(__name__)
node: "CameraBridgeNode" = None


class CameraBridgeNode(Node):
    """Node ROS2 giu publisher Image/CameraInfo cho 2 slot camera (front/left) va
    cache frame JPEG moi nhat de phuc vu HTTP; khong tu nhan du lieu, Flask route
    ben duoi goi publish_frame() moi khi co POST tu Jetson."""

    def __init__(self):
        """Tao cac publisher /camera_<slot>/image_raw + camera_info cho front/left."""
        super().__init__("camera_bridge_node")
        self.bridge = CvBridge()
        self.pub_front = self.create_publisher(Image, "/camera_front/image_raw", 5)
        self.pub_left = self.create_publisher(Image, "/camera_left/image_raw", 5)
        self.info_front = self.create_publisher(CameraInfo, "/camera_front/camera_info", 5)
        self.info_left = self.create_publisher(CameraInfo, "/camera_left/camera_info", 5)
        self.last_frame_time = {"front": 0.0, "left": 0.0}
        self.last_jpeg_bytes = {"front": None, "left": None}
        self.get_logger().info("camera_bridge_node san sang, cho frame tu Jetson qua HTTP")

    def publish_frame(self, slot: str, jpeg_bytes: bytes) -> bool:
        """Giai ma 1 frame JPEG va publish Image + CameraInfo cho `slot`
        ("front"/"left"); tra ve False neu OpenCV khong decode duoc."""
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        self.last_jpeg_bytes[slot] = jpeg_bytes

        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = f"{slot}_camera"

        info_msg = CameraInfo()
        info_msg.header = img_msg.header
        info_msg.height, info_msg.width = frame.shape[0], frame.shape[1]

        if slot == "front":
            self.pub_front.publish(img_msg)
            self.info_front.publish(info_msg)
        else:
            self.pub_left.publish(img_msg)
            self.info_left.publish(info_msg)

        self.last_frame_time[slot] = time.time()
        return True


@app.route("/frame/<slot>", methods=["POST"])
def receive_frame(slot):
    """HTTP handler POST /frame/<slot>: nhan raw JPEG bytes tu Jetson va publish
    len ROS2 qua CameraBridgeNode.publish_frame()."""
    if slot not in ("front", "left"):
        return {"error": f"unknown slot '{slot}', dung 'front' hoac 'left'"}, 400
    ok = node.publish_frame(slot, request.get_data())
    if not ok:
        return {"error": "khong decode duoc JPEG"}, 400
    return {"status": "ok"}, 200


@app.route("/frame/<slot>/latest", methods=["GET"])
def get_latest_frame(slot):
    """HTTP handler GET /frame/<slot>/latest: tra ve raw JPEG bytes cua frame
    moi nhat da nhan cho slot do (dung boi teleop_record_openarm.py)."""
    if slot not in ("front", "left"):
        return {"error": f"unknown slot '{slot}', dung 'front' hoac 'left'"}, 400
    jpeg_bytes = node.last_jpeg_bytes.get(slot)
    if jpeg_bytes is None:
        return {"error": f"chua co frame nao cho slot '{slot}'"}, 404
    return app.response_class(jpeg_bytes, mimetype="image/jpeg")


@app.route("/health", methods=["GET"])
def health():
    """HTTP handler GET /health: tra ve tuoi (giay) cua frame moi nhat tung slot,
    dung de kiem tra Jetson streamer con song khong."""
    now = time.time()
    return {
        "status": "ok",
        "last_frame_age_sec": {k: round(now - v, 2) if v else None for k, v in node.last_frame_time.items()},
    }, 200


def main():
    """Diem vao: khoi tao node ROS2 tren 1 thread rieng, chay Flask server o
    thread chinh (port 8090) cho toi khi bi ngat."""
    global node
    rclpy.init()
    node = CameraBridgeNode()

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    try:
        app.run(host="0.0.0.0", port=8090, debug=False)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
