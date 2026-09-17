"""Launches gesture_teleop_node - the real-arm side (UDP landmarks in,
rate-limited joint commands out). Runs under the system ROS install; needs
no mediapipe/cv2 (see teleop_node.py's module docstring) and works with
the robot's own camera dead, since the landmark source is the UDP link
from mock_server.py (or any other conforming sender), not an image topic.

Does NOT launch mock_server.py - that runs in its own dependency-isolated
venv (see deployment/systemd/openarm-gesture-mock.service) and is a
plain Python process, not a ROS node; see gesture_mock.launch.py for the
ExecuteProcess-based convenience launch for local/dev use.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mirror_mode_arg = DeclareLaunchArgument(
        "mirror_mode", default_value="mirror",
        description="'mirror' (person's right arm -> robot's left arm) or 'same_side'",
    )

    node = Node(
        package="gesture_teleop",
        executable="teleop_node",
        name="gesture_teleop_node",
        output="screen",
        arguments=["--mirror-mode", LaunchConfiguration("mirror_mode")],
    )

    return LaunchDescription([mirror_mode_arg, node])
