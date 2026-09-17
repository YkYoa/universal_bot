"""Convenience launch for LOCAL/DEV use only: runs mock_server.py under its
own dependency-isolated venv via ExecuteProcess (a plain Python process,
not a ROS node - it imports no rclpy, see mock_server.py's module
docstring).

Production deploys this the same way but via
deployment/systemd/openarm-gesture-mock.service directly, not through
`ros2 launch` - a systemd unit restarts on crash and starts at boot
without needing the ROS graph up first, which this file has no way to
express. Use this launch file only for a developer who already has ROS
sourced and wants a one-line way to also start the mockup server pointed
at the same venv the deployed systemd unit uses.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

DEFAULT_VENV_PYTHON = "/opt/openarm-gesture/venv/bin/python"


def generate_launch_description():
    port_arg = DeclareLaunchArgument("port", default_value="5055")
    venv_python_arg = DeclareLaunchArgument(
        "venv_python",
        default_value=os.environ.get("GESTURE_VENV_PYTHON", DEFAULT_VENV_PYTHON),
        description="Path to the mediapipe-venv python interpreter (see plan's dependency-isolation section)",
    )

    mock_server = ExecuteProcess(
        cmd=[
            LaunchConfiguration("venv_python"), "-m", "gesture_teleop.mock_server",
            "--port", LaunchConfiguration("port"),
            # --enable-real intentionally NOT passed here - see mock_server.py's
            # module docstring: the mockup must never bridge to the real
            # robot by accident.
        ],
        output="screen",
    )

    return LaunchDescription([port_arg, venv_python_arg, mock_server])
