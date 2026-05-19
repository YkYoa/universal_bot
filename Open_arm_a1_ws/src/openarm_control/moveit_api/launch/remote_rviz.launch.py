#!/usr/bin/env python3
"""
Launch RViz for remote monitoring of the OpenArm bimanual MoveIt setup.

This mirrors the RViz configuration from robot_api.launch.py without starting
MoveIt, controllers, robot_state_publisher, or the REST API.
"""

import os
import yaml

from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    moveit_config_pkg = get_package_share_directory("openarm_moveit_config")

    # -- Robot Description (URDF) --
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("openarm_description"), "urdf", "robot", "v10.urdf.xacro"]),
            " ",
            "bimanual:=true",
            " ",
            "ros2_control:=true",
            " ",
            "use_fake_hardware:=true",
            " ",
            "mobile_base:=true",
            " ",
            "mobile_base_xyz:='0 0 0.31'",
            " ",
            "mobile_base_body_xyz:='0 0 0'",
        ]
    )
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    # -- Semantic Robot Description (SRDF) --
    srdf_path = os.path.join(moveit_config_pkg, "srdf", "openarm_bimanual.srdf")
    with open(srdf_path, "r") as f:
        robot_description_semantic_content = f.read()
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_content}

    # -- MoveIt Config Files --
    kinematics_yaml_path = os.path.join(moveit_config_pkg, "config", "kinematics.yaml")
    ompl_planning_yaml_path = os.path.join(moveit_config_pkg, "config", "ompl_planning.yaml")
    pilz_planning_yaml_path = os.path.join(moveit_config_pkg, "config", "pilz_industrial_motion_planner_planning.yaml")

    with open(kinematics_yaml_path, "r") as f:
        kinematics_config = yaml.safe_load(f) or {}
    kinematics_params = kinematics_config["/**"]["ros__parameters"] if "/**" in kinematics_config else kinematics_config
    robot_description_kinematics = {"robot_description_kinematics": kinematics_params}

    rviz_config_file = os.path.join(moveit_config_pkg, "config", "moveit.rviz")

    return LaunchDescription(
        [
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2_monitor",
                output="log",
                arguments=[
                    "-d",
                    rviz_config_file,
                    "--ros-args",
                    "--log-level",
                    "class_loader:=ERROR",
                    "--log-level",
                    "rcl:=ERROR",
                ],
                parameters=[
                    robot_description,
                    robot_description_semantic,
                    robot_description_kinematics,
                    ompl_planning_yaml_path,
                    pilz_planning_yaml_path,
                ],
            ),
        ]
    )
