#!/usr/bin/env python3
"""
Launch MoveIt2 with ros2_control for the OpenArm bimanual robot.

This is the MoveIt/control-only counterpart to robot_api.launch.py. It starts
robot_state_publisher, ros2_control, controller spawners, and move_group without
the REST API or RViz nodes.
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
    joint_limits_yaml_path = os.path.join(moveit_config_pkg, "config", "joint_limits.yaml")
    moveit_controllers_yaml_path = os.path.join(moveit_config_pkg, "config", "moveit_controllers.yaml")

    with open(kinematics_yaml_path, "r") as f:
        kinematics_config = yaml.safe_load(f) or {}
    kinematics_params = kinematics_config["/**"]["ros__parameters"] if "/**" in kinematics_config else kinematics_config
    robot_description_kinematics = {"robot_description_kinematics": kinematics_params}

    # -- ros2_control controllers --
    controller_config = PathJoinSubstitution(
        [FindPackageShare("arm_control"), "config", "bimanual_controllers.yaml"]
    )

    trajectory_execution = {
        "moveit_manage_controllers": True,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }

    planning_scene_monitor = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    spawners = []
    for controller_name in [
        "joint_state_broadcaster",
        "left_arm_controller",
        "right_arm_controller",
        "left_gripper_controller",
        "right_gripper_controller",
    ]:
        spawners.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[controller_name, "-c", "/controller_manager"],
            )
        )

    return LaunchDescription(
        [
            # 1. Robot State Publisher
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="both",
                parameters=[robot_description],
            ),

            # 2. ros2_control node
            Node(
                package="controller_manager",
                executable="ros2_control_node",
                parameters=[controller_config],
                remappings=[("~/robot_description", "/robot_description")],
                output="both",
            ),

            # 3. Spawners
            *spawners,

            # 4. Move Group
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                output="screen",
                parameters=[
                    robot_description,
                    robot_description_semantic,
                    robot_description_kinematics,
                    ompl_planning_yaml_path,
                    pilz_planning_yaml_path,
                    joint_limits_yaml_path,
                    moveit_controllers_yaml_path,
                    trajectory_execution,
                    planning_scene_monitor,
                ],
            ),
        ]
    )
