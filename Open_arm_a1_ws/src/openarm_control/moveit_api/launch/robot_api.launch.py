#!/usr/bin/env python3
"""
Launch file for the OpenArm Robot REST API.

This launch file starts the MoveIt2 stack AND the REST API server together.
The API server provides HTTP endpoints that the UI team can call to control the robot.

Usage:
    ros2 launch moveit_api robot_api.launch.py

The API will be available at:
    http://<robot-ip>:5050/api/docs
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    moveit_config_pkg = get_package_share_directory("openarm_moveit_config")
    use_rviz = LaunchConfiguration("use_rviz")

    # ── Robot Description (URDF) ──
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

    # ── Semantic Robot Description (SRDF) ──
    srdf_path = os.path.join(moveit_config_pkg, "srdf", "openarm_bimanual.srdf")
    with open(srdf_path, "r") as f:
        robot_description_semantic_content = f.read()
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_content}

    # ── MoveIt Config Files ──
    kinematics_yaml_path = os.path.join(moveit_config_pkg, "config", "kinematics.yaml")
    ompl_planning_yaml_path = os.path.join(moveit_config_pkg, "config", "ompl_planning.yaml")
    joint_limits_yaml_path = os.path.join(moveit_config_pkg, "config", "joint_limits.yaml")
    moveit_controllers_yaml_path = os.path.join(moveit_config_pkg, "config", "moveit_controllers.yaml")

    # ── ros2_control controllers ──
    controller_config = PathJoinSubstitution(
        [FindPackageShare("arm_control"), "config", "bimanual_controllers.yaml"]
    )

    # ── Trajectory execution parameters ──
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

    # ─────────────────────────────────────────────
    # Nodes
    # ─────────────────────────────────────────────

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[controller_config],
        remappings=[
            ("~/robot_description", "/robot_description"),
        ],
        output="both",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    left_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_arm_controller", "-c", "/controller_manager"],
    )

    right_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_arm_controller", "-c", "/controller_manager"],
    )

    left_gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_gripper_controller", "-c", "/controller_manager"],
    )

    right_gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_gripper_controller", "-c", "/controller_manager"],
    )

    # ── MoveIt Move Group Node ──
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml_path,
            ompl_planning_yaml_path,
            joint_limits_yaml_path,
            moveit_controllers_yaml_path,
            trajectory_execution,
            planning_scene_monitor,
        ],
    )

    # ── REST API Server (Flask + ROS 2 node) ──
    robot_api_node = Node(
        package="moveit_api",
        executable="robot_api_server",
        name="robot_api_server",
        output="screen",
    )

    # ── RViz (optional, for monitoring) ──
    rviz_config_file = os.path.join(moveit_config_pkg, "config", "moveit.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_monitor",  # Renamed to avoid collision
        condition=IfCondition(use_rviz),
        output="log",
        arguments=[
            "-d", rviz_config_file,
            "--ros-args", "--log-level", "class_loader:=ERROR",  # Suppress class_loader warnings
            "--log-level", "rcl:=ERROR",
        ],
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml_path,
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_rviz",
                default_value="false",
                description="Start RViz together with the robot API stack.",
            ),
            # Core
            robot_state_publisher_node,
            ros2_control_node,
            # Controllers
            joint_state_broadcaster_spawner,
            left_arm_controller_spawner,
            right_arm_controller_spawner,
            left_gripper_controller_spawner,
            right_gripper_controller_spawner,
            # MoveIt
            move_group_node,
            # API
            robot_api_node,
            # Optional visualization
            rviz_node,
        ]
    )
