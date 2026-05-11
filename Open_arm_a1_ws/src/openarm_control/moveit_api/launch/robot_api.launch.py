#!/usr/bin/env python3
"""
Launch file for the OpenArm Robot REST API.

This launch file starts the MoveIt2 stack AND the REST API server together.
It is highly modular, allowing you to toggle specific components (MoveIt, API, RViz)
for multi-machine setups (e.g., Robot Hardware vs. PC Monitoring).

Arguments:
    use_rviz (false):    Start RViz for visualization.
    use_moveit (true):   Start the MoveGroup planning node.
    use_api (true):      Start the Flask REST API server.
    use_controllers (true): Start ros2_control and spawner nodes.
    use_rsp (true):      Start robot_state_publisher.
"""

import os
import yaml

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
    
    # ── Launch Configurations ──
    use_rviz = LaunchConfiguration("use_rviz")
    use_moveit = LaunchConfiguration("use_moveit")
    use_api = LaunchConfiguration("use_api")
    use_controllers = LaunchConfiguration("use_controllers")
    use_rsp = LaunchConfiguration("use_rsp")

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
    pilz_planning_yaml_path = os.path.join(moveit_config_pkg, "config", "pilz_industrial_motion_planner_planning.yaml")
    joint_limits_yaml_path = os.path.join(moveit_config_pkg, "config", "joint_limits.yaml")
    moveit_controllers_yaml_path = os.path.join(moveit_config_pkg, "config", "moveit_controllers.yaml")

    # Load kinematics.yaml
    with open(kinematics_yaml_path, "r") as f:
        kinematics_config = yaml.safe_load(f) or {}
    kinematics_params = kinematics_config["/**"]["ros__parameters"] if "/**" in kinematics_config else kinematics_config
    robot_description_kinematics = {"robot_description_kinematics": kinematics_params}

    # ── ros2_control controllers ──
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

    # ─────────────────────────────────────────────
    # Nodes
    # ─────────────────────────────────────────────

    # 1. Robot State Publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
        condition=IfCondition(use_rsp),
    )

    # 2. ros2_control node
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[controller_config],
        remappings=[("~/robot_description", "/robot_description")],
        output="both",
        condition=IfCondition(use_controllers),
    )

    # 3. Spawners
    spawners = []
    for controller_name in ["joint_state_broadcaster", "left_arm_controller", "right_arm_controller", 
                            "left_gripper_controller", "right_gripper_controller"]:
        spawners.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[controller_name, "-c", "/controller_manager"],
                condition=IfCondition(use_controllers),
            )
        )

    # 4. MoveIt Move Group Node
    move_group_node = Node(
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
        condition=IfCondition(use_moveit),
    )

    # 5. REST API Server
    robot_api_node = Node(
        package="moveit_api",
        executable="robot_api_server",
        name="robot_api_server",
        output="screen",
        condition=IfCondition(use_api),
    )

    # 6. RViz
    rviz_config_file = os.path.join(moveit_config_pkg, "config", "moveit.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_monitor",
        condition=IfCondition(use_rviz),
        output="log",
        arguments=[
            "-d", rviz_config_file,
            "--ros-args", "--log-level", "class_loader:=ERROR", "--log-level", "rcl:=ERROR",
        ],
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            ompl_planning_yaml_path,
            pilz_planning_yaml_path,
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="false"),
            DeclareLaunchArgument("use_moveit", default_value="true"),
            DeclareLaunchArgument("use_api", default_value="true"),
            DeclareLaunchArgument("use_controllers", default_value="true"),
            DeclareLaunchArgument("use_rsp", default_value="true"),
            
            robot_state_publisher_node,
            ros2_control_node,
            *spawners,
            move_group_node,
            robot_api_node,
            rviz_node,
        ]
    )
