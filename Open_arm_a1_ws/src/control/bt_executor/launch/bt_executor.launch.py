#!/usr/bin/env python3
"""
bt_executor.launch.py
─────────────────────────────────────────────────────────────────────────────
Launches the full VLA-BT stack:
  1. robot_state_publisher
  2. ros2_control (fake hardware)
  3. Controller spawners (arms + grippers)
  4. MoveIt2 move_group (with OMPL + Pilz pipelines)
  5. bt_executor_node  ← the BT engine
  6. (optional) RViz  via use_rviz:=true
  7. (optional) REST API via use_api:=true  (Android team telemetry + control)

Typical usage:
  ros2 launch bt_executor bt_executor.launch.py use_rviz:=true
"""

import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    moveit_cfg = get_package_share_directory("openarm_moveit_config")
    bt_cfg     = get_package_share_directory("bt_executor")
    motion_planner_cfg = get_package_share_directory("motion_planner")

    # ── Launch arguments ─────────────────────────────────────────────────────
    use_rviz    = LaunchConfiguration("use_rviz")
    use_api     = LaunchConfiguration("use_api")
    bt_xml_path = LaunchConfiguration("bt_xml_path")
    tick_rate   = LaunchConfiguration("tick_rate_hz")
    isaacsim    = LaunchConfiguration("isaacsim")
    instruction = LaunchConfiguration("instruction")
    use_sim_time = LaunchConfiguration("use_sim_time")

    # ── Robot description ────────────────────────────────────────────────────
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]), " ",
        PathJoinSubstitution([FindPackageShare("openarm_description"),
                              "urdf", "robot", "v10.urdf.xacro"]),
        " bimanual:=true ros2_control:=true use_fake_hardware:=true",
        " head_use_fake_hardware:=true",
        " mobile_base:=true",
        " mobile_base_xyz:='0 0 0.31'",
        " mobile_base_body_xyz:='0 0 0'",
    ])
    robot_description = {
        "robot_description": ParameterValue(robot_description_content,
                                             value_type=str)
    }

    # ── SRDF ─────────────────────────────────────────────────────────────────
    srdf_path = os.path.join(moveit_cfg, "srdf", "openarm_bimanual.srdf")
    with open(srdf_path) as f:
        robot_description_semantic = {"robot_description_semantic": f.read()}

    # ── MoveIt config ────────────────────────────────────────────────────────
    kinematics_yaml = os.path.join(moveit_cfg, "config", "kinematics.yaml")
    ompl_yaml       = os.path.join(moveit_cfg, "config", "ompl_planning.yaml")
    pilz_yaml       = os.path.join(moveit_cfg, "config",
                                    "pilz_industrial_motion_planner_planning.yaml")
    joint_limits    = os.path.join(moveit_cfg, "config", "joint_limits.yaml")
    moveit_ctrl     = os.path.join(moveit_cfg, "config", "moveit_controllers.yaml")

    with open(kinematics_yaml) as f:
        kin_cfg = yaml.safe_load(f) or {}
    kin_params = kin_cfg.get("/**", {}).get("ros__parameters", kin_cfg)
    # kin_params is: {"robot_description_kinematics": {"left_arm": {...}, ...}}
    # If it already has the top-level key, use it directly; otherwise wrap it.
    if "robot_description_kinematics" in kin_params:
        robot_description_kinematics = kin_params
    else:
        robot_description_kinematics = {"robot_description_kinematics": kin_params}

    controller_config = PathJoinSubstitution(
        [FindPackageShare("robot_control"), "config", "bimanual_controllers.yaml"])

    trajectory_execution = {
        "moveit_manage_controllers": True,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.0,
    }
    planning_scene_monitor = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    bounds_tolerances = {
        "start_state_max_bounds_error": 2.0,
    }

    # ── Nodes ────────────────────────────────────────────────────────────────

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
        output="both",
        condition=UnlessCondition(isaacsim),
    )

    ros2_ctrl = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[controller_config, {"use_sim_time": use_sim_time}],
        remappings=[("~/robot_description", "/robot_description")],
        output="both",
        condition=UnlessCondition(isaacsim),
    )

    spawners = [
        Node(package="controller_manager", executable="spawner",
             arguments=[c, "-c", "/controller_manager"],
             parameters=[{"use_sim_time": use_sim_time}],
             condition=UnlessCondition(isaacsim))
        for c in ["joint_state_broadcaster",
                  "left_arm_controller",  "right_arm_controller",
                  "left_gripper_controller", "right_gripper_controller"]
    ]

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            {"default_planning_pipeline": "ompl"},
            ompl_yaml, pilz_yaml, joint_limits, moveit_ctrl,
            trajectory_execution, planning_scene_monitor,
            bounds_tolerances,
            {"use_sim_time": use_sim_time},
        ],
        condition=UnlessCondition(isaacsim),
    )

    # ── Robot Skills Server (MoveItCpp action server backend) ─────────────────
    moveit_cpp_yaml = os.path.join(motion_planner_cfg, "config", "moveit_cpp.yaml")
    robot_skills = Node(
        package="robot_skills",
        executable="robot_skills_node",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            moveit_cpp_yaml,
            moveit_ctrl,
            joint_limits,
            trajectory_execution,
            planning_scene_monitor,
            bounds_tolerances,
            {"use_sim_time": use_sim_time},
        ],
        condition=UnlessCondition(isaacsim),
    )

    # ── BT executor ──────────────────────────────────────────────────────────
    bt_executor = Node(
        package="bt_executor",
        executable="bt_executor_node",
        name="bt_executor",
        output="screen",
        parameters=[{
            "bt_xml_path": bt_xml_path,
            "tick_rate_hz": tick_rate,
            "log_to_file": False,
            "vla_task_name": instruction,
            "use_sim_time": use_sim_time,
        }],
    )

    # ── Optional Web UI Telemetry & Controls ──────────────────────────────────
    bt_viewer = Node(
        package="bt_viewer",
        executable="bt_viewer_node",
        name="bt_viewer",
        output="screen",
        parameters=[{
            "bt_xml_path": bt_xml_path,
            "port": 5000,
            "use_sim_time": use_sim_time,
        }],
        condition=IfCondition(use_api),
    )

    # ── Optional RViz ─────────────────────────────────────────────────────────
    rviz_cfg = os.path.join(moveit_cfg, "config", "moveit.rviz")
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_cfg],
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            {"default_planning_pipeline": "ompl"},
            ompl_yaml,
            pilz_yaml,
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_rviz",    default_value="false"),
        DeclareLaunchArgument("use_api",     default_value="false"),
        DeclareLaunchArgument("tick_rate_hz", default_value="50.0"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "isaacsim",
            default_value="false",
            description="launch"
        ),
        DeclareLaunchArgument(
            "instruction",
            default_value="",
            description="VLA task instruction (if empty, falls back to the VLA bridge server's instruction parameter)"
        ),
        DeclareLaunchArgument(
            "bt_xml_path",
            default_value=os.path.join(bt_cfg, "bt_trees", "pick_and_place.xml"),
            description="Path to the BT XML file to load"
        ),

        rsp,
        ros2_ctrl,
        *spawners,
        move_group,
        robot_skills,
        bt_executor,
        bt_viewer,
        rviz,
    ])
