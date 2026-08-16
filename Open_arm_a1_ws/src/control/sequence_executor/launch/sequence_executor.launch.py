#!/usr/bin/env python3
"""
sequence_executor.launch.py
─────────────────────────────────────────────────────────────────────────────
Launches the full motion-sequence stack:
  1. robot_state_publisher
  2. ros2_control (fake hardware)
  3. Controller spawners (arms + grippers)
  4. MoveIt2 move_group (with OMPL + Pilz pipelines)
  5. sequence_executor_node  <- reads sequence.yaml directly, runs one
     sequences: entry (home step once, then body x repeat) - replaces
     bt_executor/BehaviorTree.CPP; event-driven, no tick loop.
  6. (optional) RViz via use_rviz:=true
  7. (ee_type:=amazing_hand only) hand_j1/j2 controller spawners - the
     closed-loop linkage solve itself runs in-process inside
     robot_hardware_interface/AmazingHandHW (see ahand.ros2_control.xacro).

Typical usage (through a project's own launch file, e.g. qvic_2026):
  ros2 launch sequence_executor sequence_executor.launch.py \
      sequence_yaml_path:=/path/to/sequence.yaml sequence_name:=my_sequence use_rviz:=true
"""

import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from openarm_moveit_config.srdf_utils import load_srdf_for_ee_type


def launch_setup(context, *args, **kwargs):
    moveit_cfg = get_package_share_directory("openarm_moveit_config")
    motion_planner_cfg = get_package_share_directory("motion_planner")

    # ── Launch arguments ─────────────────────────────────────────────────────
    use_rviz    = LaunchConfiguration("use_rviz")
    use_api     = LaunchConfiguration("use_api")
    sequence_yaml_path = LaunchConfiguration("sequence_yaml_path")
    sequence_name      = LaunchConfiguration("sequence_name")
    isaacsim    = LaunchConfiguration("isaacsim")
    use_sim_time = LaunchConfiguration("use_sim_time")
    ee_type      = LaunchConfiguration("ee_type")
    db_path      = LaunchConfiguration("db_path")
    executor_package    = LaunchConfiguration("executor_package")
    executor_executable = LaunchConfiguration("executor_executable")

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
        " ee_type:=", ee_type,
    ])
    robot_description = {
        "robot_description": ParameterValue(robot_description_content,
                                             value_type=str)
    }

    # ── SRDF ─────────────────────────────────────────────────────────────────
    # openarm_bimanual.srdf contains groups for both end-effector types; only
    # one set is valid for the URDF actually being built here. See
    # openarm_moveit_config/srdf_utils.py for why and what it strips.
    srdf_path = os.path.join(moveit_cfg, "srdf", "openarm_bimanual.srdf")
    robot_description_semantic = {
        "robot_description_semantic": load_srdf_for_ee_type(srdf_path, ee_type.perform(context))
    }

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
                  # body v2 articulated neck + head - unconditional because
                  # robot_description below never overrides xacro's own
                  # body_type default ("v2"), unlike bringup.launch.py's
                  # is_body_v2-gated spawner (real hardware also supports v1).
                  "head_controller"]
    ]

    is_openarm_hand_not_isaacsim = IfCondition(PythonExpression(
        ["'", ee_type, "' == 'openarm_hand' and '", isaacsim, "' == 'false'"]))
    gripper_spawners = [
        Node(package="controller_manager", executable="spawner",
             arguments=[c, "-c", "/controller_manager"],
             parameters=[{"use_sim_time": use_sim_time}],
             condition=is_openarm_hand_not_isaacsim)
        for c in ["left_gripper_controller", "right_gripper_controller"]
    ]

    is_amazing_hand_not_isaacsim = IfCondition(PythonExpression(
        ["'", ee_type, "' == 'amazing_hand' and '", isaacsim, "' == 'false'"]))
    hand_spawners = [
        Node(package="controller_manager", executable="spawner",
             arguments=[c, "-c", "/controller_manager"],
             parameters=[{"use_sim_time": use_sim_time}],
             condition=is_amazing_hand_not_isaacsim)
        for c in ["left_hand_j1_controller", "left_hand_j2_controller",
                  "right_hand_j1_controller", "right_hand_j2_controller",
                  "left_hand_rotate_controller", "right_hand_rotate_controller"]
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

    # ── Sequence executor ────────────────────────────────────────────────────
    sequence_executor = Node(
        package=executor_package,
        executable=executor_executable,
        name="sequence_executor_node",
        output="screen",
        parameters=[{
            "sequence_yaml_path": sequence_yaml_path,
            "sequence_name": sequence_name,
            "db_path": db_path,
            "use_sim_time": use_sim_time,
        }],
    )

    # ── Optional REST + WebSocket API and web dashboard ──────────────────────
    robot_api = Node(
        package="moveit_api",
        executable="robot_api_server",
        name="robot_api_server",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
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

    return [
        rsp,
        ros2_ctrl,
        *spawners,
        *gripper_spawners,
        *hand_spawners,
        move_group,
        robot_skills,
        sequence_executor,
        robot_api,
        rviz,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_rviz",    default_value="false"),
        DeclareLaunchArgument("use_api",     default_value="false"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "ee_type", default_value="openarm_hand",
            description="End-effector type: openarm_hand or amazing_hand (fake either way here)."
        ),
        DeclareLaunchArgument(
            "isaacsim",
            default_value="false",
            description="launch"
        ),
        DeclareLaunchArgument(
            "sequence_yaml_path", default_value="",
            description="sequence.yaml to read. Only used when the executor has no store "
                        "of its own (i.e. plain sequence_executor_node)."
        ),
        DeclareLaunchArgument(
            "sequence_name", default_value="",
            description="Sequence to run at startup. Empty leaves the FSM in IDLE, "
                        "waiting for a RunSequence goal."
        ),
        DeclareLaunchArgument(
            "db_path", default_value="",
            description="Sequence store to read. Empty uses QVIC_DB_PATH, then the "
                        "project default. Ignored by sequence_executor_node."
        ),
        DeclareLaunchArgument(
            "executor_package", default_value="sequence_executor",
            description="Package holding the executor to run."
        ),
        DeclareLaunchArgument(
            "executor_executable", default_value="sequence_executor_node",
            description="Executor executable. qvic_2026 passes qvic_fsm_node to get its "
                        "sequence store and hardcoded actions."
        ),

        OpaqueFunction(function=launch_setup),
    ])
