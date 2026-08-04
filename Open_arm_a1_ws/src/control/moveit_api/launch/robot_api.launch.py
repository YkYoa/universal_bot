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
    ee_type (amazing_hand): 'openarm_hand' (2-finger gripper) or 'amazing_hand' (5-finger).
    body_type (v2):      'v1' (rigid base) or 'v2' (articulated neck + head).
"""

import os
import re
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError


def launch_setup(context, *args, **kwargs):
    moveit_config_pkg = get_package_share_directory("openarm_moveit_config")

    # ── Launch Configurations ──
    use_rviz = LaunchConfiguration("use_rviz")
    use_moveit = LaunchConfiguration("use_moveit")
    use_api = LaunchConfiguration("use_api")
    use_controllers = LaunchConfiguration("use_controllers")
    use_rsp = LaunchConfiguration("use_rsp")
    ee_type = LaunchConfiguration("ee_type")
    body_type = LaunchConfiguration("body_type")
    ee_type_str = ee_type.perform(context)
    body_type_str = body_type.perform(context)
    run_openarm_hand = IfCondition(PythonExpression(["'", ee_type, "' == 'openarm_hand'"]))
    run_amazing_hand = IfCondition(PythonExpression(["'", ee_type, "' == 'amazing_hand'"]))
    run_body_v2 = IfCondition(PythonExpression(["'", body_type, "' == 'v2'"]))

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
            "ee_type:=", ee_type,
            " ",
            "body_type:=", body_type,
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
    # Patched the same way as moveit_bimanual.launch.py: the SRDF hardcodes
    # left_gripper/right_gripper end_effector groups (openarm_hand's 2-finger
    # gripper); under ee_type:=amazing_hand those groups don't exist in the
    # URDF at all, which segfaults RobotState::getRobotMarkers(). Retarget to
    # the finger groups that actually exist, and inject the v2 "head" group
    # (neck_joint/head_joint aren't in any static SRDF group since they don't
    # exist for body_type:=v1).
    srdf_path = os.path.join(moveit_config_pkg, "srdf", "openarm_bimanual.srdf")
    with open(srdf_path, "r") as f:
        robot_description_semantic_content = f.read()

    if ee_type_str == "amazing_hand":
        robot_description_semantic_content = (
            robot_description_semantic_content
            .replace('group="left_gripper" parent_group="left_arm"',
                      'group="left_hand_fingers" parent_group="left_arm"')
            .replace('group="right_gripper" parent_group="right_arm"',
                      'group="right_hand_fingers" parent_group="right_arm"')
        )
        robot_description_semantic_content = re.sub(
            r'  <group name="(?:left|right)_gripper">.*?</group>\n\n',
            '', robot_description_semantic_content, flags=re.DOTALL)
        robot_description_semantic_content = re.sub(
            r'  <group_state name="(?:open|close)" group="(?:left|right)_gripper">.*?</group_state>\n\n',
            '', robot_description_semantic_content, flags=re.DOTALL)
        robot_description_semantic_content = '\n'.join(
            line for line in robot_description_semantic_content.split('\n')
            if not re.search(r'openarm_(?:left|right)_(?:hand"|left_finger|right_finger)', line)
        )

    if body_type_str == "v2":
        robot_description_semantic_content = robot_description_semantic_content.replace(
            "  <!-- Virtual joints -->",
            '  <group name="head">\n'
            '    <joint name="openarm_body_neck_joint"/>\n'
            '    <joint name="openarm_body_head_joint"/>\n'
            '  </group>\n\n'
            "  <!-- Virtual joints -->"
        )
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
        [FindPackageShare("robot_control"), "config", "bimanual_controllers.yaml"]
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
    spawners = [
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[controller_name, "-c", "/controller_manager"],
            condition=IfCondition(use_controllers),
        )
        for controller_name in ["joint_state_broadcaster", "left_arm_controller", "right_arm_controller"]
    ]
    spawners += [
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[controller_name, "-c", "/controller_manager"],
            condition=IfCondition(PythonExpression([
                "'", use_controllers, "' == 'true' and '", ee_type, "' == 'openarm_hand'"
            ])),
        )
        for controller_name in ["left_gripper_controller", "right_gripper_controller"]
    ]
    spawners += [
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[controller_name, "-c", "/controller_manager"],
            condition=IfCondition(PythonExpression([
                "'", use_controllers, "' == 'true' and '", ee_type, "' == 'amazing_hand'"
            ])),
        )
        for controller_name in ["left_hand_j1_controller", "left_hand_j2_controller",
                                 "right_hand_j1_controller", "right_hand_j2_controller"]
    ]
    spawners.append(
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["head_controller", "-c", "/controller_manager"],
            condition=IfCondition(PythonExpression([
                "'", use_controllers, "' == 'true' and '", body_type, "' == 'v2'"
            ])),
        )
    )

    # 3b. amazing_hand kinematics solver (one per side; no-op unless
    # ee_type:=amazing_hand - see hand_kinematics_node.py for what it does)
    hand_kinematics_nodes = [
        Node(
            package="openarm_description",
            executable="hand_kinematics_node.py",
            name=f"{side}_hand_kinematics_node",
            parameters=[
                robot_description,
                {
                    "command_space": "knuckle",
                    "link_prefix": f"openarm_{side}_ahand_",
                    "alias_prefix": f"openarm_{side}_",
                    "joint_commands_topic": f"/{side}_ahand/joint_commands",
                    "joint_states_topic": f"/{side}_ahand/joint_states",
                },
            ],
            output="both",
            condition=run_amazing_hand,
        )
        for side in ("left", "right")
    ]

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

    # 7. Foxglove Bridge Node (Only if package is installed)
    has_foxglove = False
    try:
        get_package_share_directory("foxglove_bridge")
        has_foxglove = True
    except PackageNotFoundError:
        print("[WARNING] 'foxglove_bridge' package is not installed on this system. Skipping Foxglove node.")

    launch_entities = [
        robot_state_publisher_node,
        ros2_control_node,
        *spawners,
        *hand_kinematics_nodes,
        move_group_node,
        robot_api_node,
        rviz_node,
    ]

    if has_foxglove:
        foxglove_bridge_node = Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            parameters=[{"port": 8765}],
            condition=IfCondition(LaunchConfiguration("use_foxglove")),
            output="screen",
        )
        launch_entities.append(foxglove_bridge_node)

    return launch_entities


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_rviz", default_value="false"),
        DeclareLaunchArgument("use_moveit", default_value="true"),
        DeclareLaunchArgument("use_api", default_value="true"),
        DeclareLaunchArgument("use_controllers", default_value="true"),
        DeclareLaunchArgument("use_rsp", default_value="true"),
        DeclareLaunchArgument("use_foxglove", default_value="true"),
        DeclareLaunchArgument("ee_type", default_value="amazing_hand",
                               description="'openarm_hand' (2-finger gripper) or 'amazing_hand' (5-finger)."),
        DeclareLaunchArgument("body_type", default_value="v2",
                               description="'v1' (rigid base) or 'v2' (articulated neck + head)."),
        OpaqueFunction(function=launch_setup),
    ])
