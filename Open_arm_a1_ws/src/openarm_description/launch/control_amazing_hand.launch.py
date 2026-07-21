import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("side", default_value="right", description="Which hand: right or left."),
        DeclareLaunchArgument("use_rviz", default_value="true", description="Start RViz."),
    ]

    side = LaunchConfiguration("side")
    use_rviz = LaunchConfiguration("use_rviz")

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("openarm_description"),
                                  "urdf", "ee", "amazing_hand", "ahand_with_control.urdf.xacro"]),
            " side:=", side,
        ]
    )
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    controllers_yaml = PathJoinSubstitution([
        FindPackageShare("openarm_description"),
        "config", "hand", "amazing_hand", "ahand_controllers.yaml",
    ])

    moveit_config_dir = os.path.join(
        get_package_share_directory("openarm_description"), "config", "hand", "amazing_hand")

    # SRDF isn't a substitution-friendly format (no per-side content needed,
    # it's the same 8 joint names regardless of side), so just read it like
    # moveit_bimanual.launch.py does.
    srdf_path = os.path.join(moveit_config_dir, "amazing_hand.srdf")
    with open(srdf_path, "r") as f:
        robot_description_semantic_content = f.read()
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_content}

    joint_limits_yaml_path = os.path.join(moveit_config_dir, "joint_limits.yaml")
    ompl_planning_yaml_path = os.path.join(moveit_config_dir, "ompl_planning.yaml")
    moveit_controllers_yaml_path = os.path.join(moveit_config_dir, "moveit_controllers.yaml")

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

    rviz_config_file = PathJoinSubstitution([
        FindPackageShare("openarm_description"), "rviz", "amazing_hand_moveit.rviz",
    ])

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[controllers_yaml],
        remappings=[("~/robot_description", "/robot_description")],
        output="both",
    )

    hand_kinematics_node = Node(
        package="openarm_description",
        executable="hand_kinematics_node.py",
        name="hand_kinematics_node",
        parameters=[robot_description, {"command_space": "knuckle"}],
        output="both",
    )

    spawners = [
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[name, "--controller-manager", "/controller_manager"],
        )
        for name in ["joint_state_broadcaster", "j1_group_controller", "j2_group_controller"]
    ]

    # ── MoveIt: move_group + RViz MotionPlanning panel ──
    # hand_fingers is a joint-only planning group (see amazing_hand.srdf),
    # not a kinematic chain, so no IK solver/kinematics.yaml is needed here.
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            {"default_planning_pipeline": "ompl"},
            ompl_planning_yaml_path,
            joint_limits_yaml_path,
            moveit_controllers_yaml_path,
            trajectory_execution,
            planning_scene_monitor,
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            {"default_planning_pipeline": "ompl"},
            ompl_planning_yaml_path,
        ],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(declared_arguments + [
        robot_state_publisher_node,
        control_node,
        hand_kinematics_node,
        *spawners,
        move_group_node,
        rviz_node,
    ])
