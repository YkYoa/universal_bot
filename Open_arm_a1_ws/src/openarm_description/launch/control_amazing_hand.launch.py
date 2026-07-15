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

    rviz_config_file = PathJoinSubstitution([
        FindPackageShare("openarm_description"), "rviz", "amazing_hand.rviz",
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

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(declared_arguments + [
        robot_state_publisher_node,
        control_node,
        hand_kinematics_node,
        *spawners,
        rviz_node,
    ])
