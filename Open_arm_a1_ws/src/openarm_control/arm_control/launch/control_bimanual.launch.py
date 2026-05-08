import os

# Automatically fix CycloneDDS buffer issues for large URDFs
os.environ["CYCLONEDDS_URI"] = "<CycloneDDS><Domain><General><MaxMessageSize>65535B</MaxMessageSize><FragmentSize>4000B</FragmentSize></General></Domain></CycloneDDS>"

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Load URDF with ros2_control and fake hardware
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

    controller_config = PathJoinSubstitution(
        [FindPackageShare("arm_control"), "config", "bimanual_controllers.yaml"]
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    controller_manager = Node(
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

    # foxglove_bridge_node = Node(
    #     package="foxglove_bridge",
    #     executable="foxglove_bridge",
    #     name="foxglove_bridge",
    #     output="screen",
    #     parameters=[{
    #         "port": 8765,
    #         "address": "0.0.0.0",
    #     }]
    # )
    
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("openarm_description"), "rviz", "bimanual.rviz"]
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
    )

    return LaunchDescription(
        [
            robot_state_publisher_node,
            controller_manager,
            joint_state_broadcaster_spawner,
            left_arm_controller_spawner,
            right_arm_controller_spawner,
            left_gripper_controller_spawner,
            right_gripper_controller_spawner,
            # foxglove_bridge_node,
            rviz_node,
        ]
    )
