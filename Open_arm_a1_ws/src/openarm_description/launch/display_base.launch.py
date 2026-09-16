"""No-hardware RViz visualization of the mobile base alone (no arm/body/hand).

Entry point for `ros2 launch openarm_description display_base.launch.py`.
Xacro source: assets/robot/openarm_v1.0/urdf/base.urdf.xacro.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    """Build base.urdf.xacro (no args) and start joint_state_publisher +
    robot_state_publisher + RViz to visualize the mobile base alone."""
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("openarm_description"), "assets", "robot", "openarm_v1.0", "urdf", "base.urdf.xacro"]),
        ]
    )
    
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    rviz_config_file = PathJoinSubstitution([
        FindPackageShare("openarm_description"), 
        "rviz", 
        "base_only.rviz"
    ])

    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        parameters=[robot_description],
        remappings=[("/robot_description", "/robot_description_base")]
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
        remappings=[("/robot_description", "/robot_description_base")]
    )
    
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
    )

    nodes = [
        joint_state_publisher_node,
        robot_state_publisher_node,
        rviz_node,
    ]

    return LaunchDescription(nodes)
