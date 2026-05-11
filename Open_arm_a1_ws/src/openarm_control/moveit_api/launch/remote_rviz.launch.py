import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # 1. Robot Description (URDF)
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
            "mobile_base:=true",
        ]
    )
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    # 2. Semantic Robot Description (SRDF)
    srdf_path = PathJoinSubstitution([
        FindPackageShare("openarm_moveit_config"),
        "srdf",
        "openarm_bimanual.srdf"
    ])
    
    # We need to read the SRDF content
    # Since we are in a launch file, we can use a small trick to pass the path
    # or just let RViz load it if the package is installed.
    # However, passing it as a parameter is safer.
    
    # RViz Config
    rviz_config = PathJoinSubstitution([
        FindPackageShare("openarm_moveit_config"),
        "config",
        "moveit.rviz"
    ])

    return LaunchDescription([
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config],
            parameters=[
                robot_description,
                {"robot_description_semantic": Command(["cat ", srdf_path])}
            ],
        ),
    ])
    