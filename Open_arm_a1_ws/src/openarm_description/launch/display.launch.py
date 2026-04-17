import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    declared_arguments = []
    
    declared_arguments.append(DeclareLaunchArgument("arm_type", default_value="v10", description="Type of arm."))
    declared_arguments.append(DeclareLaunchArgument("ee_type", default_value="openarm_hand", description="Type of end-effector."))
    declared_arguments.append(DeclareLaunchArgument("bimanual", default_value="false", description="Is this a bimanual setup?"))
    declared_arguments.append(DeclareLaunchArgument("hand", default_value="true", description="Include hand?"))

    arm_type = LaunchConfiguration("arm_type")
    ee_type = LaunchConfiguration("ee_type")
    bimanual = LaunchConfiguration("bimanual")
    hand = LaunchConfiguration("hand")

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("openarm_description"), "urdf", "robot", "v10.urdf.xacro"]),
            " ",
            "arm_type:=", arm_type,
            " ",
            "ee_type:=", ee_type,
            " ",
            "bimanual:=", bimanual,
            " ",
            "hand:=", hand,
        ]
    )
    
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    rviz_config_file = PathJoinSubstitution([
        FindPackageShare("openarm_description"), 
        "rviz", 
        PythonExpression(["'bimanual.rviz' if '", bimanual, "' == 'true' else 'arm_only.rviz'"])
    ])

    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui"
    )
    
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )
    
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
    )

    nodes = [
        joint_state_publisher_gui_node,
        robot_state_publisher_node,
        rviz_node,
    ]

    return LaunchDescription(declared_arguments + nodes)
