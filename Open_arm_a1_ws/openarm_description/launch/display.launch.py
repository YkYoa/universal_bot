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
    declared_arguments.append(DeclareLaunchArgument("mobile_base", default_value="false", description="Include mobile base?"))
    declared_arguments.append(DeclareLaunchArgument("mobile_base_arm_xyz", default_value="0 0 0.31", description="Single-arm mount offset on the mobile base."))
    declared_arguments.append(DeclareLaunchArgument("mobile_base_arm_rpy", default_value="0 0 0", description="Single-arm mount rotation on the mobile base."))
    declared_arguments.append(DeclareLaunchArgument("mobile_base_body_xyz", default_value="0 0 0", description="Bimanual body mount offset on the mobile base."))
    declared_arguments.append(DeclareLaunchArgument("mobile_base_body_rpy", default_value="0 0 0", description="Bimanual body mount rotation on the mobile base."))

    arm_type = LaunchConfiguration("arm_type")
    ee_type = LaunchConfiguration("ee_type")
    bimanual = LaunchConfiguration("bimanual")
    hand = LaunchConfiguration("hand")
    mobile_base = LaunchConfiguration("mobile_base")
    mobile_base_arm_xyz = LaunchConfiguration("mobile_base_arm_xyz")
    mobile_base_arm_rpy = LaunchConfiguration("mobile_base_arm_rpy")
    mobile_base_body_xyz = LaunchConfiguration("mobile_base_body_xyz")
    mobile_base_body_rpy = LaunchConfiguration("mobile_base_body_rpy")

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
            " ",
            "mobile_base:=", mobile_base,
            " ",
            "mobile_base_arm_xyz:='", mobile_base_arm_xyz, "'",
            " ",
            "mobile_base_arm_rpy:='", mobile_base_arm_rpy, "'",
            " ",
            "mobile_base_body_xyz:='", mobile_base_body_xyz, "'",
            " ",
            "mobile_base_body_rpy:='", mobile_base_body_rpy, "'",
        ]
    )
    
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    rviz_config_file = PathJoinSubstitution([
        FindPackageShare("openarm_description"), 
        "rviz", 
        PythonExpression([
            "'mobile_base.rviz' if '", mobile_base, "' == 'true' else ",
            "('bimanual.rviz' if '", bimanual, "' == 'true' else 'arm_only.rviz')"
        ])
    ])

    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        parameters=[robot_description],
        remappings=[("/robot_description", "/robot_description_full")]
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
        remappings=[("/robot_description", "/robot_description_full")]
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

    return LaunchDescription(declared_arguments + nodes)
