import os
from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    moveit_config_pkg = get_package_share_directory("openarm_moveit_config")
    arm_control_pkg = get_package_share_directory("arm_control")

    # Robot Description (URDF)
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
    
    # Semantic Robot Description (SRDF)
    srdf_path = os.path.join(moveit_config_pkg, "srdf", "openarm_bimanual.srdf")
    with open(srdf_path, "r") as f:
        robot_description_semantic_content = f.read()
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_content}

    # MoveIt Config Files
    kinematics_yaml_path = os.path.join(moveit_config_pkg, "config", "kinematics.yaml")
    ompl_planning_yaml_path = os.path.join(moveit_config_pkg, "config", "ompl_planning.yaml")
    joint_limits_yaml_path = os.path.join(moveit_config_pkg, "config", "joint_limits.yaml")
    moveit_controllers_yaml_path = os.path.join(moveit_config_pkg, "config", "moveit_controllers.yaml")
    
    # ROS2 Control Config Files
    ros2_controllers_yaml_path = os.path.join(arm_control_pkg, "config", "bimanual_controllers.yaml")

    return LaunchDescription([
        # 1. Robot State Publisher
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[robot_description],
        ),
        # 2. Move Group
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[
                robot_description,
                robot_description_semantic,
                kinematics_yaml_path,
                ompl_planning_yaml_path,
                joint_limits_yaml_path,
                moveit_controllers_yaml_path,
            ],
        ),
        # 3. Joint State Publisher (Removes the need for manual joint_state_publisher)
        # Note: We use joint_state_broadcaster from ros2_control instead for better integration
        
        # 4. Controller Manager
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[robot_description, ros2_controllers_yaml_path],
            output="screen",
        ),

        # 5. Spawners
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["left_arm_controller", "--controller-manager", "/controller_manager"],
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["right_arm_controller", "--controller-manager", "/controller_manager"],
        ),
    ])
