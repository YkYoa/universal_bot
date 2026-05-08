from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # Define launch arguments
    mujoco_model = LaunchConfiguration("mujoco_model")
    
    # Load robot description
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("openarm_mujoco_sim"), "config", mujoco_model]),
        ]
    )
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    return LaunchDescription([
        DeclareLaunchArgument("mujoco_model", default_value="openarm.xml/mujoco_description.xml", description="Path to the MuJoCo model file."),
        
        Node(
            package="mujoco_ros2_control",
            executable="ros2_control_node",
            parameters=[{
                "robot_description": robot_description,
                "model_path": PathJoinSubstitution([FindPackageShare("openarm_mujoco_sim"), "config", mujoco_model]),
            }]
        )
    ])
