from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    interactive = LaunchConfiguration("interactive")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock if true"
    )

    interactive_arg = DeclareLaunchArgument(
        "interactive",
        default_value="true",
        description="Run the node in a separate interactive terminal window"
    )

    node_interactive = Node(
        package="openarm_head_hand",
        executable="openarm_head_hand_node",
        name="openarm_head_hand",
        output="screen",
        emulate_tty=True,
        prefix="gnome-terminal --",
        condition=IfCondition(interactive),
        parameters=[
            {
                "use_sim_time": use_sim_time,
            }
        ],
    )

    node_non_interactive = Node(
        package="openarm_head_hand",
        executable="openarm_head_hand_node",
        name="openarm_head_hand",
        output="screen",
        emulate_tty=True,
        condition=UnlessCondition(interactive),
        parameters=[
            {
                "use_sim_time": use_sim_time,
            }
        ],
    )

    return LaunchDescription([
        use_sim_time_arg,
        interactive_arg,
        node_interactive,
        node_non_interactive,
    ])
