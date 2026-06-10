import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node

def generate_launch_description():
    openarm_test_pkg = get_package_share_directory("openarm_test")
    bt_xml_path = os.path.join(openarm_test_pkg, "config", "test_bt.xml")

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
        description="Run demo node in a separate interactive terminal window"
    )

    bt_executor_node = Node(
        package="bt_executor",
        executable="bt_executor_node",
        name="bt_executor",
        output="screen",
        parameters=[
            {
                "bt_xml_path": bt_xml_path,
                "tick_rate_hz": 50.0,
                "log_to_file": False,
                "use_sim_time": use_sim_time,
            }
        ],
    )

    demo_node_interactive = Node(
        package="openarm_demo",
        executable="openarm_demo_node",
        name="some_demo",
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

    demo_node_non_interactive = Node(
        package="openarm_demo",
        executable="openarm_demo_node",
        name="some_demo",
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
        bt_executor_node,
        demo_node_interactive,
        demo_node_non_interactive,
    ])

