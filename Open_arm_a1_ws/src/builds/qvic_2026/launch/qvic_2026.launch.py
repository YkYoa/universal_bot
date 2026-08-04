import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    qvic_pkg = get_package_share_directory("qvic_2026")
    default_bt_xml = os.path.join(qvic_pkg, "config", "qvic_2026_bt.xml")

    bt_xml_path_arg = DeclareLaunchArgument(
        "bt_xml_path",
        default_value=default_bt_xml,
        description="Generated BT XML to run (see sequence_to_bt --out).",
    )
    tick_rate_hz_arg = DeclareLaunchArgument(
        "tick_rate_hz", default_value="50.0", description="bt_executor tick rate."
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="false", description="Use simulation clock if true."
    )

    bt_executor_node = Node(
        package="bt_executor",
        executable="bt_executor_node",
        name="bt_executor",
        output="screen",
        parameters=[
            {
                "bt_xml_path": LaunchConfiguration("bt_xml_path"),
                "tick_rate_hz": LaunchConfiguration("tick_rate_hz"),
                "log_to_file": False,
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }
        ],
    )

    return LaunchDescription([
        bt_xml_path_arg,
        tick_rate_hz_arg,
        use_sim_time_arg,
        bt_executor_node,
    ])
