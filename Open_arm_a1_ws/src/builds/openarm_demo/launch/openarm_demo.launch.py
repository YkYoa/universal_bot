import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    openarm_test_pkg = get_package_share_directory("openarm_test")
    bt_xml_path = os.path.join(openarm_test_pkg, "config", "test_bt.xml")

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
            }
        ],
    )

    demo_node = Node(
        package="openarm_demo",
        executable="openarm_demo_node",
        name="some_demo",
        output="screen",
    )

    return LaunchDescription([
        bt_executor_node,
        demo_node,
    ])
