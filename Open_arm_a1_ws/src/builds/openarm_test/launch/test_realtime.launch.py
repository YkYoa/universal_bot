import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # ── Retrieve package paths ──
    openarm_test_pkg = get_package_share_directory("openarm_test")
    moveit_config_pkg = get_package_share_directory("openarm_moveit_config")
    
    # Path to the test behavior tree XML
    bt_xml_path = os.path.join(openarm_test_pkg, "config", "test_bt.xml")

    # ── Include core MoveIt and Skills Server launch ──
    core_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_config_pkg, "launch", "moveit_bimanual.launch.py")
        )
    )

    # ── BT Executor Node ──
    bt_executor_node = Node(
        package="bt_executor",
        executable="bt_executor_node",
        name="bt_executor",
        output="screen",
        parameters=[{
            "bt_xml_path": bt_xml_path,
            "tick_rate_hz": 50.0,
            "log_to_file": False,
        }],
    )

    # ── Test Orchestrator Node ──
    test_node = Node(
        package="openarm_test",
        executable="test_realtime_node",
        name="test_realtime_node",
        output="screen",
    )

    return LaunchDescription([
        core_launch,
        bt_executor_node,
        test_node,
    ])
