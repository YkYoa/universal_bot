import os
import subprocess
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# arm:=left/right/both picks the matching pre-generated BT XML in config/ so
# you don't have to type the full install-space bt_xml_path by hand. Pass
# bt_xml_path explicitly to override (e.g. for a BT XML outside this map).
ARM_TO_BT_XML = {
    "left": "qvic_2026_ellipse_bt.xml",
    "right": "qvic_2026_ellipse_right_bt.xml",
    "both": "qvic_2026_ellipse_bimanual_bt.xml",
}

# The XML is a static snapshot of sequence.yaml, not read live - regenerate
# it from the SOURCE sequence.yaml on every launch (see launch_setup below)
# so editing the YAML and relaunching is enough, no separate sequence_to_bt +
# colcon build step. This is this workspace's actual source tree, not the
# install-space copy (which colcon build only refreshes when explicitly
# rebuilt) - hardcoded because this launch file, like the rest of this
# in-development repo, is only ever run from this one checkout.
SRC_SEQUENCE_YAML = "/home/hans/universal_bot/Open_arm_a1_ws/src/builds/qvic_2026/config/sequence.yaml"

# sequence_to_bt args per arm:= value (excluding --yaml/--out, filled in at
# launch time). Keep these in sync with the README's regen commands.
ARM_REGEN_ARGS = {
    "left": ["--section", "waveEllipse", "--arm", "left_arm"],
    "right": ["--section", "waveEllipseR", "--arm", "right_arm"],
    "both": [
        "--both-arms", "--left-section", "waveEllipse", "--right-section", "waveEllipseR",
        "--profile", "realtime_rrt", "--tree-name", "Both",
        "--home-section", "homePoses", "--repeat-cycles", "-1",
    ],
}


def launch_setup(context, *args, **kwargs):
    qvic_pkg = get_package_share_directory("qvic_2026")
    bt_executor_pkg = get_package_share_directory("bt_executor")

    arm = LaunchConfiguration("arm").perform(context)
    bt_xml_path = LaunchConfiguration("bt_xml_path").perform(context)
    ee_type = LaunchConfiguration("ee_type").perform(context)

    if not bt_xml_path:
        if arm not in ARM_TO_BT_XML:
            raise ValueError(
                f"arm:='{arm}' is not one of {sorted(ARM_TO_BT_XML)} - "
                "pass bt_xml_path:=... directly for anything else."
            )
        bt_xml_path = os.path.join(qvic_pkg, "config", ARM_TO_BT_XML[arm])

        sequence_to_bt_bin = os.path.join(get_package_prefix("openarm_demo"), "lib", "openarm_demo", "sequence_to_bt")
        regen_cmd = [sequence_to_bt_bin, "--yaml", SRC_SEQUENCE_YAML, "--out", bt_xml_path] + ARM_REGEN_ARGS[arm]
        result = subprocess.run(regen_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"sequence_to_bt failed regenerating {bt_xml_path}:\n{result.stdout}{result.stderr}"
            )
        print(f"[qvic_2026.launch.py] Regenerated {bt_xml_path} from {SRC_SEQUENCE_YAML}")

    # arm:=both holds an amazing_hand pose via SetHandYaw/SetHandFlex (see
    # sequence.yaml's waveHome section) - default to that ee_type only for
    # this case; pass ee_type:=... explicitly to override either way.
    if not ee_type:
        ee_type = "amazing_hand" if arm == "both" else "openarm_hand"

    bt_executor_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bt_executor_pkg, "launch", "bt_executor.launch.py")
        ),
        launch_arguments={
            "bt_xml_path": bt_xml_path,
            "tick_rate_hz": LaunchConfiguration("tick_rate_hz"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "use_rviz": LaunchConfiguration("use_rviz"),
            "use_api": LaunchConfiguration("use_api"),
            "ee_type": ee_type,
            "isaacsim": LaunchConfiguration("isaacsim"),
            "instruction": LaunchConfiguration("instruction"),
        }.items(),
    )

    return [bt_executor_launch]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "arm", default_value="left",
            description="left, right, or both - picks the matching BT XML from config/. "
                        "Ignored if bt_xml_path is set explicitly.",
        ),
        DeclareLaunchArgument(
            "bt_xml_path", default_value="",
            description="Explicit BT XML path override; leave empty to use arm:= instead.",
        ),
        DeclareLaunchArgument("tick_rate_hz", default_value="50.0"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("use_rviz", default_value="false"),
        DeclareLaunchArgument("use_api", default_value="false"),
        DeclareLaunchArgument(
            "ee_type", default_value="",
            description="openarm_hand or amazing_hand. Leave empty to auto-select "
                        "(amazing_hand for arm:=both, openarm_hand otherwise).",
        ),
        DeclareLaunchArgument("isaacsim", default_value="false"),
        DeclareLaunchArgument("instruction", default_value=""),
        OpaqueFunction(function=launch_setup),
    ])
