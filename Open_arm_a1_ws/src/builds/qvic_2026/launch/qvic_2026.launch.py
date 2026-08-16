import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from qvic_2026 import yaml_sync

# arm:=left/right/both picks a sequences: entry by name in sequence.yaml -
# sequence_executor_node reads the YAML live, no generation/build step.
# Pass sequence:=<name> directly to run any sequences: entry that doesn't
# map to the arm concept (e.g. hand_open_close).
ARM_TO_SEQUENCE = {
    "left": "qvic_2026_left",
    "right": "qvic_2026_right",
    "both": "qvic_2026_both",
}

# This workspace's actual source tree, not the install-space copy (which
# colcon build only refreshes when explicitly rebuilt) - hardcoded because
# this launch file, like the rest of this in-development repo, is only ever
# run from this one checkout. sequence_executor_node reads this path live
# at startup, so editing the YAML and relaunching is enough - no separate
# regen or colcon build step.
SRC_SEQUENCE_YAML = "/home/hans/universal_bot/Open_arm_a1_ws/src/builds/qvic_2026/config/sequence.yaml"


def launch_setup(context, *args, **kwargs):
    sequence_executor_pkg = get_package_share_directory("sequence_executor")

    arm = LaunchConfiguration("arm").perform(context)
    sequence = LaunchConfiguration("sequence").perform(context)
    ee_type = LaunchConfiguration("ee_type").perform(context)
    use_db = LaunchConfiguration("use_db").perform(context).lower() in ("true", "1")
    autostart = LaunchConfiguration("autostart").perform(context).lower() in ("true", "1")
    auto_seed_db = LaunchConfiguration("auto_seed_db").perform(context).lower() in ("true", "1")

    # qvic_fsm_node (use_db:=true) reads sequences.db, not this file directly -
    # SRC_SEQUENCE_YAML only takes effect once imported. Re-import here, every
    # launch, so editing the YAML and relaunching is enough on its own (no
    # separate `sequence_store_cli.py import` step to remember) - matches the
    # "no generation/build step" promise made for sequence_executor_node's own
    # sequence_yaml_path above. replace=True (the default) means a name that
    # also exists as an Android-app edit in the store gets overwritten by the
    # YAML's version on every launch; pass auto_seed_db:=false to preserve
    # in-store-only edits instead.
    if use_db and auto_seed_db:
        try:
            summary = yaml_sync.import_yaml(SRC_SEQUENCE_YAML)
            print(f"qvic_2026: seeded sequence store from {SRC_SEQUENCE_YAML} "
                  f"({summary['waypoints']} waypoints, sequences: "
                  f"{', '.join(summary['sequences']) or '(none)'})")
        except Exception as e:  # noqa: BLE001 - never block the launch over a bad YAML edit
            print(f"qvic_2026: WARNING - could not seed the sequence store from "
                  f"{SRC_SEQUENCE_YAML}: {e}. qvic_fsm_node will run with whatever "
                  f"is already in the store.")

    if sequence:
        sequence_name = sequence
    else:
        if arm not in ARM_TO_SEQUENCE:
            raise ValueError(
                f"arm:='{arm}' is not one of {sorted(ARM_TO_SEQUENCE)} - "
                "pass sequence:=<name> directly for anything else."
            )
        sequence_name = ARM_TO_SEQUENCE[arm]

    # arm:=both and sequence:=hand_open_close both hold/cycle an amazing_hand
    # pose via SetHandYaw/SetHandFlex - default to that ee_type only for
    # those cases; pass ee_type:=... explicitly to override either way.
    if not ee_type:
        ee_type = "amazing_hand" if (arm == "both" or sequence_name == "hand_open_close") else "openarm_hand"

    sequence_executor_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sequence_executor_pkg, "launch", "sequence_executor.launch.py")
        ),
        launch_arguments={
            # use_db:=true (the default) runs qvic_fsm_node, which reads the
            # sequence store the Android app edits and registers this project's
            # hardcoded actions. use_db:=false falls back to the plain executor
            # reading sequence.yaml, for comparing the two.
            "executor_package": "qvic_2026" if use_db else "sequence_executor",
            "executor_executable": "qvic_fsm_node" if use_db else "sequence_executor_node",
            "sequence_yaml_path": SRC_SEQUENCE_YAML,
            # Empty leaves the FSM idle, waiting for the app or the web page to
            # pick something - which is the point of having an API at all.
            "sequence_name": sequence_name if autostart else "",
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "use_rviz": LaunchConfiguration("use_rviz"),
            "use_api": LaunchConfiguration("use_api"),
            "ee_type": ee_type,
            "isaacsim": LaunchConfiguration("isaacsim"),
        }.items(),
    )

    return [sequence_executor_launch]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "arm", default_value="left",
            description="left, right, or both - picks a sequences: entry from sequence.yaml. "
                        "Ignored if sequence:= is set explicitly.",
        ),
        DeclareLaunchArgument(
            "sequence", default_value="",
            description="Explicit sequences: entry name override; leave empty to use arm:= instead.",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("use_rviz", default_value="false"),
        DeclareLaunchArgument(
            "use_api", default_value="false",
            description="Also launch the REST/WebSocket API and web dashboard on port 5050 "
                        "(FSM viewer at /dashboard/fsm.html).",
        ),
        DeclareLaunchArgument(
            "use_db", default_value="true",
            description="true: run qvic_fsm_node against the sequence store (and this "
                        "project's hardcoded actions). false: plain sequence_executor_node "
                        "reading config/sequence.yaml.",
        ),
        DeclareLaunchArgument(
            "auto_seed_db", default_value="true",
            description="true (default): re-import config/sequence.yaml into the sequence "
                        "store on every launch, so editing the YAML and relaunching is "
                        "enough. false: leave the store as-is (e.g. to preserve edits made "
                        "through the Android app / web dashboard that aren't in the YAML). "
                        "Ignored when use_db:=false.",
        ),
        DeclareLaunchArgument(
            "autostart", default_value="true",
            description="true: start the selected sequence immediately. false: sit in IDLE "
                        "and wait for a RunSequence goal from the app or the web page.",
        ),
        DeclareLaunchArgument(
            "ee_type", default_value="",
            description="openarm_hand or amazing_hand. Leave empty to auto-select "
                        "(amazing_hand for arm:=both or sequence:=hand_open_close, openarm_hand otherwise).",
        ),
        DeclareLaunchArgument("isaacsim", default_value="false"),
        OpaqueFunction(function=launch_setup),
    ])
