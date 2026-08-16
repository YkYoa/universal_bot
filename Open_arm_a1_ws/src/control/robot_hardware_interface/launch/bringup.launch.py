import os
import subprocess

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from openarm_moveit_config.srdf_utils import load_moveit_controllers_for_ee_type, load_srdf_for_ee_type

# CycloneDDS: large URDF / many joints
if "CYCLONEDDS_URI" not in os.environ:
    os.environ["CYCLONEDDS_URI"] = (
        "<CycloneDDS><Domain><General>"
        "<MaxMessageSize>10MB</MaxMessageSize>"
        "<FragmentSize>4000B</FragmentSize>"
        "</General></Domain></CycloneDDS>"
    )

# Real hardware bringup for the OpenArm v2 body when only the head/neck +
# LED board is physically present - arms/hands are not connected, so they
# run as mock_components/GenericSystem (no CAN needed) while the head runs
# robot_hardware_interface/HeadHW for real. All ros2_control + MoveIt config this
# launch needs lives in robot_hardware_interface/config/ (copied from robot_control
# and openarm_moveit_config so this bringup is self-contained), rather than
# reaching into three different packages' config dirs at launch time.
#
# hardware_config.yaml (same directory) declares, per component, whether it
# should be real hardware and which physical interface it's wired to -
# mirrors tomo_control's drive_config.yaml (port_info/port_ids) pattern.


def _interface_has_carrier(interface_name):
    """True if the NIC exists and reports a live link (cable plugged in,
    board powered). Missing interface or no-carrier both read as False -
    a stale hardware_enable=1 in the yaml should never make ros2_control
    try to activate hardware that isn't actually reachable."""
    carrier_path = f"/sys/class/net/{interface_name}/carrier"
    try:
        with open(carrier_path) as f:
            return f.read().strip() == "1"
    except OSError:
        return False


def _ensure_can_up(interface_name, bitrate, dbitrate):
    """Bring a CAN-FD interface up (sudo ip link ... type can ... fd on) if
    it isn't already - so a plain `ros2 launch` works right after plugging
    the adapter in, without a separate manual
    openarm-can-configure-socketcan-4-arms step first.

    Best-effort: any failure here (no sudo, interface missing, adapter not
    powered) is just printed - _interface_has_carrier()'s existing check
    downstream still catches it and correctly falls back to fake, so a
    failure here can't make ros2_control think hardware is present when
    it isn't."""
    if _interface_has_carrier(interface_name):
        return  # already up and live - don't reset an interface mid-use

    print(f"[bringup] {interface_name} is down, bringing it up (bitrate={bitrate}, dbitrate={dbitrate}, fd)...")
    try:
        subprocess.run(["sudo", "ip", "link", "set", interface_name, "down"], check=False)
        result = subprocess.run(
            ["sudo", "ip", "link", "set", interface_name, "type", "can",
             "bitrate", str(bitrate), "dbitrate", str(dbitrate), "fd", "on"],
            check=False)
        if result.returncode != 0:
            print(f"[bringup] WARNING: failed to configure {interface_name} (not found / adapter unplugged?)")
            return
        subprocess.run(["sudo", "ip", "link", "set", interface_name, "up"], check=False)
    except FileNotFoundError:
        print("[bringup] WARNING: 'sudo'/'ip' not found - can't auto-configure CAN, run "
              "openarm-can-configure-socketcan-4-arms manually")


def launch_setup(context, *args, **kwargs):
    hw_config_pkg = get_package_share_directory("robot_hardware_interface")
    moveit_config_pkg = get_package_share_directory("openarm_moveit_config")

    with open(os.path.join(hw_config_pkg, "config", "hardware_config.yaml")) as f:
        hw_cfg = yaml.safe_load(f)

    head_mode = LaunchConfiguration("head").perform(context)
    use_rviz = LaunchConfiguration("use_rviz")
    ee_type = LaunchConfiguration("ee_type")
    body_type = LaunchConfiguration("body_type")
    is_amazing_hand = IfCondition(PythonExpression(["'", ee_type, "' == 'amazing_hand'"]))
    is_openarm_hand = IfCondition(PythonExpression(["'", ee_type, "' == 'openarm_hand'"]))
    is_body_v2 = IfCondition(PythonExpression(["'", body_type, "' == 'v2'"]))

    # head:=auto (default) - real hardware only if hardware_config.yaml enables
    # it AND the ethernet interface actually has a live link right now.
    # head:=true/false forces the decision, skipping detection (useful for
    # testing without needing the cable plugged in).
    if head_mode == "auto":
        head_real = (
            hw_cfg["hardware_enable"]["head"] == 1
            and _interface_has_carrier(hw_cfg["interfaces"]["head_ethernet"])
        )
    else:
        head_real = head_mode == "true"

    print(f"[bringup] head hardware: {'REAL' if head_real else 'FAKE'} "
          f"(mode={head_mode}, interface={hw_cfg['interfaces']['head_ethernet']})")

    # arms:=auto (default) - same pattern as head: real only if
    # hardware_config.yaml enables it AND both CAN interfaces are up. Unless
    # arms:=false, this launch brings the CAN interfaces up itself first
    # (see _ensure_can_up) - no separate manual
    # openarm-can-configure-socketcan-4-arms step needed. arms:=true/false
    # forces the real/fake decision either way.
    left_can_interface = hw_cfg["interfaces"]["left_can"]
    right_can_interface = hw_cfg["interfaces"]["right_can"]
    can_cfg = hw_cfg.get("can", {})
    can_bitrate = can_cfg.get("bitrate", 1000000)
    can_dbitrate = can_cfg.get("dbitrate", 5000000)
    arms_mode = LaunchConfiguration("arms").perform(context)
    if arms_mode != "false":
        _ensure_can_up(left_can_interface, can_bitrate, can_dbitrate)
        _ensure_can_up(right_can_interface, can_bitrate, can_dbitrate)
    if arms_mode == "auto":
        arms_real = (
            hw_cfg["hardware_enable"]["arms"] == 1
            and _interface_has_carrier(left_can_interface)
            and _interface_has_carrier(right_can_interface)
        )
    else:
        arms_real = arms_mode == "true"

    control_mode = hw_cfg.get("control_mode", "mit")
    print(f"[bringup] arms hardware: {'REAL' if arms_real else 'FAKE'} "
          f"(mode={arms_mode}, left={left_can_interface}, right={right_can_interface}, "
          f"control_mode={control_mode})")
    if control_mode == "torque" and arms_real:
        print("[bringup] WARNING: control_mode=torque - gravity_comp_controller will be "
              "auto-loaded, activated, AND auto-enabled on both arms. Real torque will be "
              "applied with NO manual confirmation step. Support both arms by hand NOW if "
              "you have not already - the ramp starts as soon as controller_manager comes up.")

    shutdown_disable_retries = str(hw_cfg["shutdown"]["disable_retries"])
    shutdown_retry_delay_ms = str(hw_cfg["shutdown"]["retry_delay_ms"])
    shutdown_home_timeout_ms = str(hw_cfg["shutdown"].get("home_timeout_ms", 3000))
    shutdown_home_tolerance = str(hw_cfg["shutdown"].get("home_tolerance", 0.05))
    position_mode_velocity = str(hw_cfg.get("position_mode_velocity", 1.0))
    hand_rotate_ratio = str(hw_cfg.get("hand_rotate_ratio", 1.0))

    use_fake_hardware = "false" if arms_real else "true"
    head_use_fake_hardware = "false" if head_real else "true"

    # ── Robot Description (URDF) ──
    robot_description_args = [
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([FindPackageShare("openarm_description"), "urdf", "robot", "v10.urdf.xacro"]),
        " ",
        "bimanual:=true",
        " ",
        "ros2_control:=true",
        " ",
        "use_fake_hardware:=", use_fake_hardware,
        " ",
        "head_use_fake_hardware:=", head_use_fake_hardware,
        " ",
        "left_can_interface:=", left_can_interface,
        " ",
        "right_can_interface:=", right_can_interface,
        " ",
        "shutdown_disable_retries:=", shutdown_disable_retries,
        " ",
        "shutdown_retry_delay_ms:=", shutdown_retry_delay_ms,
        " ",
        "shutdown_home_timeout_ms:=", shutdown_home_timeout_ms,
        " ",
        "shutdown_home_tolerance:=", shutdown_home_tolerance,
        " ",
        "control_mode:=", control_mode,
        " ",
        "position_mode_velocity:=", position_mode_velocity,
        " ",
        "hand_rotate_ratio:=", hand_rotate_ratio,
        " ",
        "gazebo:=false",
        " ",
        "ee_type:=", ee_type,
        " ",
        "body_type:=", body_type,
        " ",
        "mobile_base:=true",
        " ",
        "mobile_base_xyz:='0 0 0.31'",
        " ",
        "mobile_base_body_xyz:='0 0 0'",
    ]
    _debug_parts = [a.perform(context) if hasattr(a, "perform") else str(a) for a in robot_description_args]
    print(f"[bringup] DEBUG resolved xacro command: {''.join(_debug_parts)!r}")
    robot_description_content = Command(robot_description_args).perform(context)
    with open("/tmp/bringup_debug_urdf.xml", "w") as f:
        f.write(robot_description_content)
    print(
        f"[bringup] DEBUG xacro output written to /tmp/bringup_debug_urdf.xml "
        f"({len(robot_description_content)} bytes, "
        f"{robot_description_content.count('ahand_connector')} ahand_connector matches)"
    )
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    # ── Semantic Robot Description (SRDF) ──
    # openarm_bimanual.srdf contains groups for both end-effector types; only
    # one set is valid for the URDF actually being built here. See
    # openarm_moveit_config/srdf_utils.py for why and what it strips.
    srdf_path = os.path.join(moveit_config_pkg, "srdf", "openarm_bimanual.srdf")
    robot_description_semantic_content = load_srdf_for_ee_type(
        srdf_path, ee_type.perform(context), body_type.perform(context))
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_content}

    # ── Config, all local to this package ──
    kinematics_yaml_path = os.path.join(hw_config_pkg, "config", "kinematics.yaml")
    ompl_planning_yaml_path = os.path.join(hw_config_pkg, "config", "ompl_planning.yaml")
    pilz_planning_yaml_path = os.path.join(hw_config_pkg, "config", "pilz_industrial_motion_planner_planning.yaml")
    chomp_planning_yaml_path = os.path.join(hw_config_pkg, "config", "chomp_planning.yaml")
    stomp_planning_yaml_path = os.path.join(hw_config_pkg, "config", "stomp_planning.yaml")
    joint_limits_yaml_path = os.path.join(hw_config_pkg, "config", "joint_limits.yaml")
    # ee_type-filtered - see load_moveit_controllers_for_ee_type's docstring
    # in openarm_moveit_config/srdf_utils.py (same left_gripper_controller vs
    # left_hand_rotate_controller "motor 8" ambiguity as moveit_bimanual.launch.py).
    moveit_controllers_params = load_moveit_controllers_for_ee_type(
        os.path.join(hw_config_pkg, "config", "moveit_controllers.yaml"),
        ee_type.perform(context),
    )
    controller_config = os.path.join(hw_config_pkg, "config", "bimanual_controllers.yaml")
    rviz_config_file = os.path.join(hw_config_pkg, "config", "moveit.rviz")

    trajectory_execution = {
        "moveit_manage_controllers": True,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }
    planning_scene_monitor = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }
    bounds_tolerances = {"start_state_max_bounds_error": 2.0}

    # ── head board processes: low-level neck driver + LED service node ──
    # head_motor_driver_node owns the real UDS<->STM32 bridge that HeadHW
    # connects to; only needed when the head is real. head_led_node is
    # independent of ros2_control (own TCP port on the board) so it runs
    # every time this bringup launches, regardless of head real/fake.
    head_motor_driver_node = Node(
        package="communication_devices",
        executable="head_motor_driver_node",
        output="screen",
    )

    head_led_node = Node(
        package="communication_devices",
        executable="head_led_node",
        output="screen",
    )

    # head/go_home service (std_srvs/Trigger) - sends a FollowJointTrajectory
    # goal to head_controller centering pan/tilt. Needs head_controller up,
    # but the node itself is harmless to start unconditionally.
    head_home_node = Node(
        package="communication_devices",
        executable="head_home_node",
        output="screen",
    )

    # ── ros2_control + MoveIt ──
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": False}],
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[controller_config, {"use_sim_time": False}],
        remappings=[("~/robot_description", "/robot_description")],
        output="both",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager"],
    )
    left_arm_controller_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["left_arm_controller", "-c", "/controller_manager"],
    )
    right_arm_controller_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["right_arm_controller", "-c", "/controller_manager"],
    )

    left_gripper_controller_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["left_gripper_controller", "-c", "/controller_manager"],
        condition=is_openarm_hand,
    )
    right_gripper_controller_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["right_gripper_controller", "-c", "/controller_manager"],
        condition=is_openarm_hand,
    )
    hand_controller_spawners = [
        Node(
            package="controller_manager", executable="spawner",
            arguments=[name, "-c", "/controller_manager"],
            condition=is_amazing_hand,
        )
        for name in ("left_hand_j1_controller", "left_hand_j2_controller",
                     "right_hand_j1_controller", "right_hand_j2_controller",
                     # "Motor 8" (openarm_<side>_finger_joint1) repurposed to
                     # spin the amazing_hand connector - see
                     # bimanual_controllers.yaml.
                     "left_hand_rotate_controller", "right_hand_rotate_controller")
    ]
    head_controller_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["head_controller", "-c", "/controller_manager"],
        condition=is_body_v2,
    )

    # amazing_hand's closed-loop finger linkage (ball/cylindrical/revolute
    # loop-closure joints, no other state source) is solved in-process by
    # robot_hardware_interface/AmazingHandHW (see ahand.ros2_control.xacro)
    # and reported as normal ros2_control state interfaces, published by
    # joint_state_broadcaster like everything else - no separate node here.
    base_controller_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["base_controller", "-c", "/controller_manager"],
    )

    # control_mode: torque only - auto load+configure+activate the gravity-
    # comp controllers AND auto-call their ~/enable service, so the arms
    # start applying real gravity-compensation torque as soon as bringup
    # finishes, no manual `ros2 control`/`ros2 service call` steps.
    #
    # DANGER: this means there is NO human-in-the-loop step between power-on
    # and the arm attempting to hold its own weight under torque control.
    # The physical-support-before-enabling precaution from the staged
    # validation procedure becomes YOUR responsibility to do BEFORE running
    # this launch command, every time - not something this launch enforces.
    # If you are not physically ready to support both arms the instant this
    # launch starts, do not use control_mode: torque, or edit
    # hardware_config.yaml back to "mit"/"position" first.
    gravity_comp_spawners = []
    gravity_comp_enable_handlers = []
    if control_mode == "torque":
        for side in ("left", "right"):
            spawner = Node(
                package="controller_manager", executable="spawner",
                arguments=[f"{side}_gravity_comp_controller", "-c", "/controller_manager"],
            )
            enable_call = ExecuteProcess(
                cmd=["ros2", "service", "call", f"/{side}_gravity_comp_controller/enable",
                     "std_srvs/srv/SetBool", "{data: true}"],
                output="screen",
            )
            gravity_comp_spawners.append(spawner)
            gravity_comp_enable_handlers.append(
                RegisterEventHandler(OnProcessExit(target_action=spawner, on_exit=[enable_call]))
            )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml_path,
            {"default_planning_pipeline": "ompl"},
            ompl_planning_yaml_path,
            pilz_planning_yaml_path,
            chomp_planning_yaml_path,
            stomp_planning_yaml_path,
            joint_limits_yaml_path,
            moveit_controllers_params,
            trajectory_execution,
            planning_scene_monitor,
            bounds_tolerances,
            {"use_sim_time": False},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml_path,
            {"default_planning_pipeline": "ompl"},
            ompl_planning_yaml_path,
            pilz_planning_yaml_path,
            chomp_planning_yaml_path,
            stomp_planning_yaml_path,
            {"use_sim_time": False},
        ],
        condition=IfCondition(use_rviz),
    )

    return [
        *([head_motor_driver_node] if head_real else []),
        head_led_node,
        head_home_node,
        robot_state_publisher_node,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        left_arm_controller_spawner,
        right_arm_controller_spawner,
        left_gripper_controller_spawner,
        right_gripper_controller_spawner,
        *hand_controller_spawners,
        head_controller_spawner,
        base_controller_spawner,
        *gravity_comp_spawners,
        *gravity_comp_enable_handlers,
        move_group_node,
        rviz_node,
    ]


def generate_launch_description():
    head_arg = DeclareLaunchArgument(
        "head",
        default_value="auto",
        description="'auto' (default): read hardware_config.yaml and auto-detect whether the "
                     "head's ethernet interface has a live link, use real hardware only if both "
                     "say yes. 'true'/'false': force real/fake for the whole robot, skipping "
                     "detection.",
    )
    arms_arg = DeclareLaunchArgument(
        "arms",
        default_value="auto",
        description="'auto' (default): read hardware_config.yaml and auto-detect whether both "
                     "CAN interfaces are up, use real hardware only if both say yes. "
                     "'true'/'false': force real/fake, skipping detection.",
    )
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz", default_value="true", description="Whether to launch RViz."
    )
    ee_type_arg = DeclareLaunchArgument(
        "ee_type", default_value="amazing_hand",
        description="End-effector type: openarm_hand or amazing_hand (fake either way here).",
    )
    body_type_arg = DeclareLaunchArgument(
        "body_type", default_value="v2",
        description="Chassis/body version. Must be 'v2' for the neck_joint/head_joint to exist.",
    )

    return LaunchDescription(
        [
            head_arg,
            arms_arg,
            use_rviz_arg,
            ee_type_arg,
            body_type_arg,
            OpaqueFunction(function=launch_setup),
        ]
    )
