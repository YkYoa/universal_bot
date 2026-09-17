import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from openarm_moveit_config.srdf_utils import load_srdf_for_ee_type

# moveit_servo real-time Cartesian/joint jogging - see servo_params.yaml's
# file header for the two servo_node instances this launches (one per arm,
# each namespaced so their ~/delta_twist_cmds etc. don't collide) and its
# own comments on which values are reasoned from this robot's real
# joint_limits.yaml/planner_profiles.yaml data vs. still-generic
# moveit_servo defaults that need live testing to tune properly.
#
# Standalone from moveit_bimanual.launch.py on purpose: that launch already
# starts robot_state_publisher/ros2_control/controllers/move_group - this
# one only adds the two servo_node instances on top of an already-running
# bimanual bringup, the same way sequence_executor.launch.py and
# robot_api.launch.py are each their own layer rather than one launch file
# trying to own everything.
#
# Lives in its own package (openarm_servo), not openarm_moveit_config:
# this is a second, always-running-alongside-bringup process for a
# different use case (live jogging) than that package's job (offline
# OMPL/Pilz/CHOMP/STOMP planning pipeline config) - reuses its
# srdf_utils.load_srdf_for_ee_type() via a normal package dependency
# rather than duplicating that logic.


def launch_setup(context, *args, **kwargs):
    moveit_config_pkg = get_package_share_directory("openarm_moveit_config")
    servo_pkg = get_package_share_directory("openarm_servo")

    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    use_sim_time = LaunchConfiguration("use_sim_time")
    ee_type = LaunchConfiguration("ee_type")
    body_type = LaunchConfiguration("body_type")
    left_arm = LaunchConfiguration("left_arm")
    right_arm = LaunchConfiguration("right_arm")

    # ── Robot Description (URDF) - same construction as moveit_bimanual.launch.py ──
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("openarm_description"), "assets", "robot", "openarm_v1.0", "urdf", "v10.urdf.xacro"]),
            " ",
            "bimanual:=true",
            " ",
            "ros2_control:=true",
            " ",
            "ee_type:=",
            ee_type,
            " ",
            "body_type:=",
            body_type,
            " ",
            "use_fake_hardware:=",
            use_fake_hardware,
            " ",
            "head_use_fake_hardware:=true",
            " ",
            "mobile_base:=true",
            " ",
            "mobile_base_xyz:='0 0 0.31'",
            " ",
            "mobile_base_body_xyz:='0 0 0'",
        ]
    )
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    srdf_path = os.path.join(moveit_config_pkg, "srdf", "openarm_bimanual.srdf")
    robot_description_semantic = {
        "robot_description_semantic": load_srdf_for_ee_type(
            srdf_path,
            ee_type.perform(context),
            body_type.perform(context),
        )
    }

    kinematics_yaml_path = os.path.join(moveit_config_pkg, "config", "kinematics.yaml")
    joint_limits_yaml_path = os.path.join(moveit_config_pkg, "config", "joint_limits.yaml")

    with open(os.path.join(servo_pkg, "config", "servo_params.yaml")) as f:
        base_servo_params = yaml.safe_load(f)

    def servo_node_for(side: str, move_group_name: str, enabled: LaunchConfiguration) -> Node:
        """One servo_node instance for `side` ("left"/"right"), namespaced
        under `~/<side>_arm_servo` so cartesian_command_in_topic/
        joint_command_in_topic/status_topic (all relative, see
        servo_params.yaml) never collide between arms. command_out_topic is
        the one field servo_params.yaml deliberately leaves unset, since
        it's the one thing that MUST differ per side - see this repo's
        left_arm_controller/right_arm_controller (JointTrajectoryController,
        bimanual_controllers.yaml)."""
        params = dict(base_servo_params)
        params["move_group_name"] = move_group_name
        params["command_out_topic"] = f"/{side}_arm_controller/joint_trajectory"
        return Node(
            package="moveit_servo",
            executable="servo_node",
            name=f"{side}_arm_servo",
            namespace=f"{side}_arm_servo",
            output="screen",
            parameters=[
                {"moveit_servo": params},
                robot_description,
                robot_description_semantic,
                kinematics_yaml_path,
                joint_limits_yaml_path,
                {"use_sim_time": use_sim_time},
            ],
            condition=IfCondition(enabled),
        )

    return [
        servo_node_for("left", "left_arm", left_arm),
        servo_node_for("right", "right_arm", right_arm),
    ]


def generate_launch_description():
    """Declares this launch file's arguments and defers node construction to
    launch_setup() via OpaqueFunction - same shape as
    moveit_bimanual.launch.py. Run alongside that launch (or bringup.launch.py),
    not standalone: servo_node needs the real controllers/move_group already
    up to actually move anything."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_fake_hardware", default_value="true",
                description="Must match whatever bringup/moveit_bimanual is already running with - "
                            "only affects the robot_description this launch builds for IK/collision "
                            "checking, not any hardware interface of its own.",
            ),
            DeclareLaunchArgument(
                "use_sim_time", default_value="false",
                description="Whether to use simulation clock (true) or not (false).",
            ),
            DeclareLaunchArgument(
                "ee_type", default_value="amazing_hand",
                description="Must match the already-running bringup's ee_type - see "
                            "moveit_bimanual.launch.py's identical argument.",
            ),
            DeclareLaunchArgument(
                "body_type", default_value="v2",
                description="Must match the already-running bringup's body_type - see "
                            "moveit_bimanual.launch.py's identical argument.",
            ),
            DeclareLaunchArgument(
                "left_arm", default_value="true",
                description="Whether to launch the left arm's servo_node instance.",
            ),
            DeclareLaunchArgument(
                "right_arm", default_value="true",
                description="Whether to launch the right arm's servo_node instance.",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
