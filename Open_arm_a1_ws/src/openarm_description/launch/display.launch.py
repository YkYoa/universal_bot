import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    declared_arguments = []
    
    declared_arguments.append(DeclareLaunchArgument("arm_type", default_value="v10", description="Type of arm."))
    declared_arguments.append(DeclareLaunchArgument("ee_type", default_value="openarm_hand", description="Type of end-effector."))
    declared_arguments.append(DeclareLaunchArgument("bimanual", default_value="false", description="Is this a bimanual setup?"))
    declared_arguments.append(DeclareLaunchArgument("hand", default_value="true", description="Include hand?"))
    declared_arguments.append(DeclareLaunchArgument("mobile_base", default_value="false", description="Include mobile base?"))
    declared_arguments.append(DeclareLaunchArgument("mobile_base_arm_xyz", default_value="0 0 0.31", description="Single-arm mount offset on the mobile base."))
    declared_arguments.append(DeclareLaunchArgument("mobile_base_arm_rpy", default_value="0 0 0", description="Single-arm mount rotation on the mobile base."))
    declared_arguments.append(DeclareLaunchArgument("mobile_base_body_xyz", default_value="0 0 0", description="Bimanual body mount offset on the mobile base."))
    declared_arguments.append(DeclareLaunchArgument("mobile_base_body_rpy", default_value="0 0 3.14159", description="Bimanual body mount rotation on the mobile base."))

    # ee mount tuning. xyz_ee/rpy_ee are consumed by BOTH ee_type's own
    # *_arguments.xacro with DIFFERENT proper defaults (openarm_hand:
    # "0 -0.0 0.1001"/"0 0 0", amazing_hand: the tuned values below), so the
    # default here has to switch on ee_type too or selecting openarm_hand
    # would silently inherit amazing_hand's mount numbers. connector_* are
    # amazing_hand-only (openarm_hand's macro never reads them, so a
    # mismatched default there is harmless either way).
    # Previously NONE of these were forwarded to the xacro command at all --
    # overrides passed on the ros2 launch command line were silently dropped.
    _ee_type_ref = LaunchConfiguration("ee_type")
    declared_arguments.append(DeclareLaunchArgument(
        "xyz_ee",
        default_value=PythonExpression(["\"0.0358 -0.0148 0.1\" if \"", _ee_type_ref, "\" == \"amazing_hand\" else \"0 -0.0 0.1001\""]),
        description="Hand mount position, relative to the connector (or link7 if no connector). Shared by both sides unless xyz_ee_right overrides it."))
    declared_arguments.append(DeclareLaunchArgument(
        "rpy_ee",
        default_value=PythonExpression(["\"0 0 -1.047\" if \"", _ee_type_ref, "\" == \"amazing_hand\" else \"0 0 0\""]),
        description="Hand mount rotation, relative to the connector (or link7 if no connector). Shared by both sides unless rpy_ee_right overrides it."))
    # Per-side overrides (xyz_ee_right etc): default to EMPTY, not a copy of
    # the shared value. amazing_hand_arguments.xacro already has its own
    # real per-side defaults, tuned independently and updated over time; a
    # non-empty default here would permanently shadow that file's default
    # (this bit twice: editing the xacro file's default silently did
    # nothing because this launch file always forwarded an explicit
    # override, even when "unset"). Empty here means "don't pass this xacro
    # arg at all", letting the xacro file's own default apply.
    for _name, _desc in [
        ("xyz_ee_right", "Right-hand-only position override (amazing_hand only). Left always uses xyz_ee. Empty = use amazing_hand_arguments.xacro's own default."),
        ("rpy_ee_right", "Right-hand-only rotation override (amazing_hand only). Left always uses rpy_ee. Empty = use amazing_hand_arguments.xacro's own default."),
        ("connector_xyz_left", "Left-connector-only position override (amazing_hand only). Empty = use amazing_hand_arguments.xacro's own default."),
        ("connector_rpy_left", "Left-connector-only rotation override (amazing_hand only). Empty = use amazing_hand_arguments.xacro's own default."),
        ("connector_xyz_right", "Right-connector-only position override (amazing_hand only). Empty = use amazing_hand_arguments.xacro's own default."),
        ("connector_rpy_right", "Right-connector-only rotation override (amazing_hand only). Empty = use amazing_hand_arguments.xacro's own default."),
    ]:
        declared_arguments.append(DeclareLaunchArgument(_name, default_value="", description=_desc))

    declared_arguments.append(DeclareLaunchArgument("connector_xyz", default_value="0 0 0.0055", description="Shared connector plate position fallback (amazing_hand only). Overridden per side by connector_xyz_left/connector_xyz_right if those are set."))
    declared_arguments.append(DeclareLaunchArgument("connector_rpy", default_value="0 0 1.047", description="Shared connector plate rotation fallback (amazing_hand only). Overridden per side by connector_rpy_left/connector_rpy_right if those are set."))
    declared_arguments.append(DeclareLaunchArgument("connector_scale", default_value="0.001 0.001 0.001", description="Connector mesh scale, STL is authored in mm, URDF is meters (amazing_hand only)."))

    def optional_xacro_arg(name):
        """Forward `name:='value'` only if the launch arg is non-empty;
        otherwise contributes nothing, letting xacro's own <xacro:arg
        default=...> apply undisturbed."""
        lc = LaunchConfiguration(name)
        return PythonExpression(["'' if '", lc, "' == '' else ' ", name, ":=\\'' + '", lc, "' + '\\''"])

    arm_type = LaunchConfiguration("arm_type")
    ee_type = LaunchConfiguration("ee_type")
    bimanual = LaunchConfiguration("bimanual")
    hand = LaunchConfiguration("hand")
    mobile_base = LaunchConfiguration("mobile_base")
    mobile_base_arm_xyz = LaunchConfiguration("mobile_base_arm_xyz")
    mobile_base_arm_rpy = LaunchConfiguration("mobile_base_arm_rpy")
    mobile_base_body_xyz = LaunchConfiguration("mobile_base_body_xyz")
    mobile_base_body_rpy = LaunchConfiguration("mobile_base_body_rpy")
    xyz_ee = LaunchConfiguration("xyz_ee")
    rpy_ee = LaunchConfiguration("rpy_ee")
    connector_xyz = LaunchConfiguration("connector_xyz")
    connector_rpy = LaunchConfiguration("connector_rpy")
    connector_scale = LaunchConfiguration("connector_scale")
    xyz_ee_right_opt = optional_xacro_arg("xyz_ee_right")
    rpy_ee_right_opt = optional_xacro_arg("rpy_ee_right")
    connector_xyz_left_opt = optional_xacro_arg("connector_xyz_left")
    connector_rpy_left_opt = optional_xacro_arg("connector_rpy_left")
    connector_xyz_right_opt = optional_xacro_arg("connector_xyz_right")
    connector_rpy_right_opt = optional_xacro_arg("connector_rpy_right")

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("openarm_description"), "urdf", "robot", "v10.urdf.xacro"]),
            " ",
            "arm_type:=", arm_type,
            " ",
            "ee_type:=", ee_type,
            " ",
            "bimanual:=", bimanual,
            " ",
            "hand:=", hand,
            " ",
            "mobile_base:=", mobile_base,
            " ",
            "mobile_base_arm_xyz:='", mobile_base_arm_xyz, "'",
            " ",
            "mobile_base_arm_rpy:='", mobile_base_arm_rpy, "'",
            " ",
            "mobile_base_body_xyz:='", mobile_base_body_xyz, "'",
            " ",
            "mobile_base_body_rpy:='", mobile_base_body_rpy, "'",
            " ",
            "xyz_ee:='", xyz_ee, "'",
            " ",
            "rpy_ee:='", rpy_ee, "'",
            " ",
            "connector_xyz:='", connector_xyz, "'",
            " ",
            "connector_rpy:='", connector_rpy, "'",
            " ",
            "connector_scale:='", connector_scale, "'",
            xyz_ee_right_opt,
            rpy_ee_right_opt,
            connector_xyz_left_opt,
            connector_rpy_left_opt,
            connector_xyz_right_opt,
            connector_rpy_right_opt,
        ]
    )
    
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    rviz_config_file = PathJoinSubstitution([
        FindPackageShare("openarm_description"), 
        "rviz", 
        PythonExpression([
            "'mobile_base.rviz' if '", mobile_base, "' == 'true' else ",
            "('bimanual.rviz' if '", bimanual, "' == 'true' else 'arm_only.rviz')"
        ])
    ])

    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        parameters=[robot_description],
        remappings=[("/robot_description", "/robot_description_full")]
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
        remappings=[("/robot_description", "/robot_description_full")]
    )
    
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
    )

    nodes = [
        joint_state_publisher_node,
        robot_state_publisher_node,
        rviz_node,
    ]

    return LaunchDescription(declared_arguments + nodes)
