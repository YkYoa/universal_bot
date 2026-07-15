from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("side", default_value="right", description="Which hand to show: right or left."),
        DeclareLaunchArgument(
            "use_gui",
            default_value="true",
            description="true: joint_state_publisher_gui with one slider per command joint "
                         "(see command_space). false: headless, all joints held at 0. "
                         "All non-command movable joints are hidden from the sliders and, when "
                         "solve_linkage:=true, computed by hand_kinematics_node.",
        ),
        DeclareLaunchArgument(
            "solve_linkage",
            default_value="true",
            description="true: hand_kinematics_node solves each finger's closed-loop rod/gimbal linkage. "
                         "false: passive joints stay locked at the assembled rest pose; "
                         "use this if hand_kinematics_node fails to start (e.g. missing scipy).",
        ),
        DeclareLaunchArgument(
            "command_space",
            default_value="knuckle",
            description="knuckle: sliders command each finger's knuckles -- yaw/abduction "
                         "(revolute_1[_1,_2,_4]) and proximal flexion (revolute_2[_1..3]); "
                         "hand_kinematics_node solves the servo angles (inverse kinematics, published "
                         "in /joint_states) and the passive linkage; the distal knuckle is mechanically "
                         "coupled and follows. "
                         "servo: sliders command the 8 SCS0009 horn angles directly (forward mode).",
        ),
    ]

    side = LaunchConfiguration("side")
    use_gui = LaunchConfiguration("use_gui")
    solve_linkage = LaunchConfiguration("solve_linkage")
    command_space = LaunchConfiguration("command_space")

    urdf_file = PythonExpression(["'r_ahand.urdf' if '", side, "' == 'right' else 'l_ahand.urdf'"])

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("openarm_description"), "urdf", "ee", "amazing_hand", urdf_file]),
        ]
    )

    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    sliders_yaml = PythonExpression(
        ["'knuckles_only.yaml' if '", command_space, "' == 'knuckle' else 'actuators_only.yaml'"])
    slider_joints_params = PathJoinSubstitution([
        FindPackageShare("openarm_description"),
        "config", "hand", "amazing_hand", sliders_yaml,
    ])

    rviz_config_file = PathJoinSubstitution([
        FindPackageShare("openarm_description"),
        "rviz",
        "amazing_hand.rviz",
    ])

    gui_and_solve = IfCondition(PythonExpression(["'", use_gui, "' == 'true' and '", solve_linkage, "' == 'true'"]))
    gui_and_not_solve = IfCondition(PythonExpression(["'", use_gui, "' == 'true' and '", solve_linkage, "' == 'false'"]))
    nogui_and_solve = IfCondition(PythonExpression(["'", use_gui, "' == 'false' and '", solve_linkage, "' == 'true'"]))
    nogui_and_not_solve = IfCondition(PythonExpression(["'", use_gui, "' == 'false' and '", solve_linkage, "' == 'false'"]))

    joint_state_publisher_gui_solved = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        parameters=[robot_description, slider_joints_params],
        remappings=[("joint_states", "joint_states_raw")],
        condition=gui_and_solve,
    )

    joint_state_publisher_gui_plain = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        parameters=[robot_description, slider_joints_params],
        condition=gui_and_not_solve,
    )

    joint_state_publisher_solved = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        parameters=[robot_description, slider_joints_params],
        remappings=[("joint_states", "joint_states_raw")],
        condition=nogui_and_solve,
    )

    joint_state_publisher_plain = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        parameters=[robot_description, slider_joints_params],
        condition=nogui_and_not_solve,
    )

    hand_kinematics_node = Node(
        package="openarm_description",
        executable="hand_kinematics_node.py",
        name="hand_kinematics_node",
        parameters=[robot_description, {"command_space": command_space}],
        condition=IfCondition(solve_linkage),
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
    )

    nodes = [
        joint_state_publisher_gui_solved,
        joint_state_publisher_gui_plain,
        joint_state_publisher_solved,
        joint_state_publisher_plain,
        hand_kinematics_node,
        robot_state_publisher_node,
        rviz_node,
    ]

    return LaunchDescription(declared_arguments + nodes)
