from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
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
            description="true: joint_state_publisher_gui with sliders for the 8 real SCS0009 servo "
                         "joints (revolute_5[_1..3], revolute_6[_1..3]) -- one flex pair per finger. "
                         "false: headless, all joints held at 0 (home/assembled pose). "
                         "The other 76 movable joints are the passive rod/gimbal linkage: they are "
                         "locked at the assembled rest pose (see config/hand/amazing_hand/actuators_only.yaml) "
                         "since this URDF does not solve the closed-loop linkage kinematics.",
        ),
    ]

    side = LaunchConfiguration("side")
    use_gui = LaunchConfiguration("use_gui")

    urdf_file = PythonExpression(["'r_ahand.urdf' if '", side, "' == 'right' else 'l_ahand.urdf'"])

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("openarm_description"), "urdf", "ee", "amazing_hand", urdf_file]),
        ]
    )

    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    actuators_only_params = PathJoinSubstitution([
        FindPackageShare("openarm_description"),
        "config", "hand", "amazing_hand", "actuators_only.yaml",
    ])

    rviz_config_file = PathJoinSubstitution([
        FindPackageShare("openarm_description"),
        "rviz",
        "amazing_hand.rviz",
    ])

    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        parameters=[robot_description, actuators_only_params],
        condition=IfCondition(use_gui),
    )

    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        parameters=[robot_description, actuators_only_params],
        condition=UnlessCondition(use_gui),
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
        joint_state_publisher_gui_node,
        joint_state_publisher_node,
        robot_state_publisher_node,
        rviz_node,
    ]

    return LaunchDescription(declared_arguments + nodes)
