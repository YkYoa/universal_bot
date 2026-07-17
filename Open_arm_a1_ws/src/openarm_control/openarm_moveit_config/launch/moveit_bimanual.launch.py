import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

# Fix CycloneDDS buffer for large URDFs
if "CYCLONEDDS_URI" not in os.environ:
    os.environ["CYCLONEDDS_URI"] = "<CycloneDDS><Domain><General><MaxMessageSize>10MB</MaxMessageSize><FragmentSize>4000B</FragmentSize></General></Domain></CycloneDDS>"



def generate_launch_description():
    moveit_config_pkg = get_package_share_directory("openarm_moveit_config")

    # Declare launch arguments
    use_fake_hardware_arg = DeclareLaunchArgument(
        "use_fake_hardware",
        default_value="true",
        description="Whether to run with fake/mock hardware (true) or real hardware (false).",
    )
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Whether to launch RViz (true) or not (false).",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Whether to use simulation clock (true) or not (false).",
    )
    ee_type_arg = DeclareLaunchArgument(
        "ee_type",
        default_value="openarm_hand",
        description="End-effector type: 'openarm_hand' (default 2-finger gripper) or "
                     "'amazing_hand' (8-DOF finger hand, adds left_hand_fingers/right_hand_fingers "
                     "MoveIt groups and per-finger hand_kinematics_node + controllers).",
    )

    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    use_rviz = LaunchConfiguration("use_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")
    ee_type = LaunchConfiguration("ee_type")
    is_amazing_hand = IfCondition(PythonExpression(["'", ee_type, "' == 'amazing_hand'"]))
    is_openarm_hand = IfCondition(PythonExpression(["'", ee_type, "' == 'openarm_hand'"]))

    # ── Robot Description (URDF) ──
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("openarm_description"), "urdf", "robot", "v10.urdf.xacro"]),
            " ",
            "bimanual:=true",
            " ",
            "ros2_control:=true",
            " ",
            "ee_type:=",
            ee_type,
            " ",
            "use_fake_hardware:=",
            use_fake_hardware,
            " ",
            "mobile_base:=true",
            " ",
            "mobile_base_xyz:='0 0 0.31'",
            " ",
            "mobile_base_body_xyz:='0 0 0'",
        ]
    )
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    # ── Semantic Robot Description (SRDF) ──
    srdf_path = os.path.join(moveit_config_pkg, "srdf", "openarm_bimanual.srdf")
    with open(srdf_path, "r") as f:
        robot_description_semantic_content = f.read()
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_content}

    # ── Kinematics ──
    kinematics_yaml_path = os.path.join(moveit_config_pkg, "config", "kinematics.yaml")
    
    # ── OMPL Planning ──
    ompl_planning_yaml_path = os.path.join(moveit_config_pkg, "config", "ompl_planning.yaml")

    # ── Pilz Industrial Planning ──
    pilz_planning_yaml_path = os.path.join(moveit_config_pkg, "config", "pilz_industrial_motion_planner_planning.yaml")

    # ── CHOMP Planning ──
    chomp_planning_yaml_path = os.path.join(moveit_config_pkg, "config", "chomp_planning.yaml")

    # ── STOMP Planning ──
    stomp_planning_yaml_path = os.path.join(moveit_config_pkg, "config", "stomp_planning.yaml")

    # ── Joint Limits ──
    joint_limits_yaml_path = os.path.join(moveit_config_pkg, "config", "joint_limits.yaml")

    # ── MoveIt Controller Manager ──
    moveit_controllers_yaml_path = os.path.join(moveit_config_pkg, "config", "moveit_controllers.yaml")

    # ── ros2_control controllers ──
    controller_config = PathJoinSubstitution(
        [FindPackageShare("arm_control"), "config", "bimanual_controllers.yaml"]
    )

    # ── Trajectory execution ──
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

    bounds_tolerances = {
        "start_state_max_bounds_error": 2.0,
    }

    # ─────────────────────────────────────────────
    # Nodes
    # ─────────────────────────────────────────────

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[controller_config, {"use_sim_time": use_sim_time}],
        remappings=[
            ("~/robot_description", "/robot_description"),
        ],
        output="both",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    left_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_arm_controller", "-c", "/controller_manager"],
    )

    right_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_arm_controller", "-c", "/controller_manager"],
    )

    left_gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_gripper_controller", "-c", "/controller_manager"],
        condition=is_openarm_hand,
    )

    right_gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_gripper_controller", "-c", "/controller_manager"],
        condition=is_openarm_hand,
    )

    # ── amazing_hand: controllers + per-side kinematics solver ──
    # (ee_type:=amazing_hand only; see hand_kinematics_node.py for what it does)
    hand_controller_spawners = [
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[name, "-c", "/controller_manager"],
            condition=is_amazing_hand,
        )
        for name in ("left_hand_j1_controller", "left_hand_j2_controller",
                     "right_hand_j1_controller", "right_hand_j2_controller")
    ]

    hand_kinematics_nodes = [
        Node(
            package="openarm_description",
            executable="hand_kinematics_node.py",
            name=f"hand_kinematics_node_{side}",
            parameters=[
                robot_description,
                {
                    "command_space": "knuckle",
                    "link_prefix": f"openarm_{side}_ahand_",
                    "alias_prefix": f"openarm_{side}_",
                    "joint_commands_topic": f"/{side}_ahand/joint_commands",
                    "joint_states_topic": f"/{side}_ahand/joint_states",
                    "use_sim_time": use_sim_time,
                },
            ],
            condition=is_amazing_hand,
        )
        for side in ("left", "right")
    ]

    # ── MoveIt Move Group Node ──
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
            moveit_controllers_yaml_path,
            trajectory_execution,
            planning_scene_monitor,
            bounds_tolerances,
            {"use_sim_time": use_sim_time},
        ],
    )

    # ── Robot Skills Node (MoveItCpp in-process backend) ──
    motion_planner_pkg = get_package_share_directory("motion_planner")
    moveit_cpp_yaml_path = os.path.join(motion_planner_pkg, "config", "moveit_cpp.yaml")
    robot_skills_node = Node(
        package="robot_skills",
        executable="robot_skills_node",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml_path,
            moveit_cpp_yaml_path,
            moveit_controllers_yaml_path,
            joint_limits_yaml_path,
            trajectory_execution,
            planning_scene_monitor,
            bounds_tolerances,
            {"use_sim_time": use_sim_time},
        ],
    )

    # ── RViz with MoveIt plugin ──
    rviz_config_file = os.path.join(moveit_config_pkg, "config", "moveit.rviz")
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
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        [
            use_fake_hardware_arg,
            use_rviz_arg,
            use_sim_time_arg,
            ee_type_arg,
            robot_state_publisher_node,
            ros2_control_node,
            joint_state_broadcaster_spawner,
            left_arm_controller_spawner,
            right_arm_controller_spawner,
            left_gripper_controller_spawner,
            right_gripper_controller_spawner,
            *hand_controller_spawners,
            *hand_kinematics_nodes,
            move_group_node,
            robot_skills_node,
            rviz_node,
        ]
    )
