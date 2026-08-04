import os
import re
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

# Fix CycloneDDS buffer for large URDFs
if "CYCLONEDDS_URI" not in os.environ:
    os.environ["CYCLONEDDS_URI"] = "<CycloneDDS><Domain><General><MaxMessageSize>10MB</MaxMessageSize><FragmentSize>4000B</FragmentSize></General></Domain></CycloneDDS>"



def launch_setup(context, *args, **kwargs):
    moveit_config_pkg = get_package_share_directory("openarm_moveit_config")

    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    head = LaunchConfiguration("head")
    use_rviz = LaunchConfiguration("use_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_robot_skills = LaunchConfiguration("use_robot_skills")
    ee_type = LaunchConfiguration("ee_type")
    body_type = LaunchConfiguration("body_type")
    is_amazing_hand = IfCondition(PythonExpression(["'", ee_type, "' == 'amazing_hand'"]))
    is_openarm_hand = IfCondition(PythonExpression(["'", ee_type, "' == 'openarm_hand'"]))
    is_body_v2 = IfCondition(PythonExpression(["'", body_type, "' == 'v2'"]))

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
            "body_type:=",
            body_type,
            " ",
            "use_fake_hardware:=",
            use_fake_hardware,
            " ",
            "head_use_fake_hardware:=",
            PythonExpression(["'false' if '", head, "' == 'true' else 'true'"]),
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

    # The SRDF's <end_effector> tags hardcode group="left_gripper"/"right_gripper"
    # (the openarm_hand 2-finger groups). Under ee_type:=amazing_hand those groups'
    # joints don't exist in the URDF at all, so MoveIt logs "Group 'left_gripper'
    # not found in model" and RobotState::getRobotMarkers() segfaults trying to
    # render markers for the dangling end-effector. Point it at the finger groups
    # that actually exist for this ee_type instead.
    if LaunchConfiguration("ee_type").perform(context) == "amazing_hand":
        robot_description_semantic_content = (
            robot_description_semantic_content
            .replace('group="left_gripper" parent_group="left_arm"',
                      'group="left_hand_fingers" parent_group="left_arm"')
            .replace('group="right_gripper" parent_group="right_arm"',
                      'group="right_hand_fingers" parent_group="right_arm"')
        )
        # The rest of the SRDF (left_gripper/right_gripper groups, their open/close
        # group_states, and disable_collisions entries for openarm_*_hand/*_finger)
        # is written for ee_type:=openarm_hand's 2-finger gripper links, which don't
        # exist under amazing_hand. Harmless but noisy: "Group 'left_gripper' is
        # empty", "Joint '...finger_joint1' ... not known to the URDF", "Link
        # '...hand' is not known to URDF. Cannot disable/enable collisons." on every
        # move_group/robot_skills_node startup. Strip those blocks/lines here rather
        # than forking the SRDF file, since it's the same dead-under-this-ee_type
        # content each time.
        robot_description_semantic_content = re.sub(
            r'  <group name="(?:left|right)_gripper">.*?</group>\n\n',
            '', robot_description_semantic_content, flags=re.DOTALL)
        robot_description_semantic_content = re.sub(
            r'  <group_state name="(?:open|close)" group="(?:left|right)_gripper">.*?</group_state>\n\n',
            '', robot_description_semantic_content, flags=re.DOTALL)
        robot_description_semantic_content = '\n'.join(
            line for line in robot_description_semantic_content.split('\n')
            if not re.search(r'openarm_(?:left|right)_(?:hand"|left_finger|right_finger)', line)
        )

    # body_type:=v2 adds an articulated neck_joint/head_joint (see openarm_body.xacro);
    # v1/v10 don't have them. Neither is in any static SRDF group, so inject one here
    # only when it actually exists in the URDF -- a static group would make v1 log
    # "Joint 'openarm_body_neck_joint' ... not known to the URDF" on every startup.
    if LaunchConfiguration("body_type").perform(context) == "v2":
        robot_description_semantic_content = robot_description_semantic_content.replace(
            "  <!-- Virtual joints -->",
            '  <group name="head">\n'
            '    <joint name="openarm_body_neck_joint"/>\n'
            '    <joint name="openarm_body_head_joint"/>\n'
            '  </group>\n\n'
            "  <!-- Virtual joints -->"
        )
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
        [FindPackageShare("robot_control"), "config", "bimanual_controllers.yaml"]
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

    # body v2 articulated neck + head
    head_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["head_controller", "-c", "/controller_manager"],
        condition=is_body_v2,
    )

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
        condition=IfCondition(use_robot_skills),
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

    return [
        robot_state_publisher_node,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        left_arm_controller_spawner,
        right_arm_controller_spawner,
        left_gripper_controller_spawner,
        right_gripper_controller_spawner,
        *hand_controller_spawners,
        *hand_kinematics_nodes,
        head_controller_spawner,
        move_group_node,
        robot_skills_node,
        rviz_node,
    ]


def generate_launch_description():
    use_fake_hardware_arg = DeclareLaunchArgument(
        "use_fake_hardware",
        default_value="true",
        description="Whether to run with fake/mock hardware (true) or real hardware (false).",
    )
    head_arg = DeclareLaunchArgument(
        "head",
        default_value="false",
        description="Whether the head/neck board is physically present and wired (true) or "
                     "not (false, default). Decoupled from use_fake_hardware so arm-only rigs "
                     "(body_type:=v2 but no head board) don't need to fake the whole robot to "
                     "avoid HeadHW blocking on a socket that's never started.",
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
        default_value="amazing_hand",
        description="End-effector type: 'openarm_hand' (2-finger gripper) or "
                     "'amazing_hand' (default, 8-DOF finger hand, adds left_hand_fingers/right_hand_fingers "
                     "MoveIt groups and per-finger hand_kinematics_node + controllers).",
    )
    body_type_arg = DeclareLaunchArgument(
        "body_type",
        default_value="v2",
        description="Chassis/body version: 'v1' (single rigid base) or "
                     "'v2' (default, new chassis mesh with an articulated neck + head).",
    )
    use_robot_skills_arg = DeclareLaunchArgument(
        "use_robot_skills",
        default_value="true",
        description="Whether to launch robot_skills_node (true) or not (false). It runs its own "
                     "independent MoveItCpp instance -- full robot model load, planning scene "
                     "monitor, and FK/collision computation, duplicating move_group's. RViz's "
                     "Plan & Execute only talks to move_group, so set false for interactive "
                     "RViz-only sessions to roughly halve planning-side CPU load.",
    )

    return LaunchDescription(
        [
            use_fake_hardware_arg,
            head_arg,
            use_rviz_arg,
            use_sim_time_arg,
            ee_type_arg,
            body_type_arg,
            use_robot_skills_arg,
            OpaqueFunction(function=launch_setup),
        ]
    )
