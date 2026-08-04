import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _setup_gazebo_resource_paths():
    """Let Gazebo find meshes from both source and install trees."""
    try:
        desc_share = get_package_share_directory("openarm_description")
        ws_src = os.path.abspath(os.path.join(desc_share, "..", "..", "..", "..", "src"))
        extra_paths = os.pathsep.join([ws_src, desc_share])
        for env_var in ["GZ_SIM_RESOURCE_PATH", "IGN_GAZEBO_RESOURCE_PATH", "GZ_SIM_MODEL_PATH"]:
            if env_var in os.environ:
                os.environ[env_var] += os.pathsep + extra_paths
            else:
                os.environ[env_var] = extra_paths
    except Exception:
        pass


def generate_launch_description():
    _setup_gazebo_resource_paths()

    # CycloneDDS: large URDF / many joints
    if "CYCLONEDDS_URI" not in os.environ:
        os.environ["CYCLONEDDS_URI"] = (
            "<CycloneDDS><Domain><General>"
            "<MaxMessageSize>10MB</MaxMessageSize>"
            "<FragmentSize>4000B</FragmentSize>"
            "</General></Domain></CycloneDDS>"
        )

    # ── Bringup modes (pick ONE via launch args) ───────────────────────────────
    # 1) Simulation (Gazebo):  use_fake_hardware:=false gazebo:=true
    # 2) Software-only fake:   use_fake_hardware:=true  gazebo:=false
    # 3) Real hardware (CAN):  use_fake_hardware:=false gazebo:=false
    #    → uses robot_hardware_interface/OpenArm_v10HW per arm (can0/can1)
    # MoveIt launch files should include this file, not duplicate ros2_control.
    use_fake_hardware_arg = DeclareLaunchArgument(
        "use_fake_hardware",
        default_value="false",
        description="Run robot with fake/mock hardware (true) or in Gazebo Sim / Real Hardware (false)."
    )
    
    # Khai báo argument để người dùng chọn bật mô phỏng Gazebo hay chạy robot thật
    gazebo_arg = DeclareLaunchArgument(
        "gazebo",
        default_value="true",
        description="Run with Gazebo simulation (true) or Real Hardware (false).",
    )

    left_can_interface_arg = DeclareLaunchArgument(
        "left_can_interface",
        default_value="can1",
        description="SocketCAN interface for left arm (real hardware only).",
    )
    right_can_interface_arg = DeclareLaunchArgument(
        "right_can_interface",
        default_value="can0",
        description="SocketCAN interface for right arm (real hardware only).",
    )
    hand_arg = DeclareLaunchArgument(
        "hand",
        default_value="true",
        description="Enable end-effector joints in ros2_control.",
    )
    ee_type_arg = DeclareLaunchArgument(
        "ee_type",
        default_value="amazing_hand",
        description="End-effector type: openarm_hand (2-finger gripper) or amazing_hand.",
    )
    body_type_arg = DeclareLaunchArgument(
        "body_type",
        default_value="v1",
        description="Chassis/body version: 'v1' (single rigid base) or 'v2' (new chassis mesh with articulated neck + head).",
    )
    head_arg = DeclareLaunchArgument(
        "head",
        default_value="false",
        description="Whether the head/neck board is physically present and wired (true) or "
                     "not (false, default). Decoupled from use_fake_hardware so body_type:=v2 "
                     "with real arm hardware but no head board doesn't crash HeadHW trying to "
                     "activate a socket that's never started.",
    )

    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    head = LaunchConfiguration("head")
    gazebo = LaunchConfiguration("gazebo")
    left_can_interface = LaunchConfiguration("left_can_interface")
    right_can_interface = LaunchConfiguration("right_can_interface")
    hand = LaunchConfiguration("hand")
    ee_type = LaunchConfiguration("ee_type")
    body_type = LaunchConfiguration("body_type")
    
    # Tính toán động giá trị use_sim_time: 'true' nếu không chạy fake hardware VÀ chạy gazebo, ngược lại là 'false'
    # Để đơn giản, ta kiểm tra xem có chạy Gazebo không: nếu gazebo == 'true' và use_fake_hardware == 'false'
    # Ta dùng một PythonExpression để xử lý
    use_sim_time = PythonExpression([
        "'true' if '", gazebo, "' == 'true' and '", use_fake_hardware, "' == 'false' else 'false'"
    ])
    
    # Load URDF với ros2_control cấu hình động theo biến use_fake_hardware và gazebo
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
            "use_fake_hardware:=", use_fake_hardware,
            " ",
            "head_use_fake_hardware:=", PythonExpression(["'false' if '", head, "' == 'true' else 'true'"]),
            " ",
            "gazebo:=", gazebo,
            " ",
            "hand:=", hand,
            " ",
            "ee_type:=", ee_type,
            " ",
            "body_type:=", body_type,
            " ",
            "left_can_interface:=", left_can_interface,
            " ",
            "right_can_interface:=", right_can_interface,
            " ",
            "mobile_base:=true",
            " ",
            "mobile_base_xyz:='0 0 0.31'",
            " ",
            "mobile_base_body_xyz:='0 0 0'",
        ]
    )
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    # Điều kiện để chạy môi trường Gazebo mô phỏng
    run_gazebo = PythonExpression(["'", use_fake_hardware, "' == 'false' and '", gazebo, "' == 'true'"])
    
    # Điều kiện để chạy node điều khiển độc lập (Fake hardware OR Real hardware)
    run_standalone_control = PythonExpression(["'", use_fake_hardware, "' == 'true' or '", gazebo, "' == 'false'"])

    # Điều kiện chọn end-effector: openarm_hand (gripper 2 ngón) hoặc amazing_hand (5 ngón)
    run_openarm_hand = PythonExpression(["'", ee_type, "' == 'openarm_hand'"])
    run_amazing_hand = PythonExpression(["'", ee_type, "' == 'amazing_hand'"])
    run_body_v2 = PythonExpression(["'", body_type, "' == 'v2'"])

    # Node Robot State Publisher để phát khung xương TF lên topic /robot_description
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[
            robot_description,
            {"use_sim_time": use_sim_time} 
        ],
    )

    # ── MÔI TRƯỜNG GAZEBO (Chỉ chạy khi use_fake_hardware:=false và gazebo:=true) ──
    gazebo_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
        condition=IfCondition(run_gazebo)
    )

    spawn_robot_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'openarm_humanoid',
            '-allow_renaming', 'true',
            '-z', '0.35'
        ],
        output='screen',
        condition=IfCondition(run_gazebo)
    )
    
    gz_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
        ],
        output='screen',
        condition=IfCondition(run_gazebo)
    )

    # ── STANDALONE NODE (Chỉ chạy khi use_fake_hardware:=true hoặc gazebo:=false) ──
    controller_config = PathJoinSubstitution(
        [FindPackageShare("robot_control"), "config", "bimanual_controllers.yaml"]
    )
    
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            controller_config,
            {"use_sim_time": False}
        ],
        remappings=[
            ("~/robot_description", "/robot_description"),
        ],
        output="both",
        condition=IfCondition(run_standalone_control)
    )

    # ── ĐỊNH NGHĨA CÁC SPAWNER CONTROLLER ──
    
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        parameters=[{"use_sim_time": use_sim_time}]
    )

    left_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_arm_controller"],
        parameters=[{"use_sim_time": use_sim_time}]
    )

    right_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_arm_controller"],
        parameters=[{"use_sim_time": use_sim_time}]
    )

    left_gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_gripper_controller"],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(run_openarm_hand)
    )

    right_gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_gripper_controller"],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(run_openarm_hand)
    )

    # ── amazing_hand (ee_type:=amazing_hand): j1/j2 group controllers per side ──
    left_hand_j1_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_hand_j1_controller"],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(run_amazing_hand)
    )

    left_hand_j2_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_hand_j2_controller"],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(run_amazing_hand)
    )

    right_hand_j1_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_hand_j1_controller"],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(run_amazing_hand)
    )

    right_hand_j2_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_hand_j2_controller"],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(run_amazing_hand)
    )

    # hand_kinematics_node solves each amazing_hand's closed-loop linkage;
    # one instance per side, scoped via link_prefix/alias_prefix like
    # control_amazing_hand.launch.py, wired to that side's ros2_control
    # topics (openarm_robot.xacro: /left_ahand, /right_ahand).
    left_hand_kinematics_node = Node(
        package="openarm_description",
        executable="hand_kinematics_node.py",
        name="left_hand_kinematics_node",
        parameters=[
            robot_description,
            {
                "command_space": "knuckle",
                "link_prefix": "openarm_left_ahand_",
                "alias_prefix": "openarm_left_",
                "joint_commands_topic": "/left_ahand/joint_commands",
                "joint_states_topic": "/left_ahand/joint_states",
                "use_sim_time": use_sim_time,
            },
        ],
        output="both",
        condition=IfCondition(run_amazing_hand)
    )

    right_hand_kinematics_node = Node(
        package="openarm_description",
        executable="hand_kinematics_node.py",
        name="right_hand_kinematics_node",
        parameters=[
            robot_description,
            {
                "command_space": "knuckle",
                "link_prefix": "openarm_right_ahand_",
                "alias_prefix": "openarm_right_",
                "joint_commands_topic": "/right_ahand/joint_commands",
                "joint_states_topic": "/right_ahand/joint_states",
                "use_sim_time": use_sim_time,
            },
        ],
        output="both",
        condition=IfCondition(run_amazing_hand)
    )

    # body v2 articulated neck + head
    head_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["head_controller"],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(run_body_v2)
    )

    base_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["base_controller"],
        parameters=[{"use_sim_time": use_sim_time}]
    )
    
    static_tf_pub_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_pub_world_to_odom",
        arguments=["0", "0", "0", "0", "0", "0", "world", "odom"]
    )
    
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("openarm_description"), "rviz", "bimanual.rviz"]
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[{"use_sim_time": use_sim_time}],
        # bimanual.rviz's RobotModel display expects /robot_description_full
        # (matches display.launch.py's remap); this launch's robot_state_publisher
        # publishes plain /robot_description (controller_manager needs that name),
        # so remap RViz's subscription instead of the publisher.
        remappings=[("/robot_description_full", "/robot_description")]
    )

    # Quy trình kích hoạt các Spawner:
    # 1. Khi dùng Gazebo: Đợi spawn_robot_node hoàn tất rồi nạp controller.
    # 2. Khi dùng Standalone (Fake hoặc Real): Kích hoạt các spawner sau khi robot_state_publisher_node chạy xong.
    
    load_controllers_event_gz = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_robot_node,
            on_exit=[
                joint_state_broadcaster_spawner,
                left_arm_controller_spawner,
                right_arm_controller_spawner,
                left_gripper_controller_spawner,
                right_gripper_controller_spawner,
                left_hand_j1_controller_spawner,
                left_hand_j2_controller_spawner,
                right_hand_j1_controller_spawner,
                right_hand_j2_controller_spawner,
                head_controller_spawner,
                base_controller_spawner,
            ],
        ),
        condition=IfCondition(run_gazebo)
    )

    load_controllers_event_standalone = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=ros2_control_node,
            on_start=[
                joint_state_broadcaster_spawner,
                left_arm_controller_spawner,
                right_arm_controller_spawner,
                left_gripper_controller_spawner,
                right_gripper_controller_spawner,
                left_hand_j1_controller_spawner,
                left_hand_j2_controller_spawner,
                right_hand_j1_controller_spawner,
                right_hand_j2_controller_spawner,
                head_controller_spawner,
                base_controller_spawner,
            ],
        ),
        condition=IfCondition(run_standalone_control)
    )

    return LaunchDescription(
        [
            use_fake_hardware_arg,
            head_arg,
            gazebo_arg,
            left_can_interface_arg,
            right_can_interface_arg,
            hand_arg,
            ee_type_arg,
            body_type_arg,
            robot_state_publisher_node,
            # Gazebo
            gazebo_sim,
            spawn_robot_node,
            gz_bridge_node,
            load_controllers_event_gz,
            # Standalone (Fake & Real)
            ros2_control_node,
            load_controllers_event_standalone,
            # amazing_hand kinematics (no-op unless ee_type:=amazing_hand)
            left_hand_kinematics_node,
            right_hand_kinematics_node,
            # Static TF world -> odom
            static_tf_pub_node,
            # RViz
            rviz_node,
        ]
    )

    # foxglove_bridge_node = Node(
    #     package="foxglove_bridge",
    #     executable="foxglove_bridge",
    #     name="foxglove_bridge",
    #     output="screen",
    #     parameters=[{
    #         "port": 8765,
    #         "address": "0.0.0.0",
    #     }]
    # )