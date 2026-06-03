import os

# Đường dẫn chứa package openarm_description (thư mục CHA của openarm_description)
_src_path = "/home/huudung/project/universal_bot/Open_arm_a1_ws/src"
_install_share_path = "/home/huudung/project/universal_bot/Open_arm_a1_ws/install/openarm_description/share"

# Thêm cả src và install/share vào resource path để Gazebo tìm thấy mesh
_extra_paths = os.pathsep.join([_src_path, _install_share_path])

for env_var in ["GZ_SIM_RESOURCE_PATH", "IGN_GAZEBO_RESOURCE_PATH", "GZ_SIM_MODEL_PATH"]:
    if env_var in os.environ:
        os.environ[env_var] += os.pathsep + _extra_paths
    else:
        os.environ[env_var] = _extra_paths
# Automatically fix CycloneDDS buffer issues for large URDFs
os.environ["CYCLONEDDS_URI"] = "<CycloneDDS><Domain><General><MaxMessageSize>65535B</MaxMessageSize><FragmentSize>4000B</FragmentSize></General></Domain></CycloneDDS>"

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

def generate_launch_description():
    # Khai báo argument để người dùng lựa chọn: false (chạy Gazebo) hoặc true (chạy Fake Hardware)
    use_fake_hardware_arg = DeclareLaunchArgument(
        "use_fake_hardware",
        default_value="false",
        description="Run robot with fake/mock hardware (true) or in Gazebo Sim / Real Hardware (false)."
    )
    
    # Khai báo argument để người dùng chọn bật mô phỏng Gazebo hay chạy robot thật
    gazebo_arg = DeclareLaunchArgument(
        "gazebo",
        default_value="true",
        description="Run with Gazebo simulation (true) or Real Hardware (false)."
    )
    
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    gazebo = LaunchConfiguration("gazebo")
    
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
            "gazebo:=", gazebo, # <--- TRUYỀN BIẾN GAZBO SANG XACRO
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
        [FindPackageShare("arm_control"), "config", "bimanual_controllers.yaml"]
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
        parameters=[{"use_sim_time": use_sim_time}]
    )

    right_gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_gripper_controller"],
        parameters=[{"use_sim_time": use_sim_time}]
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
        parameters=[{"use_sim_time": use_sim_time}]
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
                base_controller_spawner,
            ],
        ),
        condition=IfCondition(run_standalone_control)
    )

    return LaunchDescription(
        [
            use_fake_hardware_arg,
            gazebo_arg,
            robot_state_publisher_node,
            # Gazebo
            gazebo_sim,
            spawn_robot_node,
            gz_bridge_node,
            load_controllers_event_gz,
            # Standalone (Fake & Real)
            ros2_control_node,
            load_controllers_event_standalone,
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