cmake_minimum_required(VERSION 3.8)
project(bt_executor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(behaviortree_ros2 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(planning_interface REQUIRED)
find_package(openarm_messages REQUIRED)

set(DEPS
  behaviortree_cpp
  behaviortree_ros2
  rclcpp
  rclcpp_action
  rclcpp_components
  std_msgs
  std_srvs
  geometry_msgs
  sensor_msgs
  control_msgs
  trajectory_msgs
  moveit_ros_planning_interface
  planning_interface
  openarm_messages
)

# ── bt_executor node ──────────────────────────────────────────────────────────
add_executable(bt_executor_node
  src/bt_executor_node.cpp
  src/nodes/conditions/is_object_visible.cpp
  src/nodes/conditions/is_gripper_holding.cpp
  src/nodes/conditions/is_arm_at_pose.cpp
  src/nodes/conditions/is_replan_needed.cpp
  src/nodes/conditions/is_goal_changed.cpp
  src/nodes/actions/plan_to_joint_target.cpp
  src/nodes/actions/plan_to_joint_sequence.cpp
  src/nodes/actions/plan_to_named_pose.cpp
  src/nodes/actions/plan_to_pose.cpp
  src/nodes/actions/execute_trajectory.cpp
  src/nodes/actions/control_gripper.cpp
  src/nodes/actions/query_vla.cpp
  src/nodes/actions/safe_abort.cpp
  src/nodes/actions/set_planner_config.cpp
)

target_include_directories(bt_executor_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(bt_executor_node ${DEPS})

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS bt_executor_node
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/vla_bridge.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include
)

install(DIRECTORY
  launch
  config
  bt_trees
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
