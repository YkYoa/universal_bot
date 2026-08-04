cmake_minimum_required(VERSION 3.8)
project(robot_skills)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(planning_interface REQUIRED)
find_package(motion_planner REQUIRED)
find_package(openarm_messages REQUIRED)

# ── Library ──────────────────────────────────────────────────────────────────
add_library(robot_skills SHARED
  src/skills/move_to_pose_skill.cpp
  src/skills/move_to_named_pose_skill.cpp
  src/skills/move_to_joint_skill.cpp
  src/skills/move_to_joint_sequence_skill.cpp
  src/skills/cartesian_move_skill.cpp
  src/skills/gripper_skill.cpp
  src/skill_server.cpp
)

target_include_directories(robot_skills PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(robot_skills
  rclcpp
  rclcpp_action
  geometry_msgs
  moveit_msgs
  moveit_core
  planning_interface
  motion_planner
  openarm_messages
)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(robot_skills_node
  src/robot_skills_node.cpp
)

target_link_libraries(robot_skills_node
  robot_skills
)

ament_target_dependencies(robot_skills_node
  rclcpp
  rclcpp_action
  geometry_msgs
  moveit_msgs
  moveit_core
  planning_interface
  motion_planner
  openarm_messages
)

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS robot_skills robot_skills_node
  EXPORT robot_skillsTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

# Export targets
ament_export_targets(robot_skillsTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  rclcpp_action
  geometry_msgs
  moveit_msgs
  moveit_core
  planning_interface
  motion_planner
  openarm_messages
)

ament_package()
