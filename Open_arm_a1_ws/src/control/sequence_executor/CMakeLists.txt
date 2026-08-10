cmake_minimum_required(VERSION 3.8)
project(sequence_executor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(controller_manager_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(planning_interface REQUIRED)
find_package(openarm_messages REQUIRED)

set(DEPS
  ament_index_cpp
  rclcpp
  rclcpp_action
  std_msgs
  control_msgs
  trajectory_msgs
  sensor_msgs
  controller_manager_msgs
  planning_interface
  openarm_messages
)

# ── Library ──────────────────────────────────────────────────────────────────
# Exported so a project package can link its own executable against it -
# builds/qvic_2026 does exactly that to add its SQLite source and its
# hardcoded actions.
add_library(sequence_executor SHARED
  src/sequence_step.cpp
  src/step_parser.cpp
  src/sequence_yaml.cpp
  src/yaml_sequence_source.cpp
  src/skill_client.cpp
  src/hand_gripper_client.cpp
  src/scene_client.cpp
  src/control_mode_probe.cpp
  src/builtin_actions.cpp
  src/sequence_fsm.cpp
  src/robot_supervisor.cpp
  src/executor_app.cpp
)

target_include_directories(sequence_executor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(sequence_executor ${DEPS})
target_link_libraries(sequence_executor yaml-cpp)

# ── Executable ───────────────────────────────────────────────────────────────
add_executable(sequence_executor_node src/sequence_executor_node.cpp)
target_link_libraries(sequence_executor_node sequence_executor)
ament_target_dependencies(sequence_executor_node ${DEPS})

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS sequence_executor
  EXPORT sequence_executorTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS sequence_executor_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_targets(sequence_executorTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(${DEPS} yaml-cpp)

ament_package()
