cmake_minimum_required(VERSION 3.8)
project(sequence_executor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
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

add_executable(sequence_executor_node
  src/sequence_executor_node.cpp
  src/sequence_yaml.cpp
  src/skill_client.cpp
  src/hand_gripper_client.cpp
  src/sequence_interpreter.cpp
)

target_include_directories(sequence_executor_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(sequence_executor_node ${DEPS})
target_link_libraries(sequence_executor_node yaml-cpp)

install(TARGETS sequence_executor_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
