cmake_minimum_required(VERSION 3.8)
project(arm_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(get_robot_state src/get_robot_state.cpp)
target_compile_features(get_robot_state PRIVATE cxx_std_17)
ament_target_dependencies(get_robot_state
  rclcpp
  sensor_msgs
  tf2_ros
  geometry_msgs
)

install(TARGETS get_robot_state
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/bimanual_api.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
