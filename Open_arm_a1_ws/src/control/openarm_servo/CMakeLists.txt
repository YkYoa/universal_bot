cmake_minimum_required(VERSION 3.8)
project(openarm_servo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(control_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)

set(DEPS
  rclcpp
  control_msgs
  geometry_msgs
  moveit_msgs
)

add_library(openarm_servo SHARED
  src/servo_client.cpp
)

target_include_directories(openarm_servo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(openarm_servo ${DEPS})

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS openarm_servo
  EXPORT export_openarm_servo
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_targets(export_openarm_servo HAS_LIBRARY_TARGET)
ament_export_dependencies(${DEPS})

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
