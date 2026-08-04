cmake_minimum_required(VERSION 3.8)
project(planning_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(ament_index_cpp REQUIRED)

# ── Library ──────────────────────────────────────────────────────────────────
add_library(planning_interface SHARED
  src/planner_profile.cpp
  src/planner_request_response.cpp
)

target_include_directories(planning_interface PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(planning_interface PRIVATE ${YAML_CPP_INCLUDE_DIRS})
target_link_libraries(planning_interface ${YAML_CPP_LIBRARIES} moveit_core::moveit_robot_state)


ament_target_dependencies(planning_interface
  rclcpp
  geometry_msgs
  moveit_msgs
  moveit_core
  yaml-cpp
  ament_index_cpp
)

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS planning_interface
  EXPORT planning_interfaceTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

# Export targets
ament_export_targets(planning_interfaceTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  geometry_msgs
  moveit_msgs
  moveit_core
  yaml-cpp
  ament_index_cpp
)

ament_package()
