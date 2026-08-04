cmake_minimum_required(VERSION 3.8)
project(motion_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(planning_interface REQUIRED)
find_package(yaml-cpp REQUIRED)

# ── Library ──────────────────────────────────────────────────────────────────
add_library(motion_planner SHARED
  src/moveit_cpp_planner_manager.cpp
)

target_include_directories(motion_planner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(motion_planner PRIVATE ${YAML_CPP_INCLUDE_DIRS})

target_link_libraries(motion_planner ${YAML_CPP_LIBRARIES})

ament_target_dependencies(motion_planner
  rclcpp
  geometry_msgs
  moveit_msgs
  moveit_core
  moveit_ros_planning
  planning_interface
  yaml-cpp
)

# ── Install targets and headers ──────────────────────────────────────────────
install(TARGETS motion_planner
  EXPORT motion_plannerTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Export targets
ament_export_targets(motion_plannerTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  geometry_msgs
  moveit_msgs
  moveit_core
  moveit_ros_planning
  planning_interface
  yaml-cpp
)

ament_package()
