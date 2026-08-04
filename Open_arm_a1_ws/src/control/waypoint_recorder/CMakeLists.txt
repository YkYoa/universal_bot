cmake_minimum_required(VERSION 3.8)
project(waypoint_recorder)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_library(waypoint_recorder SHARED
  src/waypoint_recorder.cpp
)
target_compile_features(waypoint_recorder PRIVATE cxx_std_17)
target_include_directories(waypoint_recorder PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(waypoint_recorder
  rclcpp
  sensor_msgs
)

add_executable(record_waypoint src/record_waypoint_cli.cpp)
target_compile_features(record_waypoint PRIVATE cxx_std_17)
target_link_libraries(record_waypoint waypoint_recorder)
ament_target_dependencies(record_waypoint rclcpp)

install(TARGETS waypoint_recorder
  EXPORT export_waypoint_recorder
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS record_waypoint
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_export_targets(export_waypoint_recorder HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(waypoint_recorder)
ament_export_dependencies(rclcpp sensor_msgs)

ament_package()
