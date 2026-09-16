cmake_minimum_required(VERSION 3.8)
project(common)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)
endif()

find_package(ament_cmake REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(rclcpp REQUIRED)

# Declare a header-only library
#
# joint_names.hpp includes <sensor_msgs/msg/joint_state.hpp> but sensor_msgs
# is deliberately NOT declared/linked here: exporting an INTERFACE library's
# link against another ament package's imported target (sensor_msgs::
# sensor_msgs) broke across machines once already (worked on one dev
# checkout, failed with "the target was not found" on IQ9075's - see
# commonTargetsExport.cmake's set_target_properties error, 2026-09-16).
# Every consumer of joint_names.hpp declares sensor_msgs itself instead.
add_library(common INTERFACE)

target_include_directories(common INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(common INTERFACE yaml-cpp rclcpp::rclcpp)

# Export the library target and include directory
ament_export_targets(commonTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(yaml-cpp rclcpp)
ament_export_include_directories(include)

install(DIRECTORY include/
  DESTINATION include
)

# Export the interface library
install(TARGETS common
  EXPORT commonTargets
)

ament_package()
