cmake_minimum_required(VERSION 3.8)
project(gravity_compensation_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(orocos_kdl_vendor REQUIRED)
find_package(std_srvs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

add_library(gravity_compensation_controller SHARED
  src/gravity_compensation_controller.cpp
)
target_compile_features(gravity_compensation_controller PRIVATE cxx_std_17)
target_include_directories(gravity_compensation_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(gravity_compensation_controller
  controller_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  kdl_parser
  std_srvs
  std_msgs
)
target_link_libraries(gravity_compensation_controller orocos-kdl)

pluginlib_export_plugin_description_file(controller_interface gravity_compensation_controller.xml)

add_executable(gravity_calibration_tool src/gravity_calibration_tool.cpp)
target_compile_features(gravity_calibration_tool PRIVATE cxx_std_17)
ament_target_dependencies(gravity_calibration_tool
  rclcpp
  kdl_parser
  std_msgs
  sensor_msgs
)
target_link_libraries(gravity_calibration_tool orocos-kdl)

install(TARGETS gravity_compensation_controller
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS gravity_calibration_tool
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(include)
ament_export_libraries(gravity_compensation_controller)
ament_export_dependencies(
  controller_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  kdl_parser
  std_srvs
  std_msgs
)

ament_package()
