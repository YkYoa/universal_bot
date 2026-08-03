cmake_minimum_required(VERSION 3.8)
project(communication_devices)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(openarm_messages REQUIRED)
find_package(control_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(head_motor_driver_node head_motor_driver/head_motor_driver_node.cpp)
target_include_directories(head_motor_driver_node PRIVATE head_motor_driver)
target_compile_features(head_motor_driver_node PUBLIC cxx_std_17)
ament_target_dependencies(head_motor_driver_node ament_index_cpp)
target_link_libraries(head_motor_driver_node yaml-cpp)

add_executable(head_led_node head_led/head_led_node.cpp)
target_compile_features(head_led_node PUBLIC cxx_std_17)
ament_target_dependencies(head_led_node rclcpp openarm_messages)

add_executable(head_home_node head_home/head_home_node.cpp)
target_compile_features(head_home_node PUBLIC cxx_std_17)
ament_target_dependencies(head_home_node rclcpp rclcpp_action control_msgs std_srvs trajectory_msgs)

install(TARGETS
  head_motor_driver_node
  head_led_node
  head_home_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY head_motor_driver/config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
