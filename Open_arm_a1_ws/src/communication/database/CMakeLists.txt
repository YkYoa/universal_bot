cmake_minimum_required(VERSION 3.8)
project(database_bridge)

# Find ROS 2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

# Install Python scripts
install(PROGRAMS
  scripts/mqtt_listener.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
