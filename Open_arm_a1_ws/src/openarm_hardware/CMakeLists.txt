cmake_minimum_required(VERSION 3.8)
project(openarm_hardware)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)

# We can compile hardware_interface.cpp if it has code, otherwise skip it.
# Since it is currently 0 bytes, we just create a header-only or dummy target to keep CMake happy.
# Let's check if the file is empty. Yes, it's empty, so we won't build a library yet.

ament_package()
