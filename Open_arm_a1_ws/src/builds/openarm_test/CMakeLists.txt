cmake_minimum_required(VERSION 3.8)
project(openarm_test)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

add_executable(test_realtime_node src/test_realtime_node.cpp)
ament_target_dependencies(test_realtime_node
  rclcpp
  geometry_msgs
  std_msgs
  std_srvs
)

install(TARGETS
  test_realtime_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
