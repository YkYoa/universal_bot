cmake_minimum_required(VERSION 3.8)
project(openarm_head_hand)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(common REQUIRED)
find_package(openarm_messages REQUIRED)
find_package(ament_index_cpp REQUIRED)

add_executable(openarm_head_hand_node src/openarm_head_hand_node.cpp)
target_include_directories(openarm_head_hand_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(openarm_head_hand_node yaml-cpp)
ament_target_dependencies(openarm_head_hand_node
  rclcpp
  rclcpp_action
  geometry_msgs
  std_msgs
  std_srvs
  common
  openarm_messages
  ament_index_cpp
)

install(TARGETS
  openarm_head_hand_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
