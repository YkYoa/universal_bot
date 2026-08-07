cmake_minimum_required(VERSION 3.8)
project(robot_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(openarm_messages REQUIRED)
find_package(amazing_hand_kinematics REQUIRED)

# Optional: OpenArmCAN (openarm_can library), same pattern as
# robot_hardware_interface - keeps the workspace buildable on machines
# without the hardware stack installed (drive_control builds in stub mode).
find_package(OpenArmCAN QUIET)

add_executable(get_robot_state
  src/get_robot_state.cpp
  src/get_robot_state_node.cpp
)
target_compile_features(get_robot_state PRIVATE cxx_std_17)
target_include_directories(get_robot_state PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(get_robot_state
  rclcpp
  sensor_msgs
  tf2_ros
  geometry_msgs
)

add_executable(control_server
  src/control_server.cpp
  src/drive_control.cpp
)
target_compile_features(control_server PRIVATE cxx_std_17)
target_include_directories(control_server PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(control_server
  rclcpp
  openarm_messages
)
if(OpenArmCAN_FOUND)
  target_link_libraries(control_server OpenArmCAN::openarm_can)
  target_compile_definitions(control_server PRIVATE ROBOT_CONTROL_HAS_OPENARMCAN=1)
else()
  message(WARNING "OpenArmCAN not found. control_server/drive_control will build in stub mode (no real CAN). Install libopenarm-can-dev to enable real hardware.")
endif()

add_executable(control_client
  src/control_client.cpp
)
target_compile_features(control_client PRIVATE cxx_std_17)
target_include_directories(control_client PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(control_client
  rclcpp
  openarm_messages
)

add_executable(hand_kinematic
  src/hand_kinematic.cpp
)
target_compile_features(hand_kinematic PRIVATE cxx_std_17)
target_include_directories(hand_kinematic PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(hand_kinematic
  rclcpp
  sensor_msgs
  amazing_hand_kinematics
)

install(TARGETS get_robot_state control_server control_client hand_kinematic
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/bimanual_api.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
