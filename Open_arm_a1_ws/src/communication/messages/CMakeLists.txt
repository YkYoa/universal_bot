cmake_minimum_required(VERSION 3.8)
project(openarm_messages)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ExecuteSkill.action"
  "action/QueryVLA.action"
  "srv/SetColor.srv"
  "srv/SetMode.srv"
  "srv/SetBrightness.srv"
  DEPENDENCIES geometry_msgs moveit_msgs
)

ament_package()
