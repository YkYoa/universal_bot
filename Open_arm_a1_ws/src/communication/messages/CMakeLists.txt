cmake_minimum_required(VERSION 3.8)
project(openarm_messages)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FsmState.msg"
  "action/ExecuteSkill.action"
  "action/QueryVLA.action"
  "action/RunSequence.action"
  "srv/SetColor.srv"
  "srv/SetMode.srv"
  "srv/SetBrightness.srv"
  "srv/DriveCommand.srv"
  "srv/FsmCommand.srv"
  "srv/SceneCommand.srv"
  DEPENDENCIES geometry_msgs moveit_msgs builtin_interfaces
)

ament_package()
