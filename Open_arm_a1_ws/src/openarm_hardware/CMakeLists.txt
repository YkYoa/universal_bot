cmake_minimum_required(VERSION 3.16)
project(openarm_hardware)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

# Optional: OpenArmCAN (openarm_can library). We keep this optional so
# the workspace can build on machines without the hardware stack installed.
find_package(OpenArmCAN QUIET)

add_library(${PROJECT_NAME} SHARED
  src/v10/hardware_interface.cpp
)

target_compile_definitions(${PROJECT_NAME} PRIVATE OPENARM_HARDWARE_BUILDING_DLL)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
)

if(OpenArmCAN_FOUND)
  target_link_libraries(${PROJECT_NAME} OpenArmCAN::openarm_can)
  target_compile_definitions(${PROJECT_NAME} PRIVATE OPENARM_HARDWARE_HAS_OPENARMCAN=1)
else()
  message(WARNING "OpenArmCAN not found. openarm_hardware will build in stub mode (no real CAN). Install libopenarm-can-dev to enable real hardware.")
endif()

pluginlib_export_plugin_description_file(hardware_interface openarm_hardware.xml)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(hardware_interface pluginlib rclcpp rclcpp_lifecycle)

ament_package()
