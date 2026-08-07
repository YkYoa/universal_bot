cmake_minimum_required(VERSION 3.8)
project(trajectory_waypoint_generator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(waypoint_recorder REQUIRED)
find_package(trajectory_shapes REQUIRED)

add_executable(generate_waypoint_sequence src/generate_waypoint_sequence.cpp)
target_compile_features(generate_waypoint_sequence PRIVATE cxx_std_17)
ament_target_dependencies(generate_waypoint_sequence
  rclcpp
  sensor_msgs
  geometry_msgs
  moveit_core
  moveit_ros_planning
  ament_index_cpp
  waypoint_recorder
  trajectory_shapes
)

# Conda's bin/ being on PATH makes CMake's config-mode search also consider
# CONDA_PREFIX, which can resolve resource_retriever's transitive libcurl
# dependency to conda's (ABI-incompatible) copy instead of the system one.
# Force the system libcurl explicitly to avoid an "undefined reference to
# curl_easy_init@CURL_OPENSSL_4" link failure.
if(EXISTS "/usr/lib/x86_64-linux-gnu/libcurl.so.4")
  target_link_libraries(generate_waypoint_sequence /usr/lib/x86_64-linux-gnu/libcurl.so.4)
endif()

add_executable(check_bimanual_collision src/check_bimanual_collision.cpp)
target_compile_features(check_bimanual_collision PRIVATE cxx_std_17)
ament_target_dependencies(check_bimanual_collision
  rclcpp
  moveit_core
  moveit_ros_planning
  ament_index_cpp
  waypoint_recorder
)
if(EXISTS "/usr/lib/x86_64-linux-gnu/libcurl.so.4")
  target_link_libraries(check_bimanual_collision /usr/lib/x86_64-linux-gnu/libcurl.so.4)
endif()

install(TARGETS generate_waypoint_sequence check_bimanual_collision
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
