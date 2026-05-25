cmake_minimum_required(VERSION 3.8)
project(computation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)
endif()

find_package(ament_cmake REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(Eigen3 REQUIRED)

set(SRC_FILES
  src/computation.cpp
  src/conversion.cpp
  src/sequence_converter.cpp
)

set(DEPENDS
  ament_cmake
  tf2_ros
)

add_library(computation SHARED ${SRC_FILES})

target_include_directories(computation PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(computation PRIVATE yaml-cpp)
target_link_libraries(computation PUBLIC Eigen3::Eigen)

ament_target_dependencies(computation PUBLIC
  geometry_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
)

ament_export_targets(computationTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  geometry_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  Eigen3
)
ament_export_include_directories(include)
ament_export_libraries(computation)

install(TARGETS computation
  EXPORT computationTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
