cmake_minimum_required(VERSION 3.8)
project(trajectory_shapes)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(trajectory_shapes SHARED
  src/trajectory_shapes.cpp
)
target_compile_features(trajectory_shapes PRIVATE cxx_std_17)
target_include_directories(trajectory_shapes PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)

install(TARGETS trajectory_shapes
  EXPORT export_trajectory_shapes
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

ament_export_targets(export_trajectory_shapes HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(trajectory_shapes)

ament_package()
