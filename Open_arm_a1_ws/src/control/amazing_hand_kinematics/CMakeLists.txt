cmake_minimum_required(VERSION 3.8)
project(amazing_hand_kinematics)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(urdf REQUIRED)

add_library(amazing_hand_kinematics SHARED
  src/amazing_hand_kinematics.cpp
)
target_compile_features(amazing_hand_kinematics PRIVATE cxx_std_17)
target_include_directories(amazing_hand_kinematics PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)
ament_target_dependencies(amazing_hand_kinematics PUBLIC
  urdf
)

install(TARGETS amazing_hand_kinematics
  EXPORT export_amazing_hand_kinematics
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

ament_export_targets(export_amazing_hand_kinematics HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(amazing_hand_kinematics)
ament_export_dependencies(urdf Eigen3)

ament_package()
