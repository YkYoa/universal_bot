cmake_minimum_required(VERSION 3.8)
project(planner_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)

# ── Shared library ────────────────────────────────────────────────────────────
add_library(planner_interface SHARED
  src/planner_profile_loader.cpp
)

target_include_directories(planner_interface PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(planner_interface
  rclcpp
  ament_index_cpp
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS planner_interface
  EXPORT planner_interfaceTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

# ── Export for downstream packages (bt_executor) ──────────────────────────────
ament_export_targets(planner_interfaceTargets HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(planner_interface)
ament_export_dependencies(rclcpp ament_index_cpp)

ament_package()
