cmake_minimum_required(VERSION 3.8)
project(qvic_2026)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sequence_executor REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SQLite3 REQUIRED IMPORTED_TARGET sqlite3)

# ── Python: the sequence store ───────────────────────────────────────────────
# A plain SQLite library with no ROS imports, so the Flask API process, the
# CLI, and tests can all use it without a running graph.
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/sequence_store_cli.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── C++: this project's executor ─────────────────────────────────────────────
add_executable(qvic_fsm_node
  src/qvic_fsm_node.cpp
  src/qvic_actions.cpp
  src/sqlite_sequence_source.cpp
)

target_include_directories(qvic_fsm_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

ament_target_dependencies(qvic_fsm_node rclcpp sequence_executor)
target_link_libraries(qvic_fsm_node PkgConfig::SQLite3 yaml-cpp)

install(TARGETS qvic_fsm_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
