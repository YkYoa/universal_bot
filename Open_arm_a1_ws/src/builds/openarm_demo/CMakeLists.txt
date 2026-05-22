cmake_minimum_required(VERSION 3.8)
project(openarm_demo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)
endif()

find_package(ament_cmake REQUIRED)
find_package(computation REQUIRED)

add_executable(sequence_to_bt src/sequence_to_bt.cpp)
ament_target_dependencies(sequence_to_bt
  computation
)

install(TARGETS
  sequence_to_bt
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
