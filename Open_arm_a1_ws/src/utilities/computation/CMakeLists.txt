cmake_minimum_required(VERSION 3.8)
project(computation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

add_library(computation SHARED
  src/conversion.cpp
  src/computation.cpp
)

# Also create computation.cpp since we included it, but leave it empty or minimal
target_include_directories(computation PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_export_targets(computationTargets HAS_LIBRARY_TARGET)
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
