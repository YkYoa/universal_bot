cmake_minimum_required(VERSION 3.8)
project(motion_planner)

find_package(ament_cmake REQUIRED)

# ── Install config files ──────────────────────────────────────────────────────
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
