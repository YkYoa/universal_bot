"""qvic_2026 project-specific Python: the sequence store behind the Android
teaching-pendant API.

Nothing here imports rclpy - the store is a plain SQLite library so the Flask
API process, the CLI, and unit tests can all use it without a ROS graph.
"""
