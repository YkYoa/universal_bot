#include "sequence_executor/control_mode_probe.hpp"

#include <chrono>
#include <fstream>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <controller_manager_msgs/srv/list_controllers.hpp>
#include <yaml-cpp/yaml.h>

namespace sequence_executor {

namespace {

constexpr const char* kUnknown = "unknown";

// Substrings, not exact names: bringup spawns left_/right_ variants and the
// naming has changed once already (see bimanual_controllers.yaml).
constexpr const char* kGravityCompMarker = "gravity_comp";
constexpr const char* kArmControllerMarker = "arm_controller";

std::string resolveDefaultConfigPath()
{
  try {
    return ament_index_cpp::get_package_share_directory("robot_hardware_interface") +
           "/config/hardware_config.yaml";
  } catch (const std::exception&) {
    return {};
  }
}

}  // namespace

ControlModeProbe::ControlModeProbe(rclcpp::Node::SharedPtr node,
                                   const std::string& hardware_config_path)
  : node_(std::move(node)),
    hardware_config_path_(hardware_config_path.empty() ? resolveDefaultConfigPath()
                                                       : hardware_config_path),
    logger_(node_->get_logger())
{
}

std::string ControlModeProbe::mode() const
{
  return cached_.empty() ? kUnknown : cached_;
}

void ControlModeProbe::probe()
{
  if (!cached_.empty()) {
    return;
  }

  const std::string declared = modeFromConfig();
  const std::string running = modeFromControllers();

  if (!running.empty() && !declared.empty() && running != declared) {
    // Not fatal, but worth shouting about: it usually means hardware_config.yaml
    // was edited after bringup, so the file no longer describes the robot.
    RCLCPP_WARN(logger_,
                "control_mode mismatch: hardware_config.yaml says '%s' but the active "
                "controllers look like '%s' - trusting the controllers",
                declared.c_str(), running.c_str());
  }

  if (!running.empty()) {
    cached_ = running;
  } else if (!declared.empty()) {
    cached_ = declared;
  } else {
    cached_ = kUnknown;
    RCLCPP_WARN(logger_,
                "could not determine control mode (no hardware_config.yaml at '%s', no "
                "controller_manager) - step control-mode checks will not be enforced",
                hardware_config_path_.c_str());
  }

  RCLCPP_INFO(logger_, "Active control mode: %s", cached_.c_str());
}

std::string ControlModeProbe::modeFromConfig() const
{
  if (hardware_config_path_.empty()) {
    return {};
  }
  std::ifstream file(hardware_config_path_);
  if (!file.good()) {
    return {};
  }
  file.close();

  try {
    YAML::Node root = YAML::LoadFile(hardware_config_path_);
    if (root && root["control_mode"]) {
      return root["control_mode"].as<std::string>();
    }
  } catch (const YAML::Exception& e) {
    RCLCPP_WARN(logger_, "could not read control_mode from %s: %s",
                hardware_config_path_.c_str(), e.what());
  }
  return {};
}

std::string ControlModeProbe::modeFromControllers()
{
  using ListControllers = controller_manager_msgs::srv::ListControllers;

  // Held as a member across calls: a client destroyed while a request is still
  // in flight leaves rcl writing the late response into freed memory.
  if (!controller_client_) {
    controller_client_ = node_->create_client<ListControllers>("/controller_manager/list_controllers");
  }
  if (!controller_client_->wait_for_service(std::chrono::seconds(2))) {
    return {};
  }

  auto future = controller_client_->async_send_request(std::make_shared<ListControllers::Request>());
  if (rclcpp::spin_until_future_complete(node_, future.future, std::chrono::seconds(3)) !=
      rclcpp::FutureReturnCode::SUCCESS) {
    controller_client_->remove_pending_request(future);
    return {};
  }

  // Hold the response in a local shared_ptr rather than iterating straight off
  // future.get(), and check it: a ready-but-empty future is what a second
  // controller_manager on the same domain (a leftover from a previous launch)
  // produces, and dereferencing it walks freed memory.
  auto response = future.future.get();
  if (!response) {
    RCLCPP_WARN(logger_, "list_controllers returned an empty response");
    return {};
  }

  bool gravity_comp_active = false;
  bool arm_controller_active = false;
  for (const auto& controller : response->controller) {
    if (controller.state != "active") {
      continue;
    }
    if (controller.name.find(kGravityCompMarker) != std::string::npos) {
      gravity_comp_active = true;
    } else if (controller.name.find(kArmControllerMarker) != std::string::npos) {
      arm_controller_active = true;
    }
  }

  // Gravity comp claims effort command interfaces, which only do anything when
  // the hardware is in torque mode - so seeing it active is proof of torque,
  // and it wins over an arm controller that may merely be loaded alongside.
  if (gravity_comp_active) {
    return "torque";
  }
  if (arm_controller_active) {
    // The arm controller commands positions, which the hardware accepts in
    // both "position" and "mit" - the controller list cannot tell those apart,
    // so defer to the config file for the finer distinction.
    const std::string declared = modeFromConfig();
    return (declared == "mit" || declared == "position") ? declared : "position";
  }
  return {};
}

}  // namespace sequence_executor
