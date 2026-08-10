#include "sequence_executor/executor_app.hpp"

#include <chrono>
#include <thread>
#include <vector>

#include <controller_manager_msgs/srv/list_controllers.hpp>

#include "sequence_executor/control_mode_probe.hpp"
#include "sequence_executor/robot_supervisor.hpp"
#include "sequence_executor/yaml_sequence_source.hpp"

namespace sequence_executor {

namespace {

// joint_state_broadcaster alone is not a good enough readiness proxy: it
// activates before the arm trajectory controllers do, so a first step could
// fire while left_arm_controller/right_arm_controller were mid-activation.
// The gripper controllers are deliberately excluded - they never activate at
// all under amazing_hand and would hang this forever if required.
bool allRequiredControllersActive(
  const controller_manager_msgs::srv::ListControllers::Response& response)
{
  static const std::vector<std::string> kRequired = {
    "joint_state_broadcaster", "left_arm_controller", "right_arm_controller"};
  for (const auto& name : kRequired) {
    bool found_active = false;
    for (const auto& controller : response.controller) {
      if (controller.name == name && controller.state == "active") {
        found_active = true;
        break;
      }
    }
    if (!found_active) {
      return false;
    }
  }
  return true;
}

void waitForControllersReady(const rclcpp::Node::SharedPtr& node, double timeout_s)
{
  if (timeout_s <= 0.0) {
    return;
  }

  auto client = node->create_client<controller_manager_msgs::srv::ListControllers>(
    "/controller_manager/list_controllers");
  const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);

  if (!client->wait_for_service(std::chrono::seconds(5))) {
    RCLCPP_WARN(node->get_logger(), "controller_manager not available - starting anyway");
    return;
  }

  while (std::chrono::steady_clock::now() < deadline) {
    auto request =
      std::make_shared<controller_manager_msgs::srv::ListControllers::Request>();
    auto future = client->async_send_request(request);
    if (rclcpp::spin_until_future_complete(node, future.future, std::chrono::seconds(2)) ==
        rclcpp::FutureReturnCode::SUCCESS) {
      if (allRequiredControllersActive(*future.future.get())) {
        RCLCPP_INFO(node->get_logger(), "Required controllers active, proceeding");
        return;
      }
    } else {
      // Drop the abandoned request. Without this it stays queued inside the
      // client, and when the response finally lands - after this function has
      // returned and destroyed the client - rcl writes into freed memory and
      // the process takes a SIGSEGV. controller_manager logs it from its side
      // as "failed to send response ... client will not receive response".
      client->remove_pending_request(future);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  RCLCPP_WARN(node->get_logger(), "Required controllers not active after %.1fs - starting anyway",
              timeout_s);
}

}  // namespace

int runApp(int argc, char** argv, const std::string& node_name,
           const ConfigureCallback& configure)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(node_name);

  const std::string yaml_path = node->declare_parameter<std::string>("sequence_yaml_path", "");
  const std::string autostart = node->declare_parameter<std::string>("sequence_name", "");
  const std::string hardware_config =
    node->declare_parameter<std::string>("hardware_config_path", "");
  const double wait_seconds = node->declare_parameter<double>("wait_for_controllers", 30.0);

  auto builtins = std::make_shared<BuiltinActionRegistry>();

  std::shared_ptr<SequenceSource> source;
  if (configure) {
    try {
      source = configure(node, *builtins);
    } catch (const std::exception& e) {
      RCLCPP_FATAL(node->get_logger(), "Could not open the sequence source: %s", e.what());
      rclcpp::shutdown();
      return 1;
    }
  }

  if (!source) {
    if (yaml_path.empty()) {
      RCLCPP_FATAL(node->get_logger(),
                   "No sequence source: pass sequence_yaml_path, or run an executable that "
                   "supplies one (see builds/qvic_2026).");
      rclcpp::shutdown();
      return 1;
    }
    try {
      source = std::make_shared<YamlSequenceSource>(yaml_path);
    } catch (const std::exception& e) {
      RCLCPP_FATAL(node->get_logger(), "Could not read %s: %s", yaml_path.c_str(), e.what());
      rclcpp::shutdown();
      return 1;
    }
  }

  waitForControllersReady(node, wait_seconds);

  auto mode_probe = std::make_shared<ControlModeProbe>(node, hardware_config);
  // Once, here, while blocking is still free: after this the supervisor only
  // reads the cached value, so publishing state never waits on a service.
  mode_probe->probe();

  auto supervisor = std::make_shared<RobotSupervisor>(node, source, mode_probe, builtins);
  supervisor->start();
  supervisor->autostart(autostart);

  // Single-threaded on purpose - see the header.
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  supervisor.reset();
  rclcpp::shutdown();
  return 0;
}

}  // namespace sequence_executor
