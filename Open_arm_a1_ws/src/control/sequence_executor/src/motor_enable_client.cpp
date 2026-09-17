#include "sequence_executor/motor_enable_client.hpp"

#include <chrono>
#include <vector>

#include <lifecycle_msgs/msg/state.hpp>

namespace sequence_executor {

namespace {
// ros2_control's mock hardware plugin, used for the base and head (no real
// motors, no CAN enable/disable to lose) - see bringup.launch.py's spawner
// conditions for which components are physical vs mock depending on ee_type.
constexpr const char* kMockPluginName = "mock_components/GenericSystem";
}  // namespace

MotorEnableClient::MotorEnableClient(const rclcpp::Node::SharedPtr& node)
  : node_(node),
    logger_(node_->get_logger()),
    internal_node_(std::make_shared<rclcpp::Node>(
      std::string(node_->get_name()) + "_motor_enable_internal"))
{
  list_client_ = internal_node_->create_client<ListHardwareComponents>(
    "/controller_manager/list_hardware_components");
  set_state_client_ = internal_node_->create_client<SetHardwareComponentState>(
    "/controller_manager/set_hardware_component_state");
}

bool MotorEnableClient::setComponentState(const std::string& name, uint8_t target_id,
                                          const std::string& target_label, std::string& error)
{
  if (!set_state_client_->wait_for_service(std::chrono::seconds(2))) {
    error = "set_hardware_component_state service unavailable";
    return false;
  }

  auto request = std::make_shared<SetHardwareComponentState::Request>();
  request->name = name;
  request->target_state.id = target_id;
  request->target_state.label = target_label;

  auto future = set_state_client_->async_send_request(request);
  if (rclcpp::spin_until_future_complete(internal_node_, future.future, std::chrono::seconds(5)) !=
      rclcpp::FutureReturnCode::SUCCESS) {
    set_state_client_->remove_pending_request(future);
    error = "timed out waiting for '" + name + "' to reach '" + target_label + "'";
    return false;
  }

  auto response = future.future.get();
  if (!response) {
    error = "empty response setting '" + name + "' to '" + target_label + "'";
    return false;
  }
  if (!response->ok) {
    error = "'" + name + "' refused transition to '" + target_label + "' (now '" +
            response->state.label + "')";
    return false;
  }
  return true;
}

bool MotorEnableClient::queryMotorsEnabled(std::string& message, bool& any_physical_found,
                                           bool& mock_components_found)
{
  using lifecycle_msgs::msg::State;
  any_physical_found = false;
  mock_components_found = false;

  if (!list_client_->wait_for_service(std::chrono::seconds(2))) {
    message = "list_hardware_components service unavailable";
    return false;
  }

  auto future =
    list_client_->async_send_request(std::make_shared<ListHardwareComponents::Request>());
  if (rclcpp::spin_until_future_complete(internal_node_, future.future, std::chrono::seconds(3)) !=
      rclcpp::FutureReturnCode::SUCCESS) {
    list_client_->remove_pending_request(future);
    message = "timed out listing hardware components";
    return false;
  }

  auto response = future.future.get();
  if (!response) {
    message = "list_hardware_components returned an empty response";
    return false;
  }

  bool all_active = true;
  std::vector<std::string> states;
  for (const auto& component : response->component) {
    if (component.plugin_name == kMockPluginName ||
        component.plugin_name.find("mock") != std::string::npos ||
        component.plugin_name.find("fake") != std::string::npos) {
      mock_components_found = true;
      continue;
    }
    any_physical_found = true;
    const bool is_active = (component.state.id == State::PRIMARY_STATE_ACTIVE);
    states.push_back(component.name + ": " + component.state.label);
    if (!is_active) {
      all_active = false;
    }
  }

  if (!any_physical_found) {
    if (mock_components_found) {
      message = "mock hardware components found";
      return true;
    }
    message = "no hardware components found";
    return false;
  }

  message.clear();
  for (size_t i = 0; i < states.size(); ++i) {
    if (i > 0) {
      message += "; ";
    }
    message += states[i];
  }
  return all_active;
}

bool MotorEnableClient::queryMotorsEnabled(std::string& message, bool& any_physical_found)
{
  bool mock_found = false;
  return queryMotorsEnabled(message, any_physical_found, mock_found);
}

bool MotorEnableClient::enableAll(std::string& message)
{
  using lifecycle_msgs::msg::State;

  // If physical motors are already active, skip re-activation to avoid jerking motion
  bool any_physical = false;
  std::string current_state_msg;
  if (queryMotorsEnabled(current_state_msg, any_physical) && any_physical) {
    message = "Motors are already enabled (" + current_state_msg + ")";
    RCLCPP_INFO(logger_, "enableAll: %s, skipping re-activation", message.c_str());
    return true;
  }

  if (!list_client_->wait_for_service(std::chrono::seconds(2))) {
    message = "list_hardware_components service unavailable";
    return false;
  }

  auto future =
    list_client_->async_send_request(std::make_shared<ListHardwareComponents::Request>());
  if (rclcpp::spin_until_future_complete(internal_node_, future.future, std::chrono::seconds(3)) !=
      rclcpp::FutureReturnCode::SUCCESS) {
    list_client_->remove_pending_request(future);
    message = "timed out listing hardware components";
    return false;
  }

  auto response = future.future.get();
  if (!response) {
    message = "list_hardware_components returned an empty response";
    return false;
  }

  // All-or-nothing: a queued action may need every arm/hand it was written
  // for, so one component this robot couldn't bring back stops the whole
  // enable rather than silently letting the goal run short-handed.
  bool all_ok = true;
  std::vector<std::string> outcomes;
  for (const auto& component : response->component) {
    if (component.plugin_name == kMockPluginName) {
      continue;
    }

    // Skip components that are already active
    if (component.state.id == State::PRIMARY_STATE_ACTIVE) {
      outcomes.push_back(component.name + ": already active");
      continue;
    }

    std::string error;
    const bool ok =
      setComponentState(component.name, State::PRIMARY_STATE_INACTIVE, "inactive", error) &&
      setComponentState(component.name, State::PRIMARY_STATE_ACTIVE, "active", error);

    outcomes.push_back(component.name + ": " + (ok ? "enabled" : ("FAILED (" + error + ")")));
    if (!ok) {
      all_ok = false;
      RCLCPP_ERROR(logger_, "could not re-enable '%s': %s", component.name.c_str(),
                   error.c_str());
    } else {
      RCLCPP_INFO(logger_, "'%s' re-enabled", component.name.c_str());
    }
  }

  if (outcomes.empty()) {
    message = "no physical hardware components found";
    return false;
  }

  message.clear();
  for (size_t i = 0; i < outcomes.size(); ++i) {
    if (i > 0) {
      message += "; ";
    }
    message += outcomes[i];
  }
  return all_ok;
}

}  // namespace sequence_executor
