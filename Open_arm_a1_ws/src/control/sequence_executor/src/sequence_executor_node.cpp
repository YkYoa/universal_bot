// -----------------------------------------------------------------------------
// sequence_executor_node.cpp
//
// Entry point: reads sequence_yaml_path/sequence_name params, loads that
// sequence from sequence.yaml, and runs it via SequenceInterpreter -
// event-driven (advances on ROS action-result callbacks), no tick loop.
// -----------------------------------------------------------------------------
#include <rclcpp/rclcpp.hpp>
#include <controller_manager_msgs/srv/list_controllers.hpp>

#include "sequence_executor/sequence_yaml.hpp"
#include "sequence_executor/skill_client.hpp"
#include "sequence_executor/hand_gripper_client.hpp"
#include "sequence_executor/sequence_interpreter.hpp"

namespace {

// The old BT.CPP system's 50Hz tick loop naturally retried a goal-send
// until the target controller had finished activating (controller spawners
// run as separate launch Nodes with no explicit ordering vs. this one) -
// this event-driven single-shot node needs an explicit wait instead, or its
// very first goal can hit the same "not active yet" race and abort. A
// fixed delay was tried first but proved unreliable (system load varies run
// to run). joint_state_broadcaster alone also proved insufficient as a
// proxy: it activates before the arm trajectory controllers do, so a home
// step could still fire while left_arm_controller/right_arm_controller
// were mid-activation. Wait for all three of the controllers every
// sequence's home/body step can actually need - deliberately excluding the
// gripper controllers, which never activate at all under amazing_hand and
// would hang this forever if required.
bool allRequiredControllersActive(const controller_manager_msgs::srv::ListControllers::Response& response)
{
  static const std::vector<std::string> kRequired = {
    "joint_state_broadcaster", "left_arm_controller", "right_arm_controller"};
  for (const auto& name : kRequired) {
    bool found_active = false;
    for (const auto& c : response.controller) {
      if (c.name == name && c.state == "active") {
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
  auto client =
    node->create_client<controller_manager_msgs::srv::ListControllers>("/controller_manager/list_controllers");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);

  if (!client->wait_for_service(std::chrono::seconds(5))) {
    RCLCPP_WARN(node->get_logger(), "controller_manager not available - starting anyway");
    return;
  }

  while (std::chrono::steady_clock::now() < deadline) {
    auto request = std::make_shared<controller_manager_msgs::srv::ListControllers::Request>();
    auto future = client->async_send_request(request);
    if (rclcpp::spin_until_future_complete(node, future.future, std::chrono::seconds(2)) ==
        rclcpp::FutureReturnCode::SUCCESS) {
      if (allRequiredControllersActive(*future.future.get())) {
        RCLCPP_INFO(node->get_logger(), "Required controllers active, proceeding");
        return;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  RCLCPP_WARN(node->get_logger(), "Required controllers not active after %.1fs - starting anyway", timeout_s);
}

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("sequence_executor_node");

  node->declare_parameter<std::string>("sequence_yaml_path", "");
  node->declare_parameter<std::string>("sequence_name", "");

  const std::string yaml_path = node->get_parameter("sequence_yaml_path").as_string();
  const std::string sequence_name = node->get_parameter("sequence_name").as_string();

  if (yaml_path.empty() || sequence_name.empty()) {
    RCLCPP_ERROR(
      node->get_logger(), "sequence_yaml_path and sequence_name params are both required (got yaml_path='%s' "
                           "sequence_name='%s')",
      yaml_path.c_str(), sequence_name.c_str());
    rclcpp::shutdown();
    return 1;
  }

  std::shared_ptr<sequence_executor::SequenceYaml> yaml;
  try {
    yaml = std::make_shared<sequence_executor::SequenceYaml>(yaml_path);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(node->get_logger(), "Failed to load %s: %s", yaml_path.c_str(), e.what());
    rclcpp::shutdown();
    return 1;
  }

  auto skill_client = std::make_shared<sequence_executor::SkillClient>(node);
  auto hand_client = std::make_shared<sequence_executor::HandGripperClient>(node);
  sequence_executor::SequenceInterpreter interpreter(node, yaml, skill_client, hand_client);

  waitForControllersReady(node, 30.0);
  interpreter.run(sequence_name);

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
