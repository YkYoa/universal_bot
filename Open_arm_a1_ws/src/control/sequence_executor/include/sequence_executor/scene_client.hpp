#pragma once
// -----------------------------------------------------------------------------
// scene_client.hpp
//
// Async client for robot_skills_server/scene_command - the planning-scene
// edits behind the attach/detach/allow-collision steps.
//
// Async on purpose: the FSM is a single-threaded callback machine, so a
// blocking service call from inside a step would deadlock against the very
// executor that has to deliver the response.
// -----------------------------------------------------------------------------
#include <functional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <openarm_messages/srv/scene_command.hpp>

#include "sequence_executor/sequence_step.hpp"

namespace sequence_executor {

class SceneClient
{
public:
  using SceneCommand = openarm_messages::srv::SceneCommand;
  using ResultCallback = std::function<void(bool success, const std::string& message)>;

  explicit SceneClient(const rclcpp::Node::SharedPtr& node,
                       const std::string& service_name = "robot_skills_server/scene_command");

  // Translates a scene step (add_object, attach_object, allow_collision, ...)
  // into one service call. Reports failure through `callback` rather than
  // throwing, so the FSM handles it on the same path as a failed motion.
  void sendForStep(const Step& step, ResultCallback callback);

  void send(const SceneCommand::Request& request, ResultCallback callback);

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Client<SceneCommand>::SharedPtr client_;
  rclcpp::Logger logger_;
};

}  // namespace sequence_executor
