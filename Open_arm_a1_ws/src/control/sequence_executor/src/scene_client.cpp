#include "sequence_executor/scene_client.hpp"

#include <chrono>

namespace sequence_executor {

namespace {

constexpr const char* kDefaultFrame = "openarm_body_link0";

// Step type -> the SceneCommand action it maps to. Anything not listed here is
// not a scene step and never reaches this client.
std::string actionForStep(const std::string& type)
{
  if (type == "add_object") return "add";
  if (type == "remove_object") return "remove";
  if (type == "attach_object") return "attach";
  if (type == "detach_object") return "detach";
  if (type == "allow_collision") return "allow";
  if (type == "disallow_collision") return "disallow";
  return {};
}

}  // namespace

SceneClient::SceneClient(const rclcpp::Node::SharedPtr& node, const std::string& service_name)
  : node_(node), logger_(node->get_logger())
{
  client_ = node_->create_client<SceneCommand>(service_name);
}

void SceneClient::sendForStep(const Step& step, ResultCallback callback)
{
  const std::string action = actionForStep(step.type);
  if (action.empty()) {
    callback(false, "'" + step.type + "' is not a scene step");
    return;
  }

  SceneCommand::Request request;
  request.action = action;
  request.object_id = step.object_id;
  request.link = step.link;
  request.primitive = step.primitive;
  request.dimensions = step.dimensions;

  // "allow"/"disallow" name their partner links in `links`; attach names them
  // in `touch_links`. The service takes one list, so fold them here.
  request.touch_links = step.touch_links.empty() ? step.links : step.touch_links;

  request.pose.header.frame_id = step.frame_id.empty() ? kDefaultFrame : step.frame_id;
  if (step.position.size() == 3) {
    request.pose.pose.position.x = step.position[0];
    request.pose.pose.position.y = step.position[1];
    request.pose.pose.position.z = step.position[2];
  }
  if (step.orientation.size() == 4) {
    request.pose.pose.orientation.x = step.orientation[0];
    request.pose.pose.orientation.y = step.orientation[1];
    request.pose.pose.orientation.z = step.orientation[2];
    request.pose.pose.orientation.w = step.orientation[3];
  } else {
    request.pose.pose.orientation.w = 1.0;
  }

  send(request, std::move(callback));
}

void SceneClient::send(const SceneCommand::Request& request, ResultCallback callback)
{
  if (!client_->wait_for_service(std::chrono::milliseconds(500))) {
    callback(false,
             "scene_command service is not available - is robot_skills_node running?");
    return;
  }

  auto payload = std::make_shared<SceneCommand::Request>(request);
  const std::string action = request.action;
  client_->async_send_request(
    payload,
    [callback, action, this](rclcpp::Client<SceneCommand>::SharedFuture future) {
      auto response = future.get();
      if (!response) {
        callback(false, "scene_command '" + action + "' returned no response");
        return;
      }
      callback(response->success, response->message);
    });
}

}  // namespace sequence_executor
