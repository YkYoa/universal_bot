#include "robot_skills/scene_manager.hpp"

#include <algorithm>

#include <moveit/planning_scene_monitor/planning_scene_monitor.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

namespace robot_skills {

namespace {

constexpr const char* kServiceName = "robot_skills_server/scene_command";
constexpr const char* kDefaultFrame = "openarm_body_link0";

}  // namespace

SceneManager::SceneManager(rclcpp::Node::SharedPtr node,
                           std::shared_ptr<motion_planner::MoveItCppPlannerManager> planner)
  : node_(std::move(node)), planner_(std::move(planner)), logger_(rclcpp::get_logger("scene_manager"))
{
  logger_ = node_->get_logger();
}

bool SceneManager::start()
{
  if (!planner_ || !planner_->getMoveItCpp() ||
      !planner_->getMoveItCpp()->getPlanningSceneMonitorNonConst()) {
    RCLCPP_ERROR(logger_, "SceneManager: no PlanningSceneMonitor, scene_command not advertised.");
    return false;
  }

  service_ = node_->create_service<SceneCommand>(
    kServiceName,
    [this](const std::shared_ptr<SceneCommand::Request> req,
           std::shared_ptr<SceneCommand::Response> res) { handle(req, res); });

  RCLCPP_INFO(logger_, "Scene command service started: %s", kServiceName);
  return true;
}

void SceneManager::handle(const std::shared_ptr<SceneCommand::Request> request,
                          std::shared_ptr<SceneCommand::Response> response)
{
  const std::string& action = request->action;
  std::string error;

  if (action == "add") {
    error = addObject(*request);
  } else if (action == "remove") {
    error = removeObject(*request);
  } else if (action == "attach") {
    error = attachObject(*request);
  } else if (action == "detach") {
    error = detachObject(*request);
  } else if (action == "allow") {
    error = setAllowed(*request, true);
  } else if (action == "disallow") {
    error = setAllowed(*request, false);
  } else if (action == "clear") {
    error = clearScene();
  } else {
    error = "unknown action '" + action +
            "'; expected add|remove|attach|detach|allow|disallow|clear";
  }

  response->success = error.empty();
  if (response->success) {
    publishSceneUpdate();
    response->message = action + " ok";
    RCLCPP_INFO(logger_, "scene_command %s '%s' ok", action.c_str(), request->object_id.c_str());
  } else {
    response->message = error;
    RCLCPP_WARN(logger_, "scene_command %s failed: %s", action.c_str(), error.c_str());
  }
}

std::string SceneManager::addObject(const SceneCommand::Request& req)
{
  if (req.object_id.empty()) {
    return "add needs an object_id";
  }

  moveit_msgs::msg::CollisionObject object;
  std::string error;
  if (!buildPrimitive(req, object, error)) {
    return error;
  }

  auto psm = planner_->getMoveItCpp()->getPlanningSceneMonitorNonConst();
  {
    planning_scene_monitor::LockedPlanningSceneRW scene(psm);
    // An unknown frame_id would silently place the object at the world origin,
    // which is a collision volume in the wrong place - worse than refusing.
    if (!scene->knowsFrameTransform(object.header.frame_id)) {
      return "unknown frame_id '" + object.header.frame_id + "'";
    }
    if (!scene->processCollisionObjectMsg(object)) {
      return "planning scene rejected object '" + req.object_id + "'";
    }
  }
  return "";
}

std::string SceneManager::removeObject(const SceneCommand::Request& req)
{
  if (req.object_id.empty()) {
    return "remove needs an object_id";
  }

  moveit_msgs::msg::CollisionObject object;
  object.id = req.object_id;
  object.operation = moveit_msgs::msg::CollisionObject::REMOVE;

  auto psm = planner_->getMoveItCpp()->getPlanningSceneMonitorNonConst();
  {
    planning_scene_monitor::LockedPlanningSceneRW scene(psm);
    object.header.frame_id = scene->getPlanningFrame();
    if (!scene->processCollisionObjectMsg(object)) {
      return "no object '" + req.object_id + "' in the scene";
    }
  }
  return "";
}

std::string SceneManager::attachObject(const SceneCommand::Request& req)
{
  if (req.object_id.empty() || req.link.empty()) {
    return "attach needs both object_id and link";
  }

  moveit_msgs::msg::AttachedCollisionObject attached;
  attached.link_name = req.link;
  attached.object.id = req.object_id;
  attached.object.operation = moveit_msgs::msg::CollisionObject::ADD;
  attached.touch_links = req.touch_links.empty() ? defaultTouchLinks(req.link) : req.touch_links;

  auto psm = planner_->getMoveItCpp()->getPlanningSceneMonitorNonConst();
  {
    planning_scene_monitor::LockedPlanningSceneRW scene(psm);
    if (!scene->getRobotModel()->hasLinkModel(req.link)) {
      return "no link named '" + req.link + "' on the robot";
    }
    if (!scene->getWorld()->hasObject(req.object_id)) {
      return "object '" + req.object_id + "' is not in the scene; add it first";
    }
    attached.object.header.frame_id = req.link;
    if (!scene->processAttachedCollisionObjectMsg(attached)) {
      return "planning scene rejected attaching '" + req.object_id + "'";
    }
  }

  RCLCPP_INFO(logger_, "Attached '%s' to '%s' with %zu touch link(s)",
              req.object_id.c_str(), req.link.c_str(), attached.touch_links.size());
  return "";
}

std::string SceneManager::detachObject(const SceneCommand::Request& req)
{
  if (req.object_id.empty()) {
    return "detach needs an object_id";
  }

  moveit_msgs::msg::AttachedCollisionObject attached;
  attached.object.id = req.object_id;
  attached.link_name = req.link;
  attached.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;

  auto psm = planner_->getMoveItCpp()->getPlanningSceneMonitorNonConst();
  {
    planning_scene_monitor::LockedPlanningSceneRW scene(psm);
    // REMOVE on an attached object detaches it and drops it back into the
    // world at its current pose - it does not delete it.
    if (!scene->processAttachedCollisionObjectMsg(attached)) {
      return "object '" + req.object_id + "' is not attached";
    }
  }
  return "";
}

std::string SceneManager::setAllowed(const SceneCommand::Request& req, bool allowed)
{
  if (req.object_id.empty()) {
    return std::string(allowed ? "allow" : "disallow") + " needs an object_id";
  }

  auto psm = planner_->getMoveItCpp()->getPlanningSceneMonitorNonConst();
  {
    planning_scene_monitor::LockedPlanningSceneRW scene(psm);
    auto& acm = scene->getAllowedCollisionMatrixNonConst();
    if (req.touch_links.empty()) {
      // No partner named: one entry against everything currently known.
      acm.setEntry(req.object_id, allowed);
    } else {
      for (const auto& other : req.touch_links) {
        acm.setEntry(req.object_id, other, allowed);
      }
    }
  }
  return "";
}

std::string SceneManager::clearScene()
{
  auto psm = planner_->getMoveItCpp()->getPlanningSceneMonitorNonConst();
  {
    planning_scene_monitor::LockedPlanningSceneRW scene(psm);
    scene->removeAllCollisionObjects();
    // Rebuild the ACM from the SRDF so every entry this service added is gone,
    // while the robot's own always-disabled self-collision pairs survive.
    const auto& srdf = scene->getRobotModel()->getSRDF();
    if (srdf) {
      scene->getAllowedCollisionMatrixNonConst() =
        collision_detection::AllowedCollisionMatrix(*srdf);
    }
  }
  return "";
}

std::vector<std::string> SceneManager::defaultTouchLinks(const std::string& link) const
{
  auto model = planner_->getMoveItCpp()->getRobotModel();
  if (!model) {
    return {link};
  }

  for (const auto* group : model->getEndEffectors()) {
    if (!group) {
      continue;
    }
    const auto& names = group->getLinkModelNames();
    const bool owns_link =
      group->getEndEffectorParentGroup().second == link ||
      std::find(names.begin(), names.end(), link) != names.end();
    if (owns_link) {
      std::vector<std::string> touch = names;
      if (std::find(touch.begin(), touch.end(), link) == touch.end()) {
        touch.push_back(link);
      }
      return touch;
    }
  }
  return {link};
}

bool SceneManager::buildPrimitive(const SceneCommand::Request& req,
                                  moveit_msgs::msg::CollisionObject& object,
                                  std::string& error) const
{
  shape_msgs::msg::SolidPrimitive primitive;
  const auto& dims = req.dimensions;

  if (req.primitive == "box") {
    if (dims.size() != 3) {
      error = "box needs 3 dimensions [x, y, z]";
      return false;
    }
    primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    primitive.dimensions = {dims[0], dims[1], dims[2]};
  } else if (req.primitive == "sphere") {
    if (dims.size() != 1) {
      error = "sphere needs 1 dimension [radius]";
      return false;
    }
    primitive.type = shape_msgs::msg::SolidPrimitive::SPHERE;
    primitive.dimensions = {dims[0]};
  } else if (req.primitive == "cylinder") {
    if (dims.size() != 2) {
      error = "cylinder needs 2 dimensions [height, radius]";
      return false;
    }
    primitive.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    primitive.dimensions = {dims[0], dims[1]};
  } else {
    error = "unknown primitive '" + req.primitive + "'; expected box|sphere|cylinder";
    return false;
  }

  if (std::any_of(primitive.dimensions.begin(), primitive.dimensions.end(),
                  [](double d) { return d <= 0.0; })) {
    error = "every dimension must be > 0";
    return false;
  }

  object.id = req.object_id;
  object.header.frame_id = req.pose.header.frame_id.empty() ? kDefaultFrame
                                                            : req.pose.header.frame_id;
  object.primitives.push_back(primitive);

  // A quaternion of all zeros is what an unset geometry_msgs/Pose looks like;
  // treat it as identity rather than handing MoveIt an invalid rotation.
  geometry_msgs::msg::Pose pose = req.pose.pose;
  if (pose.orientation.x == 0.0 && pose.orientation.y == 0.0 &&
      pose.orientation.z == 0.0 && pose.orientation.w == 0.0) {
    pose.orientation.w = 1.0;
  }
  object.primitive_poses.push_back(pose);
  object.operation = moveit_msgs::msg::CollisionObject::ADD;
  return true;
}

void SceneManager::publishSceneUpdate()
{
  auto psm = planner_->getMoveItCpp()->getPlanningSceneMonitorNonConst();
  if (psm) {
    psm->triggerSceneUpdateEvent(planning_scene_monitor::PlanningSceneMonitor::UPDATE_SCENE);
  }
}

}  // namespace robot_skills
