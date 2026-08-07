#include "hand_kinematic.h"

namespace robot_control
{

HandKinematicNode::HandKinematicNode(const rclcpp::NodeOptions& options)
: rclcpp::Node("hand_kinematic_node", options)
{
  const std::string urdf_xml = declare_parameter<std::string>("robot_description", "");
  const std::string command_space = declare_parameter<std::string>("command_space", "knuckle");
  const std::string link_prefix = declare_parameter<std::string>("link_prefix", "");
  const std::string alias_prefix = declare_parameter<std::string>("alias_prefix", "");
  const std::string joint_commands_topic =
    declare_parameter<std::string>("joint_commands_topic", "ahand/joint_commands");
  const std::string joint_states_topic = declare_parameter<std::string>("joint_states_topic", "ahand/joint_states");

  try {
    solver_ = std::make_unique<amazing_hand_kinematics::HandSolver>(
      urdf_xml, link_prefix, alias_prefix, command_space);
  } catch (const std::exception& e) {
    RCLCPP_FATAL(get_logger(), "%s", e.what());
    throw;
  }

  RCLCPP_INFO(
    get_logger(), "command_space=%s, link_prefix='%s'; discovered %zu fingers", command_space.c_str(),
    link_prefix.c_str(), solver_->aliasOf().size() / 2);

  pub_ = create_publisher<JointStateMsg>("joint_states", 10);
  // queue depth 1: a solve takes tens of ms, so under a fast command stream
  // always solve the newest target and let stale ones drop.
  sub_ = create_subscription<JointStateMsg>(
    "joint_states_raw", 1, std::bind(&HandKinematicNode::onJointStates, this, std::placeholders::_1));
  cmd_sub_ = create_subscription<JointStateMsg>(
    joint_commands_topic, 1, std::bind(&HandKinematicNode::onJointStates, this, std::placeholders::_1));
  alias_pub_ = create_publisher<JointStateMsg>(joint_states_topic, 10);

  onJointStates(JointStateMsg());  // publish the solved rest pose immediately

  state_timer_ =
    create_wall_timer(std::chrono::milliseconds(50), std::bind(&HandKinematicNode::republishStates, this));
}

void HandKinematicNode::onJointStates(const JointStateMsg& msg)
{
  std::map<std::string, double> raw;
  for (std::size_t i = 0; i < msg.name.size() && i < msg.position.size(); ++i) {
    raw[msg.name[i]] = msg.position[i];
  }

  std::map<std::string, double> solved = solver_->solve(raw);

  for (const auto& gimbal : solver_->lastOutOfReachFingers()) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "finger %s: command out of mechanism reach, showing nearest feasible pose",
      gimbal.c_str());
  }

  std::vector<std::string> out_names;
  std::vector<double> out_pos;
  out_names.reserve(solved.size());
  out_pos.reserve(solved.size());
  for (const auto& [name, value] : solved) {
    out_names.push_back(name);
    out_pos.push_back(value);
  }
  last_full_state_ = {out_names, out_pos};

  JointStateMsg out;
  out.header = msg.header;
  out.header.stamp = get_clock()->now();
  out.name = out_names;
  out.position = out_pos;
  pub_->publish(out);
}

void HandKinematicNode::republishStates()
{
  const rclcpp::Time now = get_clock()->now();

  JointStateMsg msg;
  msg.header.stamp = now;
  for (const auto& [alias, val] : solver_->aliasState()) {
    msg.name.push_back(alias);
    msg.position.push_back(val);
  }
  alias_pub_->publish(msg);

  if (last_full_state_) {
    JointStateMsg full;
    full.header.stamp = now;
    full.name = last_full_state_->first;
    full.position = last_full_state_->second;
    pub_->publish(full);
  }
}

}  // namespace robot_control

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<robot_control::HandKinematicNode>());
  rclcpp::shutdown();
  return 0;
}
