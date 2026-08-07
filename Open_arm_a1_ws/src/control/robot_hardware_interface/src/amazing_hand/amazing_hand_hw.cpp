#include "amazing_hand/amazing_hand_hw.hpp"

#include <algorithm>
#include <chrono>
#include <map>
#include <mutex>
#include <optional>

#include <rclcpp/parameter_client.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace openarm_hardware {

namespace {

std::string sanitizeForNodeName(const std::string& s)
{
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back((std::isalnum(static_cast<unsigned char>(c)) || c == '_') ? c : '_');
  }
  while (!out.empty() && out.back() == '_') {
    out.pop_back();
  }
  return out;
}

// Every AmazingHandHW instance in this process (left + right hand) wants the
// exact same robot_description. Fetching it involves a fresh node's DDS
// graph discovery of robot_state_publisher's parameter service, which is
// the slow part (multiple seconds), not the RPC itself - and this whole
// call blocks controller_manager's own thread during on_init(), since
// SyncParametersClient spins its own temporary executor synchronously.
// With N hand instances each paying that cost in turn, controller_manager
// is unresponsive to spawner service calls (list_controllers etc.) long
// enough that they time out and retry. Caching the string process-wide
// means only the first instance pays it.
std::mutex g_robot_description_cache_mutex;
std::optional<std::string> g_robot_description_cache;

}  // namespace

bool AmazingHandHW::parse_config()
{
  const auto& params = info_.hardware_parameters;

  if (auto it = params.find("link_prefix"); it != params.end()) {
    link_prefix_ = it->second;
  }
  if (auto it = params.find("alias_prefix"); it != params.end()) {
    alias_prefix_ = it->second;
  }
  if (auto it = params.find("command_space"); it != params.end()) {
    command_space_ = it->second;
  }
  if (auto it = params.find("robot_description_timeout_s"); it != params.end()) {
    robot_description_timeout_s_ = std::stod(it->second);
  }

  for (const auto& joint : info_.joints) {
    command_joint_names_.push_back(joint.name);
    double initial = 0.0;
    for (const auto& iface : joint.command_interfaces) {
      if (iface.name == hardware_interface::HW_IF_POSITION && !iface.initial_value.empty()) {
        initial = std::stod(iface.initial_value);
      }
    }
    command_initial_positions_.push_back(initial);
  }
  if (command_joint_names_.empty()) {
    RCLCPP_ERROR(logger_, "No command joints declared in <ros2_control> block");
    return false;
  }

  return true;
}

bool AmazingHandHW::fetchRobotDescription(const rclcpp::Executor::WeakPtr& /*executor*/, std::string& out_urdf)
{
  // node_ is also used later (read()'s throttled logging needs a clock), so
  // it's created unconditionally here regardless of the cache below - only
  // the expensive part (discovering + querying robot_state_publisher's
  // parameter service) is skippable on a cache hit.
  const std::string node_name = "amazing_hand_hw_" +
    (sanitizeForNodeName(alias_prefix_).empty() ? "hand" : sanitizeForNodeName(alias_prefix_));
  node_ = std::make_shared<rclcpp::Node>(node_name);

  {
    std::lock_guard<std::mutex> lock(g_robot_description_cache_mutex);
    if (g_robot_description_cache) {
      out_urdf = *g_robot_description_cache;
      return true;
    }
  }

  // A brand-new, self-contained node - deliberately NOT added to the
  // controller_manager's executor (HardwareComponentInterfaceParams::executor).
  // SyncParametersClient's blocking calls each spin their own temporary
  // executor internally (rclcpp::spin_until_future_complete); adding this
  // node to an already-spinning external executor at the same time would
  // raise "node already added to an executor" the first time both tried to
  // service it. A one-shot on_init()-time fetch doesn't need the shared
  // executor - only a component with ongoing subscriptions/services beyond
  // init would.
  auto client = std::make_shared<rclcpp::SyncParametersClient>(node_, "robot_state_publisher");

  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(robot_description_timeout_s_));

  if (!client->wait_for_service(std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now()))) {
    RCLCPP_ERROR(
      logger_, "robot_state_publisher parameter service not available after %.1fs", robot_description_timeout_s_);
    return false;
  }

  try {
    out_urdf = client->get_parameter<std::string>("robot_description");
  } catch (const std::exception& e) {
    RCLCPP_ERROR(logger_, "Failed to fetch robot_description parameter: %s", e.what());
    return false;
  }

  if (out_urdf.empty()) {
    RCLCPP_ERROR(logger_, "robot_description parameter from robot_state_publisher is empty");
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(g_robot_description_cache_mutex);
    g_robot_description_cache = out_urdf;
  }
  return true;
}

hardware_interface::CallbackReturn AmazingHandHW::on_init(
  const hardware_interface::HardwareComponentInterfaceParams& params)
{
  if (hardware_interface::SystemInterface::on_init(params) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }

  if (!parse_config()) {
    return CallbackReturn::ERROR;
  }

  // Legacy manual export_*_interfaces() overrides (as used here) bypass the
  // framework's usual InterfaceDescription-based initial_value auto-seeding,
  // so this plugin owns applying it itself - command_initial_positions_ was
  // parsed from info_.joints[i].command_interfaces in parse_config().
  command_positions_ = command_initial_positions_;

  // export_state_interfaces() is called by the resource manager right after
  // on_init() - NOT after on_configure() - so state_joint_names_/state_positions_
  // must already be final here. Building the solver (and therefore knowing
  // allMovable()) this early means fetching robot_description this early too.
  std::string urdf_xml;
  if (!fetchRobotDescription(rclcpp::Executor::WeakPtr(), urdf_xml)) {
    return CallbackReturn::ERROR;
  }

  try {
    solver_ = std::make_unique<amazing_hand_kinematics::HandSolver>(
      urdf_xml, link_prefix_, alias_prefix_, command_space_);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(logger_, "Failed to build HandSolver: %s", e.what());
    return CallbackReturn::ERROR;
  }

  // command_joint_names_ (aliases, already populated above) first, then the
  // solver's passive joints (state-only - see class doc comment).
  state_joint_names_ = command_joint_names_;
  for (const auto& jn : solver_->allMovable()) {
    state_joint_names_.push_back(jn);
  }
  state_positions_.assign(state_joint_names_.size(), 0.0);

  RCLCPP_INFO(
    logger_, "command_space=%s, link_prefix='%s': %zu command joints, %zu passive state joints", command_space_.c_str(),
    link_prefix_.c_str(), command_joint_names_.size(), solver_->allMovable().size());

  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn AmazingHandHW::on_configure(const rclcpp_lifecycle::State& /*previous_state*/)
{
  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> AmazingHandHW::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  state_interfaces.reserve(state_joint_names_.size());
  for (std::size_t i = 0; i < state_joint_names_.size(); ++i) {
    state_interfaces.emplace_back(state_joint_names_[i], hardware_interface::HW_IF_POSITION, &state_positions_[i]);
  }
  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> AmazingHandHW::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  command_interfaces.reserve(command_joint_names_.size());
  for (std::size_t i = 0; i < command_joint_names_.size(); ++i) {
    command_interfaces.emplace_back(
      command_joint_names_[i], hardware_interface::HW_IF_POSITION, &command_positions_[i]);
  }
  return command_interfaces;
}

hardware_interface::return_type AmazingHandHW::read(
  const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/)
{
  std::map<std::string, double> raw;
  for (std::size_t i = 0; i < command_joint_names_.size(); ++i) {
    raw[command_joint_names_[i]] = command_positions_[i];
  }

  const std::map<std::string, double> solved = solver_->solve(raw);
  const auto& alias_state = solver_->aliasState();

  for (const auto& gimbal : solver_->lastOutOfReachFingers()) {
    RCLCPP_WARN_THROTTLE(
      logger_, *node_->get_clock(), 2000, "finger %s: command out of mechanism reach, showing nearest feasible pose",
      gimbal.c_str());
  }

  for (std::size_t i = 0; i < state_joint_names_.size(); ++i) {
    const std::string& name = state_joint_names_[i];
    if (auto ait = alias_state.find(name); ait != alias_state.end()) {
      state_positions_[i] = ait->second;
      continue;
    }
    if (auto sit = solved.find(name); sit != solved.end()) {
      state_positions_[i] = sit->second;
    }
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type AmazingHandHW::write(
  const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/)
{
  // Nothing to send to a real actuator - the command interfaces' own
  // storage (command_positions_) already holds whatever the active
  // controller last commanded; read() solves directly from that every
  // cycle. Same principle as mock_components/GenericSystem.
  return hardware_interface::return_type::OK;
}

}  // namespace openarm_hardware

PLUGINLIB_EXPORT_CLASS(openarm_hardware::AmazingHandHW, hardware_interface::SystemInterface)
