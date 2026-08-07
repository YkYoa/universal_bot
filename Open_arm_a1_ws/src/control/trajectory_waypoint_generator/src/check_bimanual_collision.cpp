// Checks EVERY point of a mirrored left/right waypoint pair for self-collision
// when combined into one both_arms state - not just the first one a real
// planner happens to hit. convertBothArmsSequenceToBt's PlanToJointSequence
// interleaves left+right per waypoint into ONE simultaneous state; OMPL only
// reports the first infeasible goal it tries during planning and then the
// whole sequence aborts, so a real planning failure tells you "there's a
// problem somewhere at or before this index," not "here are all the bad
// points." This is a pure PlanningScene::checkSelfCollision sweep over every
// index up front, independent of any planner, so it finds all of them in
// one pass. Read-only - writes nothing, no live robot needed.

#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <console_bridge/console.h>

#include "waypoint_recorder/waypoint_recorder.hpp"

#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

void printUsage(const char* prog)
{
  std::cerr << "Usage: " << prog
            << " --left-section <name> --right-section <name> --file <path/to/sequence.yaml>"
               " [--ee-type <openarm_hand|amazing_hand>]\n"
               "  Checks every point n=1,2,... of '<la><PascalLeft><n>Angle' /\n"
               "  '<ra><PascalRight><n>Angle' combined as one both_arms state for\n"
               "  self-collision. Stops at the first index missing on either side.\n"
               "  --ee-type must match whatever the real launch actually uses - hand\n"
               "  geometry differs, so a collision check against the wrong one can miss\n"
               "  or invent collisions. Default: openarm_hand (matches v10.urdf.xacro's\n"
               "  own default).\n";
}

std::string runXacro(const std::string& xacro_path, const std::string& args)
{
  const std::string cmd = "xacro " + xacro_path + " " + args;
  std::array<char, 4096> buffer{};
  std::string result;
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) return "";
  size_t n;
  while ((n = fread(buffer.data(), 1, buffer.size(), pipe)) > 0) result.append(buffer.data(), n);
  pclose(pipe);
  return result;
}

std::optional<std::vector<double>> readYamlKey(const std::string& file_path, const std::string& key)
{
  std::ifstream in(file_path);
  if (!in.good()) return std::nullopt;
  std::string line;
  while (std::getline(in, line)) {
    const size_t first_non_space = line.find_first_not_of(" \t");
    if (first_non_space == std::string::npos) continue;
    if (line.compare(first_non_space, key.size(), key) != 0) continue;
    const size_t after_key = first_non_space + key.size();
    if (after_key >= line.size() || line[after_key] != ':') continue;

    std::vector<double> values;
    std::stringstream ss(line.substr(after_key + 1));
    std::string token;
    while (std::getline(ss, token, ',')) {
      try {
        values.push_back(std::stod(token));
      } catch (...) {
        return std::nullopt;
      }
    }
    return values;
  }
  return std::nullopt;
}

}  // namespace

int main(int argc, char** argv)
{
  std::string left_section, right_section, file_path;
  std::string ee_type = "openarm_hand";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "[ERROR] " << flag << " requires a value\n";
        std::exit(1);
      }
      return argv[++i];
    };
    if (arg == "--left-section") left_section = next("--left-section");
    else if (arg == "--right-section") right_section = next("--right-section");
    else if (arg == "--file") file_path = next("--file");
    else if (arg == "--ee-type") ee_type = next("--ee-type");
    else if (arg == "--help" || arg == "-h") { printUsage(argv[0]); return 0; }
    else { std::cerr << "[ERROR] Unknown argument: " << arg << "\n"; printUsage(argv[0]); return 1; }
  }
  if (left_section.empty() || right_section.empty() || file_path.empty()) {
    std::cerr << "[ERROR] --left-section, --right-section, --file are all required\n";
    printUsage(argv[0]);
    return 1;
  }

  rclcpp::init(argc, argv);
  console_bridge::setLogLevel(console_bridge::CONSOLE_BRIDGE_LOG_NONE);

  auto model_node = std::make_shared<rclcpp::Node>(
    "check_bimanual_collision_model",
    rclcpp::NodeOptions().arguments(
      {"--ros-args", "--params-file",
       ament_index_cpp::get_package_share_directory("openarm_moveit_config") + "/config/kinematics.yaml"}));

  const std::string description_pkg = ament_index_cpp::get_package_share_directory("openarm_description");
  const std::string xacro_path = description_pkg + "/urdf/robot/v10.urdf.xacro";
  const std::string urdf_xml = runXacro(xacro_path,
    "bimanual:=true ros2_control:=true use_fake_hardware:=true mobile_base:=true "
    "mobile_base_xyz:='0 0 0.31' mobile_base_body_xyz:='0 0 0' ee_type:=" + ee_type);
  if (urdf_xml.empty()) {
    std::cerr << "[ERROR] Failed to run xacro on " << xacro_path << "\n";
    rclcpp::shutdown();
    return 1;
  }
  model_node->declare_parameter<std::string>("robot_description", urdf_xml);

  const std::string moveit_cfg_pkg = ament_index_cpp::get_package_share_directory("openarm_moveit_config");
  std::ifstream srdf_file(moveit_cfg_pkg + "/srdf/openarm_bimanual.srdf");
  if (!srdf_file.good()) {
    std::cerr << "[ERROR] Could not read openarm_bimanual.srdf\n";
    rclcpp::shutdown();
    return 1;
  }
  std::stringstream srdf_stream;
  srdf_stream << srdf_file.rdbuf();
  model_node->declare_parameter<std::string>("robot_description_semantic", srdf_stream.str());

  robot_model_loader::RobotModelLoader model_loader(model_node, "robot_description");
  auto robot_model = model_loader.getModel();
  if (!robot_model) {
    std::cerr << "[ERROR] Failed to load robot model\n";
    rclcpp::shutdown();
    return 1;
  }

  planning_scene::PlanningScene scene(robot_model);
  moveit::core::RobotState state(robot_model);
  state.setToDefaultValues();

  int checked = 0, collisions = 0;
  for (int n = 1; ; ++n) {
    const std::string la_key = "la" + waypoint_recorder::toPascalCase(left_section) + std::to_string(n) + "Angle";
    const std::string ra_key = "ra" + waypoint_recorder::toPascalCase(right_section) + std::to_string(n) + "Angle";
    auto la_raw = readYamlKey(file_path, la_key);
    auto ra_raw = readYamlKey(file_path, ra_key);
    if (!la_raw || !ra_raw) break;
    if (la_raw->size() != 7 || ra_raw->size() != 7) {
      std::cerr << "[ERROR] '" << la_key << "' or '" << ra_key << "' does not have 7 values\n";
      rclcpp::shutdown();
      return 1;
    }

    state.setJointGroupPositions("left_arm", *la_raw);
    state.setJointGroupPositions("right_arm", *ra_raw);
    state.update();

    collision_detection::CollisionRequest req;
    req.group_name = "both_arms";
    req.contacts = true;
    req.max_contacts = 5;
    collision_detection::CollisionResult res;
    scene.checkSelfCollision(req, res, state);

    ++checked;
    if (res.collision) {
      ++collisions;
      std::cout << "[COLLISION] point " << n << " (" << la_key << " / " << ra_key << ") - "
                << res.contacts.size() << " contact pair(s):";
      for (const auto& c : res.contacts) {
        std::cout << " " << c.first.first << "<->" << c.first.second;
      }
      std::cout << "\n";
    } else {
      std::cout << "[OK] point " << n << "\n";
    }
  }

  std::cout << "\nChecked " << checked << " point(s), " << collisions << " in self-collision.\n";
  rclcpp::shutdown();
  return 0;
}
