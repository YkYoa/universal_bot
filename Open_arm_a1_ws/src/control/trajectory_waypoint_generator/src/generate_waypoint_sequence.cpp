// Robot-specific glue over trajectory_shapes (pure Cartesian-path math): picks
// a shape (line, ellipse, circle, square), FK/IK-solves it against the real
// robot model, and writes the result as a numbered "<arm><Section><N>Angle"
// joint-angle waypoint run into a sequence.yaml - so it replays through the
// existing PlanToJointSequence pipeline (which does the real collision
// checking; this tool is pure FK/IK, no planning scene here - see the
// "Collision safety" note below).
//
// Run once per arm. Default is to anchor each arm's own --start/--end/
// --center to that arm's own reference pose - since each arm's own
// recorded/named poses are already correct for its own side, running the
// same shape independently per arm produces a symmetric result without any
// cross-arm math. If you don't have a reference pose for the target arm
// yet, a "mirror:" prefix (e.g. --start mirror:key:laPreWaveJoint) reflects
// the OTHER arm's pose across the model-root Y=0 plane - derived from the
// real mount geometry (v10.urdf.xacro:47-51: both arms mount to
// openarm_body_link0 with identical X/Z, only Y sign-flipped), not a
// hand-wavy guess - see mirrorAcrossY()'s comment for the exact math and
// its one known limitation (the true mounts aren't perfectly Y-symmetric,
// off by ~6mm - negligible for a gesture, verify empirically before
// anything precision-sensitive).

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <console_bridge/console.h>
#include <Eigen/Geometry>

#include "waypoint_recorder/waypoint_recorder.hpp"
#include "trajectory_shapes/trajectory_shapes.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

void printUsage(const char* prog)
{
  std::cerr
    << "Usage: " << prog
    << " --arm <la|ra> --section <name> --shape <line|ellipse|circle|square|mirror> --file <path/to/sequence.yaml>"
       " [shape-specific args] [--num-points N] [--timeout SEC]\n\n"
    << "  line/ellipse (need --start/--end):\n"
    << "    --start <live|key:NAME> --end <live|key:NAME> [--height-ratio R]  (ellipse only, default 0.5)\n"
    << "  circle (needs --center/--radius):\n"
    << "    --center <live|key:NAME> --radius R [--normal x,y,z]  (default normal 0,0,1)\n"
    << "    [--start-angle-deg D] [--end-angle-deg D]  (default 0..360, full circle)\n"
    << "  square (needs --center/--side-length):\n"
    << "    --center <live|key:NAME> --side-length L [--normal x,y,z]  (default normal 0,0,1)\n"
    << "    NOTE: --num-points means points PER SIDE for square, not a total.\n"
    << "  mirror (needs --source-section): mirrors an EXISTING numbered joint-angle run recorded\n"
    << "    on the OTHER arm (e.g. wavePoses on la) point-by-point - FK each source waypoint,\n"
    << "    reflect the pose across the model-root Y=0 plane (see mirrorAcrossY()), IK-solve it for\n"
    << "    --arm. Unlike line/ellipse/circle/square this isn't a parametric shape: it reproduces\n"
    << "    whatever the source run actually is, one point per source point. --num-points/--start/\n"
    << "    --end/--center/--height-ratio/etc. are not used with this shape.\n"
    << "    --source-section <name>: section holding the source run (e.g. 'wavePoses'); its keys\n"
    << "    must follow the usual '<otherArmPrefix><PascalSection><N>Angle' numbering starting at 1,\n"
    << "    contiguous (stops at the first missing N).\n\n"
    << "  --start/--end/--center: 'live' - capture the arm's current /joint_states now (FK'd to a\n"
    << "    pose) - or 'key:NAME' - read an existing waypoint straight out of --file, no live robot\n"
    << "    needed. Same 'Angle' vs 'Joint' convention as sequence.yaml itself: a '*Angle' key is 7\n"
    << "    joint values (FK'd); anything else (e.g. '*Joint') is read directly as a Cartesian pose\n"
    << "    (x, y, z, qx, qy, qz, qw) - no FK needed. Optional 'mirror:' prefix (e.g.\n"
    << "    mirror:key:laPreWaveJoint) reflects the OTHER arm's pose across the model-root Y=0\n"
    << "    plane instead - see this file's header comment / mirrorAcrossY() for the math and its\n"
    << "    one known limitation. mirror:live is not supported.\n"
    << "  --section must not end in a digit (ambiguous with the auto-numbered point index).\n"
    << "  Run once per arm.\n";
}

// Runs `xacro <xacro_path> <args>` and returns captured stdout, empty on failure.
std::string runXacro(const std::string& xacro_path, const std::string& args)
{
  const std::string cmd = "xacro " + xacro_path + " " + args;
  std::array<char, 4096> buffer{};
  std::string result;
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    return "";
  }
  size_t n;
  while ((n = fread(buffer.data(), 1, buffer.size(), pipe)) > 0) {
    result.append(buffer.data(), n);
  }
  pclose(pipe);
  return result;
}

// Reads a "key: v1, v2, ..., v7" waypoint straight out of a sequence.yaml
// file - no live robot connection needed. Matches the flat "key: comma,
// separated, values" shape every waypoint in this codebase uses.
std::optional<std::vector<double>> readYamlKey(const std::string& file_path, const std::string& key)
{
  std::ifstream in(file_path);
  if (!in.good()) {
    return std::nullopt;
  }
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

// Same convention sequence_converter.cpp's isJointAngle() uses.
bool isJointAngleKey(const std::string& key)
{
  std::string k = key;
  std::transform(k.begin(), k.end(), k.begin(), ::tolower);
  return k.find("angle") != std::string::npos;
}

struct PoseSource {
  bool live = false;
  bool mirror = false;
  std::string key;
};

// Optional "mirror:" prefix, then the existing live/key:NAME syntax.
// mirror:live is intentionally unsupported - if you're live-capturing
// you're already on the target arm, mirroring doesn't apply.
bool parsePoseSource(const std::string& spec_in, PoseSource& out)
{
  std::string spec = spec_in;
  if (spec.rfind("mirror:", 0) == 0) {
    out.mirror = true;
    spec = spec.substr(7);
  }
  if (spec == "live") {
    if (out.mirror) {
      return false;  // mirror:live not supported
    }
    out.live = true;
    return true;
  }
  if (spec.rfind("key:", 0) == 0) {
    out.key = spec.substr(4);
    return !out.key.empty();
  }
  return false;
}

// Reflects a pose across the model-root frame's Y=0 plane - the transform
// between this robot's left/right arm mounts (both mount to
// openarm_body_link0 with identical X/Z and only Y sign-flipped, per
// v10.urdf.xacro:47-51; the mobile-base-to-body chain is translation-only,
// no rotation, so the model-root frame's Y axis stays aligned with
// openarm_body_link0's own Y axis). Position: negate Y. Orientation: the
// standard Householder reflection-conjugation R' = F R F with
// F = diag(1,-1,1) - always yields a valid proper rotation (conjugating by
// a reflection on both sides cancels the determinant flip), unlike a
// hand-wavy "negate a quaternion component" guess.
//
// Known limitation: the real left/right mounts aren't perfectly
// Y-symmetric (0.112 vs 0.118 - a real ~6mm offset, likely a CAD/build
// tolerance), which this doesn't model. Negligible for a wave-shaped
// gesture; verify empirically (see this package's plan/verification notes)
// before trusting it for anything precision-sensitive.
Eigen::Isometry3d mirrorAcrossY(const Eigen::Isometry3d& pose)
{
  const Eigen::Matrix3d reflect = Eigen::Vector3d(1.0, -1.0, 1.0).asDiagonal();
  Eigen::Isometry3d mirrored = Eigen::Isometry3d::Identity();
  mirrored.translation() = Eigen::Vector3d(pose.translation().x(), -pose.translation().y(), pose.translation().z());
  mirrored.linear() = reflect * pose.linear() * reflect;
  return mirrored;
}

// Parses "x,y,z" into an Eigen::Vector3d.
std::optional<Eigen::Vector3d> parseVector3(const std::string& spec)
{
  std::vector<double> values;
  std::stringstream ss(spec);
  std::string token;
  while (std::getline(ss, token, ',')) {
    try {
      values.push_back(std::stod(token));
    } catch (...) {
      return std::nullopt;
    }
  }
  if (values.size() != 3) {
    return std::nullopt;
  }
  return Eigen::Vector3d(values[0], values[1], values[2]);
}

}  // namespace

int main(int argc, char** argv)
{
  std::string arm_prefix, section, shape, start_spec, end_spec, center_spec, normal_spec, file_path;
  std::string source_section;
  int num_points = 20;
  double height_ratio = 0.5;
  double radius = 0.0;
  double side_length = 0.0;
  double start_angle_deg = 0.0;
  double end_angle_deg = 360.0;
  double timeout_sec = 5.0;
  bool have_radius = false, have_side_length = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "[ERROR] " << flag << " requires a value\n";
        std::exit(1);
      }
      return argv[++i];
    };
    if (arg == "--arm") arm_prefix = next("--arm");
    else if (arg == "--section") section = next("--section");
    else if (arg == "--shape") shape = next("--shape");
    else if (arg == "--start") start_spec = next("--start");
    else if (arg == "--end") end_spec = next("--end");
    else if (arg == "--center") center_spec = next("--center");
    else if (arg == "--normal") normal_spec = next("--normal");
    else if (arg == "--file") file_path = next("--file");
    else if (arg == "--source-section") source_section = next("--source-section");
    else if (arg == "--num-points") num_points = std::stoi(next("--num-points"));
    else if (arg == "--height-ratio") height_ratio = std::stod(next("--height-ratio"));
    else if (arg == "--radius") { radius = std::stod(next("--radius")); have_radius = true; }
    else if (arg == "--side-length") { side_length = std::stod(next("--side-length")); have_side_length = true; }
    else if (arg == "--start-angle-deg") start_angle_deg = std::stod(next("--start-angle-deg"));
    else if (arg == "--end-angle-deg") end_angle_deg = std::stod(next("--end-angle-deg"));
    else if (arg == "--timeout") timeout_sec = std::stod(next("--timeout"));
    else if (arg == "--help" || arg == "-h") { printUsage(argv[0]); return 0; }
    else { std::cerr << "[ERROR] Unknown argument: " << arg << "\n"; printUsage(argv[0]); return 1; }
  }

  if (arm_prefix.empty() || section.empty() || shape.empty() || file_path.empty()) {
    std::cerr << "[ERROR] --arm, --section, --shape, --file are all required\n";
    printUsage(argv[0]);
    return 1;
  }
  if (num_points < 1) {
    std::cerr << "[ERROR] --num-points must be >= 1\n";
    return 1;
  }
  if (!section.empty() && std::isdigit(static_cast<unsigned char>(section.back()))) {
    std::cerr << "[ERROR] --section must not end in a digit (ambiguous with the auto-numbered point index)\n";
    return 1;
  }

  const bool is_start_end_shape = (shape == "line" || shape == "ellipse");
  const bool is_center_shape = (shape == "circle" || shape == "square");
  const bool is_mirror_shape = (shape == "mirror");
  if (!is_start_end_shape && !is_center_shape && !is_mirror_shape) {
    std::cerr << "[ERROR] --shape must be one of: line, ellipse, circle, square, mirror\n";
    return 1;
  }
  if (is_start_end_shape && (start_spec.empty() || end_spec.empty())) {
    std::cerr << "[ERROR] --shape " << shape << " requires --start and --end\n";
    return 1;
  }
  if (is_start_end_shape && !center_spec.empty()) {
    std::cerr << "[ERROR] --center is not used by --shape " << shape << "\n";
    return 1;
  }
  if (is_center_shape && center_spec.empty()) {
    std::cerr << "[ERROR] --shape " << shape << " requires --center\n";
    return 1;
  }
  if (is_center_shape && (!start_spec.empty() || !end_spec.empty())) {
    std::cerr << "[ERROR] --start/--end are not used by --shape " << shape << "\n";
    return 1;
  }
  if (shape == "circle" && !have_radius) {
    std::cerr << "[ERROR] --shape circle requires --radius\n";
    return 1;
  }
  if (shape == "square" && !have_side_length) {
    std::cerr << "[ERROR] --shape square requires --side-length\n";
    return 1;
  }
  if (shape != "circle" && have_radius) {
    std::cerr << "[ERROR] --radius is only used by --shape circle\n";
    return 1;
  }
  if (shape != "square" && have_side_length) {
    std::cerr << "[ERROR] --side-length is only used by --shape square\n";
    return 1;
  }
  if (is_mirror_shape && source_section.empty()) {
    std::cerr << "[ERROR] --shape mirror requires --source-section\n";
    return 1;
  }
  if (!is_mirror_shape && !source_section.empty()) {
    std::cerr << "[ERROR] --source-section is only used by --shape mirror\n";
    return 1;
  }
  if (is_mirror_shape && (!start_spec.empty() || !end_spec.empty() || !center_spec.empty())) {
    std::cerr << "[ERROR] --start/--end/--center are not used by --shape mirror\n";
    return 1;
  }

  Eigen::Vector3d normal(0.0, 0.0, 1.0);
  if (!normal_spec.empty()) {
    auto parsed = parseVector3(normal_spec);
    if (!parsed) {
      std::cerr << "[ERROR] --normal must be 'x,y,z'\n";
      return 1;
    }
    normal = *parsed;
  }

  const auto side_it = waypoint_recorder::kArmPrefixToSide.find(arm_prefix);
  if (side_it == waypoint_recorder::kArmPrefixToSide.end() || side_it->second == "left_gripper" ||
      side_it->second == "right_gripper") {
    std::cerr << "[ERROR] --arm must be 'la' or 'ra' (got '" << arm_prefix << "')\n";
    return 1;
  }
  const std::string side = side_it->second;  // "left" | "right"
  const std::string group_name = side + "_arm";
  const std::string ee_link = "openarm_" + side + "_hand_tcp";

  PoseSource start_source, end_source, center_source;
  if (is_start_end_shape &&
      (!parsePoseSource(start_spec, start_source) || !parsePoseSource(end_spec, end_source))) {
    std::cerr << "[ERROR] --start/--end must be 'live' or 'key:<yamlKeyName>'\n";
    return 1;
  }
  if (is_center_shape && !parsePoseSource(center_spec, center_source)) {
    std::cerr << "[ERROR] --center must be 'live' or 'key:<yamlKeyName>'\n";
    return 1;
  }

  static const std::vector<std::string> kJointSuffixes = {"1", "2", "3", "4", "5", "6", "7"};
  std::vector<std::string> joint_names;
  for (const auto& suffix : kJointSuffixes) {
    joint_names.push_back("openarm_" + side + "_joint" + suffix);
  }

  rclcpp::init(argc, argv);

  // openarm_bimanual.srdf is read raw (no ee_type patching, same as
  // run_skills_only.launch.py) - it declares BOTH openarm_hand and
  // amazing_hand gripper groups, only one of which matches any given URDF,
  // so urdf/srdf's parser floods stderr with "Joint ... not known to the
  // URDF" for whichever set doesn't apply. Harmless (left_arm/right_arm -
  // the only groups this tool uses - are unaffected either way), but noisy.
  // Silence console_bridge (the raw, unprefixed "Error:"/"Warning:" lines -
  // not rclcpp's own [INFO]/[WARN] logging, which is untouched by this).
  console_bridge::setLogLevel(console_bridge::CONSOLE_BRIDGE_LOG_NONE);

  // --- Load the robot model for FK/IK. Kinematics only - no planning scene,
  // no collision checking here; see header comment for why that's fine. ---
  auto model_node = std::make_shared<rclcpp::Node>(
    "generate_waypoint_sequence_model",
    rclcpp::NodeOptions().arguments(
      {"--ros-args", "--params-file",
       ament_index_cpp::get_package_share_directory("openarm_moveit_config") + "/config/kinematics.yaml"}));

  const std::string description_pkg = ament_index_cpp::get_package_share_directory("openarm_description");
  const std::string xacro_path = description_pkg + "/urdf/robot/v10.urdf.xacro";
  const std::string urdf_xml = runXacro(xacro_path,
    "bimanual:=true ros2_control:=true use_fake_hardware:=true mobile_base:=true "
    "mobile_base_xyz:='0 0 0.31' mobile_base_body_xyz:='0 0 0'");
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
  auto jmg = robot_model->getJointModelGroup(group_name);
  if (!jmg) {
    std::cerr << "[ERROR] Joint model group '" << group_name << "' not found\n";
    rclcpp::shutdown();
    return 1;
  }

  moveit::core::RobotState state(robot_model);
  state.setToDefaultValues();

  // --- Resolve a named reference to a Cartesian pose: live capture (always
  // joint-space, FK'd), or an existing yaml key - '*Angle' keys are FK'd
  // like live capture, anything else (e.g. '*Joint') is read directly as a
  // pose, matching the project-wide "Angle vs Joint" convention. ---
  auto resolvePose = [&](const PoseSource& source, const char* label) -> std::optional<Eigen::Isometry3d> {
    std::optional<std::vector<double>> joint_values;
    std::optional<Eigen::Isometry3d> resolved;

    if (source.live) {
      auto node = std::make_shared<rclcpp::Node>(std::string("generate_waypoint_sequence_capture_") + label);
      sensor_msgs::msg::JointState::SharedPtr latest;
      auto sub = node->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 10, [&latest](const sensor_msgs::msg::JointState::SharedPtr msg) { latest = msg; });
      rclcpp::executors::SingleThreadedExecutor executor;
      executor.add_node(node);
      const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<int>(timeout_sec * 1000));
      while (!latest) {
        if (std::chrono::steady_clock::now() > deadline) {
          std::cerr << "[ERROR] Timed out waiting for /joint_states (--" << label
                    << " live) - is bringup.launch.py running?\n";
          return std::nullopt;
        }
        executor.spin_some();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      std::vector<double> values(joint_names.size(), 0.0);
      for (size_t j = 0; j < joint_names.size(); ++j) {
        auto it = std::find(latest->name.begin(), latest->name.end(), joint_names[j]);
        if (it == latest->name.end()) {
          std::cerr << "[ERROR] Joint '" << joint_names[j] << "' not found in /joint_states\n";
          return std::nullopt;
        }
        values[j] = latest->position[static_cast<size_t>(std::distance(latest->name.begin(), it))];
      }
      joint_values = values;
    } else {
      auto raw = readYamlKey(file_path, source.key);
      if (!raw || raw->size() != 7) {
        std::cerr << "[ERROR] Could not read a 7-value '" << source.key << "' from " << file_path << " (for --"
                   << label << ")\n";
        return std::nullopt;
      }
      if (isJointAngleKey(source.key)) {
        joint_values = raw;
      } else {
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() = Eigen::Vector3d((*raw)[0], (*raw)[1], (*raw)[2]);
        pose.linear() = Eigen::Quaterniond((*raw)[6], (*raw)[3], (*raw)[4], (*raw)[5]).toRotationMatrix();
        resolved = pose;
      }
    }

    if (!resolved) {
      state.setJointGroupPositions(group_name, *joint_values);
      state.update();
      resolved = state.getGlobalLinkTransform(ee_link);
    }
    return source.mirror ? mirrorAcrossY(*resolved) : *resolved;
  };

  // --- Resolve reference pose(s) and build the full per-point target list.
  // Every shape ends up with one Isometry3d (position + orientation) per
  // point, so the IK loop below is shape-agnostic. ---
  std::vector<Eigen::Isometry3d> targets;

  if (is_start_end_shape) {
    const auto start_pose = resolvePose(start_source, "start");
    const auto end_pose = resolvePose(end_source, "end");
    if (!start_pose || !end_pose) {
      rclcpp::shutdown();
      return 1;
    }
    const Eigen::Quaterniond q_start(start_pose->linear());
    const Eigen::Quaterniond q_end(end_pose->linear());
    std::cout << "[INFO] --start resolved to: " << start_pose->translation().transpose() << " | qxyzw "
              << q_start.x() << " " << q_start.y() << " " << q_start.z() << " " << q_start.w() << "\n";
    std::cout << "[INFO] --end resolved to:   " << end_pose->translation().transpose() << " | qxyzw " << q_end.x()
              << " " << q_end.y() << " " << q_end.z() << " " << q_end.w() << "\n";

    std::vector<Eigen::Vector3d> positions;
    if (shape == "line") {
      positions = trajectory_shapes::line(start_pose->translation(), end_pose->translation(), num_points);
    } else {
      positions =
        trajectory_shapes::ellipseArc(start_pose->translation(), end_pose->translation(), height_ratio, num_points);
    }
    if (positions.empty()) {
      std::cerr << "[ERROR] --start and --end resolve to (nearly) the same pose - cannot draw a " << shape << "\n";
      rclcpp::shutdown();
      return 1;
    }
    for (size_t i = 0; i < positions.size(); ++i) {
      const double t = (positions.size() > 1) ? static_cast<double>(i) / static_cast<double>(positions.size() - 1) : 0.0;
      Eigen::Isometry3d target = Eigen::Isometry3d::Identity();
      target.translation() = positions[i];
      target.linear() = trajectory_shapes::slerp(q_start, q_end, t).toRotationMatrix();
      targets.push_back(target);
    }
  } else if (is_center_shape) {
    const auto center_pose = resolvePose(center_source, "center");
    if (!center_pose) {
      rclcpp::shutdown();
      return 1;
    }
    const Eigen::Quaterniond q = Eigen::Quaterniond(center_pose->linear());  // orientation held constant

    std::vector<Eigen::Vector3d> positions;
    if (shape == "circle") {
      positions = trajectory_shapes::circle(
        center_pose->translation(), radius, normal, num_points, start_angle_deg * M_PI / 180.0,
        end_angle_deg * M_PI / 180.0);
    } else {
      positions = trajectory_shapes::square(center_pose->translation(), side_length, normal, num_points);
    }
    for (const auto& p : positions) {
      Eigen::Isometry3d target = Eigen::Isometry3d::Identity();
      target.translation() = p;
      target.linear() = q.toRotationMatrix();
      targets.push_back(target);
    }
  } else {
    // mirror: read an existing numbered joint-angle run off the OTHER arm,
    // FK + mirror each point - no parametric curve, reproduces exactly
    // whatever the source run is, point for point.
    const std::string source_side = (side == "left") ? "right" : "left";
    const std::string source_arm_prefix = (source_side == "left") ? "la" : "ra";
    const std::string source_group = source_side + "_arm";
    const std::string source_ee_link = "openarm_" + source_side + "_hand_tcp";

    for (int n = 1; ; ++n) {
      const std::string source_key =
        source_arm_prefix + waypoint_recorder::toPascalCase(source_section) + std::to_string(n) + "Angle";
      auto raw = readYamlKey(file_path, source_key);
      if (!raw) {
        break;  // contiguous numbering ends here
      }
      if (raw->size() != 7) {
        std::cerr << "[ERROR] '" << source_key << "' does not have 7 values\n";
        rclcpp::shutdown();
        return 1;
      }
      state.setJointGroupPositions(source_group, *raw);
      state.update();
      const Eigen::Isometry3d source_pose = state.getGlobalLinkTransform(source_ee_link);
      targets.push_back(mirrorAcrossY(source_pose));
    }
    if (targets.empty()) {
      std::cerr << "[ERROR] No waypoints found under source-section '" << source_section << "' for prefix '"
                << source_arm_prefix << "' (expected keys like '" << source_arm_prefix
                << waypoint_recorder::toPascalCase(source_section) << "1Angle')\n";
      rclcpp::shutdown();
      return 1;
    }
    std::cout << "[INFO] Mirroring " << targets.size() << " waypoint(s) from '" << source_side << "_arm' ("
              << source_section << ")\n";
  }

  std::cout << "[INFO] Generating " << targets.size() << " " << shape << " waypoints for '" << arm_prefix
            << "'\n";

  // --- IK-solve each point in order, seeded from the previous solution to
  // keep the IK branch continuous (avoid elbow/wrist flips between points).
  // The very first point seeds from the robot's default pose - we may not
  // know a joint-space start (e.g. a direct pose key was used). ---
  moveit::core::RobotState seed_state(robot_model);
  seed_state.setToDefaultValues();
  seed_state.update();

  int written = 0;
  const size_t total = targets.size();
  for (size_t i = 0; i < total; ++i) {
    const Eigen::Isometry3d& target = targets[i];

    moveit::core::RobotState candidate(seed_state);
    // setFromIK() reports numerical convergence, not joint-limit compliance
    // - it can return true for a solution that's actually out of bounds
    // (this bit us once already: waypoint 7 of an ellipse arc passed IK
    // here but was rejected downstream by Pilz's goal-constraint check on
    // replay, "Joint openarm_left_joint2 violates joint limits"). Check
    // bounds explicitly and treat a violation the same as an IK failure.
    if (!candidate.setFromIK(jmg, target, ee_link) || !candidate.satisfiesBounds(jmg)) {
      std::cerr << "[ERROR] IK failed (or found an out-of-bounds solution) at point " << i << " of " << (total - 1)
                << " - aborting (" << written << " waypoint(s) already written to " << file_path
                << " before this failure)\n";
      rclcpp::shutdown();
      return 1;
    }
    std::vector<double> joint_values;
    candidate.copyJointGroupPositions(group_name, joint_values);

    std::ostringstream value_stream;
    // Default ostream precision (~6 significant digits) can round a value
    // that's very close to a joint limit boundary to something just past
    // the real limit once re-parsed downstream (bit us once already: a
    // solved joint7 of -1.5707963... (~-pi/2) got written as "-1.5708",
    // which is *more* negative than -pi/2, and Pilz's stricter re-check on
    // replay rejected it - even though satisfiesBounds() above passed on
    // the full-precision in-memory value). Write with enough precision that
    // the round-trip through text can't cross a boundary satisfiesBounds()
    // already confirmed.
    value_stream << std::setprecision(12);
    for (size_t j = 0; j < joint_values.size(); ++j) {
      if (j > 0) value_stream << ", ";
      value_stream << joint_values[j];
    }
    const std::string key =
      arm_prefix + waypoint_recorder::toPascalCase(section) + std::to_string(i + 1) + "Angle";
    if (!waypoint_recorder::writeWaypoint(file_path, section, key, value_stream.str())) {
      std::cerr << "[ERROR] Failed to write " << key << " to " << file_path << "\n";
      rclcpp::shutdown();
      return 1;
    }
    std::cout << "[OK] " << key << ": " << value_stream.str() << "\n";
    ++written;

    seed_state = candidate;
  }

  std::cout << "[OK] Wrote " << written << " " << shape << " waypoints under '" << section << ":' in " << file_path
            << "\n";
  rclcpp::shutdown();
  return 0;
}
