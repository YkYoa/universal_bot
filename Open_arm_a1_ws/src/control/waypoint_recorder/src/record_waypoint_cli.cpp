// Thin CLI wrapper over waypoint_recorder (control/waypoint_recorder) -
// records the arm's CURRENT joint positions (e.g. while hand-guiding under
// gravity_compensation_controller) as a named "*Angle" waypoint into
// sequence.yaml. See sequence.yaml's own header comment for the key-rule
// legend, and waypoint_recorder.hpp for the capture/write logic itself
// (shared with any other project/tool that wants to record waypoints).

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "waypoint_recorder/waypoint_recorder.hpp"

namespace
{

void printUsage(const char* prog)
{
  std::cerr << "Usage: " << prog
            << " --arm <la|ra> --section <name> --name <WaypointName> --file <path/to/sequence.yaml>"
               " [--timeout <sec>]\n"
            << "  Records the arm's current /joint_states position as '<arm><WaypointName>Angle'\n"
            << "  under '<section>:' in the given sequence.yaml.\n\n"
            << "  --loop: interactive mode instead of a single --name. Prompts for arm ('la', 'ra',\n"
            << "    or 'both' - 'both' records la+ra together from one /joint_states capture), then\n"
            << "    a section (lists existing ones, or type a new name to create it), then for each\n"
            << "    pose: Enter records the next auto-numbered '<arm><Section><N>Angle', a typed name\n"
            << "    overrides it, 'a' switches arm, 's' switches section, 'q' quits. One session can\n"
            << "    cover both arms and several sections. --arm/--section become optional and just\n"
            << "    skip their respective first prompt.\n"
            << "  --auto-convert: after --loop quits, run sequence_to_bt over the whole file (both\n"
            << "    arms, every section) - requires --bt-out.\n"
            << "  --bt-out <path>: output XML path for --auto-convert.\n";
}

}  // namespace

int main(int argc, char** argv)
{
  std::string arm_prefix, section, name, file_path, bt_out;
  double timeout_sec = 5.0;
  bool loop_mode = false;
  bool auto_convert = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "[ERROR] " << flag << " requires a value\n";
        std::exit(1);
      }
      return argv[++i];
    };
    if (arg == "--arm") {
      arm_prefix = next("--arm");
    } else if (arg == "--section") {
      section = next("--section");
    } else if (arg == "--name") {
      name = next("--name");
    } else if (arg == "--file") {
      file_path = next("--file");
    } else if (arg == "--timeout") {
      timeout_sec = std::stod(next("--timeout"));
    } else if (arg == "--loop") {
      loop_mode = true;
    } else if (arg == "--auto-convert") {
      auto_convert = true;
    } else if (arg == "--bt-out") {
      bt_out = next("--bt-out");
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "[ERROR] Unknown argument: " << arg << "\n";
      printUsage(argv[0]);
      return 1;
    }
  }

  if (file_path.empty()) {
    std::cerr << "[ERROR] --file is always required\n";
    printUsage(argv[0]);
    return 1;
  }
  if (!loop_mode && (arm_prefix.empty() || section.empty() || name.empty())) {
    std::cerr << "[ERROR] --arm, --section, and --name are required unless --loop is given\n";
    printUsage(argv[0]);
    return 1;
  }
  if (loop_mode && !name.empty()) {
    std::cerr << "[ERROR] --name and --loop are mutually exclusive (loop prompts for a name per waypoint)\n";
    return 1;
  }
  if (auto_convert && bt_out.empty()) {
    std::cerr << "[ERROR] --auto-convert requires --bt-out <path/to/output.xml>\n";
    return 1;
  }

  if (!arm_prefix.empty()) {
    const auto side_it = waypoint_recorder::kArmPrefixToSide.find(arm_prefix);
    if (side_it == waypoint_recorder::kArmPrefixToSide.end() || side_it->second == "left_gripper" ||
        side_it->second == "right_gripper") {
      std::cerr << "[ERROR] --arm must be 'la' or 'ra' (got '" << arm_prefix << "') - "
                << "gripper waypoint recording isn't supported by this tool\n";
      return 1;
    }
  }

  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("record_waypoint");
  waypoint_recorder::WaypointRecorder recorder(node, timeout_sec);

  bool ok = true;
  if (!loop_mode) {
    std::string error;
    ok = recorder.recordOne(arm_prefix, section, name, file_path, error);
    if (!ok) {
      std::cerr << "[ERROR] " << error << "\n";
    }
  } else {
    // arm_prefix/section here are just optional starting defaults - the loop
    // can switch either interactively ('a'/'s'), covering both arms and
    // multiple sections in one session.
    recorder.recordLoop(arm_prefix, section, file_path);
  }

  rclcpp::shutdown();

  if (!ok) {
    return 1;
  }

  if (auto_convert) {
    // Loop mode may have touched either/both arms and several sections -
    // convert the whole file (empty --section/--arm filters let
    // sequence_to_bt auto-derive both per key) rather than just what this
    // process started with.
    const std::string section_filter = loop_mode ? "" : section;
    const std::string arm_filter = loop_mode ? "" : waypoint_recorder::kArmPrefixToSide.at(arm_prefix) + "_arm";
    std::ostringstream cmd;
    cmd << "ros2 run openarm_demo sequence_to_bt"
        << " --yaml " << waypoint_recorder::shellQuote(file_path)
        << " --out " << waypoint_recorder::shellQuote(bt_out)
        << " --section " << waypoint_recorder::shellQuote(section_filter)
        << " --arm " << waypoint_recorder::shellQuote(arm_filter);
    std::cout << "[INFO] Auto-converting: " << cmd.str() << "\n";
    int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
      std::cerr << "[ERROR] sequence_to_bt failed (exit " << rc << ")\n";
      return 1;
    }
  }

  return 0;
}
