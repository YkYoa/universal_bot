// Records the arm's CURRENT joint positions (e.g. while hand-guiding under
// gravity_compensation_controller) as a named "*Angle" waypoint into
// sequence.yaml, in the format sequence_converter.cpp / PlanToJointTarget
// expect. See sequence.yaml's own header comment for the key-rule legend.
//
// Deliberately does NOT round-trip the file through yaml-cpp's emitter -
// that would silently discard every comment in the file (yaml-cpp has no
// comment-preserving write path). Instead does a targeted text insertion:
// finds the named section, finds where that section's block ends, splices
// the new (or updated) key line in, leaves every other line untouched.

#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

namespace {

const std::map<std::string, std::string> ARM_PREFIX_TO_SIDE = {
  {"la", "left"}, {"ra", "right"}, {"lh", "left_gripper"}, {"rh", "right_gripper"},
};

void printUsage(const char* prog)
{
  std::cerr << "Usage: " << prog
            << " --arm <la|ra> --section <name> --name <WaypointName> --file <path/to/sequence.yaml>"
               " [--timeout <sec>]\n"
            << "  Records the arm's current /joint_states position as '<arm><WaypointName>Angle'\n"
            << "  under '<section>:' in the given sequence.yaml.\n";
}

// Matches sequence_converter.cpp's toPascalCase(), duplicated here rather
// than shared since it's a few lines and pulling in a cross-package header
// for this one helper isn't worth the coupling.
std::string toPascalCase(const std::string& s)
{
  std::string result;
  bool capitalize_next = true;
  for (char c : s) {
    if (c == '_' || std::isspace(static_cast<unsigned char>(c))) {
      capitalize_next = true;
    } else if (capitalize_next) {
      result += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
      capitalize_next = false;
    } else {
      result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
  }
  return result;
}

bool isBlank(const std::string& line)
{
  return line.find_first_not_of(" \t\r\n") == std::string::npos;
}

bool startsWithWhitespace(const std::string& line)
{
  return !line.empty() && (line[0] == ' ' || line[0] == '\t');
}

// Inserts or updates `key: value` under `section:` in the file at file_path.
// Returns true on success.
bool writeWaypoint(
  const std::string& file_path, const std::string& section, const std::string& key, const std::string& value)
{
  std::ifstream in(file_path);
  if (!in.good()) {
    std::cerr << "[ERROR] Could not open " << file_path << " for reading\n";
    return false;
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    lines.push_back(line);
  }
  in.close();

  const std::string new_entry = "  " + key + ": " + value;
  const std::string section_header = section + ":";

  int section_idx = -1;
  for (size_t i = 0; i < lines.size(); ++i) {
    if (lines[i] == section_header) {
      section_idx = static_cast<int>(i);
      break;
    }
  }

  if (section_idx < 0) {
    // Section doesn't exist yet - append a new one at the end of the file.
    if (!lines.empty() && !isBlank(lines.back())) {
      lines.push_back("");
    }
    lines.push_back(section_header);
    lines.push_back(new_entry);
  } else {
    // Scan the section's body: everything blank or indented, starting right
    // after the header, until the first unindented non-blank line (the next
    // top-level key) or EOF.
    int last_content_idx = section_idx;
    int existing_key_idx = -1;
    size_t i = static_cast<size_t>(section_idx) + 1;
    for (; i < lines.size(); ++i) {
      if (isBlank(lines[i])) {
        continue;
      }
      if (!startsWithWhitespace(lines[i])) {
        break;  // next top-level section
      }
      last_content_idx = static_cast<int>(i);
      // Match "  key:" (possibly with trailing content) for an update-in-place.
      std::string trimmed = lines[i];
      size_t first_non_space = trimmed.find_first_not_of(" \t");
      if (first_non_space != std::string::npos && trimmed.compare(first_non_space, key.size(), key) == 0 &&
          trimmed.size() > first_non_space + key.size() && trimmed[first_non_space + key.size()] == ':') {
        existing_key_idx = static_cast<int>(i);
      }
    }

    if (existing_key_idx >= 0) {
      lines[static_cast<size_t>(existing_key_idx)] = new_entry;
    } else {
      lines.insert(lines.begin() + last_content_idx + 1, new_entry);
    }
  }

  std::ofstream out(file_path);
  if (!out.good()) {
    std::cerr << "[ERROR] Could not open " << file_path << " for writing\n";
    return false;
  }
  for (const auto& l : lines) {
    out << l << "\n";
  }
  return true;
}

}  // namespace

int main(int argc, char** argv)
{
  std::string arm_prefix, section, name, file_path;
  double timeout_sec = 5.0;

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
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "[ERROR] Unknown argument: " << arg << "\n";
      printUsage(argv[0]);
      return 1;
    }
  }

  if (arm_prefix.empty() || section.empty() || name.empty() || file_path.empty()) {
    std::cerr << "[ERROR] --arm, --section, --name, and --file are all required\n";
    printUsage(argv[0]);
    return 1;
  }

  auto side_it = ARM_PREFIX_TO_SIDE.find(arm_prefix);
  if (side_it == ARM_PREFIX_TO_SIDE.end() || side_it->second == "left_gripper" || side_it->second == "right_gripper") {
    std::cerr << "[ERROR] --arm must be 'la' or 'ra' (got '" << arm_prefix << "') - "
              << "gripper waypoint recording isn't supported by this tool\n";
    return 1;
  }
  const std::string side = side_it->second;  // "left" | "right"

  static const std::vector<std::string> kJointSuffixes = {"1", "2", "3", "4", "5", "6", "7"};
  std::vector<std::string> joint_names;
  for (const auto& suffix : kJointSuffixes) {
    joint_names.push_back("openarm_" + side + "_joint" + suffix);
  }

  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("record_waypoint");

  std::promise<sensor_msgs::msg::JointState::SharedPtr> msg_promise;
  auto msg_future = msg_promise.get_future();
  bool promise_set = false;
  auto sub = node->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", 10,
    [&msg_promise, &promise_set](const sensor_msgs::msg::JointState::SharedPtr msg) {
      if (!promise_set) {
        promise_set = true;
        msg_promise.set_value(msg);
      }
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::milliseconds(static_cast<int>(timeout_sec * 1000));
  while (msg_future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
    if (std::chrono::steady_clock::now() > deadline) {
      std::cerr << "[ERROR] Timed out waiting for /joint_states (" << timeout_sec << "s) - "
                << "is bringup.launch.py running?\n";
      rclcpp::shutdown();
      return 1;
    }
    executor.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  auto msg = msg_future.get();
  rclcpp::shutdown();

  std::vector<double> values(joint_names.size(), 0.0);
  for (size_t j = 0; j < joint_names.size(); ++j) {
    auto it = std::find(msg->name.begin(), msg->name.end(), joint_names[j]);
    if (it == msg->name.end()) {
      std::cerr << "[ERROR] Joint '" << joint_names[j] << "' not found in /joint_states - "
                << "is the right arm/side actually publishing?\n";
      return 1;
    }
    values[j] = msg->position[static_cast<size_t>(std::distance(msg->name.begin(), it))];
  }

  std::ostringstream value_stream;
  for (size_t j = 0; j < values.size(); ++j) {
    if (j > 0) value_stream << ", ";
    value_stream << values[j];
  }
  const std::string key = arm_prefix + toPascalCase(name) + "Angle";

  if (!writeWaypoint(file_path, section, key, value_stream.str())) {
    return 1;
  }

  std::cout << "[OK] Recorded " << key << ": " << value_stream.str() << "\n"
            << "     -> " << section << ": in " << file_path << "\n";
  return 0;
}
