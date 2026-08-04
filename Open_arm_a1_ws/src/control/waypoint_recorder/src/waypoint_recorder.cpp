#include "waypoint_recorder/waypoint_recorder.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

namespace waypoint_recorder
{

const std::map<std::string, std::string> kArmPrefixToSide = {
  {"la", "left"}, {"ra", "right"}, {"lh", "left_gripper"}, {"rh", "right_gripper"},
};

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

std::string trim(const std::string& s)
{
  size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

std::string shellQuote(const std::string& s)
{
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out += c;
    }
  }
  out += "'";
  return out;
}

namespace
{

bool isBlank(const std::string& line)
{
  return line.find_first_not_of(" \t\r\n") == std::string::npos;
}

bool startsWithWhitespace(const std::string& line)
{
  return !line.empty() && (line[0] == ' ' || line[0] == '\t');
}

}  // namespace

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

WaypointRecorder::WaypointRecorder(rclcpp::Node::SharedPtr node, double timeout_sec)
: node_(std::move(node)), timeout_sec_(timeout_sec)
{
  executor_.add_node(node_);
  sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", 10, [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
      latest_msg_ = msg;
      got_new_ = true;
    });
}

sensor_msgs::msg::JointState::SharedPtr WaypointRecorder::captureJointState()
{
  got_new_ = false;
  const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<int>(timeout_sec_ * 1000));
  while (!got_new_) {
    if (std::chrono::steady_clock::now() > deadline) {
      return nullptr;
    }
    executor_.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return latest_msg_;
}

bool WaypointRecorder::recordOne(
  const std::string& arm_prefix, const std::string& section, const std::string& waypoint_name,
  const std::string& file_path, std::string& out_error)
{
  auto side_it = kArmPrefixToSide.find(arm_prefix);
  if (side_it == kArmPrefixToSide.end() || side_it->second == "left_gripper" || side_it->second == "right_gripper") {
    out_error = "--arm must be 'la' or 'ra' (got '" + arm_prefix + "') - gripper waypoint recording isn't supported";
    return false;
  }
  const std::string side = side_it->second;  // "left" | "right"

  static const std::vector<std::string> kJointSuffixes = {"1", "2", "3", "4", "5", "6", "7"};
  std::vector<std::string> joint_names;
  for (const auto& suffix : kJointSuffixes) {
    joint_names.push_back("openarm_" + side + "_joint" + suffix);
  }

  auto msg = captureJointState();
  if (!msg) {
    out_error = "Timed out waiting for /joint_states (" + std::to_string(timeout_sec_) +
      "s) - is bringup.launch.py running?";
    return false;
  }

  std::vector<double> values(joint_names.size(), 0.0);
  for (size_t j = 0; j < joint_names.size(); ++j) {
    auto it = std::find(msg->name.begin(), msg->name.end(), joint_names[j]);
    if (it == msg->name.end()) {
      out_error = "Joint '" + joint_names[j] + "' not found in /joint_states - is the right arm/side publishing?";
      return false;
    }
    values[j] = msg->position[static_cast<size_t>(std::distance(msg->name.begin(), it))];
  }

  std::ostringstream value_stream;
  for (size_t j = 0; j < values.size(); ++j) {
    if (j > 0) value_stream << ", ";
    value_stream << values[j];
  }
  const std::string key = arm_prefix + toPascalCase(waypoint_name) + "Angle";

  if (!writeWaypoint(file_path, section, key, value_stream.str())) {
    out_error = "Failed to write waypoint to " + file_path;
    return false;
  }

  std::cout << "[OK] Recorded " << key << ": " << value_stream.str() << "\n"
            << "     -> " << section << ": in " << file_path << "\n";
  return true;
}

int WaypointRecorder::recordLoop(const std::string& arm_prefix, const std::string& section, const std::string& file_path)
{
  std::cout << "Loop mode - arm '" << arm_prefix << "', section '" << section << "'.\n"
            << "Enter a waypoint name and press Enter to record; empty line to quit.\n";
  int count = 0;
  std::string line;
  while (true) {
    std::cout << "waypoint name> " << std::flush;
    if (!std::getline(std::cin, line)) {
      break;  // EOF (e.g. Ctrl-D)
    }
    const std::string name = trim(line);
    if (name.empty()) {
      break;
    }
    std::string error;
    if (recordOne(arm_prefix, section, name, file_path, error)) {
      ++count;
    } else {
      std::cerr << "[ERROR] " << error << "\n";
    }
  }
  std::cout << "Loop finished - recorded " << count << " waypoint(s).\n";
  return count;
}

}  // namespace waypoint_recorder
