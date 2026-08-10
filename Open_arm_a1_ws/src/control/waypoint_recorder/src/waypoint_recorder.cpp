#include "waypoint_recorder/waypoint_recorder.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
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
  char prev = '\0';
  for (char c : s) {
    if (c == '_' || std::isspace(static_cast<unsigned char>(c))) {
      capitalize_next = true;
      continue;
    }
    // An uppercase letter right after a lowercase one is a word boundary in
    // the input's own casing (camelCase/PascalCase) - preserve it instead of
    // forcing it lowercase, so "wavePoses" -> "WavePoses", not "Waveposes".
    const bool camel_boundary =
      std::isupper(static_cast<unsigned char>(c)) && std::islower(static_cast<unsigned char>(prev));
    if (capitalize_next || camel_boundary) {
      result += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    } else {
      result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    capitalize_next = false;
    prev = c;
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

// Reserved top-level keys that aren't a poses/waypoints section (so they're
// left out of the --loop section picker's suggestion list).
const std::set<std::string> kReservedTopLevelKeys = {"speed"};

// Top-level ("unindented, bare 'name:'") section headers already present in
// file_path, in file order, skipping kReservedTopLevelKeys. Empty (not an
// error) if the file can't be read - callers just get no suggestions.
std::vector<std::string> listSections(const std::string& file_path)
{
  std::vector<std::string> sections;
  std::ifstream in(file_path);
  if (!in.good()) {
    return sections;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || startsWithWhitespace(line) || line[0] == '#' || line.back() != ':') {
      continue;
    }
    const std::string key = line.substr(0, line.size() - 1);
    if (key.empty() || kReservedTopLevelKeys.count(key)) {
      continue;
    }
    sections.push_back(key);
  }
  return sections;
}

// Scans `section`'s body in file_path for keys already following the
// "<arm_prefix><PascalCase(section)><N>Angle" auto-numbering scheme and
// returns the next unused N (1 if the section doesn't exist yet, or none of
// its keys match the scheme).
int nextOrdinal(const std::string& file_path, const std::string& arm_prefix, const std::string& section)
{
  std::ifstream in(file_path);
  if (!in.good()) {
    return 1;
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    lines.push_back(line);
  }

  const std::string section_header = section + ":";
  int section_idx = -1;
  for (size_t i = 0; i < lines.size(); ++i) {
    if (lines[i] == section_header) {
      section_idx = static_cast<int>(i);
      break;
    }
  }
  if (section_idx < 0) {
    return 1;
  }

  const std::regex key_re("^\\s*" + arm_prefix + toPascalCase(section) + "(\\d+)Angle\\s*:");
  int max_n = 0;
  for (size_t i = static_cast<size_t>(section_idx) + 1; i < lines.size(); ++i) {
    if (isBlank(lines[i])) {
      continue;
    }
    if (!startsWithWhitespace(lines[i])) {
      break;
    }
    std::smatch m;
    if (std::regex_search(lines[i], m, key_re)) {
      max_n = std::max(max_n, std::stoi(m[1].str()));
    }
  }
  return max_n + 1;
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

namespace
{

// Extracts arm_prefix's 7 joint positions from an already-captured
// JointState message and writes them as a waypoint. Shared by recordOne()
// (captures then writes once) and recordBoth() (captures once, writes
// twice from the same message so both arms' keys reflect the exact same
// instant).
bool writeArmFromMessage(
  const sensor_msgs::msg::JointState::SharedPtr& msg, const std::string& arm_prefix, const std::string& section,
  const std::string& waypoint_name, const std::string& file_path, std::string& out_error)
{
  auto side_it = kArmPrefixToSide.find(arm_prefix);
  if (side_it == kArmPrefixToSide.end() || side_it->second == "left_gripper" || side_it->second == "right_gripper") {
    out_error = "arm must be 'la' or 'ra' (got '" + arm_prefix + "') - gripper waypoint recording isn't supported";
    return false;
  }
  const std::string side = side_it->second;  // "left" | "right"

  static const std::vector<std::string> kJointSuffixes = {"1", "2", "3", "4", "5", "6", "7"};
  std::vector<std::string> joint_names;
  for (const auto& suffix : kJointSuffixes) {
    joint_names.push_back("openarm_" + side + "_joint" + suffix);
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

}  // namespace

bool WaypointRecorder::recordOne(
  const std::string& arm_prefix, const std::string& section, const std::string& waypoint_name,
  const std::string& file_path, std::string& out_error)
{
  auto msg = captureJointState();
  if (!msg) {
    out_error = "Timed out waiting for /joint_states (" + std::to_string(timeout_sec_) +
      "s) - is bringup.launch.py running?";
    return false;
  }
  return writeArmFromMessage(msg, arm_prefix, section, waypoint_name, file_path, out_error);
}

bool WaypointRecorder::recordBoth(
  const std::string& section, const std::string& waypoint_name, const std::string& file_path, std::string& out_error)
{
  auto msg = captureJointState();
  if (!msg) {
    out_error = "Timed out waiting for /joint_states (" + std::to_string(timeout_sec_) +
      "s) - is bringup.launch.py running?";
    return false;
  }
  std::string left_error, right_error;
  const bool left_ok = writeArmFromMessage(msg, "la", section, waypoint_name, file_path, left_error);
  const bool right_ok = writeArmFromMessage(msg, "ra", section, waypoint_name, file_path, right_error);
  if (!left_ok || !right_ok) {
    out_error = left_error + (!left_ok && !right_ok ? "; " : "") + right_error;
    return false;
  }
  return true;
}

int WaypointRecorder::recordLoop(
  const std::string& default_arm, const std::string& default_section, const std::string& file_path)
{
  std::string current_arm = default_arm;
  std::string current_section = default_section;
  int count = 0;
  std::string line;
  while (true) {
    if (current_arm.empty()) {
      // Arm-select phase. 'both' records the same pose on both arms at once
      // from a single capture (matching bimanual joint counts/timing).
      std::cout << "Enter arm ('la', 'ra', or 'both'), empty line to quit> " << std::flush;
      if (!std::getline(std::cin, line)) {
        break;  // EOF (e.g. Ctrl-D)
      }
      const std::string entry = trim(line);
      if (entry.empty()) {
        break;
      }
      if (entry != "both") {
        const auto side_it = kArmPrefixToSide.find(entry);
        if (side_it == kArmPrefixToSide.end() || side_it->second == "left_gripper" ||
            side_it->second == "right_gripper") {
          std::cerr << "[ERROR] Arm must be 'la', 'ra', or 'both' (got '" << entry << "')\n";
          continue;
        }
      }
      current_arm = entry;
      continue;
    }

    if (current_section.empty()) {
      // Section-select phase: show what's already in the file, let the user
      // pick one or type a new name to create it.
      const auto sections = listSections(file_path);
      std::cout << "[" << current_arm << "] Existing sections: ";
      if (sections.empty()) {
        std::cout << "(none yet)";
      } else {
        for (size_t i = 0; i < sections.size(); ++i) {
          if (i > 0) std::cout << ", ";
          std::cout << sections[i];
        }
      }
      std::cout << "\nEnter a section to record into (existing or new), empty line to quit> " << std::flush;
      if (!std::getline(std::cin, line)) {
        break;
      }
      const std::string entry = trim(line);
      if (entry.empty()) {
        break;
      }
      current_section = entry;
      continue;
    }

    // Pose-recording phase: default is the next auto-numbered
    // "<arm><PascalSection><N>Angle" key; a typed name overrides it. For
    // 'both', take the max ordinal across la/ra so laWavePoses3Angle and
    // raWavePoses3Angle stay paired even if one side has more entries.
    const bool both = (current_arm == "both");
    const int n = both ? std::max(nextOrdinal(file_path, "la", current_section), nextOrdinal(file_path, "ra", current_section))
                        : nextOrdinal(file_path, current_arm, current_section);
    const std::string auto_name = current_section + std::to_string(n);
    const std::string preview_key =
      both ? "la/ra" + toPascalCase(auto_name) + "Angle" : current_arm + toPascalCase(auto_name) + "Angle";
    std::cout << "[" << current_arm << "/" << current_section << "] Enter records '" << preview_key
              << "' - or type a custom name; 'a' switches arm, 's' switches section, 'q' quits> " << std::flush;
    if (!std::getline(std::cin, line)) {
      break;
    }
    const std::string entry = trim(line);
    if (entry == "q") {
      break;
    }
    if (entry == "a") {
      current_arm.clear();
      continue;
    }
    if (entry == "s") {
      current_section.clear();
      continue;
    }
    const std::string waypoint_name = entry.empty() ? auto_name : entry;

    std::string error;
    const bool recorded = both ? recordBoth(current_section, waypoint_name, file_path, error)
                                : recordOne(current_arm, current_section, waypoint_name, file_path, error);
    if (recorded) {
      ++count;
    } else {
      std::cerr << "[ERROR] " << error << "\n";
    }
  }
  std::cout << "Loop finished - recorded " << count << " waypoint(s).\n";
  return count;
}

}  // namespace waypoint_recorder
