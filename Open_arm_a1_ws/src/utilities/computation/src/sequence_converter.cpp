#include "computation/sequence_converter.h"

// Standard Library Headers
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Third-Party Headers
#include <yaml-cpp/yaml.h>

namespace utilities {
namespace computation {

namespace {

const std::map<std::string, std::string> ARM_PREFIX = {
    {"la", "left_arm"},
    {"ra", "right_arm"},
    {"lh", "left_gripper"},
    {"rh", "right_gripper"}
};

const std::map<std::string, std::string> PLANNER_TOKEN = {
    {"_Lin", "linear_approach"},
    {"_PTP", "fast_ptp"}
};

std::string toPascalCase(const std::string& s) {
    std::string result;
    bool capitalizeNext = true;
    for (char c : s) {
        if (c == '_' || std::isspace(c)) {
            capitalizeNext = true;
        } else {
            if (capitalizeNext) {
                result += std::toupper(c);
                capitalizeNext = false;
            } else {
                result += std::tolower(c);
            }
        }
    }
    return result;
}

std::string armFromKey(const std::string& key) {
    for (const auto& pair : ARM_PREFIX) {
        if (key.size() >= pair.first.size()) {
            std::string prefix = key.substr(0, pair.first.size());
            std::transform(prefix.begin(), prefix.end(), prefix.begin(), ::tolower);
            if (prefix == pair.first) {
                return pair.second;
            }
        }
    }
    return "";
}

std::pair<std::string, std::vector<std::string>> plannerFromValues(const std::vector<std::string>& tokens) {
    std::string profile = "fast_ptp";
    std::vector<std::string> remaining = tokens;
    if (!remaining.empty()) {
        std::string first = remaining[0];
        // Trim whitespace
        first.erase(0, first.find_first_not_of(" \t\n\r"));
        first.erase(first.find_last_not_of(" \t\n\r") + 1);
        
        auto it = PLANNER_TOKEN.find(first);
        if (it != PLANNER_TOKEN.end()) {
            profile = it->second;
            remaining.erase(remaining.begin());
        }
    }
    return {profile, remaining};
}

bool isJointAngle(const std::string& key) {
    std::string k = key;
    std::transform(k.begin(), k.end(), k.begin(), ::tolower);
    return k.find("angle") != std::string::npos;
}

// Parses waypoint_recorder's "<prefix><N>Angle" auto-numbering convention
// (e.g. "laWavePoses1Angle" -> prefix="laWavePoses", number=1). Requires the
// literal "Angle" suffix (case-sensitive, matching what the recorder always
// emits) - keys that don't match this exact shape (e.g. "laHomeAngle", no
// trailing digits) simply aren't run-merge candidates and fall back to a
// normal single PlanToJointTarget.
bool parseNumberedAngleKey(const std::string& key, std::string& prefix, long& number) {
    static const std::regex re("^(.+?)([0-9]+)Angle$");
    std::smatch m;
    if (!std::regex_match(key, m, re)) {
        return false;
    }
    prefix = m[1].str();
    try {
        number = std::stol(m[2].str());
    } catch (...) {
        return false;
    }
    return true;
}

std::vector<std::string> parseValue(const std::string& raw) {
    std::vector<std::string> tokens;
    std::stringstream ss(raw);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Trim whitespace
        item.erase(0, item.find_first_not_of(" \t\n\r"));
        item.erase(item.find_last_not_of(" \t\n\r") + 1);
        if (!item.empty()) {
            tokens.push_back(item);
        }
    }
    return tokens;
}

std::string getYamlValueAsString(const YAML::Node& node) {
    if (node.IsScalar()) {
        return node.as<std::string>();
    } else if (node.IsSequence()) {
        std::string result;
        for (size_t i = 0; i < node.size(); ++i) {
            if (i > 0) result += ", ";
            result += node[i].as<std::string>();
        }
        return result;
    }
    return "";
}

std::string formatScale(double v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f", v);
    return std::string(buf);
}

// speed: <section>: <velocity>[, <acceleration>] - up to two MoveIt scaling
// factors [0-1] per section. Missing/absent -> -1.0 (caller omits the
// attribute, skill falls back to its profile default).
struct SectionSpeed {
    double velocity = -1.0;
    double acceleration = -1.0;
};

SectionSpeed speedForSection(const std::map<std::string, std::string>& speed_dict, const std::string& section) {
    SectionSpeed speed;
    auto it = speed_dict.find(section);
    if (it == speed_dict.end()) {
        return speed;
    }
    std::vector<std::string> tokens = parseValue(it->second);
    if (tokens.empty()) {
        return speed;
    }
    try {
        speed.velocity = std::stod(tokens[0]);
    } catch (...) {
    }
    if (tokens.size() > 1) {
        try {
            speed.acceleration = std::stod(tokens[1]);
        } catch (...) {
        }
    }
    return speed;
}

} // namespace

void convertSequenceToBt(const std::string& yaml_path, 
                         const std::string& out_path, 
                         const std::string& section_filter, 
                         const std::string& arm_override) {
    
    // Check if YAML file exists
    std::ifstream yaml_file(yaml_path);
    if (!yaml_file.good()) {
        throw std::runtime_error("YAML file not found: " + yaml_path);
    }
    yaml_file.close();

    YAML::Node data;
    try {
        data = YAML::LoadFile(yaml_path);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse YAML file: " + std::string(e.what()));
    }

    if (data.IsNull() || !data.IsMap()) {
        throw std::runtime_error("YAML file is empty or invalid format: " + yaml_path);
    }

    // Extract speed dictionary
    std::map<std::string, std::string> speed_dict;
    if (data["speed"] && data["speed"].IsMap()) {
        for (auto it = data["speed"].begin(); it != data["speed"].end(); ++it) {
            speed_dict[it->first.as<std::string>()] = getYamlValueAsString(it->second);
        }
    }

    // Filter and collect sections
    std::vector<std::pair<std::string, YAML::Node>> sections;
    if (!section_filter.empty()) {
        if (data[section_filter]) {
            sections.push_back({section_filter, data[section_filter]});
        } else {
            throw std::runtime_error("Section '" + section_filter + "' not found in YAML.");
        }
    } else {
        for (auto it = data.begin(); it != data.end(); ++it) {
            std::string key = it->first.as<std::string>();
            if (key != "speed" && it->second.IsMap()) {
                sections.push_back({key, it->second});
            }
        }
    }

    if (sections.empty()) {
        throw std::runtime_error("No convertible sections (dict values) found in YAML.");
    }

    // Extract filename for comments
    size_t last_slash = yaml_path.find_last_of("/\\");
    std::string filename = (last_slash == std::string::npos) ? yaml_path : yaml_path.substr(last_slash + 1);

    // Build XML content
    std::stringstream xml;
    xml << "<?xml version=\"1.0\" ?>\n";
    xml << "<!-- Auto-generated from " << filename << " by sequence_to_bt.py.\n";
    xml << "     DO NOT edit manually — regenerate from the YAML instead.\n";
    xml << "     Wire BB keys for Cartesian poses before executing. -->\n";
    
    std::string first_section_id = toPascalCase(sections[0].first);
    xml << "<root BTCPP_format=\"4\" main_tree_to_execute=\"" << first_section_id << "\">\n";

    for (const auto& section_pair : sections) {
        std::string section_name = section_pair.first;
        YAML::Node section_data = section_pair.second;
        
        std::string bt_id = toPascalCase(section_name);
        xml << "  <BehaviorTree ID=\"" << bt_id << "\">\n";
        xml << "    <Sequence name=\"" << section_name << "\">\n";
        
        SectionSpeed section_speed = speedForSection(speed_dict, section_name);
        std::string last_profile = "";

        // Materialize entries up front (instead of a straight-through
        // iterator loop) so run-merge detection below can look ahead.
        std::vector<std::pair<std::string, YAML::Node>> section_entries;
        for (auto it = section_data.begin(); it != section_data.end(); ++it) {
            section_entries.emplace_back(it->first.as<std::string>(), it->second);
        }

        size_t idx = 0;
        while (idx < section_entries.size()) {
            const std::string& key = section_entries[idx].first;
            std::string raw_val = getYamlValueAsString(section_entries[idx].second);
            std::vector<std::string> tokens = parseValue(raw_val);
            if (tokens.empty()) {
                ++idx;
                continue;
            }

            std::string arm = (!arm_override.empty()) ? arm_override : armFromKey(key);

            // ── Gripper keys ───────────────────────────────────────────────────
            if (arm == "left_gripper" || arm == "right_gripper") {
                std::string lower_key = key;
                std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(), ::tolower);
                std::string arm_name = (arm.find("left") != std::string::npos) ? "left_arm" : "right_arm";

                xml << "      <!-- " << key << " -->\n";
                if (lower_key.find("open") != std::string::npos || lower_key.find("release") != std::string::npos) {
                    xml << "      <OpenGripper arm=\"" << arm_name << "\" open_position=\"0.044\" duration=\"1.0\"/>\n";
                } else {
                    xml << "      <CloseGripper arm=\"" << arm_name << "\" close_position=\"0.0\" duration=\"1.5\"/>\n";
                }
                ++idx;
                continue;
            }

            // ── Head keys — skip ───────────────────────────────────────────────
            std::string lower_key = key;
            std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(), ::tolower);
            if (lower_key.rfind("head", 0) == 0) {
                std::string token_str;
                for (size_t i = 0; i < tokens.size(); ++i) {
                    if (i > 0) token_str += ", ";
                    token_str += tokens[i];
                }
                xml << "      <!-- " << key << ": head move (not yet wired to BT node) — values: " << token_str << " -->\n";
                ++idx;
                continue;
            }

            // ── Arm keys ───────────────────────────────────────────────────────
            if (arm.empty()) {
                std::string token_str;
                for (size_t i = 0; i < tokens.size(); ++i) {
                    if (i > 0) token_str += ", ";
                    token_str += tokens[i];
                }
                xml << "      <!-- config: " << key << " = " << token_str << " -->\n";
                ++idx;
                continue;
            }

            auto planner_res = plannerFromValues(tokens);
            std::string profile = planner_res.first;
            std::vector<std::string> value_tokens = planner_res.second;
            if (value_tokens.empty()) {
                ++idx;
                continue;
            }

            // ── Numbered *Angle run merge ─────────────────────────────────────
            // A consecutive run (N, N+1, N+2...) of waypoint_recorder's
            // "<prefix><N>Angle" auto-numbered keys, sharing arm + planner
            // profile, becomes ONE <PlanToJointSequence> (single blended
            // trajectory, no per-waypoint stop) instead of N independent
            // <PlanToJointTarget> nodes. A run of length 1, non-consecutive
            // numbering, an arm/profile change, or a key that doesn't match
            // the numbered convention at all falls through unchanged to the
            // normal single-key emission below.
            std::string run_prefix;
            long run_number = 0;
            if (isJointAngle(key) && parseNumberedAngleKey(key, run_prefix, run_number)) {
                std::vector<std::vector<std::string>> run_values;
                run_values.push_back(value_tokens);
                long expected_number = run_number + 1;
                size_t j = idx + 1;
                while (j < section_entries.size()) {
                    const std::string& next_key = section_entries[j].first;
                    std::string next_prefix;
                    long next_number = 0;
                    if (!isJointAngle(next_key) || !parseNumberedAngleKey(next_key, next_prefix, next_number)) {
                        break;
                    }
                    if (next_prefix != run_prefix || next_number != expected_number) {
                        break;
                    }
                    std::string next_arm = (!arm_override.empty()) ? arm_override : armFromKey(next_key);
                    if (next_arm != arm) {
                        break;
                    }
                    std::string next_raw = getYamlValueAsString(section_entries[j].second);
                    std::vector<std::string> next_tokens = parseValue(next_raw);
                    if (next_tokens.empty()) {
                        break;
                    }
                    auto next_planner_res = plannerFromValues(next_tokens);
                    if (next_planner_res.first != profile || next_planner_res.second.empty()) {
                        break;
                    }
                    run_values.push_back(next_planner_res.second);
                    ++expected_number;
                    ++j;
                }

                if (run_values.size() >= 2) {
                    if (profile != last_profile) {
                        xml << "      <SetPlannerConfig profile=\"" << profile << "\"/>\n";
                        last_profile = profile;
                    }
                    xml << "      <!-- " << run_prefix << run_number << ".." << (expected_number - 1)
                        << "Angle: " << run_values.size() << " waypoints, blended into one trajectory -->\n";
                    std::string flat_tokens;
                    for (size_t k = 0; k < run_values.size(); ++k) {
                        for (size_t v = 0; v < run_values[k].size(); ++v) {
                            if (k > 0 || v > 0) flat_tokens += ";";
                            flat_tokens += run_values[k][v];
                        }
                    }
                    xml << "      <PlanToJointSequence arm=\"" << arm << "\" joint_sequence=\"" << flat_tokens << "\"";
                    if (section_speed.velocity >= 0.0) {
                        xml << " velocity_scaling=\"" << formatScale(section_speed.velocity) << "\"";
                    }
                    if (section_speed.acceleration >= 0.0) {
                        xml << " acceleration_scaling=\"" << formatScale(section_speed.acceleration) << "\"";
                    }
                    xml << " output_trajectory=\"{plan_trajectory}\"/>\n";

                    idx = j;
                    continue;
                }
                // Run length 1 - fall through to the normal single-key path.
            }

            if (profile != last_profile) {
                xml << "      <SetPlannerConfig profile=\"" << profile << "\"/>\n";
                last_profile = profile;
            }

            std::string val_token_str;
            for (size_t i = 0; i < value_tokens.size(); ++i) {
                if (i > 0) val_token_str += ", ";
                val_token_str += value_tokens[i];
            }

            if (isJointAngle(key)) {
                // PlanToJointTarget takes raw joint values directly (via the
                // "move_to_joint" skill, robot_skills/MoveToJointSkill) - no
                // SRDF named_state needed. joint_targets is a
                // BT::InputPort<std::vector<double>>, semicolon-separated
                // per BT.CPP's default vector parsing convention (sequence.yaml
                // itself stays comma-separated - only the emitted XML attribute
                // uses semicolons).
                std::string semicolon_tokens;
                for (size_t i = 0; i < value_tokens.size(); ++i) {
                    if (i > 0) semicolon_tokens += ";";
                    semicolon_tokens += value_tokens[i];
                }
                xml << "      <!-- " << key << ": joint angles [" << val_token_str << "] -->\n";
                xml << "      <PlanToJointTarget arm=\"" << arm << "\" joint_targets=\"" << semicolon_tokens << "\"";
                if (section_speed.velocity >= 0.0) {
                    xml << " velocity_scaling=\"" << formatScale(section_speed.velocity) << "\"";
                }
                if (section_speed.acceleration >= 0.0) {
                    xml << " acceleration_scaling=\"" << formatScale(section_speed.acceleration) << "\"";
                }
                xml << " output_trajectory=\"{plan_trajectory}\"/>\n";
            } else {
                xml << "      <!-- " << key << ": [" << val_token_str << "]  → set BB key \"" << key << "_pose\" before this node -->\n";
                xml << "      <PlanToPose arm=\"" << arm << "\" target_pose=\"{" << key << "_pose}\"";
                xml << " velocity_scaling=\"" << formatScale(section_speed.velocity >= 0.0 ? section_speed.velocity : 0.3) << "\"";
                if (section_speed.acceleration >= 0.0) {
                    xml << " acceleration_scaling=\"" << formatScale(section_speed.acceleration) << "\"";
                }
                xml << " position_only=\"false\" output_trajectory=\"{plan_trajectory}\"/>\n";
            }
            ++idx;
        }

        xml << "    </Sequence>\n";
        xml << "  </BehaviorTree>\n";
    }
    xml << "</root>\n";

    // Write to file
    std::ofstream out_file(out_path);
    if (!out_file.good()) {
        throw std::runtime_error("Failed to open output file: " + out_path);
    }
    out_file << xml.str();
    out_file.close();
}

} // namespace computation
} // namespace utilities
