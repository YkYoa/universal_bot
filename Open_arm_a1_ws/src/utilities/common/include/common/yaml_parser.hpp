#pragma once

#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

namespace common
{
    /**
     * @brief Parse a waypoint key from a YAML configuration node.
     * Searches all mapped sections for the given key and returns its value as a std::vector<double>.
     * 
     * Supports both yaml sequence formats: [1.0, 2.0] and comma-separated strings "1.0, 2.0".
     */
    inline std::vector<double> getWaypointFromConfig(const YAML::Node & config, const std::string & key)
    {
        std::vector<double> values;
        if (!config || !config.IsMap()) {
            return values;
        }
        for (auto const & section : config) {
            if (section.second.IsMap() && section.second[key]) {
                YAML::Node node = section.second[key];
                if (node.IsSequence()) {
                    for (size_t i = 0; i < node.size(); ++i) {
                        values.push_back(node[i].as<double>());
                    }
                    return values;
                } else if (node.IsScalar()) {
                    std::string raw = node.as<std::string>();
                    std::stringstream ss(raw);
                    std::string item;
                    while (std::getline(ss, item, ',')) {
                        // Remove leading/trailing spaces
                        item.erase(0, item.find_first_not_of(" \t"));
                        item.erase(item.find_last_not_of(" \t") + 1);
                        if (!item.empty()) {
                            try {
                                values.push_back(std::stod(item));
                            } catch (...) {
                                // Ignore non-numeric strings
                            }
                        }
                    }
                    return values;
                }
            }
        }
        return values;
    }

    /**
     * @brief Print details of a waypoint node.
     */
    inline void printWaypoint(const rclcpp::Logger & logger, const std::string & section, const std::string & name, const YAML::Node & node)
    {
        std::string raw_val;
        if (node.IsScalar()) {
            raw_val = node.as<std::string>();
        } else if (node.IsSequence()) {
            std::stringstream ss;
            for (size_t i = 0; i < node.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << node[i].as<double>();
            }
            raw_val = ss.str();
        }

        // Auto-detect planning group from key prefix
        std::string group = "unknown";
        if (name.rfind("la", 0) == 0) group = "left_arm";
        else if (name.rfind("ra", 0) == 0) group = "right_arm";
        else if (name.rfind("lh", 0) == 0) group = "left_gripper";
        else if (name.rfind("rh", 0) == 0) group = "right_gripper";
        else if (name.rfind("head", 0) == 0) group = "head";

        // Auto-detect targets type
        std::string type = "pose (x, y, z, qx, qy, qz, qw)";
        std::string lower_name = name;
        for (auto & c : lower_name) c = std::tolower(c);
        if (lower_name.find("angle") != std::string::npos || lower_name.find("joint") != std::string::npos) {
            type = "joint angles";
        }

        RCLCPP_INFO(logger, "  -> Waypoint: '%s' in section '%s' (Group: %s, Type: %s)", name.c_str(), section.c_str(), group.c_str(), type.c_str());
        RCLCPP_INFO(logger, "     Values:   [%s]", raw_val.c_str());
    }

    /**
     * @brief Load waypoints YAML file and sketch sequence in logs.
     */
    inline YAML::Node loadAndSketchSequence(const rclcpp::Logger & logger, const std::string & yaml_path)
    {
        YAML::Node config;
        try {
            config = YAML::LoadFile(yaml_path);
        } catch (const std::exception & e) {
            RCLCPP_ERROR(logger, "Failed to load YAML file: %s", e.what());
            return config;
        }

        RCLCPP_INFO(logger, "==================================================");
        RCLCPP_INFO(logger, "       SKETCHING WAYPOINTS FROM YAML CONFIG       ");
        RCLCPP_INFO(logger, "==================================================");

        for (auto const& section : config) {
            std::string section_name = section.first.as<std::string>();
            if (section_name == "speed") {
                RCLCPP_INFO(logger, "[Speed Profiles]");
                for (auto const& speed_entry : section.second) {
                    RCLCPP_INFO(logger, "  - %s: %s", 
                        speed_entry.first.as<std::string>().c_str(),
                        speed_entry.second.as<std::string>().c_str());
                }
                continue;
            }

            RCLCPP_INFO(logger, "[Section: %s]", section_name.c_str());
            if (section.second.IsMap()) {
                for (auto const& pose_entry : section.second) {
                    std::string key = pose_entry.first.as<std::string>();
                    printWaypoint(logger, section_name, key, pose_entry.second);
                }
            } else {
                RCLCPP_INFO(logger, "  (Value is not a map, skipping visualization)");
            }
        }
        RCLCPP_INFO(logger, "==================================================");
        RCLCPP_INFO(logger, "Base initialization completed. Ready for your custom waypoint execution code!");
        return config;
    }
} // namespace common
