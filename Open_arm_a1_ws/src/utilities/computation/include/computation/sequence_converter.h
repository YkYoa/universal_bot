#pragma once

#include <string>

namespace utilities {
namespace computation {

/**
 * @brief Convert an OpenArm sequence YAML config file into a BehaviorTree.CPP v4 XML file.
 * 
 * @param yaml_path Path to the input sequence.yaml file
 * @param out_path Path to the output behavior tree XML file
 * @param section_filter Optional name of specific section to convert (e.g. "homePoses")
 * @param arm_override Optional string to override the MoveIt planning group/arm for all nodes
 */
void convertSequenceToBt(const std::string& yaml_path, 
                         const std::string& out_path, 
                         const std::string& section_filter = "", 
                         const std::string& arm_override = "");

} // namespace computation
} // namespace utilities
