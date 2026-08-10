#pragma once
// -----------------------------------------------------------------------------
// sequence_yaml.hpp
//
// Reads sequence.yaml directly - both the pre-existing flat waypoint-data
// sections (homePoses, waveEllipse, ... - la/ra/lh/rh prefix + Angle/Joint
// suffix convention, unchanged from what waypoint_recorder/
// trajectory_waypoint_generator already write) and the new `sequences:`
// block (flat per-entry fields: arm/planner_profile/home_section/
// body_section/body_right_section/body_sections/repeat).
//
// Parsing logic here mirrors utilities/computation/src/sequence_converter.cpp's
// private helpers (parseValue, speedForSection, the home-section key
// detection in convertBothArmsSequenceToBt) - reimplemented directly here
// rather than exposing them from computation's public API, since they were
// anonymous-namespace/private there and this keeps the two packages
// independent during the BT.CPP -> sequence_executor transition.
// -----------------------------------------------------------------------------
#include <string>
#include <vector>
#include <stdexcept>

namespace sequence_executor {

struct SequenceDef
{
  std::string arm;                          // left_arm | right_arm | both_arms
  std::string planner_profile;
  std::string home_section;                 // optional, empty = no home step
  std::string body_section;                 // required unless body_sections is set
  std::string body_right_section;            // optional: interleave with body_section
  std::vector<std::string> body_sections;    // optional, alternative to body_section: hand-pose cycle
  int repeat = 1;                            // -1 = infinite
  std::vector<int> exclude_points;           // optional, 1-indexed waypoint numbers to skip (e.g. a
                                              // known self-collision between the interleaved arms at
                                              // that point - see check_bimanual_collision)
};

// Yaw+flex (4 values each) found in a section under one side prefix
// ("lh"/"rh"), via a generalized `<prefix><anything>Yaw`/`<prefix><anything>Flex`
// key match (e.g. "lhHomeYaw", "lhOpenFlex") - NOT tied to one literal
// middle word, since sections used by body_sections aren't literally "home".
struct HandPose
{
  bool has_yaw = false;
  bool has_flex = false;
  std::vector<double> yaw;
  std::vector<double> flex;
};

struct SectionSpeed
{
  double velocity = -1.0;      // -1 = not set, caller omits the override
  double acceleration = -1.0;
};

class SequenceYaml
{
public:
  explicit SequenceYaml(const std::string& yaml_path);

  // Throws std::runtime_error if `name` isn't in the `sequences:` block.
  SequenceDef sequence(const std::string& name) const;

  // Every entry under `sequences:`, in file order.
  std::vector<std::string> sequenceNames() const;

  // One key's parsed values, looked up section-first because names repeat
  // across sections (laHomeAngle exists in both homePoses and waveHome).
  // Empty if the section or key is absent.
  std::vector<double> value(const std::string& section, const std::string& key) const;

  bool hasSection(const std::string& section) const;

  // Literal "laHomeAngle"/"raHomeAngle"-style lookup (side_prefix "la" or
  // "ra") - used only for home_section's fixed arm pose, matching
  // --home-section's exact-key-name convention. Empty if not present.
  std::vector<double> armAngle(const std::string& section, const std::string& side_prefix) const;

  // Generalized hand yaw/flex scan for one side ("lh" or "rh") within a
  // section - used for both home_section's hand hold and body_sections'
  // per-step hand pose.
  HandPose handPose(const std::string& section, const std::string& side_prefix) const;

  // Every *Angle-valued entry in a section, in file order, as parsed
  // doubles - the waypoint list for a joint_sequence body step.
  std::vector<std::vector<double>> waypoints(const std::string& section) const;

  SectionSpeed speedForSection(const std::string& section) const;

private:
  std::string yaml_path_;
};

}  // namespace sequence_executor
