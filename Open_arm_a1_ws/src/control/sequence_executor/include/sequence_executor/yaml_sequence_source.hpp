#pragma once
// -----------------------------------------------------------------------------
// yaml_sequence_source.hpp
//
// Reads the legacy config/sequence.yaml and unrolls its fixed shape - home
// once, then a body repeated - into the generic step list the FSM walks:
//
//   home_section          -> move_joint (+ hand_pose if the section has lh/rh keys)
//   body_section[+right]  -> move_joint_sequence
//   body_sections         -> one hand_pose per section
//
// The same mapping exists in Python in qvic_2026/yaml_sync.py, which is what
// imports the YAML into the store. Both have to agree, so a change to one is a
// change to the other; the mapping is spelled out in the qvic_2026 README.
//
// Kept because `sequence_yaml_path` is what every existing launch file passes,
// and because a dry run should not need a database.
// -----------------------------------------------------------------------------
#include <memory>
#include <string>
#include <vector>

#include "sequence_executor/sequence_source.hpp"
#include "sequence_executor/sequence_yaml.hpp"

namespace sequence_executor {

class YamlSequenceSource : public SequenceSource
{
public:
  explicit YamlSequenceSource(const std::string& yaml_path);

  std::vector<std::string> listSequences() override;
  SequenceSpec loadSequence(const std::string& name) override;
  std::vector<double> loadWaypoint(const std::string& ref) override;
  std::vector<std::vector<double>> loadSection(const std::string& section) override;
  bool hasWaypoint(const std::string& ref) override;
  bool hasSection(const std::string& section) override;
  std::string describe() const override;

private:
  std::string yaml_path_;
  std::shared_ptr<SequenceYaml> yaml_;
};

}  // namespace sequence_executor
