#pragma once
// -----------------------------------------------------------------------------
// sequence_source.hpp
//
// Where sequences come from. Two implementations exist:
//
//   YamlSequenceSource  - here, reads config/sequence.yaml through SequenceYaml
//                         and unrolls the old fixed home/body shape into steps.
//                         Kept for dry runs and for launching without a store.
//   SqliteSequenceSource - in builds/qvic_2026, reads the store the Android app
//                         edits. It lives there so this generic package never
//                         takes a SQLite dependency.
//
// Everything throws std::runtime_error with a message meant for a human
// reading a fault banner, not a status code.
// -----------------------------------------------------------------------------
#include <memory>
#include <string>
#include <vector>

#include "sequence_executor/sequence_step.hpp"

namespace sequence_executor {

class SequenceSource
{
public:
  virtual ~SequenceSource() = default;

  virtual std::vector<std::string> listSequences() = 0;
  virtual SequenceSpec loadSequence(const std::string& name) = 0;

  // Waypoint refs are section-qualified ("homePoses/laHomeAngle") because
  // sequence.yaml reuses names across sections - laHomeAngle exists in both
  // homePoses and waveHome with different values.
  virtual std::vector<double> loadWaypoint(const std::string& ref) = 0;

  // Every *Angle waypoint in a section, in order.
  virtual std::vector<std::vector<double>> loadSection(const std::string& section) = 0;

  // Used by the VALIDATING phase to check every reference up front, so a typo
  // fails before the arm moves rather than three steps in.
  virtual bool hasWaypoint(const std::string& ref) = 0;
  virtual bool hasSection(const std::string& section) = 0;

  // Human-readable description of where this source reads from, for logs.
  virtual std::string describe() const = 0;
};

}  // namespace sequence_executor
