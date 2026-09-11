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

/// Where a sequence and its waypoints come from - see the file header
/// comment for the two implementations (YamlSequenceSource here,
/// SqliteSequenceSource in qvic_2026).
class SequenceSource
{
public:
  virtual ~SequenceSource() = default;

  /// Names of every sequence this source can load.
  virtual std::vector<std::string> listSequences() = 0;
  /// Loads the full SequenceSpec (metadata + steps) for `name`; throws if unknown.
  virtual SequenceSpec loadSequence(const std::string& name) = 0;

  // Waypoint refs are section-qualified ("homePoses/laHomeAngle") because
  // sequence.yaml reuses names across sections - laHomeAngle exists in both
  // homePoses and waveHome with different values.
  /// Loads one waypoint's joint values by its section-qualified ref.
  virtual std::vector<double> loadWaypoint(const std::string& ref) = 0;

  // Every *Angle waypoint in a section, in order.
  /// Loads every waypoint in `section`, in order.
  virtual std::vector<std::vector<double>> loadSection(const std::string& section) = 0;

  // Used by the VALIDATING phase to check every reference up front, so a typo
  // fails before the arm moves rather than three steps in.
  /// True if `ref` resolves to a real waypoint.
  virtual bool hasWaypoint(const std::string& ref) = 0;
  /// True if `section` resolves to a real section.
  virtual bool hasSection(const std::string& section) = 0;

  // Human-readable description of where this source reads from, for logs.
  virtual std::string describe() const = 0;
};

}  // namespace sequence_executor
