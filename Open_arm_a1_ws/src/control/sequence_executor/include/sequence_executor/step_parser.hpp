#pragma once
// -----------------------------------------------------------------------------
// step_parser.hpp
//
// Builds a Step from the (type, params_json) pair a store row holds.
//
// The JSON is parsed with yaml-cpp, which this package already links: YAML is
// a superset of JSON, so `YAML::Load(json)` handles it and no new dependency
// appears just to read a params blob. Shared by every SequenceSource - the
// SQLite source in qvic_2026 hands its params column straight to this.
//
// Params are written by qvic_2026/store.py, which validates them against
// step_types.py before the row is stored. This side re-checks only what would
// crash or silently misbehave (wrong vector lengths, missing ids), not the
// whole schema.
// -----------------------------------------------------------------------------
#include <string>

#include "sequence_executor/sequence_step.hpp"

namespace sequence_executor {

// Throws std::runtime_error naming the sequence and step index on bad input.
Step parseStep(const std::string& sequence_name, int index, const std::string& name,
               const std::string& type, const std::string& params_json,
               const std::string& required_control_mode, bool enabled);

}  // namespace sequence_executor
