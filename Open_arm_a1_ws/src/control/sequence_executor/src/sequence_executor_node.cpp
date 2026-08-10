// -----------------------------------------------------------------------------
// sequence_executor_node
//
// The generic executor: reads sequences from a config/sequence.yaml given by
// the `sequence_yaml_path` parameter, and registers no hardcoded actions.
//
// Pass `sequence_name` to run one sequence at startup, the way this node
// behaved before it became a server. Leave it empty and the node sits in IDLE
// waiting for a RunSequence goal.
//
// A project that wants a sequence store or hardcoded actions builds its own
// executable on top of runApp - see builds/qvic_2026/src/qvic_fsm_node.cpp.
// -----------------------------------------------------------------------------
#include "sequence_executor/executor_app.hpp"

int main(int argc, char** argv)
{
  return sequence_executor::runApp(argc, argv, "sequence_executor_node");
}
