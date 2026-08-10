#pragma once
// -----------------------------------------------------------------------------
// qvic_actions.hpp
//
// The ten hardcoded competition actions.
//
// This is the slot for behaviour that is easier to write as C++ than to
// express as a stored step list - anything that branches on sensor state, or
// that has to be exactly reproducible on the day. Everything else belongs in
// the store, where the Android app can edit it.
//
// Registering one makes it appear everywhere a stored sequence appears, under
// the name "builtin:<id>": in GET /api/actions, in the web page's list, and as
// a RunSequence goal. There is no second code path.
//
// See qvic_actions.cpp for how to fill one in.
// -----------------------------------------------------------------------------
#include "sequence_executor/builtin_actions.hpp"

namespace qvic_2026 {

// Adds action_01 .. action_10 to `registry`.
void registerQvicActions(sequence_executor::BuiltinActionRegistry& registry);

}  // namespace qvic_2026
