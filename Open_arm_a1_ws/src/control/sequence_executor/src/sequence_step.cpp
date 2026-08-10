#include "sequence_executor/sequence_step.hpp"

#include <sstream>

namespace sequence_executor {

bool modeIsCompatible(const std::string& required, const std::string& active)
{
  if (required.empty() || required == kModeAny) {
    return true;
  }
  // A failed probe must not brick every sequence - it just stops being a guard.
  if (active.empty() || active == "unknown") {
    return true;
  }

  // `required` is a '|'-separated set, e.g. "position|mit": a motion step runs
  // in either, because both drive the arm through position commands.
  std::istringstream stream(required);
  std::string option;
  while (std::getline(stream, option, '|')) {
    if (option == active) {
      return true;
    }
  }
  return false;
}

}  // namespace sequence_executor
