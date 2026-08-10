#include "sequence_executor/builtin_actions.hpp"

#include <cstring>

namespace sequence_executor {

void BuiltinActionRegistry::add(BuiltinAction action)
{
  if (action.id.empty()) {
    return;
  }
  if (action.required_control_mode.empty()) {
    action.required_control_mode = kModeAny;
  }

  auto it = by_id_.find(action.id);
  if (it != by_id_.end()) {
    ordered_[it->second] = std::move(action);
    return;
  }
  by_id_[action.id] = ordered_.size();
  ordered_.push_back(std::move(action));
}

const BuiltinAction* BuiltinActionRegistry::find(const std::string& id) const
{
  auto it = by_id_.find(id);
  return it == by_id_.end() ? nullptr : &ordered_[it->second];
}

std::string BuiltinActionRegistry::idFromSequenceName(const std::string& name)
{
  const std::size_t prefix_len = std::strlen(kBuiltinPrefix);
  if (name.rfind(kBuiltinPrefix, 0) != 0) {
    return {};
  }
  return name.substr(prefix_len);
}

}  // namespace sequence_executor
