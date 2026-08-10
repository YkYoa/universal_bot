// -----------------------------------------------------------------------------
// qvic_fsm_node
//
// This project's executor: the shared state machine, pointed at this project's
// sequence store, with this project's hardcoded actions registered.
//
// Parameters (on top of the ones runApp declares):
//   db_path  - the sequence store; empty uses QVIC_DB_PATH, then the
//              source-tree default that store.py also uses.
//
// Falling back: if the store cannot be opened, this exits rather than quietly
// running from the YAML. The two can disagree - the store is what the Android
// app edits - and silently replaying the older one is exactly the surprise
// nobody wants on competition day. Run sequence_executor_node directly for the
// YAML path.
// -----------------------------------------------------------------------------
#include <memory>
#include <string>

#include "qvic_2026/qvic_actions.hpp"
#include "qvic_2026/sqlite_sequence_source.hpp"
#include "sequence_executor/executor_app.hpp"

int main(int argc, char** argv)
{
  return sequence_executor::runApp(
    argc, argv, "sequence_executor_node",
    [](const rclcpp::Node::SharedPtr& node,
       sequence_executor::BuiltinActionRegistry& builtins)
      -> std::shared_ptr<sequence_executor::SequenceSource> {
      qvic_2026::registerQvicActions(builtins);

      const std::string db_path = node->declare_parameter<std::string>("db_path", "");
      auto source = std::make_shared<qvic_2026::SqliteSequenceSource>(db_path);
      RCLCPP_INFO(node->get_logger(), "qvic_2026: %zu builtin action(s), %s",
                  builtins.all().size(), source->describe().c_str());
      return source;
    });
}
