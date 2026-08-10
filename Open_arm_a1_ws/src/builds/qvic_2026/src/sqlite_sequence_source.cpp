#include "qvic_2026/sqlite_sequence_source.hpp"

#include <cstdlib>
#include <stdexcept>

#include <sqlite3.h>
#include <yaml-cpp/yaml.h>

#include "sequence_executor/step_parser.hpp"

namespace qvic_2026 {

using sequence_executor::SequenceSpec;
using sequence_executor::Step;

namespace {

// Mirrors store.py's DEFAULT_DB_PATH. Hardcoded for the same reason
// launch/qvic_2026.launch.py hardcodes SRC_SEQUENCE_YAML: this repo only ever
// runs from this one checkout, and deriving it from the install space would
// point at a directory colcon can wipe.
constexpr const char* kSourceTreeDb =
  "/home/hans/universal_bot/Open_arm_a1_ws/src/builds/qvic_2026/data/sequences.db";

std::string columnText(sqlite3_stmt* stmt, int index)
{
  const auto* text = sqlite3_column_text(stmt, index);
  return text ? reinterpret_cast<const char*>(text) : std::string();
}

// Statement handle that finalises itself, so a throw mid-loop cannot leak it.
class Statement
{
public:
  Statement(sqlite3* db, const std::string& sql)
  {
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt_, nullptr) != SQLITE_OK) {
      throw std::runtime_error(std::string("sqlite: ") + sqlite3_errmsg(db));
    }
  }
  ~Statement() { sqlite3_finalize(stmt_); }
  Statement(const Statement&) = delete;
  Statement& operator=(const Statement&) = delete;

  void bind(int index, const std::string& value)
  {
    sqlite3_bind_text(stmt_, index, value.c_str(), -1, SQLITE_TRANSIENT);
  }
  bool step() { return sqlite3_step(stmt_) == SQLITE_ROW; }
  sqlite3_stmt* get() const { return stmt_; }

private:
  sqlite3_stmt* stmt_ = nullptr;
};

// The store writes value vectors as a JSON array; yaml-cpp parses JSON, so no
// separate JSON library appears just to read a column.
std::vector<double> parseValues(const std::string& json, const std::string& where)
{
  try {
    YAML::Node node = YAML::Load(json);
    if (!node.IsSequence()) {
      throw std::runtime_error("expected a JSON array");
    }
    std::vector<double> values;
    for (const auto& item : node) {
      values.push_back(item.as<double>());
    }
    return values;
  } catch (const std::exception& e) {
    throw std::runtime_error(where + ": bad values_json (" + e.what() + ")");
  }
}

std::pair<std::string, std::string> splitRef(const std::string& ref)
{
  const auto slash = ref.find('/');
  if (slash == std::string::npos) {
    return {std::string(), ref};
  }
  return {ref.substr(0, slash), ref.substr(slash + 1)};
}

}  // namespace

SqliteSequenceSource::Connection::Connection(const std::string& path)
{
  // READONLY so a bug here can never corrupt what the Android app is editing.
  if (sqlite3_open_v2(path.c_str(), &handle_, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
    const std::string message =
      handle_ ? sqlite3_errmsg(handle_) : "could not open the database";
    sqlite3_close(handle_);
    handle_ = nullptr;
    throw std::runtime_error("sqlite: " + message + " (" + path + ")");
  }
  sqlite3_busy_timeout(handle_, 5000);
}

SqliteSequenceSource::Connection::~Connection()
{
  sqlite3_close(handle_);
}

std::string SqliteSequenceSource::defaultPath()
{
  if (const char* env = std::getenv("QVIC_DB_PATH")) {
    if (env[0] != '\0') {
      return env;
    }
  }
  return kSourceTreeDb;
}

SqliteSequenceSource::SqliteSequenceSource(const std::string& db_path)
  : db_path_(db_path.empty() ? defaultPath() : db_path)
{
  Connection conn(db_path_);
  Statement check(conn.get(),
                  "SELECT name FROM sqlite_master WHERE type='table' AND name='sequences'");
  if (!check.step()) {
    throw std::runtime_error(
      db_path_ + " has no 'sequences' table - seed it with: ros2 run qvic_2026 "
                 "sequence_store_cli.py import --file <pkg>/config/sequence.yaml");
  }
}

std::vector<std::string> SqliteSequenceSource::listSequences()
{
  Connection conn(db_path_);
  Statement query(conn.get(), "SELECT name FROM sequences ORDER BY name");
  std::vector<std::string> names;
  while (query.step()) {
    names.push_back(columnText(query.get(), 0));
  }
  return names;
}

SequenceSpec SqliteSequenceSource::loadSequence(const std::string& name)
{
  Connection conn(db_path_);

  SequenceSpec spec;
  long long sequence_id = 0;
  {
    Statement query(conn.get(),
                    "SELECT id, description, arm, planner_profile, required_control_mode, "
                    "repeat, velocity, acceleration, builtin FROM sequences WHERE name = ?");
    query.bind(1, name);
    if (!query.step()) {
      throw std::runtime_error("no sequence named '" + name + "' in " + db_path_);
    }
    sequence_id = sqlite3_column_int64(query.get(), 0);
    spec.name = name;
    spec.description = columnText(query.get(), 1);
    spec.arm = columnText(query.get(), 2);
    spec.planner_profile = columnText(query.get(), 3);
    spec.required_control_mode = columnText(query.get(), 4);
    spec.repeat = sqlite3_column_int(query.get(), 5);
    spec.velocity = sqlite3_column_double(query.get(), 6);
    spec.acceleration = sqlite3_column_double(query.get(), 7);
    spec.builtin = sqlite3_column_int(query.get(), 8) != 0;
  }

  Statement steps(conn.get(),
                  "SELECT idx, name, type, params_json, required_control_mode, enabled "
                  "FROM steps WHERE sequence_id = ? ORDER BY idx");
  sqlite3_bind_int64(steps.get(), 1, sequence_id);
  while (steps.step()) {
    spec.steps.push_back(sequence_executor::parseStep(
      name,
      sqlite3_column_int(steps.get(), 0),
      columnText(steps.get(), 1),
      columnText(steps.get(), 2),
      columnText(steps.get(), 3),
      columnText(steps.get(), 4),
      sqlite3_column_int(steps.get(), 5) != 0));
  }

  if (spec.steps.empty()) {
    throw std::runtime_error("sequence '" + name + "' has no steps");
  }
  return spec;
}

std::vector<double> SqliteSequenceSource::loadWaypoint(const std::string& ref)
{
  const auto [section, key] = splitRef(ref);
  Connection conn(db_path_);

  // Names repeat across sections, so an unqualified ref is only usable when it
  // happens to be unique - refusing beats silently picking one.
  if (section.empty()) {
    Statement query(conn.get(), "SELECT values_json, section FROM waypoints WHERE name = ?");
    query.bind(1, key);
    if (!query.step()) {
      throw std::runtime_error("no waypoint named '" + key + "'");
    }
    const std::string values = columnText(query.get(), 0);
    const std::string found_section = columnText(query.get(), 1);
    if (query.step()) {
      throw std::runtime_error("waypoint '" + key +
                               "' exists in several sections - qualify it as 'section/name'");
    }
    return parseValues(values, found_section + "/" + key);
  }

  Statement query(conn.get(),
                  "SELECT values_json FROM waypoints WHERE section = ? AND name = ?");
  query.bind(1, section);
  query.bind(2, key);
  if (!query.step()) {
    throw std::runtime_error("no waypoint '" + ref + "' in " + db_path_);
  }
  return parseValues(columnText(query.get(), 0), ref);
}

std::vector<std::vector<double>> SqliteSequenceSource::loadSection(const std::string& section)
{
  Connection conn(db_path_);
  // Insertion order (id), matching the file order the YAML reader used - the
  // waypoints in a section are a path, so the order is the data.
  Statement query(conn.get(),
                  "SELECT name, values_json FROM waypoints "
                  "WHERE section = ? AND kind = 'angle' ORDER BY id");
  query.bind(1, section);

  std::vector<std::vector<double>> waypoints;
  while (query.step()) {
    const std::string name = columnText(query.get(), 0);
    waypoints.push_back(parseValues(columnText(query.get(), 1), section + "/" + name));
  }
  if (waypoints.empty()) {
    throw std::runtime_error("section '" + section + "' has no joint-angle waypoints");
  }
  return waypoints;
}

bool SqliteSequenceSource::hasWaypoint(const std::string& ref)
{
  const auto [section, key] = splitRef(ref);
  Connection conn(db_path_);
  if (section.empty()) {
    Statement query(conn.get(), "SELECT 1 FROM waypoints WHERE name = ?");
    query.bind(1, key);
    return query.step();
  }
  Statement query(conn.get(), "SELECT 1 FROM waypoints WHERE section = ? AND name = ?");
  query.bind(1, section);
  query.bind(2, key);
  return query.step();
}

bool SqliteSequenceSource::hasSection(const std::string& section)
{
  Connection conn(db_path_);
  Statement query(conn.get(),
                  "SELECT 1 FROM waypoints WHERE section = ? AND kind = 'angle' LIMIT 1");
  query.bind(1, section);
  return query.step();
}

std::string SqliteSequenceSource::describe() const
{
  return "sequence store at " + db_path_;
}

}  // namespace qvic_2026
