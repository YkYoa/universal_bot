#pragma once
// -----------------------------------------------------------------------------
// sqlite_sequence_source.hpp
//
// Reads sequences straight out of the store the Android app edits
// (qvic_2026/store.py's schema). Lives in this project package rather than in
// control/sequence_executor so the shared executor never takes a SQLite
// dependency - a different project can point it at something else entirely.
//
// Read-only on purpose: writes go through the store's Python API, which
// validates steps against step_types.py before anything lands in a row. This
// side trusts the schema and only reports what it cannot make sense of.
//
// Opens a connection per call rather than holding one: the Flask API writes to
// the same file from another process, and SQLite in WAL mode handles that
// cleanly as long as readers do not sit on stale snapshots. A sequence is
// loaded once at the start of a run, so the cost is irrelevant.
// -----------------------------------------------------------------------------
#include <string>
#include <vector>

#include "sequence_executor/sequence_source.hpp"

struct sqlite3;

namespace qvic_2026 {

class SqliteSequenceSource : public sequence_executor::SequenceSource
{
public:
  // Throws std::runtime_error if the file cannot be opened or does not carry
  // the expected tables.
  explicit SqliteSequenceSource(const std::string& db_path);

  std::vector<std::string> listSequences() override;
  sequence_executor::SequenceSpec loadSequence(const std::string& name) override;
  std::vector<double> loadWaypoint(const std::string& ref) override;
  std::vector<std::vector<double>> loadSection(const std::string& section) override;
  bool hasWaypoint(const std::string& ref) override;
  bool hasSection(const std::string& section) override;
  std::string describe() const override;

  // Where the store lives when nothing overrides it: QVIC_DB_PATH, else the
  // same source-tree path store.py's DEFAULT_DB_PATH names.
  static std::string defaultPath();

private:
  // RAII around one sqlite3 handle so every early return closes it.
  class Connection
  {
  public:
    explicit Connection(const std::string& path);
    ~Connection();
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    sqlite3* get() const { return handle_; }

  private:
    sqlite3* handle_ = nullptr;
  };

  std::string db_path_;
};

}  // namespace qvic_2026
