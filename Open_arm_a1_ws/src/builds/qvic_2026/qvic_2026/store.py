"""SQLite store for waypoints, sequences, and their steps.

Replaces config/sequence.yaml as the runtime source of truth. The YAML stays
supported through yaml_sync.py, because waypoint_recorder and
trajectory_waypoint_generator still write it - the workflow is record to YAML,
then import.

Why SQLite and not the YAML: the Android app does live CRUD (add, reorder,
delete steps) while the robot may be running. The YAML path edits the file
with a comment-preserving text splice (waypoint_recorder.cpp), which has no
stable ids, no transactions, and corrupts under concurrent writes.

Pure stdlib, no ROS. Every public function opens and closes its own
connection, so this is safe to call from Flask request threads.
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager

from . import step_types

# The source tree, not the install-space copy. Deriving this from __file__
# would land the database inside site-packages, where the next `colcon build`
# can wipe it. Hardcoded for the same reason launch/qvic_2026.launch.py
# hardcodes SRC_SEQUENCE_YAML: this in-development repo only ever runs from
# this one checkout. Override with QVIC_DB_PATH.
DEFAULT_DB_PATH = "/home/hans/universal_bot/Open_arm_a1_ws/src/builds/qvic_2026/data/sequences.db"

WAYPOINT_KINDS = ("angle", "pose", "hand_yaw", "hand_flex")
CONTROL_MODES = ("any", "position|mit", "torque")

SCHEMA = """
-- Waypoint names are only unique *within* a section: sequence.yaml reuses
-- laHomeAngle in both homePoses and waveHome, and the C++ reader has always
-- resolved them section-first (SequenceYaml::armAngle takes a section). A
-- global UNIQUE(name) here would silently collapse those two rows into one.
CREATE TABLE IF NOT EXISTS waypoints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    section     TEXT NOT NULL,
    arm_prefix  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    values_json TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    UNIQUE(section, name)
);
CREATE INDEX IF NOT EXISTS idx_waypoints_section ON waypoints(section);
CREATE INDEX IF NOT EXISTS idx_waypoints_name ON waypoints(name);

CREATE TABLE IF NOT EXISTS sequences (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    name                  TEXT NOT NULL UNIQUE,
    description           TEXT NOT NULL DEFAULT '',
    arm                   TEXT NOT NULL DEFAULT 'left_arm',
    planner_profile       TEXT NOT NULL DEFAULT '',
    required_control_mode TEXT NOT NULL DEFAULT 'any',
    repeat                INTEGER NOT NULL DEFAULT 1,
    velocity              REAL NOT NULL DEFAULT 0.0,
    acceleration          REAL NOT NULL DEFAULT 0.0,
    builtin               INTEGER NOT NULL DEFAULT 0,
    created_at            REAL NOT NULL,
    updated_at            REAL NOT NULL,
    version               INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS steps (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_id           INTEGER NOT NULL REFERENCES sequences(id) ON DELETE CASCADE,
    idx                   INTEGER NOT NULL,
    name                  TEXT NOT NULL DEFAULT '',
    type                  TEXT NOT NULL,
    params_json           TEXT NOT NULL,
    required_control_mode TEXT NOT NULL DEFAULT 'any',
    enabled               INTEGER NOT NULL DEFAULT 1,
    UNIQUE(sequence_id, idx)
);

CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_name   TEXT NOT NULL,
    started_at      REAL NOT NULL,
    ended_at        REAL,
    result          TEXT NOT NULL DEFAULT 'running',
    fault_reason    TEXT NOT NULL DEFAULT '',
    steps_completed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);
"""


class StoreError(Exception):
    """Anything the caller did wrong: missing row, duplicate name, bad step."""


def db_path():
    return os.environ.get("QVIC_DB_PATH", DEFAULT_DB_PATH)


@contextmanager
def connect(path=None):
    """Open the DB, creating the file and schema if needed.

    Wrapped in a transaction: the body either commits whole or rolls back
    whole, which is the entire reason for moving off the YAML splice.
    """
    path = path or db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── waypoints ─────────────────────────────────────────────────────────────

def upsert_waypoint(name, section, arm_prefix, kind, values, path=None):
    """Insert or replace one waypoint. `values` is a list of floats: 7 for a
    joint angle, 7 for a pose (3 position + 4 quaternion), 4 for a hand pose.
    """
    if kind not in WAYPOINT_KINDS:
        raise StoreError(f"kind must be one of {WAYPOINT_KINDS}, got '{kind}'")
    if not values:
        raise StoreError("waypoint needs at least one value")
    now = time.time()
    with connect(path) as conn:
        conn.execute(
            """
            INSERT INTO waypoints (name, section, arm_prefix, kind, values_json,
                                   created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(section, name) DO UPDATE SET
                arm_prefix  = excluded.arm_prefix,
                kind        = excluded.kind,
                values_json = excluded.values_json,
                updated_at  = excluded.updated_at
            """,
            (name, section, arm_prefix, kind,
             json.dumps([float(v) for v in values]), now, now),
        )
    return get_waypoint(name, section=section, path=path)


def waypoint_ref(section, name):
    """The qualified form a step stores: 'homePoses/laHomeAngle'."""
    return f"{section}/{name}"


def split_ref(ref):
    """Accept either 'section/name' or a bare 'name'."""
    if "/" in ref:
        section, _, name = ref.partition("/")
        return section, name
    return None, ref


def get_waypoint(ref, section=None, path=None):
    """Look up one waypoint by qualified ref, or by name plus explicit section.

    A bare name is allowed only when it is unambiguous, because sequence.yaml
    reuses names across sections.
    """
    parsed_section, name = split_ref(ref)
    section = section or parsed_section
    with connect(path) as conn:
        if section:
            rows = conn.execute(
                "SELECT * FROM waypoints WHERE section = ? AND name = ?", (section, name)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM waypoints WHERE name = ?", (name,)).fetchall()
    if not rows:
        where = f" in section '{section}'" if section else ""
        raise StoreError(f"no waypoint named '{name}'{where}")
    if len(rows) > 1:
        sections = sorted(r["section"] for r in rows)
        raise StoreError(
            f"waypoint '{name}' exists in {sections}; qualify it as 'section/{name}'"
        )
    return _waypoint_dict(rows[0])


def list_waypoints(section=None, path=None):
    with connect(path) as conn:
        if section:
            rows = conn.execute(
                "SELECT * FROM waypoints WHERE section = ? ORDER BY id", (section,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM waypoints ORDER BY section, id").fetchall()
    return [_waypoint_dict(r) for r in rows]


def list_sections(path=None):
    """Section name -> per-kind waypoint counts.

    A 'play section' step needs `angle` > 0; the hand and pose counts are there
    so the UI can tell a replayable section apart from a hand-only one or a
    Cartesian reference block.
    """
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT section, kind, COUNT(*) AS n FROM waypoints"
            " GROUP BY section, kind ORDER BY section"
        ).fetchall()
    out = {}
    for r in rows:
        entry = out.setdefault(r["section"], {k: 0 for k in WAYPOINT_KINDS})
        entry[r["kind"]] = r["n"]
    return out


def delete_waypoint(ref, section=None, path=None):
    parsed_section, name = split_ref(ref)
    section = section or parsed_section
    if not section:
        section = get_waypoint(name, path=path)["section"]
    with connect(path) as conn:
        cur = conn.execute(
            "DELETE FROM waypoints WHERE section = ? AND name = ?", (section, name)
        )
    if cur.rowcount == 0:
        raise StoreError(f"no waypoint named '{name}' in section '{section}'")


def _waypoint_dict(row):
    return {
        "ref": waypoint_ref(row["section"], row["name"]),
        "name": row["name"],
        "section": row["section"],
        "arm_prefix": row["arm_prefix"],
        "kind": row["kind"],
        "values": json.loads(row["values_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# ── sequences ─────────────────────────────────────────────────────────────

def create_sequence(name, description="", arm="left_arm", planner_profile="",
                    repeat=1, velocity=0.0, acceleration=0.0, builtin=False,
                    steps=None, path=None):
    """Create a sequence, optionally with its whole step list in one shot.

    Steps are validated before anything is written, so a bad step list leaves
    no half-built sequence behind.
    """
    prepared = [_prepare_step(s, i) for i, s in enumerate(steps or [])]
    now = time.time()
    with connect(path) as conn:
        if conn.execute("SELECT 1 FROM sequences WHERE name = ?", (name,)).fetchone():
            raise StoreError(f"sequence '{name}' already exists")
        cur = conn.execute(
            """
            INSERT INTO sequences (name, description, arm, planner_profile,
                                   required_control_mode, repeat, velocity,
                                   acceleration, builtin, created_at, updated_at, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (name, description, arm, planner_profile,
             _rollup_mode(prepared), int(repeat), float(velocity),
             float(acceleration), 1 if builtin else 0, now, now),
        )
        _insert_steps(conn, cur.lastrowid, prepared)
    return get_sequence(name, path=path)


def get_sequence(name, path=None):
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM sequences WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise StoreError(f"no sequence named '{name}'")
        steps = conn.execute(
            "SELECT * FROM steps WHERE sequence_id = ? ORDER BY idx", (row["id"],)
        ).fetchall()
    return _sequence_dict(row, steps)


def list_sequences(path=None):
    with connect(path) as conn:
        rows = conn.execute(
            """
            SELECT s.*, (SELECT COUNT(*) FROM steps WHERE sequence_id = s.id) AS num_steps
            FROM sequences s ORDER BY s.name
            """
        ).fetchall()
    return [
        {
            "name": r["name"],
            "description": r["description"],
            "arm": r["arm"],
            "planner_profile": r["planner_profile"],
            "required_control_mode": r["required_control_mode"],
            "repeat": r["repeat"],
            "velocity": r["velocity"],
            "acceleration": r["acceleration"],
            "builtin": bool(r["builtin"]),
            "num_steps": r["num_steps"],
            "version": r["version"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]


UPDATABLE_FIELDS = ("description", "arm", "planner_profile", "repeat",
                    "velocity", "acceleration")


def update_sequence(name, path=None, **fields):
    unknown = set(fields) - set(UPDATABLE_FIELDS)
    if unknown:
        raise StoreError(
            f"cannot update {sorted(unknown)}; updatable: {list(UPDATABLE_FIELDS)}"
        )
    if not fields:
        return get_sequence(name, path=path)
    assignments = ", ".join(f"{k} = ?" for k in fields)
    with connect(path) as conn:
        seq_id = _sequence_id(conn, name)
        conn.execute(
            f"UPDATE sequences SET {assignments}, updated_at = ?, version = version + 1"
            " WHERE id = ?",
            (*fields.values(), time.time(), seq_id),
        )
    return get_sequence(name, path=path)


def rename_sequence(name, new_name, path=None):
    with connect(path) as conn:
        seq_id = _sequence_id(conn, name)
        if conn.execute("SELECT 1 FROM sequences WHERE name = ?", (new_name,)).fetchone():
            raise StoreError(f"sequence '{new_name}' already exists")
        conn.execute(
            "UPDATE sequences SET name = ?, updated_at = ?, version = version + 1 WHERE id = ?",
            (new_name, time.time(), seq_id),
        )
    return get_sequence(new_name, path=path)


def delete_sequence(name, path=None):
    with connect(path) as conn:
        cur = conn.execute("DELETE FROM sequences WHERE name = ?", (name,))
    if cur.rowcount == 0:
        raise StoreError(f"no sequence named '{name}'")


def duplicate_sequence(name, new_name, path=None):
    src = get_sequence(name, path=path)
    return create_sequence(
        new_name,
        description=src["description"],
        arm=src["arm"],
        planner_profile=src["planner_profile"],
        repeat=src["repeat"],
        velocity=src["velocity"],
        acceleration=src["acceleration"],
        steps=[{k: s[k] for k in ("name", "type", "params", "enabled")} for s in src["steps"]],
        path=path,
    )


# ── steps ─────────────────────────────────────────────────────────────────

def add_step(sequence_name, step, index=None, path=None):
    """Insert a step. `index` None appends; otherwise everything from that
    index down shifts by one."""
    prepared = _prepare_step(step, 0)
    with connect(path) as conn:
        seq_id = _sequence_id(conn, sequence_name)
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM steps WHERE sequence_id = ?", (seq_id,)
        ).fetchone()["n"]
        at = count if index is None else max(0, min(int(index), count))
        # Shift upward through a negative window so the UNIQUE(sequence_id, idx)
        # index never sees a duplicate mid-statement.
        conn.execute(
            "UPDATE steps SET idx = -(idx + 1) WHERE sequence_id = ? AND idx >= ?",
            (seq_id, at),
        )
        conn.execute(
            "UPDATE steps SET idx = -idx WHERE sequence_id = ? AND idx < 0", (seq_id,)
        )
        prepared["idx"] = at
        _insert_steps(conn, seq_id, [prepared])
        _touch(conn, seq_id)
        # Raises (and rolls the whole insert back) if this step's control mode
        # contradicts the ones already in the sequence.
        _resync_mode(conn, seq_id)
    return get_sequence(sequence_name, path=path)


def update_step(sequence_name, index, step, path=None):
    """Replace the step at `index`. `step` may carry only the fields that
    change; type and params are re-validated together."""
    with connect(path) as conn:
        seq_id = _sequence_id(conn, sequence_name)
        row = conn.execute(
            "SELECT * FROM steps WHERE sequence_id = ? AND idx = ?", (seq_id, index)
        ).fetchone()
        if row is None:
            raise StoreError(f"sequence '{sequence_name}' has no step at index {index}")

        merged = {
            "name": step.get("name", row["name"]),
            "type": step.get("type", row["type"]),
            "params": step.get("params", json.loads(row["params_json"])),
            "enabled": step.get("enabled", bool(row["enabled"])),
        }
        prepared = _prepare_step(merged, index)
        conn.execute(
            """
            UPDATE steps SET name = ?, type = ?, params_json = ?,
                             required_control_mode = ?, enabled = ?
            WHERE id = ?
            """,
            (prepared["name"], prepared["type"], prepared["params_json"],
             prepared["required_control_mode"], prepared["enabled"], row["id"]),
        )
        _touch(conn, seq_id)
        _resync_mode(conn, seq_id)
    return get_sequence(sequence_name, path=path)


def delete_step(sequence_name, index, path=None):
    with connect(path) as conn:
        seq_id = _sequence_id(conn, sequence_name)
        cur = conn.execute(
            "DELETE FROM steps WHERE sequence_id = ? AND idx = ?", (seq_id, index)
        )
        if cur.rowcount == 0:
            raise StoreError(f"sequence '{sequence_name}' has no step at index {index}")
        conn.execute(
            "UPDATE steps SET idx = idx - 1 WHERE sequence_id = ? AND idx > ?",
            (seq_id, index),
        )
        _touch(conn, seq_id)
        _resync_mode(conn, seq_id)
    return get_sequence(sequence_name, path=path)


def reorder_steps(sequence_name, order, path=None):
    """`order` is the current indices in their new order: [2, 0, 1] moves the
    third step to the front. This is what a drag-and-drop reorder posts."""
    with connect(path) as conn:
        seq_id = _sequence_id(conn, sequence_name)
        rows = conn.execute(
            "SELECT id, idx FROM steps WHERE sequence_id = ? ORDER BY idx", (seq_id,)
        ).fetchall()
        if sorted(order) != list(range(len(rows))):
            raise StoreError(
                f"order must be a permutation of 0..{len(rows) - 1}, got {order}"
            )
        by_idx = {r["idx"]: r["id"] for r in rows}
        # Park every row in the negative range first, same reason as add_step.
        conn.execute(
            "UPDATE steps SET idx = -(idx + 1) WHERE sequence_id = ?", (seq_id,)
        )
        for new_idx, old_idx in enumerate(order):
            conn.execute("UPDATE steps SET idx = ? WHERE id = ?", (new_idx, by_idx[old_idx]))
        _touch(conn, seq_id)
    return get_sequence(sequence_name, path=path)


def replace_steps(sequence_name, steps, path=None):
    """Overwrite the whole step list. Used by yaml_sync on re-import."""
    prepared = [_prepare_step(s, i) for i, s in enumerate(steps)]
    with connect(path) as conn:
        seq_id = _sequence_id(conn, sequence_name)
        conn.execute("DELETE FROM steps WHERE sequence_id = ?", (seq_id,))
        _insert_steps(conn, seq_id, prepared)
        conn.execute(
            "UPDATE sequences SET required_control_mode = ?, updated_at = ?,"
            " version = version + 1 WHERE id = ?",
            (_rollup_mode(prepared), time.time(), seq_id),
        )
    return get_sequence(sequence_name, path=path)


# ── run history ───────────────────────────────────────────────────────────

def start_run(sequence_name, path=None):
    with connect(path) as conn:
        cur = conn.execute(
            "INSERT INTO runs (sequence_name, started_at) VALUES (?, ?)",
            (sequence_name, time.time()),
        )
        return cur.lastrowid


def finish_run(run_id, result, fault_reason="", steps_completed=0, path=None):
    with connect(path) as conn:
        conn.execute(
            "UPDATE runs SET ended_at = ?, result = ?, fault_reason = ?,"
            " steps_completed = ? WHERE id = ?",
            (time.time(), result, fault_reason, int(steps_completed), run_id),
        )


def list_runs(limit=50, path=None):
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(r) for r in rows]


# ── internals ─────────────────────────────────────────────────────────────

def _sequence_id(conn, name):
    row = conn.execute("SELECT id FROM sequences WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise StoreError(f"no sequence named '{name}'")
    return row["id"]


def _prepare_step(step, idx):
    if not isinstance(step, dict):
        raise StoreError("step must be an object")
    step_type = step.get("type")
    if not step_type:
        raise StoreError("step is missing 'type'")
    try:
        params = step_types.validate_step(step_type, step.get("params") or {})
    except step_types.StepValidationError as exc:
        raise StoreError(str(exc)) from exc
    return {
        "idx": idx,
        "name": step.get("name") or step_types.STEP_TYPES[step_type]["label"],
        "type": step_type,
        "params_json": json.dumps(params),
        "required_control_mode": step_types.control_mode_for(step_type),
        "enabled": 1 if step.get("enabled", True) else 0,
    }


def _insert_steps(conn, seq_id, prepared):
    conn.executemany(
        """
        INSERT INTO steps (sequence_id, idx, name, type, params_json,
                           required_control_mode, enabled)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (seq_id, p["idx"], p["name"], p["type"], p["params_json"],
             p["required_control_mode"], p["enabled"])
            for p in prepared
        ],
    )


def _rollup_mode(prepared):
    """A sequence needs whatever its strictest enabled step needs. Mixing
    motion and torque steps is a contradiction the hardware can't satisfy in
    one run, so that is reported here rather than discovered mid-motion."""
    modes = {
        p["required_control_mode"] for p in prepared
        if p["enabled"] and p["required_control_mode"] != step_types.MODE_ANY
    }
    if len(modes) > 1:
        raise StoreError(
            "sequence mixes incompatible control modes " + str(sorted(modes)) +
            "; the arm's mode is fixed at startup, so one sequence can't use both"
        )
    return modes.pop() if modes else step_types.MODE_ANY


def _resync_mode(conn, seq_id):
    rows = conn.execute(
        "SELECT required_control_mode, enabled FROM steps WHERE sequence_id = ?", (seq_id,)
    ).fetchall()
    prepared = [
        {"required_control_mode": r["required_control_mode"], "enabled": r["enabled"]}
        for r in rows
    ]
    conn.execute(
        "UPDATE sequences SET required_control_mode = ? WHERE id = ?",
        (_rollup_mode(prepared), seq_id),
    )


def _touch(conn, seq_id):
    conn.execute(
        "UPDATE sequences SET updated_at = ?, version = version + 1 WHERE id = ?",
        (time.time(), seq_id),
    )


def _sequence_dict(row, step_rows):
    return {
        "name": row["name"],
        "description": row["description"],
        "arm": row["arm"],
        "planner_profile": row["planner_profile"],
        "required_control_mode": row["required_control_mode"],
        "repeat": row["repeat"],
        "velocity": row["velocity"],
        "acceleration": row["acceleration"],
        "builtin": bool(row["builtin"]),
        "version": row["version"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "steps": [
            {
                "index": s["idx"],
                "name": s["name"],
                "type": s["type"],
                "params": json.loads(s["params_json"]),
                "required_control_mode": s["required_control_mode"],
                "enabled": bool(s["enabled"]),
            }
            for s in step_rows
        ],
    }
