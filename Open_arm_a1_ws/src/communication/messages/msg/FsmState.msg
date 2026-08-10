# One snapshot of the two-layer robot FSM, published on every transition
# (not on a timer) by sequence_executor_node's ~/state topic. Latched
# (transient_local) so a client that connects mid-run sees the current state
# immediately instead of waiting for the next transition.

builtin_interfaces/Time stamp

# ── Supervisory layer ─────────────────────────────────────────
string robot_state
# "BOOTING" | "IDLE" | "RUNNING" | "PAUSED" | "FAULT" | "ESTOP" | "TEACHING"

# ── Sequence layer ────────────────────────────────────────────
string sequence_name        # empty when robot_state is not RUNNING/PAUSED
string sequence_state
# "" | "LOADING" | "VALIDATING" | "STEP_PLANNING" | "STEP_EXECUTING"
# | "STEP_DONE" | "LOOP_CHECK" | "COMPLETED" | "FAILED" | "CANCELLED"

int32 step_index            # 0-based; -1 when no step is current
int32 step_total
string step_name
string step_type            # matches the step `type` column, e.g. "move_joint"

int32 loop_index            # 0-based iteration counter
int32 loop_total            # -1 = infinite

# ── Context ───────────────────────────────────────────────────
string control_mode_active
# "position" | "mit" | "velocity" | "torque" | "unknown"
# Hardware-fixed at startup (damiao register written once in init), so this
# is what every step's required_control_mode is validated against.

float32 progress            # 0.0 → 1.0 across the whole sequence
string fault_reason         # empty unless FAULT/FAILED
