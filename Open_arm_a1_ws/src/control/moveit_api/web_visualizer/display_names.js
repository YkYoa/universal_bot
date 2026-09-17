// display_names.js
// -----------------------------------------------------------------------------
// Friendly labels for the raw call names /api/sequences and /api/actions
// return - `builtin:action_02`, `qvic_2026_both`, etc. Purely cosmetic: the
// call name itself (what gets POSTed to /api/sequence/run) never changes,
// only what the picker option's text reads. Kept as one shared table instead
// of duplicated per-page so fsm.html (and anything else that lists
// sequences/actions later) shows the same names.
//
// Source of truth for each entry is the actual behavior in
// builds/qvic_2026/src/qvic_actions.cpp (BuiltinAction.label/.description)
// and builds/qvic_2026/config/sequence.yaml's `sequences:` block - see those
// files if a name here stops matching what the action/sequence actually
// does.
(function (global) {
  'use strict';

  var DISPLAY_NAMES = {
    // ── builtin actions (qvic_actions.cpp) ──────────────────────────────────
    'builtin:action_01': 'Home Both Arms',
    'builtin:action_02': 'Wave – Left Arm (Ellipse Arc)',
    'builtin:action_03': 'Head Sweep (±10°, Continuous)',
    'builtin:action_04': 'Wave – Left Arm (Simple)',
    'builtin:action_05': 'Wave – Right Arm (Simple)',
    'builtin:action_06': 'Loop – Right Arm',
    'builtin:action_07': 'Loop – Left Arm',
    'builtin:action_08': 'Wave – Right Arm (Ellipse Arc)',
    'builtin:action_09': 'Show Pose (Both Arms)',
    'builtin:action_10': 'Head Look Left',
    'builtin:action_11': 'Head Look Right',
    'builtin:action_12': 'Head Center',

    // ── YAML sequences (config/sequence.yaml `sequences:`) ──────────────────
    // qvic_2026_left/right declare ee_type: openarm_hand - this robot boots
    // amazing_hand, so SequenceFsm::validate() rejects them. The qualifier
    // makes that legible in the picker instead of just failing after Run.
    'qvic_2026_left': 'Left Arm Wave (needs openarm_hand)',
    'qvic_2026_right': 'Right Arm Wave (needs openarm_hand)',
    'qvic_2026_both': 'Both Arms – Full Wave (Loop)',
    'qvic_2026_wavepose_bimanual': 'Both Arms – Pose Wave (Loop)',
    'hand_open_close': 'Hands – Open/Close (Loop)',
  };

  /// Friendly label for `callName`, or `callName` itself if it's not in the
  /// table (e.g. a sequence created through the Android app).
  function displayNameFor(callName) {
    return DISPLAY_NAMES[callName] || callName;
  }

  global.DISPLAY_NAMES = DISPLAY_NAMES;
  global.displayNameFor = displayNameFor;
})(window);
