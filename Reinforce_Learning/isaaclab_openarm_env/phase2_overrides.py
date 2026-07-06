# Copyright 2026 Enactic, Inc.
#
# Phase-2 gate overrides — một nguồn duy nhất cho demo, train, eval.
# Làm từng công đoạn: --stage reach | grasp | lift | all

from __future__ import annotations

ENV_CFG_SNAPSHOT_KEYS = (
    "task_phase",
    "episode_length_s",
)

# Assist / OSC mechanics shared by all phase-2 stages
PHASE2_BASE = {
    "reach_advance_use_finger_lateral": True,
    "reach_advance_max_lateral": 0.100,
    "reach_advance_max_z_error": 0.08,
    "reach_advance_min_finger_level": 0.80,
    "reach_descent_max_z_error": 0.15,
    "grasp_descent_world_down": True,
    "grasp_descent_in_reach": True,
    "grasp_descent_blend": 1.0,
    "grasp_descent_xy_blend": 0.85,
    "grasp_descent_align_blend": 1.0,
    "grasp_descent_z_finger_loop": True,
    "grasp_descent_force_override": True,
    "grasp_descent_force_in_grasp": True,
    "grasp_descent_min_top_down_grasp": 0.44,
    "grasp_body_height_ratio": 0.52,
    "grasp_pregrasp_freeze_arm": True,
    "grasp_hold_closed": True,
    "grasp_lift_world_up": True,
    "grasp_lift_assist_blend": 1.0,
    "grasp_close_ramp_steps": 0,
    "grasp_close_squeeze_m": 0.0,
    "grasp_slip_reopen_z_finger": 0.0,
    "grasp_reward_lift_scale": 2.0,
    "grasp_reward_lift_hold_scale": 2.0,
    "grasp_success_max_tilt_deg": 12.0,
    "episode_length_s": 20.0,
}

# Công đoạn 1 — REACH: căn ngón; vào GRASP khi đủ gần (GRASP sẽ hạ tiếp tới z_target)
PHASE2_REACH = {
    "reach_advance_min_top_down": 0.58,
    "reach_advance_max_lateral_f": 0.100,
    "reach_advance_max_z_finger": 0.072,
    "grasp_descent_min_top_down": 0.50,
    "grasp_descent_world_m": 0.014,
    "grasp_descent_force_in_reach": True,
    "grasp_grip_arm_damp": 0.12,
}

# Công đoạn 2 — GRASP: hạ tới z_target, căn XY, latch + đóng gripper
PHASE2_GRASP = {
    "grasp_close_min_top_down": 0.55,
    # train_osc_phase2: lat_f ~0.023–0.025 khi hạ; gate 0.022 chặn latch mãi
    "grasp_close_max_lat": 0.032,
    "grasp_close_max_z_err": 0.018,
    "grasp_latch_max_z_finger": 0.016,
    "grasp_close_max_dist_finger": 0.055,
    "grasp_descend_hold_steps": 6,
    "grasp_descent_z_target": 0.012,
    "grasp_descent_world_m": 0.012,
    "grasp_descent_max_bottle_tilt_deg": 10.0,
    "grasp_auto_close": True,
    "grasp_close_ramp_steps": 20,
    "grasp_lift_start_max_tilt_deg": 22.0,
}

# Công đoạn 3 — LIFT: settle → pre-hold → nhấc world +Z
PHASE2_LIFT = {
    "grasp_lift_max_z_finger": 0.026,
    "grasp_lift_max_dist_f": 0.065,
    "grasp_lift_max_lat_f": 0.110,
    "grasp_pre_lift_hold_steps": 2,
    "grasp_lift_settle_steps": 3,
    "grasp_lift_world_m": 0.018,
    "grasp_lift_align_blend": 0.25,
    "grasp_lift_start_max_tilt_deg": 25.0,
    "grasp_lift_max_bottle_tilt_deg": 28.0,
    "grasp_lift_hold_steps": 5,
    "grasp_lift_threshold": 0.03,
}

_VALID_STAGES = frozenset({"reach", "grasp", "lift", "all"})


def parse_stages(stage: str | None) -> frozenset[str]:
    """Map CLI --stage to enabled override blocks."""
    if not stage or stage == "all" or stage == "lift":
        return frozenset({"reach", "grasp", "lift"})
    if stage == "grasp":
        return frozenset({"reach", "grasp"})
    if stage == "reach":
        return frozenset({"reach"})
    raise ValueError(f"Unknown phase2 stage {stage!r}; use reach | grasp | lift | all")


def _apply_overrides(env_cfg, overrides: dict) -> None:
    for key, val in overrides.items():
        setattr(env_cfg, key, val)


def apply_phase2_overrides(
    env_cfg,
    *,
    stages: frozenset[str] | None = None,
    assist_curriculum: bool = False,
) -> frozenset[str]:
    """Apply phase-2 gates. Demo, train, eval dùng cùng hàm này."""
    if getattr(env_cfg, "task_phase", 1) < 2:
        return frozenset()

    active = stages or frozenset({"reach", "grasp", "lift"})
    overrides = dict(PHASE2_BASE)

    if "reach" in active:
        overrides.update(PHASE2_REACH)
    if "grasp" in active:
        overrides.update(PHASE2_GRASP)
    if "lift" in active:
        overrides.update(PHASE2_LIFT)

    has_grasp = "grasp" in active
    has_lift = "lift" in active

    overrides["grasp_descent_assist_enabled"] = has_grasp or assist_curriculum
    overrides["grasp_descent_assist_grasp"] = has_grasp
    overrides["grasp_lift_assist_enabled"] = has_lift

    if assist_curriculum:
        overrides["grasp_assist_schedule_enabled"] = True
        env_cfg.episode_length_s = 25.0

    _apply_overrides(env_cfg, overrides)
    return active


# Back-compat wrappers (demo / train cũ)
def apply_phase2_demo_gates(env_cfg, model_path: str = "", stage: str | None = None) -> str:
    del model_path  # không còn profile theo tên checkpoint
    active = apply_phase2_overrides(env_cfg, stages=parse_stages(stage or "all"))
    return "+".join(sorted(active))


def apply_phase2_train_overrides(
    env_cfg,
    *,
    assist_curriculum: bool = False,
    assist_profile: str | None = None,
    stage: str | None = None,
) -> None:
    del assist_profile  # deprecated
    apply_phase2_overrides(
        env_cfg,
        stages=parse_stages(stage or "all"),
        assist_curriculum=assist_curriculum,
    )
