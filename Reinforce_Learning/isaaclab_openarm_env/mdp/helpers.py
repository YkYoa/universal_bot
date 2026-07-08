# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply

# ── Curriculum Stage Constants ─────────────────────────────────────────────
STAGE_REACH = 0   # Move EE above bottle, avoid table
STAGE_GRASP = 1   # Close gripper and lift bottle
STAGE_PLACE = 2   # Transport bottle to bowl and release

def uses_grasp_lift(task_phase: int | None = None) -> bool:
    """Phase 2: reach + grasp + lift in one episode."""
    return task_phase == 2


def finger_grasp_ready(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """True when fingertip is aligned with bottle body (close / stage gates)."""
    max_lat = getattr(env.cfg, "grasp_close_max_lat", 0.040)
    max_z = getattr(env.cfg, "grasp_close_max_z_err", 0.04)
    min_top = getattr(env.cfg, "grasp_close_min_top_down", 0.88)
    max_dist_f = getattr(
        env.cfg, "grasp_close_max_dist_finger",
        getattr(env.cfg, "grasp_success_max_dist_finger", 0.10),
    )
    return (
        (s["lateral_finger_xy"] < max_lat)
        & (s["z_error_finger"] > -0.04)
        & (s["z_error_finger"] < max_z)
        & (s["top_down_align"] > min_top)
        & (s["dist_finger_body"] < max_dist_f)
    )


def finger_symmetric_ready(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Both pads near bottle body with similar distance/height (mimic gripper safety)."""
    max_close = float(getattr(env.cfg, "grasp_close_max_dist_finger", 0.058))
    max_each = float(getattr(env.cfg, "grasp_sym_max_dist_each", 0.0))
    if max_each <= 0.0:
        max_each = max_close
    max_delta = float(getattr(env.cfg, "grasp_sym_max_dist_delta", 0.012))
    max_z_delta = float(getattr(env.cfg, "grasp_sym_max_z_delta", 0.012))
    dl = s["dist_left_body"]
    dr = s["dist_right_body"]
    zl = s["z_error_finger_left"]
    zr = s["z_error_finger_right"]
    dist_balanced = (dl - dr).abs() < max_delta
    height_balanced = (zl - zr).abs() < max_z_delta
    # Cả hai pad đủ gần (dùng cùng ngưỡng close) + cân bằng L/R
    both_near = (dl < max_each) & (dr < max_each)
    return both_near & dist_balanced & height_balanced


def finger_pad_asymmetric(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """True when L/R pad distances diverge (mimic gripper closing off-center)."""
    max_delta = float(getattr(env.cfg, "grasp_reopen_asym_dist_delta", 0.022))
    return (s["dist_left_body"] - s["dist_right_body"]).abs() > max_delta


def finger_pad_severe_asymmetric(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """L/R lệch nặng — dùng pause ramp / mid-ramp reopen (ngưỡng cao hơn asym nhẹ)."""
    max_delta = float(getattr(env.cfg, "grasp_reopen_severe_asym_delta", 0.032))
    return (s["dist_left_body"] - s["dist_right_body"]).abs() > max_delta


def grip_close_ramp_active(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gripper đang ramp khép (0 < gc < 99%) — tránh descent/lift chen ngang."""
    term = env.action_manager._terms.get("gripper_action")
    ramp_steps = int(getattr(env.cfg, "grasp_close_ramp_steps", 0))
    if ramp_steps <= 0 or term is None or not hasattr(term, "_close_progress"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return (
        term._want_close
        & (term._close_progress > 0.05)
        & (term._close_progress < 0.99)
    )


def finger_ready_for_close(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Align + optional two-pad symmetry before auto-close / latch."""
    ready = finger_grasp_ready(env, s)
    if getattr(env.cfg, "grasp_symmetry_gate_enabled", False):
        ready = ready & finger_symmetric_ready(env, s)
    return ready


def finger_descended_for_close(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """Ngón đã hạ đủ thấp quanh thân chai trước khi đóng gripper."""
    max_z = float(getattr(env.cfg, "grasp_close_max_z_err", 0.04))
    z_target = float(getattr(env.cfg, "grasp_descent_z_target", max_z))
    # Descent assist dùng z_target (~0.004); close_max_z thường lỏng hơn (~0.030).
    # Dùng ngưỡng thấp hơn — nếu không, need_down tắt sớm → kẹp cổ chai, lift fail.
    threshold = min(max_z, z_target) if z_target > 0.0 else max_z
    # z_f đo bằng pad thấp hơn — thêm ε để không kẹt đúng tại z_target (vd 0.010 vs 0.010)
    return s["z_error_finger"] < threshold + 1.5e-3


def grasp_lift_success_ready(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """True when bottle is lifted Δz from spawn AND still held (not knocked away)."""
    grip_thresh = getattr(env.cfg, "grasp_grip_threshold", 0.4)
    lift_thresh = getattr(env.cfg, "grasp_lift_threshold", 0.03)
    max_dist_f = getattr(env.cfg, "grasp_success_max_dist_finger", 0.10)
    max_dist_ee = getattr(env.cfg, "grasp_success_max_dist_ee", 0.12)
    max_lat = getattr(env.cfg, "grasp_success_max_lateral", 0.10)
    min_top = getattr(env.cfg, "grasp_success_min_top_down", 0.50)
    max_speed = getattr(env.cfg, "grasp_success_max_bottle_speed", 0.35)
    max_tilt = getattr(env.cfg, "grasp_success_max_tilt_deg", 12.0)

    gripped = s["gripper_state"] > grip_thresh
    lifted = s["bottle_lift"] > lift_thresh
    finger_close = s["dist_finger_body"] < max_dist_f
    ee_close = s["dist_ee_bottle"] < max_dist_ee
    lat_ok = s["lateral_finger_xy"] < max_lat
    top_ok = s["top_down_align"] > min_top
    bottle_vel = env._bottle.data.root_lin_vel_w
    speed_ok = torch.norm(bottle_vel, dim=-1) < max_speed
    tilt_ok = s["bottle_tilt_deg"] < max_tilt

    return gripped & lifted & finger_close & ee_close & lat_ok & top_ok & speed_ok & tilt_ok


def _reach_lateral_ok(env: ManagerBasedRLEnv, s: dict, tol: float) -> torch.Tensor:
    """Lateral gate for REACH — finger XY when configured (OSC EE offset drifts)."""
    if getattr(env.cfg, "reach_advance_use_finger_lateral", False):
        max_lat = getattr(
            env.cfg, "reach_advance_max_lateral_f",
            getattr(env.cfg, "grasp_close_max_lat", 0.08),
        ) * tol
        return s["lateral_finger_xy"] < max_lat
    max_lat = getattr(env.cfg, "reach_advance_max_lateral", 0.04) * tol
    return s["lateral_dist_xy"] < max_lat


def reach_align_ready(
    env: ManagerBasedRLEnv, s: dict, *, soft: bool = False
) -> torch.Tensor:
    """Gate for REACH→GRASP (strict) or early descent (soft = +tolerance%)."""
    tol = getattr(env.cfg, "reach_advance_tolerance", 1.10) if soft else 1.0
    min_top = getattr(env.cfg, "reach_advance_min_top_down", 0.92)
    max_z = getattr(env.cfg, "reach_advance_max_z_error", 0.065) * tol
    max_z_f = getattr(env.cfg, "reach_advance_max_z_finger", 0.05) * tol
    min_lvl = getattr(env.cfg, "reach_advance_min_finger_level", 0.85)
    return (
        (s["top_down_align"] > min_top)
        & _reach_lateral_ok(env, s, tol)
        & (s["z_error"] < max_z)
        & (s["z_error"] > -0.02)
        & (s["z_error_finger"] < max_z_f)
        & (s["z_error_finger"] > -0.03)
        & (s["finger_level"] > min_lvl)
    )


def reach_descent_ready(env: ManagerBasedRLEnv, s: dict) -> torch.Tensor:
    """XY/top/EE aligned — z_finger excluded; z_err relaxed while descending."""
    tol = getattr(env.cfg, "reach_advance_tolerance", 1.10)
    min_top = getattr(env.cfg, "reach_advance_min_top_down", 0.92)
    max_z = getattr(
        env.cfg, "reach_descent_max_z_error",
        getattr(env.cfg, "reach_advance_max_z_error", 0.08) * 1.5,
    )
    min_lvl = getattr(env.cfg, "reach_advance_min_finger_level", 0.85)
    return (
        (s["top_down_align"] > min_top)
        & _reach_lateral_ok(env, s, tol)
        & (s["z_error"] < max_z)
        & (s["z_error"] > -0.02)
        & (s["finger_level"] > min_lvl)
    )


def check_init_buffers(env: ManagerBasedRLEnv):
    """Dynamically initialize tracking, visualization, and reference buffers on the env instance."""
    if hasattr(env, "_arm_joint_ids"):
        if not hasattr(env, "_bottle_local_xy_offset"):
            env._bottle_local_xy_offset = default_bottle_local_xy_offset(env)
        return

    # Cache robot, bottle, and bowl references
    env._robot = env.scene["robot"]
    env._bottle = env.scene["bottle"]
    env._bowl = env.scene["bowl"]

    # Joint index arrays
    env._arm_joint_ids, _ = env._robot.find_joints([
        "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
        "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
        "openarm_left_joint7"
    ])
    env._gripper_joint_ids, _ = env._robot.find_joints([
        "openarm_left_finger_joint1", "openarm_left_finger_joint2"
    ])

    # End-effector body index — prefer the dedicated TCP link when available
    env._use_tcp_body = False
    try:
        env._ee_body_ids, _ = env._robot.find_bodies(["openarm_left_hand_tcp"])
        env._ee_body_id = env._ee_body_ids[0]
        env._use_tcp_body = True
    except Exception:
        env._ee_body_ids, _ = env._robot.find_bodies(["openarm_left_link7"])
        env._ee_body_id = env._ee_body_ids[0]

    try:
        env._hand_body_ids, _ = env._robot.find_bodies(["openarm_left_hand"])
        env._hand_body_id = env._hand_body_ids[0]
    except Exception:
        env._hand_body_id = env._ee_body_id

    # Left/right finger collision links (URDF: joint1→right, joint2→left mimic)
    env._use_finger_bodies = False
    try:
        env._left_finger_body_ids, _ = env._robot.find_bodies(["openarm_left_left_finger"])
        env._right_finger_body_ids, _ = env._robot.find_bodies(["openarm_left_right_finger"])
        env._left_finger_body_id = env._left_finger_body_ids[0]
        env._right_finger_body_id = env._right_finger_body_ids[0]
        env._use_finger_bodies = True
    except Exception:
        env._left_finger_body_id = env._hand_body_id
        env._right_finger_body_id = env._hand_body_id

    # Fallback TCP offset when the dedicated TCP body is unavailable
    hand_body_name = env._robot.data.body_names[env._hand_body_id]
    if hand_body_name == "openarm_left_hand":
        env._tcp_offset_val = 0.08
    elif hand_body_name == "openarm_left_link7":
        env._tcp_offset_val = 0.1801
    else:
        env._tcp_offset_val = 0.0

    # Bottle geometry — root prim is treated as geometric center (see config.bottle_height_m)
    env._bottle_height = getattr(env.cfg, "bottle_height_m", 0.12)

    # Nominal spawn (sync from scene cfg so config.py init_state is single source of truth)
    spawn = env.cfg.scene.bottle.init_state.pos
    env._bottle_nominal_pos = torch.tensor(list(spawn), dtype=torch.float32, device=env.device)
    bowl_spawn = env.cfg.scene.bowl.init_state.pos
    env._bowl_nominal_pos = torch.tensor(list(bowl_spawn), dtype=torch.float32, device=env.device)

    # TCP safety floor and curriculum variables (aligned with physical table surface at 0.626)
    env._table_z = 0.626
    env.step_counter = 0
    env._stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._steps_near_grasp = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._steps_bottle_lifted = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._steps_hovering_in_grasp = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._steps_in_contact = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._steps_in_grasp = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._lift_settle_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    env._prev_dist_ee_bottle = torch.full((env.num_envs,), 1.0, device=env.device)
    env._prev_dist_bottle_bowl = torch.full((env.num_envs,), 1.0, device=env.device)
    env._bottle_rest_z = env._bottle_nominal_pos[2].expand(env.num_envs).clone()
    env._assist_blend_scale = 1.0
    env._dbg_prev_lift_steps = 0
    env._dbg_logged_grip = False
    env._dbg_logged_lift = False

    # Initialize Coordinate Frame Visualizers (markers)
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG
    
    ee_marker_cfg = FRAME_MARKER_CFG.copy()
    ee_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    env._ee_markers = VisualizationMarkers(ee_marker_cfg.replace(prim_path="/Visuals/ee_marker"))

    # tcp_marker_cfg = FRAME_MARKER_CFG.copy()
    # tcp_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    # env._tcp_markers = VisualizationMarkers(tcp_marker_cfg.replace(prim_path="/Visuals/tcp_marker"))

    bottle_marker_cfg = FRAME_MARKER_CFG.copy()
    bottle_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    env._bottle_markers = VisualizationMarkers(bottle_marker_cfg.replace(prim_path="/Visuals/bottle_marker"))

    # Grasp BODY marker — điểm ngón tay thực sự đặt (body_z + z_offset), nhỏ hơn để phân biệt
    grasp_body_marker_cfg = FRAME_MARKER_CFG.copy()
    grasp_body_marker_cfg.markers["frame"].scale = (0.07, 0.07, 0.07)
    env._grasp_body_markers = VisualizationMarkers(grasp_body_marker_cfg.replace(prim_path="/Visuals/grasp_body_marker"))


def default_bottle_local_xy_offset(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Known mesh-vs-root offset in Bottle local frame (fallback when USD calib skipped)."""
    x = getattr(env.cfg, "bottle_grasp_local_xy_x", 0.0)
    y = getattr(env.cfg, "bottle_grasp_local_xy_y", 0.0)
    return torch.tensor([x, y], dtype=torch.float32, device=env.device)


def calibrate_bottle_local_xy_offset(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-shot USD bbox read — call only during scene setup, never inside step/obs."""
    fallback = default_bottle_local_xy_offset(env)
    if not getattr(env.cfg, "bottle_use_usd_mesh_xy", False):
        return fallback
    try:
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return fallback
        bottle_path = "/World/envs/env_0/Scene/Bottle"
        bottle = stage.GetPrimAtPath(bottle_path)
        if not bottle.IsValid():
            return fallback

        root_xf = UsdGeom.Xformable(bottle).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        root_inv = root_xf.GetInverse()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])

        local_min = Gf.Vec3d(1e9, 1e9, 1e9)
        local_max = Gf.Vec3d(-1e9, -1e9, -1e9)
        found = False
        for prim in Usd.PrimRange(bottle):
            if not prim.IsA(UsdGeom.Boundable):
                continue
            bbox = cache.ComputeWorldBound(prim)
            rng = bbox.GetRange()
            for corner in (rng.GetMin(), rng.GetMax()):
                local = root_inv.Transform(corner)
                local_min = Gf.Vec3d(
                    min(local_min[0], local[0]),
                    min(local_min[1], local[1]),
                    min(local_min[2], local[2]),
                )
                local_max = Gf.Vec3d(
                    max(local_max[0], local[0]),
                    max(local_max[1], local[1]),
                    max(local_max[2], local[2]),
                )
                found = True
        if not found:
            return fallback

        cap_xy = Gf.Vec3d(
            0.5 * (local_min[0] + local_max[0]),
            0.5 * (local_min[1] + local_max[1]),
            local_max[2],
        )
        print(
            f"[BottleCalib] USD local cap XY: ({cap_xy[0]:.4f}, {cap_xy[1]:.4f}) "
            f"| bbox z [{local_min[2]:.3f}, {local_max[2]:.3f}]"
        )
        return torch.tensor([float(cap_xy[0]), float(cap_xy[1])], device=env.device)
    except Exception as exc:
        print(f"[BottleCalib] fallback static offset: {exc}")
        return fallback


def get_gripper_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return gripper state in [0, 1]: 0=open, 1=closed."""
    terms = env.action_manager._terms
    cmd_state = torch.zeros(env.num_envs, device=env.device)
    if "openarm_action" in terms:
        return terms["openarm_action"].gripper_state[:, 0]
    if "gripper_action" in terms:
        raw = terms["gripper_action"].raw_actions[:, 0]
        cmd_state = (raw <= 0.0).float()
    if hasattr(env, "_gripper_joint_ids"):
        finger_pos = env._robot.data.joint_pos[:, env._gripper_joint_ids].mean(dim=1)
        actual = (1.0 - finger_pos / 0.044).clamp(0.0, 1.0)
        return torch.maximum(cmd_state, actual)
    return cmd_state


def _finger_tip_link_offsets(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Collision-pad center in each finger link frame (v10.urdf)."""
    lo = getattr(env.cfg, "finger_tip_local_left", (0.0, -0.05, -0.673001))
    ro = getattr(env.cfg, "finger_tip_local_right", (0.0, 0.05, -0.673001))
    left = torch.tensor(lo, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    right = torch.tensor(ro, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    return left, right


def _finger_tips_hand_frame(
    env: ManagerBasedRLEnv, hand_pos_w: torch.Tensor, hand_quat_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grip-pad centers in hand frame (v10.usd measured 2026-07-02).

    Pads sit z_below from the hand origin along hand-local +Z — the SAME direction
    as the TCP body_offset (0,0,+0.08), which points toward the fingertips.
    (Verified in-sim 2026-07-02: with −z the virtual pads land 8.5cm ABOVE the hand
    and z_f reads +0.20 while the EE is already at the grasp height.)
    Joint travel j∈[0,0.044] moves each pad OUTWARD from ±y_closed to ≈±0.05.
    """
    y_closed = getattr(env.cfg, "finger_tip_hand_y_closed_m", 0.006)
    z_below = getattr(env.cfg, "finger_tip_hand_z_below_m", 0.085)
    n = env.num_envs
    dev = env.device
    if hasattr(env, "_gripper_joint_ids"):
        j1 = env._robot.data.joint_pos[:, env._gripper_joint_ids[0]]  # right finger joint
        j2 = env._robot.data.joint_pos[:, env._gripper_joint_ids[1]]  # left finger (mimic)
    else:
        j1 = j2 = torch.zeros(n, device=dev)
    zeros = torch.zeros(n, device=dev)
    z_col = z_below * torch.ones(n, device=dev)
    left_off = torch.stack([zeros, -(y_closed + j2), z_col], dim=1)
    right_off = torch.stack([zeros, (y_closed + j1), z_col], dim=1)
    left_w = hand_pos_w + quat_apply(hand_quat_w, left_off)
    right_w = hand_pos_w + quat_apply(hand_quat_w, right_off)
    return left_w, right_w


def _finger_tips_world(
    env: ManagerBasedRLEnv,
    hand_pos_w: torch.Tensor,
    hand_quat_w: torch.Tensor,
    tool_z_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Return (left_tip_w, right_tip_w, proxy_tip_w, finger_tip_mode)."""
    finger_reach = getattr(env.cfg, "finger_reach_m", 0.12)
    finger_a_w = hand_pos_w - tool_z_w * finger_reach
    finger_b_w = hand_pos_w + tool_z_w * finger_reach
    proxy_w = torch.where(finger_a_w[:, 2:3] < finger_b_w[:, 2:3], finger_a_w, finger_b_w)

    mode = getattr(env.cfg, "finger_tip_mode", "hand_frame")
    if mode == "proxy":
        return proxy_w, proxy_w, proxy_w, "proxy"

    if mode == "hand_frame":
        left_w, right_w = _finger_tips_hand_frame(env, hand_pos_w, hand_quat_w)
        return left_w, right_w, proxy_w, "hand_frame"

    if not getattr(env, "_use_finger_bodies", False):
        left_w, right_w = _finger_tips_hand_frame(env, hand_pos_w, hand_quat_w)
        return left_w, right_w, proxy_w, "hand_frame"

    left_off, right_off = _finger_tip_link_offsets(env)
    left_quat = env._robot.data.body_quat_w[:, env._left_finger_body_id, :]
    right_quat = env._robot.data.body_quat_w[:, env._right_finger_body_id, :]
    left_origin = env._robot.data.body_pos_w[:, env._left_finger_body_id, :]
    right_origin = env._robot.data.body_pos_w[:, env._right_finger_body_id, :]
    if mode == "collision_offset":
        left_w = left_origin + quat_apply(left_quat, left_off)
        right_w = right_origin + quat_apply(right_quat, right_off)
    else:
        left_w = left_origin
        right_w = right_origin
    return left_w, right_w, proxy_w, mode


def reset_action_terms(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Zero raw actions on all registered action terms after env reset."""
    for name, term in env.action_manager._terms.items():
        if hasattr(term, "reset"):
            term.reset(env_ids)
        elif hasattr(term, "_raw_actions"):
            term._raw_actions[env_ids] = 0.0
        if name == "openarm_action" and hasattr(term, "gripper_state"):
            term.gripper_state[env_ids] = 0.0
        if name == "gripper_action" and hasattr(term, "_raw_actions"):
            # Binary gripper: +1 = open, -1 = close
            term._raw_actions[env_ids, 0] = 1.0
        if name == "gripper_action" and hasattr(term, "_grasp_latched"):
            term._grasp_latched[env_ids] = False
        if name == "gripper_action" and hasattr(term, "_descend_hold"):
            term._descend_hold[env_ids] = 0
        if name == "gripper_action" and hasattr(term, "_sym_hold"):
            term._sym_hold[env_ids] = 0
        if name == "gripper_action" and hasattr(term, "_reopen_cooldown"):
            term._reopen_cooldown[env_ids] = 0
        if name == "gripper_action" and hasattr(term, "_reopen_count"):
            term._reopen_count[env_ids] = 0
        if name == "gripper_action" and hasattr(term, "_close_progress"):
            term._close_progress[env_ids] = 0.0
        if name == "gripper_action" and hasattr(term, "_want_close"):
            term._want_close[env_ids] = False
    if hasattr(env, "_lift_settle_steps"):
        env._lift_settle_steps[env_ids] = 0


def compute_state(env: ManagerBasedRLEnv) -> dict:
    """Compute and cache all coordinate transformations and task state vectors."""
    origins = env.scene.env_origins

    hand_pos_w = env._robot.data.body_pos_w[:, env._hand_body_id, :]
    hand_quat_w = env._robot.data.body_quat_w[:, env._hand_body_id, :]
    ee_quat_w = env._robot.data.body_quat_w[:, env._ee_body_id, :]
    if env._use_tcp_body:
        ee_world = env._robot.data.body_pos_w[:, env._ee_body_id, :]
    else:
        tcp_offset = torch.tensor([0.0, 0.0, env._tcp_offset_val], device=env.device).repeat(env.num_envs, 1)
        ee_world = hand_pos_w + quat_apply(hand_quat_w, tcp_offset)
    ee_pos = ee_world - origins

    bottle_world = env._bottle.data.root_pos_w
    bottle_quat_w = env._bottle.data.root_quat_w
    bottle_pos = bottle_world - origins

    bowl_world = env._bowl.data.root_pos_w
    bowl_pos = bowl_world - origins

    h = env._bottle_height
    half_h = h * 0.5
    table_z = env._table_z
    body_ratio = getattr(env.cfg, "grasp_body_height_ratio", 0.38)
    root_at_base = getattr(env.cfg, "bottle_root_at_base", True)

    # qvic.usd Bottle root = bottom center (NOT geometric center)
    if root_at_base:
        bottom_z = torch.clamp(bottle_pos[:, 2], min=table_z)
    else:
        bottom_z = torch.clamp(bottle_pos[:, 2] - half_h, min=table_z)
    top_ratio = getattr(env.cfg, "bottle_cap_height_ratio", 1.0)
    top_z = bottom_z + h * top_ratio
    body_z = bottom_z + h * body_ratio

    # Grasp offset in bottle LOCAL frame (XY precomputed at setup + user fine-tune)
    if not hasattr(env, "_bottle_local_xy_offset"):
        env._bottle_local_xy_offset = default_bottle_local_xy_offset(env)
    tune_x = getattr(env.cfg, "bottle_grasp_xy_offset_x", 0.0)
    tune_y = getattr(env.cfg, "bottle_grasp_xy_offset_y", 0.0)
    tune_z = getattr(env.cfg, "bottle_grasp_z_offset", 0.0)
    off_local = torch.zeros(env.num_envs, 3, device=env.device)
    off_local[:, 0] = env._bottle_local_xy_offset[0] + tune_x
    off_local[:, 1] = env._bottle_local_xy_offset[1] + tune_y
    off_local[:, 2] = tune_z
    off_world = quat_apply(bottle_quat_w, off_local)

    grasp_pos = bottle_pos.clone()
    grasp_pos[:, 0] += off_world[:, 0]
    grasp_pos[:, 1] += off_world[:, 1]
    grasp_pos[:, 2] = top_z + off_world[:, 2]

    grasp_body_pos = bottle_pos.clone()
    grasp_body_pos[:, 0] += off_world[:, 0]
    grasp_body_pos[:, 1] += off_world[:, 1]
    grasp_body_pos[:, 2] = body_z + off_world[:, 2]
    grasp_body_pos_w = grasp_body_pos + origins

    bottle_bottom_z = bottom_z
    bottle_resting_z = table_z

    # Stage 0 approach target: 15cm above bottle top
    grasp_target_pos = grasp_pos.clone()
    grasp_target_pos[:, 2] += 0.15

    # EE descend target in GRASP: body minus extra offset (TCP must go lower than body center)
    descend_off = getattr(env.cfg, "grasp_ee_descend_offset_m", 0.05)
    grasp_ee_descend_pos = grasp_body_pos.clone()
    grasp_ee_descend_pos[:, 2] = grasp_body_pos[:, 2] - descend_off
    grasp_ee_descend_pos_w = grasp_ee_descend_pos + origins

    hand_pos = hand_pos_w - origins

    # Gripper orientation from TCP body frame (needed for finger-tip geometry)
    z_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    y_local = torch.tensor([0.0, 1.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    tool_z_w = quat_apply(ee_quat_w, z_local)
    tool_y_w = quat_apply(ee_quat_w, y_local)

    left_finger_w, right_finger_w, proxy_finger_w, finger_tip_mode = _finger_tips_world(
        env, hand_pos_w, hand_quat_w, tool_z_w
    )
    # Midpoint for lateral; lower-Z tip for height (top-down contact)
    finger_mid_w = 0.5 * (left_finger_w + right_finger_w)
    lower_mask = left_finger_w[:, 2:3] < right_finger_w[:, 2:3]
    finger_lower_w = torch.where(lower_mask, left_finger_w, right_finger_w)
    finger_pos_w = finger_mid_w if finger_tip_mode != "proxy" else proxy_finger_w
    finger_pos = finger_pos_w - origins
    left_finger_pos = left_finger_w - origins
    right_finger_pos = right_finger_w - origins

    # TCP height when fingers are at body (world +Z, top-down grasp — NOT max along tool_z)
    tcp_above_body = getattr(env.cfg, "grasp_tcp_above_body_m", 0.10)
    grasp_tcp_pos = grasp_body_pos.clone()
    grasp_tcp_pos[:, 2] = grasp_body_pos[:, 2] + tcp_above_body
    grasp_tcp_pos_w = grasp_tcp_pos + origins

    ee_to_grasp = grasp_target_pos - ee_pos
    ee_to_grasp_body = grasp_pos - ee_pos
    ee_to_grasp_finger = grasp_body_pos - hand_pos
    bottle_to_bowl = bowl_pos - bottle_pos

    dist_ee_grasp = torch.norm(ee_to_grasp, dim=-1)
    dist_ee_bottle = torch.norm(ee_to_grasp_body, dim=-1)
    lateral_dist_xy = torch.norm(ee_pos[:, :2] - grasp_pos[:, :2], dim=-1)
    z_error = ee_pos[:, 2] - grasp_pos[:, 2]

    dist_hand_body = torch.norm(ee_to_grasp_finger, dim=-1)
    lateral_hand_xy = torch.norm(hand_pos[:, :2] - grasp_body_pos[:, :2], dim=-1)
    lateral_finger_xy = torch.norm(finger_pos[:, :2] - grasp_body_pos[:, :2], dim=-1)
    z_error_hand = hand_pos[:, 2] - grasp_body_pos[:, 2]
    finger_lower_pos = finger_lower_w - origins
    z_error_finger = finger_lower_pos[:, 2] - grasp_body_pos[:, 2]
    z_error_finger_left = left_finger_pos[:, 2] - grasp_body_pos[:, 2]
    z_error_finger_right = right_finger_pos[:, 2] - grasp_body_pos[:, 2]
    z_error_tcp_grasp = ee_pos[:, 2] - grasp_tcp_pos[:, 2]
    dist_left_body = torch.norm(grasp_body_pos - left_finger_pos, dim=-1)
    dist_right_body = torch.norm(grasp_body_pos - right_finger_pos, dim=-1)
    dist_finger_body = torch.minimum(dist_left_body, dist_right_body)
    finger_span_xy = torch.norm(left_finger_pos[:, :2] - right_finger_pos[:, :2], dim=-1)
    dist_bottle_bowl = torch.norm(bottle_to_bowl, dim=-1)

    table_clearance = ee_pos[:, 2] - env._table_z
    if not hasattr(env, "_bottle_rest_z"):
        env._bottle_rest_z = env._bottle_nominal_pos[2].expand(env.num_envs).clone()
    # Δz so với lúc spawn — không dùng độ cao tuyệt đối (chai đặt sẵn trên bàn ~>0.03m)
    bottle_lift = bottle_pos[:, 2] - env._bottle_rest_z
    is_lifted = bottle_lift > getattr(env.cfg, "grasp_lift_threshold", 0.03)

    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    bottle_up_w = quat_apply(bottle_quat_w, up_local)
    bottle_tilt_deg = torch.rad2deg(torch.acos(bottle_up_w[:, 2].clamp(-1.0, 1.0)))
    bottle_lin_speed = torch.norm(env._bottle.data.root_lin_vel_w, dim=-1)
    bottle_ang_speed = torch.norm(env._bottle.data.root_ang_vel_w, dim=-1)

    # +Z or -Z of tool frame pointing down (handles either URDF convention)
    top_down_pos = (-tool_z_w[:, 2]).clamp(-1.0, 1.0)
    top_down_neg = tool_z_w[:, 2].clamp(-1.0, 1.0)
    top_down_align = torch.maximum(top_down_pos, top_down_neg)

    # Fingers should be horizontal when approaching from above
    finger_level = (1.0 - tool_y_w[:, 2].abs()).clamp(0.0, 1.0)

    # Target should be below TCP (downward approach, not sideways)
    approach_dir = ee_to_grasp_body / (dist_ee_bottle.unsqueeze(-1) + 1e-6)
    descent_align = torch.clamp(-approach_dir[:, 2], min=0.0, max=1.0)

    # Legacy key kept for demo logs (no longer used as primary reward)
    approach_align = torch.sum(tool_z_w * approach_dir, dim=-1).clamp(-1.0, 1.0)

    gripper_state = get_gripper_state(env)

    grasp_pos_w = grasp_pos + origins
    grasp_body_pos_w = grasp_body_pos + origins
    grasp_tcp_pos_w = grasp_tcp_pos + origins

    return {
        "origins": origins,
        "ee_world": ee_world,
        "hand_pos_w": hand_pos_w,
        "hand_pos": hand_pos,
        "finger_pos_w": finger_pos_w,
        "finger_pos": finger_pos,
        "left_finger_pos_w": left_finger_w,
        "right_finger_pos_w": right_finger_w,
        "left_finger_pos": left_finger_pos,
        "right_finger_pos": right_finger_pos,
        "dist_left_body": dist_left_body,
        "dist_right_body": dist_right_body,
        "z_error_finger_left": z_error_finger_left,
        "z_error_finger_right": z_error_finger_right,
        "finger_span_xy": finger_span_xy,
        "finger_tip_mode": finger_tip_mode,
        "hand_quat_w": hand_quat_w,
        "ee_quat_w": ee_quat_w,
        "ee_pos": ee_pos,
        "bottle_world": bottle_world,
        "bottle_quat_w": bottle_quat_w,
        "grasp_pos_w": grasp_pos_w,
        "grasp_body_pos": grasp_body_pos,
        "grasp_body_pos_w": grasp_body_pos_w,
        "grasp_ee_descend_pos": grasp_ee_descend_pos,
        "grasp_ee_descend_pos_w": grasp_ee_descend_pos_w,
        "grasp_tcp_pos": grasp_tcp_pos,
        "grasp_tcp_pos_w": grasp_tcp_pos_w,
        "bottle_pos": bottle_pos,
        "bowl_pos": bowl_pos,
        "grasp_target_pos": grasp_target_pos,
        "grasp_pos": grasp_pos,
        "ee_to_grasp": ee_to_grasp,
        "ee_to_grasp_finger": ee_to_grasp_finger,
        "bottle_to_bowl": bottle_to_bowl,
        "dist_ee_grasp": dist_ee_grasp,
        "dist_ee_bottle": dist_ee_bottle,
        "dist_hand_body": dist_hand_body,
        "dist_finger_body": dist_finger_body,
        "lateral_dist_xy": lateral_dist_xy,
        "lateral_hand_xy": lateral_hand_xy,
        "lateral_finger_xy": lateral_finger_xy,
        "z_error": z_error,
        "z_error_hand": z_error_hand,
        "z_error_finger": z_error_finger,
        "z_error_tcp_grasp": z_error_tcp_grasp,
        "dist_bottle_bowl": dist_bottle_bowl,
        "table_clearance": table_clearance,
        "bottle_lift": bottle_lift,
        "is_lifted": is_lifted,
        "bottle_tilt_deg": bottle_tilt_deg,
        "bottle_lin_speed": bottle_lin_speed,
        "bottle_ang_speed": bottle_ang_speed,
        "tool_z_w": tool_z_w,
        "tool_y_w": tool_y_w,
        "top_down_align": top_down_align,
        "finger_level": finger_level,
        "descent_align": descent_align,
        "approach_align": approach_align,
        "gripper_state": gripper_state,
    }


def format_finger_debug(s: dict, env_id: int = 0, env: ManagerBasedRLEnv | None = None) -> str:
    """One-line left/right fingertip debug (requires finger body links)."""
    i = env_id
    if s.get("finger_tip_mode", "proxy") == "proxy":
        return "fingers:proxy"
    dl = float(s["dist_left_body"][i].item())
    dr = float(s["dist_right_body"][i].item())
    zl = float(s["z_error_finger_left"][i].item())
    zr = float(s["z_error_finger_right"][i].item())
    span = float(s["finger_span_xy"][i].item())
    asym = " ⚠️asym" if abs(dl - dr) > 0.015 or abs(zl - zr) > 0.012 else ""
    joint_str = ""
    if env is not None and hasattr(env, "_gripper_joint_ids"):
        jp = env._robot.data.joint_pos[i, env._gripper_joint_ids]
        j1, j2 = float(jp[0].item()), float(jp[1].item())
        joint_str = f" j1:{j1:.4f} j2:{j2:.4f}"
    return (
        f"fingers L d:{dl:.3f} z:{zl:+.3f} | R d:{dr:.3f} z:{zr:+.3f}"
        f" span:{span:.3f}m{joint_str}{asym}"
    )


def format_bottle_debug(env: ManagerBasedRLEnv, s: dict, env_id: int = 0) -> str:
    """One-line bottle physics debug for demo logs."""
    i = env_id
    lift_m = float(s["bottle_lift"][i].item())
    tilt = float(s["bottle_tilt_deg"][i].item())
    spd = float(s["bottle_lin_speed"][i].item())
    ang = float(s["bottle_ang_speed"][i].item())
    bx, by, bz = (float(s["bottle_pos"][i, j].item()) for j in range(3))
    tipped = " ⚠️TIPPED" if tilt > 15.0 else (" ⚠️lean" if tilt > 8.0 else "")
    return (
        f"bottle:[{bx:.3f},{by:.3f},{bz:.3f}] Δz:{lift_m:+.3f}m"
        f" tilt:{tilt:.1f}° spd:{spd:.2f} ang:{ang:.2f}{tipped}"
    )


def save_pregrasp_joints(env: ManagerBasedRLEnv, env_id: int = 0, path: str | None = None) -> dict:
    """Save arm joints when aligned near bottle (debug / future calibration)."""
    check_init_buffers(env)
    s = compute_state(env)
    names = [
        "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
        "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
        "openarm_left_joint7",
        "openarm_left_finger_joint1", "openarm_left_finger_joint2",
    ]
    q = env._robot.data.joint_pos[env_id]
    out: dict = {"joints": {}, "ee_offset_from_grasp_body": []}
    for name in names:
        jids, _ = env._robot.find_joints([name])
        out["joints"][name] = float(q[jids[0]].item())
    off = (s["ee_pos"][env_id] - s["grasp_body_pos"][env_id]).detach().cpu().tolist()
    out["ee_offset_from_grasp_body"] = off
    if path:
        import json
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"  [Calib] Saved pregrasp calibration → {path}")
        print(f"  [Calib] ee_offset_from_grasp_body={off}")
    return out


def load_pregrasp_calibration(path: str) -> dict | None:
    """Load full calibration JSON (joints + optional ee_offset_from_grasp_body)."""
    import json
    import os
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_pregrasp_joints(path: str) -> dict | None:
    data = load_pregrasp_calibration(path)
    if data is None:
        return None
    if isinstance(data, dict) and "joints" in data:
        return data["joints"]
    return data if isinstance(data, dict) else None
