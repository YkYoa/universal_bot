#!/usr/bin/env python3
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

"""
isaaclab_demo.py
================
Playback script to run and visualize a trained Stable-Baselines3 policy locally or via streaming.
Converted to the world-class Manager-based RL workflow.
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import hashlib
import pickle
import torch
import numpy as np
import builtins
import atexit
import sys

_last_print_was_cr = False
_original_print = builtins.print

def print(*args, **kwargs):
    global _last_print_was_cr
    end = kwargs.get("end", "\n")
    
    is_cr_manipulation = (
        args 
        and isinstance(args[0], str) 
        and args[0].startswith("\r")
    )
    if _last_print_was_cr and end != "\r" and not is_cr_manipulation:
        _original_print()
        _last_print_was_cr = False
        
    if end == "\r":
        _last_print_was_cr = True
    else:
        _last_print_was_cr = False
    _original_print(*args, **kwargs)

builtins.print = print

def _cleanup_newline():
    if _last_print_was_cr:
        _original_print()

atexit.register(_cleanup_newline)

_original_excepthook = sys.excepthook
def _custom_excepthook(exctype, value, traceback):
    if _last_print_was_cr:
        _original_print()
    _original_excepthook(exctype, value, traceback)
sys.excepthook = _custom_excepthook

# Dynamic Python path registration to make isaaclab_openarm_env package discoverable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

# Force NVIDIA Vulkan ICD if available, to bypass any integrated GPU conflicts
if os.path.exists("/usr/share/vulkan/icd.d/nvidia_icd.json"):
    os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

# ── 1. Parse arguments and launch Isaac Sim first (must happen before importing isaaclab) ──
parser = argparse.ArgumentParser(description="Isaac Lab OpenArm Apple Pick-and-Place Demo — Manager-Based workflow")
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1, help="Number of parallel environments to run")
parser.add_argument("--model-path", "--model_path", dest="model_path", type=str, default=os.path.join(_THIS_DIR, "logs", "train", "best_policy.pt"), help="Path to trained policy (.pt weights or SB3 .zip checkpoint)")
parser.add_argument("--seed",       type=int, default=42, help="Random seed")
parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=0, help="Maximum simulation steps to run before exiting (0 for infinite)")
parser.add_argument("--joint-space", "--joint_space", dest="joint_space", action="store_true",
                    help="Use legacy 8-D joint-space actions (must match trained checkpoint)")
parser.add_argument("--action-smooth", "--action_smooth", dest="action_smooth", type=float, default=0.0,
                    help="EMA action smoothing alpha (0=off; thử 0.25 nếu demo rung).")
parser.add_argument("--assist", action="store_true",
                    help="Enable REACH/GRASP descent + lift assist (bootstrap only).")
parser.add_argument("--no-bottle-rand", action="store_true",
                    help="Disable bottle XY randomization on reset (easier marker tuning).")
parser.add_argument("--debug-marker", action="store_true",
                    help="Print bottle root vs marker position each episode reset.")
parser.add_argument("--marker-offset-x", type=float, default=None,
                    help="Override bottle_grasp_xy_offset_x (metres) for this demo run.")
parser.add_argument("--marker-offset-y", type=float, default=None,
                    help="Override bottle_grasp_xy_offset_y (metres) for this demo run.")
parser.add_argument("--marker-offset-z", type=float, default=None,
                    help="Override bottle_grasp_z_offset (metres). Negative = hạ marker/target xuống.")
parser.add_argument("--task-phase", "--task_phase", dest="task_phase", type=int, default=None,
                    help="Curriculum: 1=reach, 2=reach+grasp+lift")
parser.add_argument("--stage", type=str, default="all",
                    choices=("reach", "grasp", "lift", "all"),
                    help="Phase 2 sub-stage: reach | grasp | lift | all (default: all)")

# Add Isaac Sim AppLauncher args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── 2. Import standard libraries and environment configuration ──
from stable_baselines3 import PPO
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_openarm_env.env import ApplePickPlaceEnv
from isaaclab_openarm_env.config import ApplePickPlaceEnvCfg
from isaaclab_openarm_env.mdp.helpers import finger_grasp_ready, save_pregrasp_joints, compute_state, format_bottle_debug, format_finger_debug
from isaaclab_openarm_env.scene import patch_qvic_usd_once

# Permanently patch qvic.usd to remove duplicate robot prims and nested RigidBodyAPIs.
patch_qvic_usd_once()


def _file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


from isaaclab_openarm_env.phase2_overrides import (
    ENV_CFG_SNAPSHOT_KEYS,
    apply_phase2_demo_gates,
)


def _find_env_cfg_snapshot(model_path: str) -> str | None:
    """Locate env_cfg.pkl saved alongside a fetched checkpoint."""
    log_dir = os.path.dirname(os.path.abspath(model_path))
    base = os.path.basename(model_path)
    candidates = [os.path.join(log_dir, "env_cfg.pkl")]
    if base.startswith("best_policy_") and base.endswith(".pt"):
        run = base[len("best_policy_") : -3]
        candidates.append(os.path.join(log_dir, f"env_cfg_{run}.pkl"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _apply_env_cfg_snapshot(env_cfg, model_path: str) -> str | None:
    """Apply training-time scalar config from env_cfg.pkl (if present)."""
    cfg_path = _find_env_cfg_snapshot(model_path)
    if cfg_path is None:
        return None
    with open(cfg_path, "rb") as f:
        saved = pickle.load(f)
    for key in ENV_CFG_SNAPSHOT_KEYS:
        if hasattr(saved, key):
            setattr(env_cfg, key, getattr(saved, key))
    return cfg_path


def _warn_model_mismatch(model_path: str) -> None:
    name = os.path.basename(model_path).lower()
    if "phase2c" in name or "_2c" in name:
        print("")
        print("  ⚠️  Model phase2c (task_phase=22) — Phase 2C đã bỏ khỏi code.")
        print("      Dùng checkpoint phase 2 thật: train_osc_phase2_v3")
        print("      ./fetch_model.sh naiscorp-4090 train_osc_phase2_v3")
        print("      ./run_local.sh demo --model ./logs/active_policy.pt --phase 2")
        print("")


def _load_pt_policy(model_path: str, env, device: str) -> PPO:
    """Load a .pt state_dict or .zip checkpoint into a PPO policy."""
    if model_path.endswith(".zip"):
        return PPO.load(model_path, env=env, device=device)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.Tanh),
        device=device,
    )
    state_dict = torch.load(model_path, map_location=device)
    model.policy.load_state_dict(state_dict)
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Resolve paths relative to workspace directory
    model_path = os.path.abspath(args.model_path)

    print("=" * 70)
    print("  Isaac Lab OpenArm Apple Pick-and-Place Policy Player (Manager-Based)")
    print(f"  Model Path : {model_path}")
    if os.path.exists(model_path):
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(model_path)))
        print(f"  Model MD5  : {_file_md5(model_path)}  (modified {mtime})")
    _warn_model_mismatch(model_path)
    print(f"  Envs       : {args.num_envs}")
    print(f"  Device     : {device.upper()}")
    print("=" * 70)

    # ── Build environment ──
    print("\n  Building environment...")
    env_cfg = ApplePickPlaceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.seed = args.seed
    env_cfg.use_joint_space_actions = args.joint_space
    cfg_snapshot = _apply_env_cfg_snapshot(env_cfg, model_path)
    if cfg_snapshot:
        print(f"  Env cfg    : loaded snapshot → {os.path.basename(cfg_snapshot)}")
    if args.task_phase is not None:
        env_cfg.task_phase = args.task_phase
        if args.task_phase >= 2:
            env_cfg.episode_length_s = 20.0
    demo_profile = "n/a"
    if env_cfg.task_phase >= 2:
        env_cfg.debug_success_log = True
        stage = "all" if args.assist else args.stage
        demo_profile = apply_phase2_demo_gates(env_cfg, model_path, stage=stage)
    if args.no_bottle_rand:
        env_cfg.bottle_pos_noise = 0.0
    # Mimic gripper: moderate stiffness — symmetric close via joint1 only
    if env_cfg.task_phase >= 2:
        env_cfg.scene.robot.actuators["gripper"].stiffness = 900.0
        env_cfg.scene.robot.actuators["gripper"].damping = 45.0
    if args.marker_offset_x is not None:
        env_cfg.bottle_grasp_xy_offset_x = args.marker_offset_x
    if args.marker_offset_y is not None:
        env_cfg.bottle_grasp_xy_offset_y = args.marker_offset_y
    if args.marker_offset_z is not None:
        env_cfg.bottle_grasp_z_offset = args.marker_offset_z
    
    print(f"  Task phase : {env_cfg.task_phase}")
    if env_cfg.task_phase >= 2:
        print(f"  Phase2 stage: {demo_profile}")
    print(f"  Descent assist: {env_cfg.grasp_descent_assist_enabled}"
          f" (grasp={getattr(env_cfg, 'grasp_descent_assist_grasp', False)},"
          f" force={getattr(env_cfg, 'grasp_descent_force_override', False)})")
    print(f"  Lift assist   : {env_cfg.grasp_lift_assist_enabled}")
    if env_cfg.task_phase >= 2:
        print(
            f"  Grasp gates  : top↓ advance>{env_cfg.reach_advance_min_top_down}"
            f" close>{env_cfg.grasp_close_min_top_down}"
            f" z_f<{env_cfg.reach_advance_max_z_finger}"
        )
    print(f"  Action smooth : {args.action_smooth}")
    print(
        f"  Bottle grasp XY: usd_auto={env_cfg.bottle_use_usd_mesh_xy} "
        f"local=({env_cfg.bottle_grasp_local_xy_x}, {env_cfg.bottle_grasp_local_xy_y}) "
        f"tune=({env_cfg.bottle_grasp_xy_offset_x}, {env_cfg.bottle_grasp_xy_offset_y}, {env_cfg.bottle_grasp_z_offset}) "
        f"| noise: {env_cfg.bottle_pos_noise}"
    )
    
    env = ApplePickPlaceEnv(cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)
    from isaaclab_openarm_env.mdp.helpers import check_init_buffers
    check_init_buffers(env.unwrapped)
    ft_mode = getattr(env_cfg, "finger_tip_mode", "hand_frame")
    print(f"  Finger metrics: {ft_mode} (L/R pads from hand kinematics + joint travel)")

    # ── Load Model / Policy ──
    print(f"\n  Loading weights from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"  ❌ File not found: {model_path}")
        print("  Please run './fetch_model.sh' first or check the path.")
        env.close()
        return

    try:
        model = _load_pt_policy(model_path, env, device)
        print("  ✅ Loaded grasp/lift policy.")
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        env.close()
        return

    task_phase = getattr(env_cfg, "task_phase", 1)
    calib_path = os.path.join(_THIS_DIR, "logs", "pregrasp_joints.json")

    total_rewards = np.zeros(args.num_envs)
    episode_steps = np.zeros(args.num_envs)
    value_histories = [[] for _ in range(args.num_envs)]
    total_steps = 0
    was_table_colliding = np.zeros(args.num_envs, dtype=bool)
    was_object_colliding = np.zeros(args.num_envs, dtype=bool)
    prev_stage_arr = np.zeros(args.num_envs, dtype=int)
    action_smooth = None

    def _sync_stage_tracking():
        nonlocal prev_stage_arr
        if hasattr(env.unwrapped, "_stage"):
            prev_stage_arr = env.unwrapped._stage.cpu().numpy().copy()

    # ── Playback Loop ──
    print("\n  Running simulation. Press Ctrl+C in the terminal to stop.")
    obs = env.reset()
    _sync_stage_tracking()

    try:
        while simulation_app.is_running():
            # Force play/unpause programmatically to ensure simulation steps forward
            try:
                if hasattr(env.unwrapped, "sim"):
                    sim_ctx = env.unwrapped.sim
                    if not sim_ctx.is_playing():
                        print(f"  [DEBUG] Sim timeline is PAUSED. Forcing Play...")
                        sim_ctx.play()
            except Exception as e:
                print(f"  [DEBUG] Error forcing sim play: {e}")
                
            with torch.no_grad():
                raw_action, _ = model.predict(obs, deterministic=True)
                if args.action_smooth > 0.0:
                    if action_smooth is None:
                        action_smooth = raw_action.copy()
                    else:
                        a = args.action_smooth
                        action_smooth = a * raw_action + (1.0 - a) * action_smooth
                    action = action_smooth
                else:
                    action = raw_action
                action = np.clip(action, -1.0, 1.0)
                obs_tensor, _ = model.policy.obs_to_tensor(obs)
                values = model.policy.predict_values(obs_tensor).flatten().cpu().numpy()
            ee_pos = obs[:, 14:17]
            ee_to_grasp = obs[:, 17:20]
            dist_ee_grasp = np.linalg.norm(ee_to_grasp, axis=-1)   # distance to stage-dynamic target
            table_clearance = ee_pos[:, 2] - 0.626

            # Fetch metrics from env state cache
            dist_ee_bottle_arr = np.full(args.num_envs, 999.0)
            lateral_dist_arr = np.full(args.num_envs, 999.0)
            z_error_arr = np.zeros(args.num_envs)
            z_error_hand_arr = np.zeros(args.num_envs)
            z_error_finger_arr = np.zeros(args.num_envs)
            lateral_hand_arr = np.zeros(args.num_envs)
            lateral_finger_arr = np.zeros(args.num_envs)
            top_down_arr = np.zeros(args.num_envs)
            finger_level_arr = np.zeros(args.num_envs)
            approach_align_arr = np.zeros(args.num_envs)
            grasp_pos_arr = np.zeros((args.num_envs, 3))
            contact_steps_arr = np.zeros(args.num_envs, dtype=int)
            lift_steps_arr = np.zeros(args.num_envs, dtype=int)
            bottle_lift_arr = np.zeros(args.num_envs)
            gripper_arr = np.zeros(args.num_envs)
            bottle_tilt_arr = np.zeros(args.num_envs)
            bottle_spd_arr = np.zeros(args.num_envs)
            stage_arr = np.zeros(args.num_envs, dtype=int)
            contact_threshold = getattr(env.unwrapped.cfg, "success_dist_threshold", 0.06)
            hold_steps = getattr(env.unwrapped.cfg, "success_hold_steps", 5)
            task_phase = getattr(env.unwrapped.cfg, "task_phase", 1)
            lift_hold_steps = getattr(env.unwrapped.cfg, "grasp_lift_hold_steps", 5)
            if hasattr(env.unwrapped, "_last_state") and env.unwrapped._last_state is not None:
                ls = env.unwrapped._last_state
                dist_ee_bottle_arr = ls["dist_ee_bottle"].cpu().numpy()
                lateral_dist_arr = ls["lateral_dist_xy"].cpu().numpy()
                z_error_arr = ls["z_error"].cpu().numpy()
                z_error_hand_arr = ls["z_error_hand"].cpu().numpy()
                z_error_finger_arr = ls["z_error_finger"].cpu().numpy()
                lateral_hand_arr = ls["lateral_hand_xy"].cpu().numpy()
                lateral_finger_arr = ls["lateral_finger_xy"].cpu().numpy()
                top_down_arr = ls["top_down_align"].cpu().numpy()
                finger_level_arr = ls["finger_level"].cpu().numpy()
                approach_align_arr = ls["approach_align"].cpu().numpy()
                grasp_pos_arr = ls["grasp_pos"].cpu().numpy()
                bottle_lift_arr = ls["bottle_lift"].cpu().numpy()
                gripper_arr = ls["gripper_state"].cpu().numpy()
                if "bottle_tilt_deg" in ls:
                    bottle_tilt_arr = ls["bottle_tilt_deg"].cpu().numpy()
                if "bottle_lin_speed" in ls:
                    bottle_spd_arr = ls["bottle_lin_speed"].cpu().numpy()
            if hasattr(env.unwrapped, "_steps_in_contact"):
                contact_steps_arr = env.unwrapped._steps_in_contact.cpu().numpy()
            if hasattr(env.unwrapped, "_steps_bottle_lifted"):
                lift_steps_arr = env.unwrapped._steps_bottle_lifted.cpu().numpy()
            if hasattr(env.unwrapped, "_stage"):
                stage_arr = env.unwrapped._stage.cpu().numpy()
            
            for i in range(args.num_envs):
                value_histories[i].append(values[i])
            
            # Detect collisions
            is_table_colliding = (table_clearance <= -0.02)
            # "Object contact" = TCP inside contact zone (hysteresis if available)
            if hasattr(env.unwrapped, "_in_contact_zone"):
                is_object_colliding = env.unwrapped._in_contact_zone.cpu().numpy()
            else:
                is_object_colliding = (dist_ee_bottle_arr <= contact_threshold)
            
            # Detect if an event transition is about to occur
            has_event = False
            for i in range(args.num_envs):
                if (is_table_colliding[i] and not was_table_colliding[i]) or \
                   (not is_table_colliding[i] and was_table_colliding[i]) or \
                   (is_object_colliding[i] and not was_object_colliding[i]) or \
                   (not is_object_colliding[i] and was_object_colliding[i]):
                     has_event = True
                     break
            
            if has_event:
                print("\r\033[K", end="")  # Clear the current carriage-return line
            
            for i in range(args.num_envs):
                tcp_str = f"[{ee_pos[i,0]:.3f}, {ee_pos[i,1]:.3f}, {ee_pos[i,2]:.3f}]"
                # 1. Table collision transitions
                if is_table_colliding[i] and not was_table_colliding[i]:
                    stage_name = "UNKNOWN"
                    dist_to_bottle = 0.0
                    dist_to_hover = 0.0
                    grip_state = 0.0
                    if hasattr(env.unwrapped, "_last_state") and env.unwrapped._last_state is not None:
                        ls = env.unwrapped._last_state
                        stage_idx = int(env.unwrapped._stage[i].item())
                        stage_name = ["REACH (Stage 0)", "GRASP (Stage 1)", "PLACE (Stage 2)"][stage_idx] if stage_idx in [0, 1, 2] else f"STAGE_{stage_idx}"
                        dist_to_bottle = ls["dist_ee_bottle"][i].item()
                        dist_to_hover = ls["dist_ee_grasp"][i].item()
                        grip_state = ls["gripper_state"][i].item()
                    
                    print(f"  [Env {i:02d} Step {int(episode_steps[i]):3d}] 💥 TABLE COLLISION DETECTED!")
                    print(f"    - Current Stage: {stage_name}")
                    print(f"    - TCP Position: {tcp_str}")
                    print(f"    - Table Clearance: {table_clearance[i]:.4f}m")
                    print(f"    - Distance to Bottle Body: {dist_to_bottle:.4f}m")
                    print(f"    - Distance to Hover Target: {dist_to_hover:.4f}m")
                    print(f"    - Gripper State: {grip_state:.4f} (0=Open, 1=Closed)")
                    print(f"    - Action Output: {np.array2string(action[i], precision=3, separator=',')}")
                    print(f"    - Critic Value Estimate: {values[i]:.4f}")
                elif not is_table_colliding[i] and was_table_colliding[i]:
                    print(f"  [Env {i:02d} Step {int(episode_steps[i]):3d}] Clear of Table. Critic Value: {values[i]:.4f} | TCP Pos: {tcp_str}")
                
                # 2. Object contact/collision transitions
                if is_object_colliding[i] and not was_object_colliding[i]:
                    bottle_dbg = ""
                    if hasattr(env.unwrapped, "_last_state") and env.unwrapped._last_state is not None:
                        bottle_dbg = f" | {format_bottle_debug(env.unwrapped, env.unwrapped._last_state, i)}"
                    print(
                        f"  [Env {i:02d} Step {int(episode_steps[i]):3d}] 🎯 BOTTLE CONTACT!"
                        f" | dist: {dist_ee_bottle_arr[i]:.4f}m | lat: {lateral_dist_arr[i]:.4f}m"
                        f" | z_err: {z_error_arr[i]:+.4f}m | z_f: {z_error_finger_arr[i]:+.4f}m"
                        f" | top↓: {top_down_arr[i]:.2f}"
                        f" | lvl: {finger_level_arr[i]:.2f} | Critic: {values[i]:.2f}"
                        f" | TCP: {tcp_str} | grip:{gripper_arr[i]:.2f}{bottle_dbg}"
                    )
                elif not is_object_colliding[i] and was_object_colliding[i]:
                    bottle_dbg = ""
                    if hasattr(env.unwrapped, "_last_state") and env.unwrapped._last_state is not None:
                        bottle_dbg = f" | {format_bottle_debug(env.unwrapped, env.unwrapped._last_state, i)}"
                    print(
                        f"  [Env {i:02d} Step {int(episode_steps[i]):3d}] Lost Bottle Contact."
                        f" Critic Value: {values[i]:.4f} | TCP Pos: {tcp_str}{bottle_dbg}"
                    )
                
                # Update history
                was_table_colliding[i] = is_table_colliding[i]
                was_object_colliding[i] = is_object_colliding[i]
            
            # Print continuous real-time coordinate tracking on a single updating line
            if args.num_envs == 1 and not has_event:
                stage_name = ["REACH", "GRASP", "PLACE"][stage_arr[0]] if stage_arr[0] < 3 else f"S{stage_arr[0]}"
                line = (
                    f"  [Env 00 Step {int(episode_steps[0]):3d}] {stage_name} "
                    f"dist:{dist_ee_bottle_arr[0]:.3f} lat_f:{lateral_finger_arr[0]:.3f} "
                    f"z_f:{z_error_finger_arr[0]:+.3f} top↓:{top_down_arr[0]:.2f} "
                    f"grip:{gripper_arr[0]:.2f} lift:{bottle_lift_arr[0]:+.3f}m "
                    f"tilt:{bottle_tilt_arr[0]:.1f}° spd:{bottle_spd_arr[0]:.2f} "
                )
                if task_phase >= 2:
                    line += f"lift_hold:{lift_steps_arr[0]}/{lift_hold_steps} "
                    if stage_arr[0] == 0 and hasattr(env.unwrapped, "_assist_want_descend"):
                        if bool(env.unwrapped._assist_want_descend[0].item()):
                            line += "desc↓ "
                    if stage_arr[0] == 1:
                        grip_term = env.unwrapped.action_manager._terms.get("gripper_action")
                        if grip_term is not None and hasattr(grip_term, "_descend_hold"):
                            dh = int(grip_term._descend_hold[0].item())
                            dh_req = int(getattr(env.unwrapped.cfg, "grasp_descend_hold_steps", 8))
                            latched = bool(grip_term._grasp_latched[0].item())
                            line += f"dh:{dh}/{dh_req}{'L' if latched else ''} "
                            if grip_term is not None and hasattr(grip_term, "_close_progress"):
                                ramp = int(getattr(env.unwrapped.cfg, "grasp_close_ramp_steps", 0))
                                if ramp > 0:
                                    pct = int(grip_term._close_progress[0].item() * 100)
                                    line += f"gc:{pct}% "
                            if hasattr(env.unwrapped, "_lift_settle_steps"):
                                settle_req = int(getattr(env.unwrapped.cfg, "grasp_lift_settle_steps", 8))
                                st = int(env.unwrapped._lift_settle_steps[0].item())
                                line += f"st:{st}/{settle_req} "
                            if hasattr(env.unwrapped, "_pre_lift_hold_steps"):
                                ph_req = int(getattr(env.unwrapped.cfg, "grasp_pre_lift_hold_steps", 0))
                                if ph_req > 0:
                                    ph = int(env.unwrapped._pre_lift_hold_steps[0].item())
                                    line += f"ph:{ph}/{ph_req} "
                            if hasattr(env.unwrapped, "_assist_want_descend"):
                                if bool(env.unwrapped._assist_want_descend[0].item()):
                                    line += "desc↓ "
                            if hasattr(env.unwrapped, "_assist_want_lift"):
                                if bool(env.unwrapped._assist_want_lift[0].item()):
                                    line += "lift↑ "
                                elif latched and getattr(env.unwrapped.cfg, "grasp_pregrasp_freeze_arm", False):
                                    line += "hold "
                else:
                    line += f"hold:{contact_steps_arr[0]}/{hold_steps} "
                line += f"V:{values[0]:.1f}\033[K"
                print(line, end="\r", flush=True)
            
            obs, rewards, dones, infos = env.step(action)

            if args.debug_marker and hasattr(env.unwrapped, "_last_state"):
                ls = env.unwrapped._last_state
                cfg = env.unwrapped.cfg
                for i in range(args.num_envs):
                    if int(episode_steps[i]) > 2:
                        continue
                    bx = float(ls["bottle_pos"][i, 0].item())
                    by = float(ls["bottle_pos"][i, 1].item())
                    mx = float(ls["grasp_pos"][i, 0].item())
                    my = float(ls["grasp_pos"][i, 1].item())
                    mz = float(ls["grasp_pos"][i, 2].item())
                    local_xy = getattr(env.unwrapped, "_bottle_local_xy_offset", None)
                    base_x = float(local_xy[0].item()) if local_xy is not None else 0.0
                    base_y = float(local_xy[1].item()) if local_xy is not None else 0.0
                    tune_x = float(getattr(cfg, "bottle_grasp_xy_offset_x", 0.0))
                    tune_y = float(getattr(cfg, "bottle_grasp_xy_offset_y", 0.0))
                    tune_z = float(getattr(cfg, "bottle_grasp_z_offset", 0.0))
                    bz = float(ls["bottle_pos"][i, 2].item())
                    spawn = ""
                    if hasattr(env.unwrapped, "_bottle_spawn_env_pos"):
                        sx = float(env.unwrapped._bottle_spawn_env_pos[i, 0].item())
                        sy = float(env.unwrapped._bottle_spawn_env_pos[i, 1].item())
                        spawn = f" | spawn_xy=({sx:.3f},{sy:.3f})"
                    print(
                        f"  [MarkerDbg Env {i:02d} ep_step {int(episode_steps[i])}] "
                        f"bottle=({bx:.3f},{by:.3f},{bz:.3f}) marker=({mx:.3f},{my:.3f},{mz:.3f}) "
                        f"local_base=({base_x:+.3f},{base_y:+.3f}) tune=({tune_x:+.3f},{tune_y:+.3f},{tune_z:+.3f}) "
                        f"world_delta=({mx-bx:+.3f},{my-by:+.3f},{mz-bz:+.3f}){spawn}"
                    )
            if hasattr(env.unwrapped, "_stage"):
                stage_arr = env.unwrapped._stage.cpu().numpy()
                for i in range(args.num_envs):
                    if stage_arr[i] != prev_stage_arr[i] and not dones[i]:
                        names = ["REACH", "GRASP", "PLACE"]
                        old_n = names[prev_stage_arr[i]] if prev_stage_arr[i] < 3 else f"S{prev_stage_arr[i]}"
                        new_n = names[stage_arr[i]] if stage_arr[i] < 3 else f"S{stage_arr[i]}"
                        print(
                            f"\n  [Env {i:02d}] Stage {old_n} → {new_n} @ step {int(episode_steps[i])}"
                            f" | z_f:{z_error_finger_arr[i]:+.3f} grip:{gripper_arr[i]:.2f}"
                        )
                        if task_phase == 2 and new_n == "GRASP" and i == 0:
                            save_pregrasp_joints(env.unwrapped, env_id=0, path=calib_path)
                    prev_stage_arr[i] = stage_arr[i]
            # Synchronize CUDA to prevent async GPU race conditions
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            total_rewards += rewards
            episode_steps += 1
            total_steps += 1
            
            for i, done in enumerate(dones):
                if done:
                    print("\r\033[K", end="")  # Clear the current carriage-return line
                    info = infos[i] if isinstance(infos, (list, tuple)) else infos
                    is_truncated = bool(info.get("TimeLimit.truncated", info.get("time_outs", False)))
                    dist_end = float(dist_ee_bottle_arr[i])
                    lat_end = float(lateral_dist_arr[i])
                    hold_end = int(contact_steps_arr[i])
                    lift_end = int(lift_steps_arr[i])
                    stage_end = int(stage_arr[i])
                    if task_phase >= 2:
                        is_success = lift_end >= lift_hold_steps
                    else:
                        is_success = (hold_end >= hold_steps) or ((dist_end < contact_threshold) and not is_truncated)
                    if is_success:
                        status = "SUCCESS"
                        if task_phase >= 2:
                            print(
                                f"  [Success] 🎉 EPISODE SUCCESS"
                                f" | lift:{float(bottle_lift_arr[i]):.3f}m"
                                f" grip:{float(gripper_arr[i]):.2f}"
                                f" lift_hold:{lift_end}/{lift_hold_steps}"
                                f" stage:{stage_end}"
                            )
                        else:
                            print(
                                f"  [Success] 🎉 EPISODE SUCCESS (Phase1)"
                                f" | contact_hold:{hold_end}/{hold_steps}"
                                f" dist:{dist_end:.3f}m"
                            )
                    elif is_truncated:
                        status = "TIMEOUT"
                    else:
                        status = "TERMINATED"

                    if len(value_histories[i]) > 0:
                        vals = np.array(value_histories[i])
                        avg_val = np.mean(vals)
                        min_val = np.min(vals)
                        max_val = np.max(vals)
                        val_str = f" | Critic Val (Avg/Min/Max): {avg_val:.2f}/{min_val:.2f}/{max_val:.2f}"
                    else:
                        val_str = ""

                    print(
                        f"  [Env {i:02d}] Done! Steps: {int(episode_steps[i])} | Return: {total_rewards[i]:.2f}"
                        f"{val_str} | Status: {status}"
                        f" | stage:{stage_end} dist:{dist_end:.3f} lat:{lat_end:.3f}"
                        f" | top↓:{float(top_down_arr[i]):.2f} grip:{float(gripper_arr[i]):.2f}"
                        f" | z_f:{float(z_error_finger_arr[i]):+.3f}"
                        f" | lift:{float(bottle_lift_arr[i]):.3f}m"
                        + (f" lift_hold:{lift_end}/{lift_hold_steps}" if task_phase >= 2
                           else f" hold:{hold_end}/{hold_steps}")
                    )
                    total_rewards[i] = 0.0
                    episode_steps[i] = 0
                    value_histories[i] = []
                    
                    # Reset collision tracking flags for this environment
                    was_table_colliding[i] = False
                    was_object_colliding[i] = False
                    _sync_stage_tracking()
            
            if args.max_steps > 0 and total_steps >= args.max_steps:
                print(f"\n  [INFO] Reached maximum steps limit ({args.max_steps}). Exiting playback loop...")
                break
                
            if not args.headless:
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n  Exiting playback...")

    print("\n  Closing environment...")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
