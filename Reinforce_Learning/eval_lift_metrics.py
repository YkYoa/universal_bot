#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
#
# Headless phase-2 evaluation — prints one LIFT_METRICS JSON line.

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

if os.path.exists("/usr/share/vulkan/icd.d/nvidia_icd.json"):
    os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

parser = argparse.ArgumentParser(description="Headless lift eval — outputs LIFT_METRICS JSON")
parser.add_argument("--model-path", "--model_path", dest="model_path", type=str,
                    default=os.path.join(_THIS_DIR, "logs", "active_policy.pt"))
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run-name", "--run_name", dest="run_name", type=str, default="")
parser.add_argument("--iteration", type=int, default=0)
parser.add_argument("--task-phase", "--task_phase", dest="task_phase", type=int, default=2)
parser.add_argument("--stage", type=str, default="all",
                    choices=("reach", "grasp", "lift", "all"),
                    help="Phase 2 sub-stage gates")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not hasattr(args, "headless") or not args.headless:
    args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
from stable_baselines3 import PPO
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from isaaclab_openarm_env.env import ApplePickPlaceEnv
from isaaclab_openarm_env.config import ApplePickPlaceEnvCfg
from isaaclab_openarm_env.mdp.helpers import STAGE_GRASP, STAGE_REACH, check_init_buffers, finger_descended_for_close, finger_grasp_ready
from isaaclab_openarm_env.phase2_overrides import (
    ENV_CFG_SNAPSHOT_KEYS,
    apply_phase2_demo_gates,
)
from isaaclab_openarm_env.scene import patch_qvic_usd_once

patch_qvic_usd_once()


def _find_env_cfg_snapshot(model_path: str) -> str | None:
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


def _apply_env_cfg_snapshot(env_cfg, model_path: str) -> None:
    cfg_path = _find_env_cfg_snapshot(model_path)
    if cfg_path is None:
        return
    with open(cfg_path, "rb") as f:
        saved = pickle.load(f)
    for key in ENV_CFG_SNAPSHOT_KEYS:
        if hasattr(saved, key):
            setattr(env_cfg, key, getattr(saved, key))


def _load_pt_policy(model_path: str, env, device: str) -> PPO:
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


def _classify_fail_mode(
    *,
    is_success: bool,
    is_truncated: bool,
    stage_end: int,
    gripper: float,
    bottle_lift: float,
    bottle_tilt: float,
    lift_hold: int,
    lift_hold_steps: int,
    grip_thresh: float,
    lift_thresh: float,
    tipped_deg: float,
) -> str:
    if is_success:
        return "success"
    if bottle_tilt >= tipped_deg * 0.9:
        return "tilt"
    if is_truncated:
        if stage_end == STAGE_REACH:
            return "reach"
        if stage_end == STAGE_GRASP:
            if gripper < grip_thresh:
                return "grasp"
            if bottle_lift < lift_thresh or lift_hold < lift_hold_steps:
                return "grasp"
        return "timeout"
    if stage_end == STAGE_REACH:
        return "reach"
    if stage_end == STAGE_GRASP and gripper < grip_thresh:
        return "grasp"
    if stage_end == STAGE_GRASP and bottle_lift < lift_thresh:
        return "grasp"
    return "timeout"


def main() -> None:
    model_path = os.path.abspath(args.model_path)
    if not os.path.isfile(model_path):
        print(f"LIFT_METRICS {json.dumps({'error': f'model not found: {model_path}'})}")
        env = None
        simulation_app.close()
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = ApplePickPlaceEnvCfg()
    env_cfg.sim.log_dir = os.path.join(_THIS_DIR, "logs", "sim_logs")
    env_cfg.scene.num_envs = 1
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.seed = args.seed
    env_cfg.task_phase = args.task_phase
    if args.task_phase >= 2:
        env_cfg.episode_length_s = 20.0

    _apply_env_cfg_snapshot(env_cfg, model_path)
    apply_phase2_demo_gates(env_cfg, model_path, stage=args.stage)

    env_cfg.bottle_pos_noise = 0.0
    env_cfg.align_log_interval = 999999
    env_cfg.debug_success_log = False
    env_cfg.suppress_align_log = True
    if env_cfg.task_phase >= 2:
        env_cfg.scene.robot.actuators["gripper"].stiffness = 550.0
        env_cfg.scene.robot.actuators["gripper"].damping = 35.0

    env = ApplePickPlaceEnv(cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)
    check_init_buffers(env.unwrapped)

    model = _load_pt_policy(model_path, env, device)
    cfg = env.unwrapped.cfg
    lift_hold_steps = getattr(cfg, "grasp_lift_hold_steps", 5)
    grip_thresh = getattr(cfg, "grasp_grip_threshold", 0.4)
    lift_thresh = getattr(cfg, "grasp_lift_threshold", 0.03)
    tipped_deg = getattr(cfg, "bottle_tipped_termination_deg", 60.0)

    episodes_done = 0
    results: list[dict] = []
    obs = env.reset()

    # SB3 VecEnv auto-resets on done — capture per-episode peaks before state is cleared.
    ep_max_stage = 0
    ep_max_lift = 0.0
    ep_max_lift_hold = 0
    ep_last_gripper = 0.0
    ep_max_gripper = 0.0
    ep_last_tilt = 0.0
    ep_last_z_f = 0.0
    ep_last_top = 0.0
    ep_last_lat_f = 0.0
    ep_last_dist_f = 0.0

    while simulation_app.is_running() and episodes_done < args.episodes:
        with torch.no_grad():
            action, _ = model.predict(obs, deterministic=True)
            action = np.clip(action, -1.0, 1.0)

        obs, _, dones, infos = env.step(action)

        ls = env.unwrapped._last_state
        # SB3 auto-resets on done — skip ls on done step (stale reset pose poisons peaks).
        if ls is not None and not dones[0]:
            stage_now = int(env.unwrapped._stage[0].item())
            lift_now = float(ls["bottle_lift"][0].item())
            grip_now = float(ls["gripper_state"][0].item())
            tilt_now = float(ls.get("bottle_tilt_deg", torch.zeros(1))[0].item())
            hold_now = int(env.unwrapped._steps_bottle_lifted[0].item())
            ep_max_stage = max(ep_max_stage, stage_now)
            ep_max_lift = max(ep_max_lift, lift_now)
            ep_max_lift_hold = max(ep_max_lift_hold, hold_now)
            ep_last_gripper = grip_now
            ep_max_gripper = max(ep_max_gripper, grip_now)
            ep_last_tilt = tilt_now
            ep_last_z_f = float(ls["z_error_finger"][0].item())
            ep_last_top = float(ls["top_down_align"][0].item())
            ep_last_lat_f = float(ls["lateral_finger_xy"][0].item())
            ep_last_dist_f = float(ls["dist_finger_body"][0].item())

        if not dones[0]:
            continue

        if ls is not None:
            print(
                f"  [EvalDbg] last z_f:{ep_last_z_f:+.3f} top↓:{ep_last_top:.2f} lat_f:{ep_last_lat_f:.3f}"
                f" dist_f:{ep_last_dist_f:.3f} grip:{ep_last_gripper:.3f} max_grip:{ep_max_gripper:.3f}"
                f" stage:{ep_max_stage}"
            )

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        is_truncated = bool(
            info.get("TimeLimit.truncated", info.get("time_outs", info.get("terminated", False)))
        )
        # Timeout episodes usually lack explicit truncated flag — treat non-success as timeout.
        if not is_truncated and cfg.task_phase >= 2:
            is_truncated = True

        stage_end = ep_max_stage
        gripper = ep_max_gripper
        bottle_lift = ep_max_lift
        bottle_tilt = ep_last_tilt
        lift_hold = ep_max_lift_hold
        is_success = lift_hold >= lift_hold_steps if cfg.task_phase >= 2 else False

        fail_mode = _classify_fail_mode(
            is_success=is_success,
            is_truncated=is_truncated,
            stage_end=stage_end,
            gripper=gripper,
            bottle_lift=bottle_lift,
            bottle_tilt=bottle_tilt,
            lift_hold=lift_hold,
            lift_hold_steps=lift_hold_steps,
            grip_thresh=grip_thresh,
            lift_thresh=lift_thresh,
            tipped_deg=tipped_deg,
        )

        results.append({
            "success": is_success,
            "lift_m": bottle_lift,
            "lift_hold": lift_hold,
            "gripper": gripper,
            "tilt_deg": bottle_tilt,
            "stage": stage_end,
            "fail_mode": fail_mode,
        })
        episodes_done += 1
        ep_max_stage = 0
        ep_max_lift = 0.0
        ep_max_lift_hold = 0
        ep_last_gripper = 0.0
        ep_last_tilt = 0.0

    env.close()

    n = max(len(results), 1)
    successes = sum(1 for r in results if r["success"])
    fail_modes: dict[str, int] = {"reach": 0, "grasp": 0, "tilt": 0, "timeout": 0}
    for r in results:
        if r["fail_mode"] != "success":
            fail_modes[r["fail_mode"]] = fail_modes.get(r["fail_mode"], 0) + 1

    run_name = args.run_name or os.path.basename(model_path)
    if run_name.startswith("best_policy_"):
        run_name = run_name[len("best_policy_") : -3] if run_name.endswith(".pt") else run_name

    metrics = {
        "success_rate": successes / len(results) if results else 0.0,
        "mean_lift_m": float(np.mean([r["lift_m"] for r in results])) if results else 0.0,
        "mean_lift_hold": float(np.mean([r["lift_hold"] for r in results])) if results else 0.0,
        "grasp_rate": float(np.mean([1.0 if r["stage"] >= STAGE_GRASP else 0.0 for r in results])) if results else 0.0,
        "fail_modes": fail_modes,
        "episodes": len(results),
        "per_episode": results,
        "run": run_name,
        "iteration": args.iteration,
        "model": os.path.basename(model_path),
    }
    # flush: Isaac's close() hard-exits and drops buffered stdout when piped.
    print(f"LIFT_METRICS {json.dumps(metrics)}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
