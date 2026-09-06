#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
#
# Headless phase-2 evaluation — prints one LIFT_METRICS JSON line.
#
# Runs N parallel envs with randomised bottle placement so the episodes are
# actually independent samples. `--assist-scale` separates "the script lifts"
# from "the policy lifts": eval at 1.0 / 0.5 / 0.0 and compare.

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
parser.add_argument("--episodes", type=int, default=30)
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=8,
                    help="Parallel envs. Episodes are collected across all of them.")
parser.add_argument("--bottle-noise", "--bottle_noise", dest="bottle_noise", type=float, default=0.05,
                    help="Bottle spawn XY noise (m). 0.0 makes every episode identical.")
parser.add_argument("--assist-scale", "--assist_scale", dest="assist_scale", type=float, default=1.0,
                    help="Scripted assist authority: 1.0 = full script, 0.0 = pure policy.")
parser.add_argument("--stochastic", action="store_true",
                    help="Sample from the policy instead of taking the mean action.")
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
from isaaclab_openarm_env.mdp.helpers import STAGE_GRASP, STAGE_REACH, check_init_buffers
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
    try:
        with open(cfg_path, "rb") as f:
            saved = pickle.load(f)
    except Exception as e:
        print(f"  ⚠️  Bỏ qua env_cfg snapshot ({os.path.basename(cfg_path)}): {e}")
        return
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
    latched: bool,
    lift_cmd_steps: int,
    gripper: float,
    bottle_lift: float,
    bottle_tilt: float,
    lift_hold: int,
    lift_hold_steps: int,
    grip_thresh: float,
    lift_thresh: float,
    tipped_deg: float,
) -> str:
    """Name the failure so a fix is falsifiable.

    `no_lift_command` is the mode this pipeline exhibits today: the bottle is
    grasped and latched but the lift state machine never armed, so the assist
    never issued an upward command.
    """
    if is_success:
        return "success"
    if bottle_tilt >= tipped_deg * 0.9:
        return "tilt"
    if stage_end == STAGE_REACH:
        return "reach"
    if not latched:
        return "grasp"
    if lift_cmd_steps == 0:
        return "no_lift_command"
    if bottle_lift < lift_thresh:
        return "lift_too_low"
    if lift_hold < lift_hold_steps:
        return "hold_unstable"
    return "timeout"


class _EpisodeAccumulator:
    """Per-env running peaks, harvested when that env reports done."""

    __slots__ = (
        "max_stage", "max_lift", "max_lift_hold", "last_gripper", "max_gripper",
        "last_tilt", "last_z_f", "last_top", "last_lat_f", "last_dist_f",
        "max_gc", "min_gc_after_exhaust", "min_span", "latched", "steps",
        "lift_cmd_steps", "first_lift_cmd_step", "abort_reasons",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.max_stage = 0
        # -999 not 0.0: bottle_lift sits negative when the datum is stale, and a
        # 0.0 floor would silently clip every real reading to zero.
        self.max_lift = -999.0
        self.max_lift_hold = 0
        self.last_gripper = 0.0
        self.max_gripper = 0.0
        self.last_tilt = 0.0
        self.last_z_f = 0.0
        self.last_top = 0.0
        self.last_lat_f = 0.0
        self.last_dist_f = 0.0
        self.max_gc = 0.0
        self.min_gc_after_exhaust = 1.0
        self.min_span = 999.0
        self.latched = False
        self.steps = 0
        self.lift_cmd_steps = 0
        self.first_lift_cmd_step = -1
        self.abort_reasons: dict[str, int] = {}


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def main() -> None:
    model_path = os.path.abspath(args.model_path)
    if not os.path.isfile(model_path):
        print(f"LIFT_METRICS {json.dumps({'error': f'model not found: {model_path}'})}")
        simulation_app.close()
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = max(1, args.num_envs)

    env_cfg = ApplePickPlaceEnvCfg()
    env_cfg.sim.log_dir = os.path.join(_THIS_DIR, "logs", "sim_logs")
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.seed = args.seed
    env_cfg.task_phase = args.task_phase
    if args.task_phase >= 2:
        env_cfg.episode_length_s = 20.0

    _apply_env_cfg_snapshot(env_cfg, model_path)
    apply_phase2_demo_gates(env_cfg, model_path, stage=args.stage)

    env_cfg.bottle_pos_noise = args.bottle_noise
    _show_align = os.environ.get("EVAL_SHOW_ALIGN_LOG") == "1"
    env_cfg.align_log_interval = 50 if _show_align else 999999
    env_cfg.debug_success_log = False
    env_cfg.suppress_align_log = not _show_align

    env = ApplePickPlaceEnv(cfg=env_cfg)
    env.unwrapped._assist_blend_scale = float(args.assist_scale)
    env = Sb3VecEnvWrapper(env)
    check_init_buffers(env.unwrapped)

    model = _load_pt_policy(model_path, env, device)
    cfg = env.unwrapped.cfg
    lift_hold_steps = getattr(cfg, "grasp_lift_hold_steps", 5)
    grip_thresh = getattr(cfg, "grasp_grip_threshold", 0.4)
    lift_thresh = getattr(cfg, "grasp_lift_threshold", 0.03)
    tipped_deg = getattr(cfg, "bottle_tipped_termination_deg", 60.0)

    print(
        f"  [Eval] envs={num_envs} episodes={args.episodes} noise={args.bottle_noise}"
        f" assist_scale={args.assist_scale} stage={args.stage}"
    )

    results: list[dict] = []
    acc = [_EpisodeAccumulator() for _ in range(num_envs)]
    obs = env.reset()

    term_gripper = env.unwrapped.action_manager._terms.get("gripper_action")
    reopen_max = int(getattr(cfg, "grasp_reopen_max_count", 3))

    while simulation_app.is_running() and len(results) < args.episodes:
        with torch.no_grad():
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            action = np.clip(action, -1.0, 1.0)

        obs, _, dones, infos = env.step(action)
        u = env.unwrapped
        ls = u._last_state

        # Harvest live state for every env that did NOT just reset; the post-reset
        # pose would poison the peaks.
        if ls is not None:
            stage_t = u._stage.detach().cpu().numpy()
            lift_t = ls["bottle_lift"].detach().cpu().numpy()
            grip_t = ls["gripper_state"].detach().cpu().numpy()
            tilt_t = ls["bottle_tilt_deg"].detach().cpu().numpy() if "bottle_tilt_deg" in ls else np.zeros(num_envs)
            hold_t = u._steps_bottle_lifted.detach().cpu().numpy()
            z_f_t = ls["z_error_finger"].detach().cpu().numpy()
            top_t = ls["top_down_align"].detach().cpu().numpy()
            lat_t = ls["lateral_finger_xy"].detach().cpu().numpy()
            dist_t = ls["dist_finger_body"].detach().cpu().numpy()
            span_t = ls["finger_span_xy"].detach().cpu().numpy() if "finger_span_xy" in ls else None
            gc_t = (
                term_gripper._close_progress.detach().cpu().numpy()
                if term_gripper is not None and hasattr(term_gripper, "_close_progress")
                else None
            )
            reopen_t = (
                term_gripper._reopen_count.detach().cpu().numpy()
                if term_gripper is not None and hasattr(term_gripper, "_reopen_count")
                else None
            )
            latch_t = (
                term_gripper._grasp_latched.detach().cpu().numpy()
                if term_gripper is not None and hasattr(term_gripper, "_grasp_latched")
                else None
            )
            want_lift_t = (
                u._assist_want_lift.detach().cpu().numpy()
                if hasattr(u, "_assist_want_lift")
                else None
            )
            abort_t = getattr(u, "_lift_abort_reason", None)

            for i in range(num_envs):
                if dones[i]:
                    continue
                a = acc[i]
                a.steps += 1
                a.max_stage = max(a.max_stage, int(stage_t[i]))
                a.max_lift = max(a.max_lift, float(lift_t[i]))
                a.max_lift_hold = max(a.max_lift_hold, int(hold_t[i]))
                a.last_gripper = float(grip_t[i])
                a.max_gripper = max(a.max_gripper, float(grip_t[i]))
                a.last_tilt = float(tilt_t[i])
                a.last_z_f = float(z_f_t[i])
                a.last_top = float(top_t[i])
                a.last_lat_f = float(lat_t[i])
                a.last_dist_f = float(dist_t[i])
                if span_t is not None:
                    a.min_span = min(a.min_span, float(span_t[i]))
                if gc_t is not None:
                    a.max_gc = max(a.max_gc, float(gc_t[i]))
                    if reopen_t is not None and int(reopen_t[i]) >= reopen_max:
                        a.min_gc_after_exhaust = min(a.min_gc_after_exhaust, float(gc_t[i]))
                if latch_t is not None and bool(latch_t[i]):
                    a.latched = True
                if want_lift_t is not None and bool(want_lift_t[i]):
                    a.lift_cmd_steps += 1
                    if a.first_lift_cmd_step < 0:
                        a.first_lift_cmd_step = a.steps
                if abort_t is not None:
                    reason = abort_t[i] if isinstance(abort_t, (list, tuple)) else None
                    if reason:
                        a.abort_reasons[reason] = a.abort_reasons.get(reason, 0) + 1

        if not np.any(dones):
            continue

        # `success` / `time_outs` buffers persist through the auto-reset, so they
        # still describe the transition that produced `dones`.
        try:
            success_t = u.termination_manager.get_term("success").detach().cpu().numpy()
        except (KeyError, ValueError, AttributeError):
            success_t = np.zeros(num_envs, dtype=bool)
        try:
            timeout_t = u.termination_manager.time_outs.detach().cpu().numpy()
        except AttributeError:
            timeout_t = np.zeros(num_envs, dtype=bool)

        for i in range(num_envs):
            if not dones[i] or len(results) >= args.episodes:
                continue
            a = acc[i]
            if a.max_lift <= -998.0:
                a.max_lift = 0.0  # episode produced no state samples
            is_success = bool(success_t[i]) or (
                a.max_lift_hold >= lift_hold_steps and cfg.task_phase >= 2
            )
            fail_mode = _classify_fail_mode(
                is_success=is_success,
                is_truncated=bool(timeout_t[i]),
                stage_end=a.max_stage,
                latched=a.latched,
                lift_cmd_steps=a.lift_cmd_steps,
                gripper=a.max_gripper,
                bottle_lift=a.max_lift,
                bottle_tilt=a.last_tilt,
                lift_hold=a.max_lift_hold,
                lift_hold_steps=lift_hold_steps,
                grip_thresh=grip_thresh,
                lift_thresh=lift_thresh,
                tipped_deg=tipped_deg,
            )
            results.append({
                "success": is_success,
                "truncated": bool(timeout_t[i]),
                "lift_m": a.max_lift,
                "lift_hold": a.max_lift_hold,
                "gripper": a.max_gripper,
                "tilt_deg": a.last_tilt,
                "stage": a.max_stage,
                "latched": a.latched,
                "lift_cmd_steps": a.lift_cmd_steps,
                "first_lift_cmd_step": a.first_lift_cmd_step,
                "fail_mode": fail_mode,
                "max_gc": a.max_gc,
                "min_gc_after_exhaust": a.min_gc_after_exhaust if a.min_gc_after_exhaust < 1.0 else None,
                "min_span_m": a.min_span if a.min_span < 999.0 else None,
                "abort_reasons": a.abort_reasons or None,
                "steps": a.steps,
            })
            a.reset()

    env.close()

    if not results:
        print(f"LIFT_METRICS {json.dumps({'error': 'no episodes collected'})}", flush=True)
        simulation_app.close()
        return

    n = len(results)
    successes = sum(1 for r in results if r["success"])
    fail_modes: dict[str, int] = {}
    abort_hist: dict[str, int] = {}
    for r in results:
        if r["fail_mode"] != "success":
            fail_modes[r["fail_mode"]] = fail_modes.get(r["fail_mode"], 0) + 1
        for reason, cnt in (r.get("abort_reasons") or {}).items():
            abort_hist[reason] = abort_hist.get(reason, 0) + cnt

    lifts = [r["lift_m"] for r in results]
    lift_cmd = [r["lift_cmd_steps"] for r in results]
    first_cmd = [r["first_lift_cmd_step"] for r in results if r["first_lift_cmd_step"] >= 0]

    run_name = args.run_name or os.path.basename(model_path)
    if run_name.startswith("best_policy_"):
        run_name = run_name[len("best_policy_") : -3] if run_name.endswith(".pt") else run_name

    metrics = {
        "success_rate": successes / n,
        "grasp_rate": float(np.mean([1.0 if r["stage"] >= STAGE_GRASP else 0.0 for r in results])),
        "latch_rate": float(np.mean([1.0 if r["latched"] else 0.0 for r in results])),
        "lift_start_rate": float(np.mean([1.0 if r["lift_cmd_steps"] > 0 else 0.0 for r in results])),
        "mean_lift_m": float(np.mean(lifts)),
        "p50_max_lift_m": _percentile(lifts, 50),
        "p90_max_lift_m": _percentile(lifts, 90),
        "mean_lift_commanded_steps": float(np.mean(lift_cmd)),
        "mean_steps_to_first_lift_cmd": float(np.mean(first_cmd)) if first_cmd else None,
        "mean_lift_hold": float(np.mean([r["lift_hold"] for r in results])),
        "fail_modes": fail_modes,
        "abort_reasons": abort_hist or None,
        "episodes": n,
        "num_envs": num_envs,
        "assist_scale": args.assist_scale,
        "bottle_noise": args.bottle_noise,
        "seed": args.seed,
        "per_episode": results,
        "run": run_name,
        "iteration": args.iteration,
        "model": os.path.basename(model_path),
    }

    print(
        f"  [Eval] success={metrics['success_rate']:.2f} grasp={metrics['grasp_rate']:.2f}"
        f" latch={metrics['latch_rate']:.2f} lift_start={metrics['lift_start_rate']:.2f}"
        f" p50_lift={metrics['p50_max_lift_m']:+.4f} lift_cmd_steps={metrics['mean_lift_commanded_steps']:.1f}"
    )
    print(f"  [Eval] fail_modes={fail_modes}")
    # flush: Isaac's close() hard-exits and drops buffered stdout when piped.
    print(f"LIFT_METRICS {json.dumps(metrics)}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
