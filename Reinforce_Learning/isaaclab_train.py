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
isaaclab_train.py
==================
Training script for Apple Pick-and-Place using Isaac Lab + Stable-Baselines3.
Converted to the world-class Manager-based RL workflow.
"""

from __future__ import annotations

import os
import sys
import glob
import argparse
import pickle

# Dynamic Python path registration to make isaaclab_openarm_env package discoverable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

# Force NVIDIA Vulkan ICD if available, to bypass any integrated GPU conflicts
if os.path.exists("/usr/share/vulkan/icd.d/nvidia_icd.json"):
    os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

# ── 1. Parse arguments and launch Isaac Sim FIRST (must happen before any isaaclab import) ──
parser = argparse.ArgumentParser(description="OpenArm Isaac Lab PPO Training — Manager-Based workflow")
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1024, help="Number of parallel envs")
parser.add_argument("--timesteps",   type=int,   default=1_000_000, help="Total training timesteps")
parser.add_argument("--num-envs-eval", "--num_envs_eval", dest="num_envs_eval", type=int, default=16, help="Eval envs")
parser.add_argument("--log-dir", "--log_dir", dest="log_dir", type=str, default=os.path.join(_THIS_DIR, "logs", "train"), help="Output directory")
parser.add_argument("--model-name", "--model_name", dest="model_name", type=str, default="ppo_openarm_pick_place")
parser.add_argument("--resume",      action="store_true", help="Resume from latest checkpoint in log_dir")
parser.add_argument("--checkpoint",  type=str, default=None,
                    help="Load policy weights from .pt (fine-tune, e.g. best_policy_train_osc_phase2_v3.pt)")
parser.add_argument("--progress",    action="store_true", help="Enable tqdm progress bar")
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--joint-space", "--joint_space", dest="joint_space", action="store_true",
                    help="Use legacy 8-D joint-space actions instead of OSC 6-DOF (default)")
parser.add_argument("--task-phase", "--task_phase", dest="task_phase", type=int, default=None,
                    help="Curriculum: 1=reach, 2=reach+grasp+lift")
parser.add_argument("--descent-assist", "--descent_assist", dest="descent_assist", action="store_true",
                    help="Enable REACH+GRASP descent assist during training")
parser.add_argument("--assist-schedule", "--assist_schedule", dest="assist_schedule", action="store_true",
                    help="Phase 2: decay assist blend 1.0→0.0 over training (Option C)")
parser.add_argument("--stage", type=str, default="all",
                    choices=("reach", "grasp", "lift", "all"),
                    help="Phase 2 sub-stage gates: reach | grasp | lift | all")

# Isaac Sim AppLauncher args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch standalone Isaac Sim
app_launcher  = AppLauncher(args)
simulation_app = app_launcher.app

# ── 2. Import standard libraries and environment configuration ────────────────────────────────────────────
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)
# Linear LR schedule: decays from initial_value to final_value as progress_remaining goes 1→0
def linear_lr_schedule(initial_value: float, final_value: float):
    """Returns a callable suitable for SB3 learning_rate parameter."""
    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from isaaclab_openarm_env.env import ApplePickPlaceEnv
from isaaclab_openarm_env.config import ApplePickPlaceEnvCfg
from isaaclab_openarm_env.phase2_overrides import apply_phase2_train_overrides
from isaaclab_openarm_env.scene import patch_qvic_usd_once

# Permanently patch qvic.usd to remove duplicate robot prims and nested RigidBodyAPIs.
patch_qvic_usd_once()


def find_latest_checkpoint(log_dir: str) -> str | None:
    """Scan for the most recent rl_model_*_steps.zip checkpoint."""
    pattern = os.path.join(log_dir, "checkpoints", "rl_model_*_steps.zip")
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort numerically by step count
    def get_steps(f):
        try:
            return int(os.path.basename(f).split("rl_model_")[1].split("_steps.zip")[0])
        except (IndexError, ValueError):
            return 0
    files.sort(key=get_steps)
    return files[-1]


def make_env(num_envs: int) -> Sb3VecEnvWrapper:
    """Build and wrap the Isaac Lab environment for SB3."""
    env_cfg = ApplePickPlaceEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args.seed
    env_cfg.use_joint_space_actions = args.joint_space
    if args.task_phase is not None:
        env_cfg.task_phase = args.task_phase
        if args.task_phase >= 2:
            env_cfg.episode_length_s = 20.0
    phase = args.task_phase if args.task_phase is not None else env_cfg.task_phase
    if phase >= 2:
        env_cfg.grasp_lift_assist_enabled = True
        if args.descent_assist or args.assist_schedule:
            env_cfg.grasp_descent_assist_enabled = True
            env_cfg.grasp_descent_assist_grasp = True
        if args.assist_schedule:
            env_cfg.grasp_assist_schedule_enabled = True
    elif args.descent_assist:
        env_cfg.grasp_descent_assist_enabled = True
        env_cfg.grasp_descent_assist_grasp = True
    if phase >= 2:
        apply_phase2_train_overrides(
            env_cfg,
            assist_curriculum=args.assist_schedule,
            stage=args.stage,
        )
    env = ApplePickPlaceEnv(cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)
    return env


def keep_only_last_n_tb_logs(tb_dir: str, n: int = 2):
    """Scan and keep only the last n PPO folders in TensorBoard directory."""
    import shutil
    if not os.path.exists(tb_dir):
        return
    ppo_dirs = []
    for entry in os.listdir(tb_dir):
        full_path = os.path.join(tb_dir, entry)
        if os.path.isdir(full_path) and entry.startswith("PPO_"):
            try:
                num = int(entry.split("PPO_")[1])
                ppo_dirs.append((num, full_path))
            except ValueError:
                pass
    ppo_dirs.sort(key=lambda x: x[0])
    if len(ppo_dirs) > n:
        to_delete = ppo_dirs[:-n]
        print(f"  🧹 Auto-cleaning {len(to_delete)} old TensorBoard runs (keeping last {n})...")
        for num, full_path in to_delete:
            try:
                shutil.rmtree(full_path)
                print(f"    - Deleted old run: PPO_{num}")
            except Exception as e:
                print(f"    - Warning: Failed to delete PPO_{num}: {e}")


def keep_only_last_n_checkpoints(ckpt_dir: str, n: int = 5):
    """Delete old rl_model_*_steps.zip files, keep only the newest n."""
    pattern = os.path.join(ckpt_dir, "rl_model_*_steps.zip")
    files = glob.glob(pattern)
    if len(files) <= n:
        return

    def get_steps(f):
        try:
            return int(os.path.basename(f).split("rl_model_")[1].split("_steps.zip")[0])
        except (IndexError, ValueError):
            return 0

    files.sort(key=get_steps)
    to_delete = files[:-n]
    print(f"  🧹 Pruning {len(to_delete)} old checkpoints (keeping last {n})...")
    for f in to_delete:
        try:
            os.remove(f)
        except OSError as e:
            print(f"    - Warning: failed to delete {f}: {e}")


class AssistScheduleCallback(BaseCallback):
    """Decay grasp assist blend from start→end over training (Option C curriculum)."""

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self._total = max(total_timesteps, 1)

    def _on_step(self) -> bool:
        env = self.training_env
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env.envs[0].unwrapped
        cfg = unwrapped.cfg
        if not getattr(cfg, "grasp_assist_schedule_enabled", False):
            return True
        progress = 1.0 - (self.num_timesteps / self._total)
        start = getattr(cfg, "grasp_assist_blend_start", 1.0)
        end = getattr(cfg, "grasp_assist_blend_end", 0.0)
        unwrapped._assist_blend_scale = end + progress * (start - end)
        return True


class PruningCheckpointCallback(CheckpointCallback):
    """CheckpointCallback that keeps only the last N zip files on disk."""

    def __init__(self, keep_last: int = 5, **kwargs):
        super().__init__(**kwargs)
        self._keep_last = keep_last

    def _on_step(self) -> bool:
        ret = super()._on_step()
        keep_only_last_n_checkpoints(self.save_path, self._keep_last)
        return ret


def train():
    log_dir  = args.log_dir
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    tb_dir   = os.path.normpath(os.path.join(log_dir, "..", "tensorboard"))

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir,   exist_ok=True)

    # Keep disk usage bounded: 2 TB runs, 5 checkpoints
    keep_only_last_n_tb_logs(tb_dir, n=2)
    keep_only_last_n_checkpoints(ckpt_dir, n=5)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("  Isaac Lab Manager-Based: OpenArm Apple Pick-and-Place PPO Training")
    print(f"  Device     : {device.upper()}")
    print(f"  Envs       : {args.num_envs} parallel")
    print(f"  Timesteps  : {args.timesteps:,}")
    print(f"  Log dir    : {log_dir}")
    print(f"  TensorBoard: {tb_dir}")
    print(f"  Resume     : {args.resume}")
    print(f"  Actions    : {'joint-space 8D' if args.joint_space else 'OSC 6-DOF + gripper (7D)'}")
    phase = args.task_phase if args.task_phase is not None else ApplePickPlaceEnvCfg().task_phase
    print(f"  Task phase : {phase}")
    print("=" * 65)

    # Build environment
    print("\n  Building training environment...")
    train_env = make_env(args.num_envs)

    # Save environment configurations
    env_cfg_path = os.path.join(log_dir, "env_cfg.pkl")
    with open(env_cfg_path, "wb") as f:
        pickle.dump(train_env.unwrapped.cfg, f)
    print(f"  ✅ Env config saved: {env_cfg_path}")

    # Checkpoint Callback setup
    checkpoint_cb = PruningCheckpointCallback(
        keep_last=5,
        save_freq=max(1_000_000 // args.num_envs, 1),
        save_path=ckpt_dir,
        name_prefix="rl_model",
        verbose=1,
    )
    callbacks = CallbackList([checkpoint_cb])
    if getattr(train_env.unwrapped.cfg, "grasp_assist_schedule_enabled", False):
        callbacks = CallbackList([checkpoint_cb, AssistScheduleCallback(args.timesteps)])
        print("  ✅ Assist schedule enabled (blend 1.0 → 0.0)")

    # Model definition
    latest_ckpt = find_latest_checkpoint(log_dir) if args.resume else None

    if args.resume and latest_ckpt:
        print(f"\n  ✅ Resuming from: {latest_ckpt}")
        model = PPO.load(
            latest_ckpt,
            env=train_env,
            device=device,
            tensorboard_log=tb_dir,
        )
        done_steps = int(latest_ckpt.split("_steps.zip")[0].split("_")[-1])
        remaining  = max(args.timesteps - done_steps, 0)
        print(f"  Steps done: {done_steps:,} / Remaining: {remaining:,}")
    else:
        if args.resume:
            print("  ⚠️  No checkpoint found — starting from scratch.")
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=linear_lr_schedule(3e-4, 1e-5),
            n_steps=64,
            batch_size=4096,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=[256, 256],
                activation_fn=torch.nn.Tanh,
            ),
            verbose=1,
            seed=args.seed,
            device=device,
            tensorboard_log=tb_dir,
        )
        remaining = args.timesteps

        if args.checkpoint:
            ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(_THIS_DIR, args.checkpoint)
            if not os.path.isfile(ckpt_path):
                ckpt_path = os.path.abspath(args.checkpoint)
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
            print(f"\n  ✅ Fine-tuning from: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=device)
            model.policy.load_state_dict(state_dict)

    # Launch RL learning loop
    print(f"\n  🚀 Starting training for {remaining:,} timesteps...\n")
    model.learn(
        total_timesteps=remaining,
        callback=callbacks,
        reset_num_timesteps=not args.resume,
        tb_log_name="PPO",
        progress_bar=args.progress,
    )

    # Save outputs
    final_path = os.path.join(log_dir, args.model_name)
    model.save(final_path)

    policy_state = model.policy.state_dict()
    torch.save(policy_state, os.path.join(log_dir, "final_policy.pt"))

    best_model_zip = os.path.join(log_dir, "best_model.zip")
    if os.path.exists(best_model_zip):
        print(f"\n  💾 Extracting best policy state_dict from {best_model_zip}...")
        try:
            best_model = PPO.load(best_model_zip, device=device)
            best_policy_state = best_model.policy.state_dict()
            torch.save(best_policy_state, os.path.join(log_dir, "best_policy.pt"))
            print(f"  ✅ Extracted best policy weights: {log_dir}/best_policy.pt")
        except Exception as e:
            print(f"  ⚠️ Error: {e}. Saving final policy as best_policy.pt")
            torch.save(policy_state, os.path.join(log_dir, "best_policy.pt"))
    else:
        torch.save(policy_state, os.path.join(log_dir, "best_policy.pt"))
        print(f"  ⚠️ Saved final policy weights to: {log_dir}/best_policy.pt")

    print("\n" + "=" * 65)
    print(f"  ✅ Training Complete!")
    print(f"  Final model  : {final_path}.zip")
    print(f"  Policy (.pt) : {os.path.join(log_dir, 'best_policy.pt')}")
    print(f"  TensorBoard  : tensorboard --logdir {tb_dir}")
    print("=" * 65)

    train_env.close()


if __name__ == "__main__":
    train()
    simulation_app.close()
