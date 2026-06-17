#!/usr/bin/env python3
"""
isaaclab_train.py
==================
Training script for Apple Pick-and-Place using Isaac Lab + Stable-Baselines3.

Run on server (headless):
    /path/to/isaac-sim/python.sh isaaclab_train.py \\
        --headless \\
        --num_envs 1024 \\
        --timesteps 1000000 \\
        --log_dir /data21tb/huyhoang/openarm_train_ws/logs/train

Run locally with GUI (small test):
    /path/to/isaac-sim/python.sh isaaclab_train.py \\
        --num_envs 4 \\
        --timesteps 50000

Resume interrupted training:
    /path/to/isaac-sim/python.sh isaaclab_train.py \\
        --headless --resume \\
        --log_dir /data21tb/huyhoang/openarm_train_ws/logs/train
"""

from __future__ import annotations

import os
import glob
import argparse
import pickle

# ── 1. Parse args and launch Isaac Sim FIRST (must happen before any isaaclab import) ──
parser = argparse.ArgumentParser(description="OpenArm Isaac Lab PPO Training")
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1024, help="Number of parallel envs")
parser.add_argument("--timesteps",   type=int,   default=1_000_000,help="Total training timesteps")
parser.add_argument("--num-envs-eval", "--num_envs_eval", dest="num_envs_eval", type=int, default=16, help="Eval envs")
parser.add_argument("--log-dir", "--log_dir", dest="log_dir", type=str, default="./logs/train", help="Output directory")
parser.add_argument("--model-name", "--model_name", dest="model_name", type=str, default="ppo_openarm_pick_place")
parser.add_argument("--resume",      action="store_true", help="Resume from latest checkpoint")
parser.add_argument("--progress",    action="store_true", help="Enable tqdm progress bar (warning: can bloat background log files)")
parser.add_argument("--seed",        type=int,   default=42)

# Isaac Sim AppLauncher args (--headless, --enable_cameras, etc.)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim (must be done before importing anything else from isaaclab)
app_launcher  = AppLauncher(args)
simulation_app = app_launcher.app

# ── 2. Now import everything else ────────────────────────────────────────────
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

# Import our environment
from isaaclab_openarm_env import ApplePickPlaceEnv, ApplePickPlaceEnvCfg

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(log_dir: str) -> str | None:
    """Scan for the most recent rl_model_*_steps.zip checkpoint."""
    pattern = os.path.join(log_dir, "checkpoints", "rl_model_*_steps.zip")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def make_env(num_envs: int) -> Sb3VecEnvWrapper:
    """Build and wrap the Isaac Lab environment for SB3."""
    env_cfg = ApplePickPlaceEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed            = args.seed
    env = ApplePickPlaceEnv(cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train():
    log_dir  = args.log_dir
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    tb_dir   = os.path.join(log_dir, "..", "tensorboard")  # sibling dir

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir,   exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("  Isaac Lab: OpenArm Apple Pick-and-Place PPO Training")
    print(f"  Device     : {device.upper()}")
    print(f"  Envs       : {args.num_envs} parallel")
    print(f"  Timesteps  : {args.timesteps:,}")
    print(f"  Log dir    : {log_dir}")
    print(f"  TensorBoard: {tb_dir}")
    print(f"  Resume     : {args.resume}")
    print("=" * 65)

    # ── Build training environment ────────────────────────────────────────────
    print("\n  Building training environment...")
    train_env = make_env(args.num_envs)

    # Save env config for later loading during demo
    env_cfg_path = os.path.join(log_dir, "env_cfg.pkl")
    with open(env_cfg_path, "wb") as f:
        pickle.dump(train_env.unwrapped.cfg, f)
    print(f"  ✅ Env config saved: {env_cfg_path}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    # Save a checkpoint every 1,000,000 timesteps (decreased frequency to save disk space)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1_000_000 // args.num_envs, 1),
        save_path=ckpt_dir,
        name_prefix="rl_model",
        verbose=1,
    )
    callbacks = CallbackList([checkpoint_cb])

    # ── PPO Model ─────────────────────────────────────────────────────────────
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
            learning_rate=get_linear_fn(3e-4, 1e-5, 1.0),
            n_steps=2048,
            batch_size=256,
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

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"\n  🚀 Starting training for {remaining:,} timesteps...\n")
    model.learn(
        total_timesteps=remaining,
        callback=callbacks,
        reset_num_timesteps=not args.resume,
        tb_log_name="PPO",
        progress_bar=args.progress,
    )

    # ── Save final model ───────────────────────────────────────────────────────
    final_path = os.path.join(log_dir, args.model_name)
    model.save(final_path)

    # Also save policy weights as .pt for easy loading in demos
    policy_state = model.policy.state_dict()
    torch.save(policy_state, os.path.join(log_dir, "final_policy.pt"))

    # Also save as best_policy.pt for compatibility with demo scripts
    torch.save(policy_state, os.path.join(log_dir, "best_policy.pt"))
    print(f"  ✅ Best policy weights: {log_dir}/best_policy.pt")

    print("\n" + "=" * 65)
    print(f"  ✅ Training Complete!")
    print(f"  Final model  : {final_path}.zip")
    print(f"  Policy (.pt) : {os.path.join(log_dir, 'best_policy.pt')}")
    print(f"  TensorBoard  : tensorboard --logdir {tb_dir}")
    print("=" * 65)

    train_env.close()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
    simulation_app.close()
