#!/usr/bin/env python3
"""
isaaclab_demo.py
================
Playback script to run and visualize a trained Stable-Baselines3 policy locally or via streaming.

Usage (Local GUI viewport):
    /home/hans/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \\
        src/isaacsim_setup/isaaclab_demo.py \\
        --model_path src/isaacsim_setup/logs/best_policy.pt \\
        --num_envs 1

Usage (Headless WebRTC livestreaming from server):
    /data21tb/huyhoang/isaacsim/python.sh \
        src/isaacsim_setup/isaaclab_demo.py \
        --model_path logs_openarm/train/best_policy.pt \
        --num_envs 1 \
        --headless \
        --livestream 1
"""

from __future__ import annotations

import os
import time
import argparse
import torch
import numpy as np
from stable_baselines3 import PPO

# ── 1. Parse arguments and launch Isaac Sim first (must happen before importing isaaclab) ──
parser = argparse.ArgumentParser(description="Isaac Lab OpenArm Apple Pick-and-Place Demo")
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1, help="Number of parallel environments to run")
parser.add_argument("--model-path", "--model_path", dest="model_path", type=str, default="./logs/best_policy.pt", help="Path to trained policy (.pt weights or SB3 .zip checkpoint)")
parser.add_argument("--seed",       type=int, default=42, help="Random seed")

# Add Isaac Sim AppLauncher args (like --headless, --livestream, etc.)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── 2. Now import Isaac Lab and gymnasium wrappers ──
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_openarm_env import ApplePickPlaceEnv, ApplePickPlaceEnvCfg

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Resolve paths relative to workspace directory
    model_path = os.path.abspath(args.model_path)

    print("=" * 70)
    print("  Isaac Lab OpenArm Apple Pick-and-Place Policy Player")
    print(f"  Model Path : {model_path}")
    print(f"  Envs       : {args.num_envs}")
    print(f"  Device     : {device.upper()}")
    print("=" * 70)

    # ── Build environment ──
    print("\n  Building environment...")
    env_cfg = ApplePickPlaceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    # Enable rendering at every step for smooth visual playback
    env_cfg.sim.render_interval = 1
    env_cfg.seed = args.seed
    
    env = ApplePickPlaceEnv(cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)

    # ── Load Model / Policy ──
    print(f"\n  Loading weights from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"  ❌ File not found: {model_path}")
        print("  Please run './fetch_model.sh' first or check the path.")
        env.close()
        return

    # Instantiate SB3 PPO model structure
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.Tanh,
        ),
        device=device,
    )

    if model_path.endswith(".zip"):
        try:
            model = PPO.load(model_path, env=env, device=device)
            print("  ✅ Loaded full SB3 PPO model (.zip).")
        except Exception as e:
            print(f"  ❌ Error loading .zip file: {e}")
            env.close()
            return
    else:
        # Load state_dict from saved policy weights (.pt file)
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.policy.load_state_dict(state_dict)
            print("  ✅ Loaded PyTorch policy state_dict (.pt).")
        except Exception as e:
            print(f"  ⚠️  Failed to load as state_dict directly: {e}")
            print("  Retrying load as a standard SB3 model...")
            try:
                model = PPO.load(model_path, env=env, device=device)
                print("  ✅ Loaded model successfully using PPO.load")
            except Exception as e_zip:
                print(f"  ❌ Error loading model: {e_zip}")
                env.close()
                return

    # ── Playback Loop ──
    print("\n  Running simulation. Press Ctrl+C in the terminal to stop.")
    obs = env.reset()
    
    total_rewards = np.zeros(args.num_envs)
    episode_steps = np.zeros(args.num_envs)
    
    step = 0
    try:
        while simulation_app.is_running():
            # Get action predictions
            with torch.no_grad():
                action, _ = model.predict(obs, deterministic=True)
            
            # Step simulation
            obs, rewards, dones, infos = env.step(action)
            
            total_rewards += rewards
            episode_steps += 1
            
            # Reset logs for finished episodes
            for i, done in enumerate(dones):
                if done:
                    success = infos[i].get("success", False)
                    status = "SUCCESS" if success else "TIMEOUT"
                    print(f"  [Env {i:02d}] Done! Steps: {int(episode_steps[i])} | Return: {total_rewards[i]:.2f} | Status: {status}")
                    total_rewards[i] = 0.0
                    episode_steps[i] = 0
            
            # Control playback speed to be human-digestible when running with GUI
            if not args.headless:
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n  Exiting playback...")

    print("\n  Closing environment...")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
