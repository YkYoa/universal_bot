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
import torch
import numpy as np

# Dynamic Python path registration to make isaaclab_openarm_env package discoverable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

# ── 1. Parse arguments and launch Isaac Sim first (must happen before importing isaaclab) ──
parser = argparse.ArgumentParser(description="Isaac Lab OpenArm Apple Pick-and-Place Demo — Manager-Based workflow")
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1, help="Number of parallel environments to run")
parser.add_argument("--model-path", "--model_path", dest="model_path", type=str, default=os.path.join(_THIS_DIR, "logs", "train", "best_policy.pt"), help="Path to trained policy (.pt weights or SB3 .zip checkpoint)")
parser.add_argument("--seed",       type=int, default=42, help="Random seed")
parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=0, help="Maximum simulation steps to run before exiting (0 for infinite)")

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
from isaaclab_openarm_env.scene import patch_qvic_usd_once

# Permanently patch qvic.usd to remove duplicate robot prims and nested RigidBodyAPIs.
patch_qvic_usd_once()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Resolve paths relative to workspace directory
    model_path = os.path.abspath(args.model_path)

    print("=" * 70)
    print("  Isaac Lab OpenArm Apple Pick-and-Place Policy Player (Manager-Based)")
    print(f"  Model Path : {model_path}")
    print(f"  Envs       : {args.num_envs}")
    print(f"  Device     : {device.upper()}")
    print("=" * 70)

    # ── Build environment ──
    print("\n  Building environment...")
    env_cfg = ApplePickPlaceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.render_interval = env_cfg.decimation
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
    value_histories = [[] for _ in range(args.num_envs)]
    total_steps = 0
    
    # Initialize collision tracking
    was_table_colliding = np.zeros(args.num_envs, dtype=bool)
    was_object_colliding = np.zeros(args.num_envs, dtype=bool)
    
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
                action, _ = model.predict(obs, deterministic=True)
                # Predict value from policy (state value estimates)
                obs_tensor, _ = model.policy.obs_to_tensor(obs)
                values = model.policy.predict_values(obs_tensor).flatten().cpu().numpy()
            
            # Extract observation features directly (safe, fast, and no USD locks)
            ee_pos = obs[:, 14:17]
            ee_to_grasp = obs[:, 17:20]
            dist_ee_grasp = np.linalg.norm(ee_to_grasp, axis=-1)   # distance to stage-dynamic target
            table_clearance = ee_pos[:, 2] - 0.58

            # Fetch real dist_ee_bottle (distance to actual grasp point) from env state cache
            dist_ee_bottle_arr = np.full(args.num_envs, 999.0)
            if hasattr(env.unwrapped, "_last_state") and env.unwrapped._last_state is not None:
                dist_ee_bottle_arr = env.unwrapped._last_state["dist_ee_bottle"].cpu().numpy()
            
            for i in range(args.num_envs):
                value_histories[i].append(values[i])
            
            # Detect collisions
            is_table_colliding = (table_clearance <= -0.02)
            # "Object contact" = TCP within 6cm of the actual grasp point on bottle body
            is_object_colliding = (dist_ee_bottle_arr <= 0.06)
            
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
                    print(f"  [Env {i:02d} Step {int(episode_steps[i]):3d}] 🎯 BOTTLE CONTACT! Critic Value: {values[i]:.4f} | dist_bottle: {dist_ee_bottle_arr[i]:.4f}m | TCP Pos: {tcp_str}")
                elif not is_object_colliding[i] and was_object_colliding[i]:
                    print(f"  [Env {i:02d} Step {int(episode_steps[i]):3d}] Lost Bottle Contact. Critic Value: {values[i]:.4f} | TCP Pos: {tcp_str}")
                
                # Update history
                was_table_colliding[i] = is_table_colliding[i]
                was_object_colliding[i] = is_object_colliding[i]
            
            # Print continuous real-time coordinate tracking on a single updating line
            if args.num_envs == 1 and not has_event:
                tcp_str = f"[{ee_pos[0,0]:.3f}, {ee_pos[0,1]:.3f}, {ee_pos[0,2]:.3f}]"
                act_str = np.array2string(action[0], precision=3, separator=',').replace('\n', '')
                obs_str = np.array2string(obs[0][:6], precision=3, separator=',').replace('\n', '') # just show first 6 features
                dq_norm = np.linalg.norm(obs[0][7:14])
                print(f"  [Env 00 Step {int(episode_steps[0]):3d}] TCP Pos: {tcp_str} | Obs: {obs_str} | DQ Norm: {dq_norm:.4f} | Critic: {values[0]:.4f}\033[K", end="\r", flush=True)
            
            obs, rewards, dones, infos = env.step(action)
            # Synchronize CUDA to prevent async GPU race conditions
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            total_rewards += rewards
            episode_steps += 1
            total_steps += 1
            
            for i, done in enumerate(dones):
                if done:
                    print("\r\033[K", end="")  # Clear the current carriage-return line
                    success = infos[i].get("success", False)
                    status = "SUCCESS" if success else "TIMEOUT"
                    
                    if len(value_histories[i]) > 0:
                        vals = np.array(value_histories[i])
                        avg_val = np.mean(vals)
                        min_val = np.min(vals)
                        max_val = np.max(vals)
                        val_str = f" | Critic Val (Avg/Min/Max): {avg_val:.2f}/{min_val:.2f}/{max_val:.2f}"
                    else:
                        val_str = ""
                    
                    print(f"  [Env {i:02d}] Done! Steps: {int(episode_steps[i])} | Return: {total_rewards[i]:.2f}{val_str} | Status: {status}")
                    total_rewards[i] = 0.0
                    episode_steps[i] = 0
                    value_histories[i] = []
                    
                    # Reset collision tracking flags for this environment
                    was_table_colliding[i] = False
                    was_object_colliding[i] = False
            
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
