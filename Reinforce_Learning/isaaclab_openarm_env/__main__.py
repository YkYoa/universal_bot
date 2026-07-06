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

import sys
import os
import argparse
import torch

# Ensure package root is in sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PARENT_DIR not in sys.path:
    sys.path.append(_PARENT_DIR)

# Parse standalone execution arguments
parser = argparse.ArgumentParser(description="OpenArm Apple Pick-and-Place — Standalone Manager-Based Env Loader")
parser.add_argument("--num_envs",  type=int, default=4,   help="Parallel envs")
parser.add_argument("--num_steps", type=int, default=200, help="RL steps")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch simulator application
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import package components from submodules
from isaaclab_openarm_env.env import ApplePickPlaceEnv
from isaaclab_openarm_env.config import ApplePickPlaceEnvCfg, STAGE_REACH, STAGE_GRASP, STAGE_PLACE

# Configure and instantiate environment directly (best practice for custom standalone tasks)
print("\n  Instantiating environment 'ApplePickPlaceEnv' directly...")
env_cfg = ApplePickPlaceEnvCfg()
env_cfg.scene.num_envs = args.num_envs

env = ApplePickPlaceEnv(cfg=env_cfg)

print(f"\n{'='*60}")
print(f"  Isaac Lab Manager-Based RL Standalone Load Test")
print(f"  Num envs  : {env.num_envs}")
print(f"  Device    : {env.device}")
print(f"  Obs space : {env.observation_space}")
print(f"  Act space : {env.action_space}")
print(f"{'='*60}\n")

obs, _ = env.reset()
print(f"Initial observation shape: {obs['policy'].shape}")

for step in range(args.num_steps):
    # Send random actions in [-1, 1]
    act_dim = env.action_space.shape[0]
    actions = torch.rand(env.num_envs, act_dim, device=env.device) * 2.0 - 1.0
    obs, rewards, terminated, truncated, infos = env.step(actions)

    if (step + 1) % 50 == 0 or step == args.num_steps - 1:
        mean_reward = rewards.mean().item()
        num_done    = (terminated | truncated).sum().item()
        
        stage_dist  = [
            (env._stage == STAGE_REACH).sum().item(),
            (env._stage == STAGE_GRASP).sum().item(),
            (env._stage == STAGE_PLACE).sum().item(),
        ]
        print(f"  Step {step+1:4d} | mean_reward={mean_reward:+.3f} | envs_done={num_done}/{env.num_envs} | stages={stage_dist}")

print("\n✅  Test complete — Manager-based environment ran successfully.")
env.close()
simulation_app.close()
