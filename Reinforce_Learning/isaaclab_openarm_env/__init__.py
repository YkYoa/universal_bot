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

import gymnasium as gym

# Expose constants that do not depend on Omniverse/pxr
STAGE_REACH = 0   # Move EE above bottle, avoid table
STAGE_GRASP = 1   # Close gripper and lift bottle
STAGE_PLACE = 2   # Transport bottle to bowl and release

# Register the environment with Gymnasium using string entry points to prevent premature imports
# of pxr/Isaac Sim before AppLauncher is initialized.
gym.register(
    id="Isaac-OpenArm-Apple-Pick-Place-v0",
    entry_point="isaaclab_openarm_env.env:ApplePickPlaceEnv",
    kwargs={
        "env_cfg_entry_point": "isaaclab_openarm_env.config:ApplePickPlaceEnvCfg",
    },
    disable_env_checker=True,
)
