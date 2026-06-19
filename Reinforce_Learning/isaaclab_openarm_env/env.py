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

from __future__ import annotations

import omni.usd
from pxr import UsdPhysics, Usd
from isaaclab.envs import ManagerBasedRLEnv
from .config import ApplePickPlaceEnvCfg


class ApplePickPlaceEnv(ManagerBasedRLEnv):
    """
    Apple Pick-and-Place environment using Isaac Lab ManagerBasedRLEnv workflow.
    Physics and collision parameters are pre-configured during backdrop loading.
    """

    cfg: ApplePickPlaceEnvCfg

    def __init__(self, cfg: ApplePickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        # Deactivate Realsense and duplicate static prims BEFORE super().__init__() starts
        # the physics simulation. Isaac Lab calls _setup_scene() inside __init__, and the
        # physics engine is started afterward. However the USD stage is already populated by
        # the time spawn functions run, so we hook into _setup_scene via override below.
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        """Override to deactivate problematic prims right after scene assets are spawned
        but BEFORE the physics simulation is initialized."""
        # Run the normal scene setup (spawns all USD assets)
        super()._setup_scene()

        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        for i in range(self.scene.cfg.num_envs):
            # Deactivate active robot's Realsense camera prims if active
            for arm in ["left", "right"]:
                realsense_path = (
                    f"/World/envs/env_{i}/openarm/openarm_{arm}_hand"
                    f"/openarm_{arm}_hand_tcp/Realsense"
                )
                realsense_prim = stage.GetPrimAtPath(realsense_path)
                if realsense_prim.IsValid() and realsense_prim.IsActive():
                    realsense_prim.SetActive(False)
                    if i == 0:
                        print(f"[ApplePickPlaceEnv] Deactivated active robot Realsense camera: {realsense_path}")
