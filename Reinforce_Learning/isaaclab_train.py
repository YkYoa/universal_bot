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

# Isaac Lab writes file logs under tempfile.gettempdir()/isaaclab/logs — use per-user TMPDIR
# (shared /tmp/isaaclab is often owned by another account on multi-user servers).
_il_tmp = os.environ.get("ISAACLAB_TMPDIR") or os.path.join(
    os.path.expanduser("~"), ".cache", "isaaclab_tmp"
)
os.makedirs(_il_tmp, exist_ok=True)
os.environ.setdefault("TMPDIR", _il_tmp)

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
parser.add_argument("--lr-start", "--lr_start", dest="lr_start", type=float, default=3e-4,
                    help="LR đầu. Fine-tune sau khi đổi thang reward: dùng 1e-4.")
parser.add_argument("--lr-end", "--lr_end", dest="lr_end", type=float, default=1e-5)
parser.add_argument("--clip-range", "--clip_range", dest="clip_range", type=float, default=0.2,
                    help="Fine-tune: 0.1 để update đầu không phá reach/grasp đang tốt.")
parser.add_argument("--ent-coef", "--ent_coef", dest="ent_coef", type=float, default=0.005)
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


def _freeze_scripted_gripper_log_std(model, always_freeze_idx=None, threshold=4.0):
    """Đóng băng vĩnh viễn gradient của log_std cho các dim không bao giờ học
    được gì thật (gripper_action luôn bị scripted override trong actions.py,
    xác nhận bằng grep "_assist_blend_scale" — 0 kết quả). Không đóng băng thì
    entropy bonus đẩy log_std của dim đó tăng vô hạn (đã đo: 16.0 ≈ std 9tr).

    Dùng cho CẢ 2 đường nạp model: fine-tune (`--checkpoint`, load_state_dict)
    VÀ resume (`--resume`, PPO.load) — hook là runtime-only, PPO.load() không
    khôi phục nó nên phải áp lại mỗi lần, không chỉ lần đầu.

    `always_freeze_idx`: đóng băng các dim này VÔ ĐIỀU KIỆN (không cần giá trị
    hiện tại đã hỏng — dùng cho resume, khi giá trị có thể đã hợp lý từ lần
    đóng băng trước nhưng sẽ trôi dạt tiếp nếu không áp lại hook).
    `threshold`: NGOÀI RA, tự phát hiện thêm bất kỳ dim nào khác đã hỏng
    (log_std > threshold) — phòng vệ, không giả định trước dim nào.
    """
    with torch.no_grad():
        log_std = model.policy.log_std
        freeze_mask = log_std.data > threshold  # dim thực sự hỏng — cần reset giá trị
        if always_freeze_idx:
            for i in always_freeze_idx:
                freeze_mask[i] = True  # chỉ khoá gradient, KHÔNG ép giá trị nếu đang hợp lý
        if not freeze_mask.any():
            return
        reset_mask = log_std.data > threshold  # chỉ reset dim THẬT SỰ vượt ngưỡng
        if reset_mask.any():
            bad_dims = reset_mask.nonzero(as_tuple=False).flatten().tolist()
            vals_before = log_std.data[reset_mask].tolist()
            log_std.data[reset_mask] = 0.0
            print(f"  ⚠️ log_std hỏng ở dim {bad_dims} (giá trị trước: {vals_before}) — reset về 0.0.")
        frozen_dims = freeze_mask.nonzero(as_tuple=False).flatten().tolist()
        print(f"  🔒 Đóng băng gradient log_std vĩnh viễn ở dim {frozen_dims} "
              f"(giá trị hiện tại: {log_std.data[freeze_mask].tolist()}) — "
              f"dim bị scripted override, không bao giờ nhận gradient thật.")

        def _freeze_grad(grad, mask=freeze_mask.clone()):
            grad = grad.clone()
            grad[mask] = 0.0
            return grad

        log_std.register_hook(_freeze_grad)


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
    env_cfg.sim.log_dir = os.path.join(args.log_dir, "sim_logs")
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
        # Khớp demo/eval — mimic gripper khép nhẹ hơn
        env_cfg.scene.robot.actuators["gripper"].stiffness = 550.0
        env_cfg.scene.robot.actuators["gripper"].damping = 35.0
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
        # Clamp [0,1]: với --resume, num_timesteps tiếp tục từ checkpoint nhưng
        # _total là budget MỚI → progress âm mạnh → _assist_blend_scale < 0 →
        # early-out `scale <= 1e-6` trong grasp_assist tắt sạch assist âm thầm.
        frac = min(max(self.num_timesteps / self._total, 0.0), 1.0)
        start = getattr(cfg, "grasp_assist_blend_start", 1.0)
        end = getattr(cfg, "grasp_assist_blend_end", 0.0)
        # Anneal về `end` trong anneal_frac đầu rồi GIỮ nguyên. Lịch cũ tuyến
        # tính suốt run nên assist vẫn khác 0 tới bước cuối → policy CHƯA BAO
        # GIỜ được train trên chính action nó thực thi, tức credit assignment
        # của PPO cho việc nhấc chưa bao giờ hợp lệ. Giữ phần cuối on-policy sạch.
        anneal_frac = float(getattr(cfg, "grasp_assist_anneal_frac", 0.4))
        w = start + (end - start) * min(frac / max(anneal_frac, 1e-6), 1.0)
        unwrapped._assist_blend_scale = w
        if self.num_timesteps % 200000 < (self.training_env.num_envs or 1):
            print(f"  [Assist] scale={w:.3f} @ {self.num_timesteps:,} steps", flush=True)
        return True


class TrainMetricsCallback(BaseCallback):
    """Log tỉ lệ thành công thật sự trong lúc train.

    Trước đây tín hiệu duy nhất là ``ep_rew_mean``, mà nó NGHỊCH BIẾN với thành
    công: episode thành công kết thúc sớm nên gom ít reward hơn episode timeout.
    Đường cong reward 255→785 hôm trước chính là agent học cách nằm lì lâu hơn.

    QUAN TRỌNG: phải TÍCH LUỸ theo từng env trong suốt episode. Env tự reset khi
    done, và ``reset_action_terms``/``reset_robot`` xoá ``_grasp_latched``,
    ``_steps_bottle_lifted``, ``_lift_phase``, ``_stage`` TRƯỚC khi callback chạy
    — đọc lúc đó chỉ ra toàn số 0, kể cả khi episode vừa thành công.

    KIỂM TRA TỈNH TÁO: train/success_rate và ep_rew_mean phải đi CÙNG chiều. Nếu
    reward tăng mà success_rate phẳng → reward redesign chưa ăn, dừng run.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._n = None
        self._done_succ: list[float] = []
        self._done_grasp: list[float] = []
        self._done_latch: list[float] = []
        self._done_lift: list[float] = []
        self._done_maxlift: list[float] = []

    def _reset_acc(self, n: int) -> None:
        import numpy as _np
        self._n = n
        self._ep_succ = _np.zeros(n, dtype=bool)
        self._ep_grasp = _np.zeros(n, dtype=bool)
        self._ep_latch = _np.zeros(n, dtype=bool)
        self._ep_lift = _np.zeros(n, dtype=bool)
        self._ep_maxlift = _np.full(n, -9.9)

    def _on_step(self) -> bool:
        import numpy as _np
        env = self.training_env
        u = env.unwrapped if hasattr(env, "unwrapped") else env.envs[0].unwrapped
        dones = self.locals.get("dones")
        if dones is None:
            return True
        n = len(dones)
        if self._n != n:
            self._reset_acc(n)

        hold_req = int(getattr(u.cfg, "grasp_lift_hold_steps", 5))
        stage = u._stage.detach().cpu().numpy()
        lifted_steps = u._steps_bottle_lifted.detach().cpu().numpy()
        phase = (
            u._lift_phase.detach().cpu().numpy()
            if hasattr(u, "_lift_phase") else _np.zeros(n, dtype=int)
        )
        t = u.action_manager._terms.get("gripper_action")
        latched = (
            t._grasp_latched.detach().cpu().numpy()
            if t is not None and hasattr(t, "_grasp_latched") else _np.zeros(n, dtype=bool)
        )
        ls = getattr(u, "_last_state", None)
        lift_m = (
            ls["bottle_lift"].detach().cpu().numpy() if ls is not None
            else _np.zeros(n, dtype=float)
        )

        # Tích luỹ cho các env CHƯA done (env đã done thì state là của episode mới)
        live = ~_np.asarray(dones, dtype=bool)
        self._ep_succ |= live & (lifted_steps >= hold_req)
        self._ep_grasp |= live & (stage >= 1)
        self._ep_latch |= live & latched
        self._ep_lift |= live & (phase != 0)
        self._ep_maxlift = _np.where(live, _np.maximum(self._ep_maxlift, lift_m), self._ep_maxlift)

        idx = _np.nonzero(dones)[0]
        for i in idx:
            self._done_succ.append(float(self._ep_succ[i]))
            self._done_grasp.append(float(self._ep_grasp[i]))
            self._done_latch.append(float(self._ep_latch[i]))
            self._done_lift.append(float(self._ep_lift[i]))
            self._done_maxlift.append(float(max(self._ep_maxlift[i], 0.0)))
            self._ep_succ[i] = False
            self._ep_grasp[i] = False
            self._ep_latch[i] = False
            self._ep_lift[i] = False
            self._ep_maxlift[i] = -9.9

        for buf in (self._done_succ, self._done_grasp, self._done_latch,
                    self._done_lift, self._done_maxlift):
            del buf[:-200]
        if self._done_succ:
            self.logger.record("train/success_rate", float(_np.mean(self._done_succ)))
            self.logger.record("train/grasp_rate", float(_np.mean(self._done_grasp)))
            self.logger.record("train/latch_rate", float(_np.mean(self._done_latch)))
            self.logger.record("train/lift_start_rate", float(_np.mean(self._done_lift)))
            self.logger.record("train/mean_max_lift_m", float(_np.mean(self._done_maxlift)))
            self.logger.record("train/episodes_seen", len(self._done_succ))
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
    callbacks = CallbackList([checkpoint_cb, TrainMetricsCallback()])
    if getattr(train_env.unwrapped.cfg, "grasp_assist_schedule_enabled", False):
        callbacks = CallbackList([checkpoint_cb, TrainMetricsCallback(), AssistScheduleCallback(args.timesteps)])
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
            learning_rate=linear_lr_schedule(args.lr_start, args.lr_end),
            n_steps=64,
            batch_size=4096,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
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
            # log_std dim gripper (index 6) đóng băng ~16.0 (std≈9tr) từ lịch sử
            # train cũ. ĐÃ THỬ 3 cách sửa-rồi-để-gradient-tự-do (clamp mỗi bước
            # 2 trần khác nhau, reset một lần không đóng băng) — CẢ 3 đều làm
            # latch/grasp tệ hơn không đụng gì (xem verify_exploit5b/5c/5d).
            #
            # NGUYÊN NHÂN GỐC (xác nhận bằng grep "_assist_blend_scale" trong
            # actions.py — KHÔNG XUẤT HIỆN LẦN NÀO): logic đóng/mở kẹp trong
            # actions.py luôn chạy kịch bản (scripted), độc lập hoàn toàn với
            # lịch assist — override raw action của gripper_action ở MỌI thời
            # điểm, kể cả sau khi assist_scale=0. Nghĩa là dim này KHÔNG BAO
            # GIỜ nhận gradient thật tương quan với reward, ở bất kỳ giai đoạn
            # nào — entropy bonus đẩy log_std của nó tăng vô hạn, không có
            # điểm dừng tự nhiên (xác nhận: std vẫn tăng chậm liên tục 1.35e6→
            # 1.40e6 xuyên suốt run 5M kể cả rất lâu sau handoff).
            #
            # Lý do 3 lần sửa trước thất bại: chỉ RESET giá trị nhưng để
            # gradient tiếp tục tự do — dim vừa "yên tĩnh" (gradient gần 0 do
            # std cũ khổng lồ làm mẫu số log-prob triệt tiêu) bị bơm gradient
            # mạnh trở lại đột ngột (Adam optimizer state cho param này bắt
            # đầu từ 0 nên phản ứng tức thời), gây nhiễu loạn shared trunk
            # (mean network dùng chung cho cả 7 dim) — đúng cơ chế "đấu tay
            # đôi với optimizer" đã ghi nhận.
            #
            # FIX ĐÚNG: reset giá trị VỀ MỘT LẦN (để không neo ở 16.0 vĩnh
            # viễn) NHƯNG đồng thời ĐÓNG BĂNG GRADIENT vĩnh viễn cho đúng dim
            # đó bằng backward hook trên chính param — không phải ghi đè
            # `.data` mỗi bước (cách cũ, đấu với Adam), mà chặn gradient TRƯỚC
            # khi tới optimizer, nên Adam không bao giờ "thấy" có gì để đấu.
            _freeze_scripted_gripper_log_std(model)

    # --resume dùng PPO.load() — KHÔNG đi qua nhánh --checkpoint ở trên nên
    # hook đóng băng không tự áp lại (hook là runtime-only, không nằm trong
    # state SB3 lưu/khôi phục). Dim gripper vẫn CHƯA BAO GIỜ nhận gradient
    # thật ở bất kỳ giai đoạn nào (cấu trúc actions.py không đổi), nên áp lại
    # vô điều kiện — không cần đợi phát hiện giá trị đã hỏng (resume thường
    # giá trị vẫn còn hợp lý từ lần đóng băng trước, nhưng SẼ trôi dạt tiếp
    # nếu không đóng băng lại).
    if args.resume and latest_ckpt:
        _freeze_scripted_gripper_log_std(model, always_freeze_idx=[6])

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
