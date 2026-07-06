# OpenArm RL — Pick & Place (Isaac Lab + SB3)

Pipeline manager-based RL cho robot OpenArm: **REACH → GRASP → LIFT** chai.

---

## Tổng quan workflow

| Mục tiêu | Lệnh chính |
|----------|------------|
| Deploy code lên server | `./deploy_training.sh [user@server]` |
| Train trên server (4090) | SSH + `isaaclab_train.py` (xem §2) |
| Tải model về máy local | `./fetch_model.sh [user@server] [run_name]` |
| Demo visual local | `./run_local.sh demo --model ./logs/active_policy.pt --phase 2` |
| Train backup local (4050) | `./run_local.sh train --phase 2 ...` (xem §3) |

**Biến môi trường (tùy chọn):**

```bash
export OPENARM_REMOTE_ROOT=/data21tb/users/huyhoang/openarm_train_ws
export ISAAC_SIM_PYTHON=/path/to/isaacsim/python.sh
```

---

## Làm từng công đoạn (khuyến nghị)

Dùng `--stage` để bật đúng phần gate/assist. **Demo và train dùng cùng file gate** → `isaaclab_openarm_env/phase2_overrides.py`.

| Công đoạn | `--stage` | Mục tiêu |
|-----------|-----------|----------|
| 1. REACH | `reach` | Tay tới chai, căn top↓ / z_f, vào GRASP |
| 2. GRASP | `grasp` | Hạ ngón, latch, đóng grip (`grip:1.0`), chưa nhấc |
| 3. LIFT | `all` hoặc `lift` | Nhấc chai Δz ≥ 3cm, giữ 5 bước |

### Demo từng công đoạn (local)

```bash
cd Reinforce_Learning

# Công đoạn 1 — chỉ REACH (phase 1, không cần --stage)
./run_local.sh demo --model ./logs/active_policy.pt --phase 1

# Công đoạn 2 — REACH + GRASP, chưa bật lift assist
./run_local.sh demo --model ./logs/active_policy.pt --phase 2 --stage grasp

# Công đoạn 3 — đủ pipeline (reach + grasp + lift)
./run_local.sh demo --model ./logs/active_policy.pt --phase 2 --stage all
# hoặc tương đương:
./run_local.sh demo --model ./logs/active_policy.pt --phase 2 --assist
```

**Log GRASP cần thấy:** `desc↓` → `dh:12/12L` → `grip:1.00` với `z_f < 0.014`  
**Log LIFT cần thấy:** `st:2/2` → `ph:2/2` → `lift↑` → `lift > +0.03m`

### Chỉnh gate (khi cần tune)

**Chỉ sửa một file:** `isaaclab_openarm_env/phase2_overrides.py`

| Dict | Khi nào chỉnh |
|------|----------------|
| `PHASE2_REACH` | Vào GRASP quá sớm / quá muộn (`reach_advance_max_z_finger`, `reach_advance_min_top_down`) |
| `PHASE2_GRASP` | Kẹp cổ chai, không đóng được (`grasp_descent_z_target`, `grasp_latch_max_z_finger`, `grasp_close_max_z_err`) |
| `PHASE2_LIFT` | Không nhấc / trượt khi nhấc (`grasp_lift_world_m`, `grasp_pre_lift_hold_steps`, `grasp_lift_settle_steps`) |

**Bug fix quan trọng (giữ nguyên):** `mdp/helpers.py` — `finger_descended_for_close` dùng `min(close_max_z, z_target)` để descent không dừng sớm.

Sau khi sửa gate → chạy lại demo cùng `--stage`, **chưa cần train** cho đến khi demo ổn.

---

## §1 — Server train + Local demo (workflow chính)

### Bước 1: Deploy code lên server

```bash
./deploy_training.sh naiscorp-4090
# hoặc: ./deploy_training.sh user@192.168.x.x /path/to/isaacsim/python.sh
```

### Bước 2: Train trên server (background)

```bash
# Phase 1 — REACH only
ssh naiscorp-4090 "nohup /data21tb/users/huyhoang/isaacsim/python.sh \
  /data21tb/users/huyhoang/openarm_train_ws/Reinforce_Learning/isaaclab_train.py \
  --headless --task_phase 1 --num-envs 1024 --timesteps 50000000 \
  --log_dir /data21tb/users/huyhoang/openarm_train_ws/logs_openarm/train_osc \
  > /data21tb/users/huyhoang/openarm_train_ws/logs_openarm/train_osc.log 2>&1 &"

# Phase 2 — reach + grasp + lift (assist giảm dần khi train)
ssh naiscorp-4090 "nohup /data21tb/users/huyhoang/isaacsim/python.sh \
  /data21tb/users/huyhoang/openarm_train_ws/Reinforce_Learning/isaaclab_train.py \
  --headless --task_phase 2 --assist-schedule --descent-assist \
  --stage all --num-envs 1024 --timesteps 50000000 \
  --log_dir /data21tb/users/huyhoang/openarm_train_ws/logs_openarm/train_osc_phase2 \
  > /data21tb/users/huyhoang/openarm_train_ws/logs_openarm/train_osc_phase2.log 2>&1 &"

# Fine-tune từ checkpoint .pt đã fetch
ssh naiscorp-4090 "nohup .../isaaclab_train.py \
  --headless --task_phase 2 --assist-schedule --descent-assist --stage all \
  --checkpoint /data21tb/users/huyhoang/openarm_train_ws/logs_openarm/train_osc_phase2/best_policy.pt \
  --log_dir .../logs_openarm/train_osc_phase2_ft ..."
```

**Theo dõi / dừng train:**

```bash
ssh naiscorp-4090 "tail -f /data21tb/users/huyhoang/openarm_train_ws/logs_openarm/train_osc_phase2.log"
ssh naiscorp-4090 "pgrep -u \$(whoami) -f 'isaaclab_train.py'"
ssh naiscorp-4090 "pgrep -u \$(whoami) -f 'isaaclab_train.py' | xargs -r kill"
```

**TensorBoard server:** `http://<server_ip>:6008` (tự khởi động khi deploy)

### Bước 3: Fetch model về local

```bash
./fetch_model.sh naiscorp-4090 train_osc              # phase 1
./fetch_model.sh naiscorp-4090 train_osc_phase2      # phase 2
./list_models.sh naiscorp-4090                         # xem run nào có checkpoint
```

Tạo symlink: `logs/active_policy.pt` → `best_policy_<run_name>.pt`

### Bước 4: Demo trên laptop (4050)

```bash
./run_local.sh demo --model ./logs/active_policy.pt --phase 2 --stage all
```

---

## §2 — Backup: Server offline → train & demo local

Khi server chưa online, dùng RTX 4050 local (ít env hơn, steps ít hơn).

```bash
# Phase 1 — REACH
./run_local.sh train --phase 1 --envs 16 --steps 2000000

# Phase 2 — full pipeline (assist schedule)
./run_local.sh train --phase 2 --stage all --assist-schedule \
  --envs 16 --steps 5000000 \
  --checkpoint ./logs/active_policy.pt   # fine-tune nếu đã có weights

# Demo ngay trên local
./run_local.sh demo --model ./logs/train/best_policy.pt --phase 2 --stage grasp
./run_local.sh demo --model ./logs/train/best_policy.pt --phase 2 --stage all
```

**Lưu ý local:** log mặc định `./logs/train/`. Server dùng `logs_openarm/<run_name>/`.

---

## §3 — Eval headless (tùy chọn)

```bash
$ISAAC_SIM_PYTHON eval_lift_metrics.py --headless \
  --model_path ./logs/active_policy.pt --episodes 5 --stage all
```

In một dòng `LIFT_METRICS {...}` — không cần GUI.

---

## Tham chiếu lệnh nhanh

### `run_local.sh train`

| Option | Mô tả | Default |
|--------|--------|---------|
| `--envs N` | Số env song song | 16 |
| `--steps N` | Tổng timesteps | 1_000_000 |
| `--phase 1\|2` | Task phase | (từ config) |
| `--stage reach\|grasp\|lift\|all` | Gate phase 2 | all |
| `--assist-schedule` | Assist 1.0→0.0 khi train | off |
| `--checkpoint path` | Fine-tune từ .pt | — |
| `--resume` | Tiếp checkpoint zip | off |
| `--gui` | Hiện Isaac Sim | headless |

### `run_local.sh demo`

| Option | Mô tả | Default |
|--------|--------|---------|
| `--model path` | File .pt | `./logs/active_policy.pt` |
| `--phase 1\|2` | Task phase | — |
| `--stage reach\|grasp\|lift\|all` | Gate phase 2 | all |
| `--assist` | = `--stage all` | — |

### `isaaclab_train.py` (server / direct)

Cùng options với `run_local.sh train`, thêm:

```bash
--headless --num-envs 1024 --timesteps 50000000 \
--log_dir /path/to/logs_openarm/my_run \
--descent-assist    # bật descent assist khi train phase 2
```

---

## Cấu trúc file quan trọng

```
Reinforce_Learning/
├── run_local.sh              # train + demo local
├── deploy_training.sh        # rsync → server
├── fetch_model.sh            # scp model ← server
├── list_models.sh            # inventory checkpoints
├── isaaclab_train.py         # training entry
├── isaaclab_demo.py          # visual playback
├── eval_lift_metrics.py      # headless eval (optional)
└── isaaclab_openarm_env/
    ├── config.py             # defaults
    ├── phase2_overrides.py   # ★ CHỈNH GATE Ở ĐÂY
    └── mdp/
        ├── helpers.py        # metrics, finger_descended_for_close
        ├── grasp_assist.py   # descent/lift assist logic
        ├── actions.py        # auto-close gripper
        └── rewards.py        # stage transitions, rewards
```

---

## Debug khi demo

Console in real-time khi `--phase 2`:

- `desc↓` — assist đang hạ ngón
- `dh:N/12L` — descend hold / latched
- `lift↑` — assist đang nhấc
- `lift_hold:N/5` — đếm bước success

Collision / contact events in `z_f`, `top↓`, TCP position.
