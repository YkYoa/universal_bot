# OpenArm RL — Pick & Place (Isaac Lab + SB3)

Pipeline: **REACH → GRASP → LIFT**

---

## Quick start

```bash
cd Reinforce_Learning
cp openarm.env.example openarm.env

./rl.sh deploy
./rl.sh train-fg test_run --num-envs 16 --timesteps 2000 --progress
./rl.sh fetch train_osc_phase2
./rl.sh demo --model ./logs/active_policy.pt --phase 2 --stage all
```

Chỉ **2 file shell**: `rl.sh` (mọi lệnh) + `server_resolve_env.sh` (chạy trên server).

---

## `./rl.sh` commands

| Lệnh | Mô tả |
|------|--------|
| `deploy` | Sync code + USD lên server, TensorBoard :6008 |
| `train <run> [args]` | Train background trên 4090 |
| `train-fg <run> [args]` | Train foreground (test) |
| `fetch [run]` | Tải model → `logs/active_policy.pt` |
| `list` | Inventory checkpoint |
| `cleanup [--dry-run]` | Dọn log server |
| `demo [args]` | Demo visual local |
| `local-train [args]` | Train backup laptop 4050 |

### Train phase 2

```bash
./rl.sh train train_osc_phase2 \
  --task_phase 2 --assist-schedule --descent-assist --stage all \
  --num-envs 1024 --timesteps 50000000
```

### Theo dõi / dừng train

```bash
ssh huyhoang-4090 "tail -f /data21tb/users/\$(whoami)/openarm_train_ws/logs_openarm/train_osc_phase2.log"
ssh huyhoang-4090 "pgrep -u \$(whoami) -f 'isaaclab_train.py' | xargs -r kill"
```

---

## Cấu trúc thư mục

**Local**
```
Reinforce_Learning/
├── rl.sh                  # ★ duy nhất cần nhớ
├── openarm.env            # config (copy từ .example)
├── server_resolve_env.sh  # sync lên server (tự gọi)
├── logs/active_policy.pt
└── isaaclab_openarm_env/phase2_overrides.py  # ★ chỉnh gate
```

**Server** (`/data21tb/users/huyhoang/openarm_train_ws/`)
```
Reinforce_Learning/    logs_openarm/<run>/    .cache/isaaclab_tmp/
```

---

## Config `openarm.env`

```bash
export OPENARM_SERVER=huyhoang-4090
export ISAAC_SIM_PYTHON=/home/hans/isaacsim/.../python.sh
```

---

## Phase 2 từng công đoạn

```bash
./rl.sh demo --model ./logs/active_policy.pt --phase 2 --stage grasp
./rl.sh demo --model ./logs/active_policy.pt --phase 2 --stage all
```

Chỉnh gate: `isaaclab_openarm_env/phase2_overrides.py`

---

## Local train (4050 backup)

```bash
./rl.sh local-train --phase 2 --stage all --assist-schedule --envs 16 --steps 5000000
```

Log local: `./logs/train/` — server: `logs_openarm/<run>/`.
