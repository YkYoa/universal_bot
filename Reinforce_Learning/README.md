# OpenArm RL Pick-and-Place: Commands Guide

This workspace provides a manager-based RL pipeline using **Isaac Lab** and **Stable-Baselines3** for the Apple Pick-and-Place task.

---

## 🚀 Quick Command Overview

| Goal | Command | Description |
| :--- | :--- | :--- |
| **Deploy** | `./deploy_training.sh [server_ip]` | Sync code & robot assets to high-end GPU server (RTX 4090). |
| **Fetch** | `./fetch_model.sh [server_ip]` | Download trained models, checkpoints, & TensorBoard logs locally. |
| **Train (Local)** | `./run_local.sh train` | Run training locally optimized for laptop GPU (RTX 4050). |
| **Demo (Local)** | `./run_local.sh demo` | Playback & visualize a trained policy locally. |

---

## 💻 1. Local Training & Visual Playback

Use `./run_local.sh` to run training or visualization locally (e.g., on your RTX 4050 laptop).

### Train Locally
Runs training in headless mode by default.
```bash
./run_local.sh train --envs 16 --steps 1000000
```
* **Local Script Options (`./run_local.sh train`):**
  * `--envs <num>` : Number of parallel environments (default: `16`).
  * `--steps <num>` : Total training timesteps (default: `1000000`).
  * `--gui` : Make the Isaac Sim window visible during training (runs headless by default).
  * `--resume` : Resume from the latest local checkpoint.
  * `--progress` : Enable tqdm progress bar in terminal.

---

### Direct Python Training Options (`isaaclab_train.py`)
If you are running the python script directly using Isaac Sim's python executable, you can use these arguments:
```bash
/path/to/isaacsim/python.sh isaaclab_train.py --num_envs 1024 --timesteps 1000000 --headless
```
* **Arguments:**
  * `--num_envs` / `--num-envs` : Number of parallel environments (default: `1024`).
  * `--timesteps` : Total training timesteps (default: `1_000_000`).
  * `--resume` : Resume training from the latest checkpoint.
  * `--progress` : Enable tqdm progress bar.
  * `--headless` : Run without opening the local GUI window.
  * `--log_dir` : Output directory for policies and checkpoints.

### Run Demo (Visual Playback)
Visualizes your trained model in the local Isaac Sim viewport:
* **For locally trained models**:
  ```bash
  ./run_local.sh demo --model ./logs/train/best_policy.pt
  ```
* **For fetched remote server models**:
  ```bash
  ./run_local.sh demo --model ./logs/best_policy.pt
  ```

---

## 🖥️ 2. Remote Server Training (RTX 4090)

For high-performance training, deploy to a remote server and run headless.

### Deploy Training to Server
Compresses and synchronizes the training codebase and robot URDF/USD assets to your server. It also automatically launches a remote TensorBoard server on port `6008`.
```bash
./deploy_training.sh user@server_ip
```

### Start Remote Training
After deploying, connect to your server to start training in the background (headless). 
> [!NOTE]
> Do NOT use the `--progress` flag for background runs, as it will suppress the text logging tables and buffer stdout.

* **Start training from scratch**:
  ```bash
  ssh user@server_ip "nohup /data21tb/huyhoang/isaacsim/python.sh /data21tb/huyhoang/openarm_train_ws/Reinforce_Learning/isaaclab_train.py --headless --num-envs 1024 --timesteps 50000000 --log_dir /data21tb/huyhoang/openarm_train_ws/logs_openarm/train > /data21tb/huyhoang/openarm_train_ws/logs_openarm/train.log 2>&1 &"
  ```
* **Resume training from the latest checkpoint**:
  ```bash
  ssh user@server_ip "nohup /data21tb/huyhoang/isaacsim/python.sh /data21tb/huyhoang/openarm_train_ws/Reinforce_Learning/isaaclab_train.py --headless --resume --num-envs 1024 --timesteps 100000000 --log_dir /data21tb/huyhoang/openarm_train_ws/logs_openarm/train > /data21tb/huyhoang/openarm_train_ws/logs_openarm/train.log 2>&1 &"
  ```

* **See if training is running**:
  ```bash
  ssh user@server_ip "ps aux | grep isaaclab_train.py"
  ```
* **Monitor logs**:
  ```bash
  ssh user@server_ip "tail -f /data21tb/huyhoang/openarm_train_ws/logs_openarm/train.log"
  ```
* **Stop training**:
  ```bash
  ssh user@server_ip "pgrep -u \$(whoami) -f 'isaaclab_train.py' | grep -v '\$\$' | xargs -r kill -9"
  ```

---

## 📥 3. Fetch Results to Local Machine

Once training is complete (or to check intermediate progress), download the policy weights, logs, and config back to your laptop:

```bash
./fetch_model.sh user@server_ip
```
* Downloads `best_policy.pt`, `final_policy.pt`, `env_cfg.pkl`, checkpoints, and TensorBoard logs to `./logs/`.
* Offers to launch **TensorBoard locally** on `http://localhost:6007` automatically.

---

## 🔬 Debugging RL Values & Collisions

During local playback (`./run_local.sh demo`), the console is configured to be silent during smooth flight but will output real-time **Critic/Q-Value Estimates** and the **exact 3D coordinates `[X, Y, Z]` of the Tool Center Point (TCP)** whenever transition or collision events happen:
* **Table Collisions**: Triggers when the tool-center point contacts/penetrates the table.
* **Object Contact**: Triggers when the gripper reaches grasping range with the apple/bottle.
* **Object Knocked Over**: Triggers if the bottle tilts $>15^\circ$.
