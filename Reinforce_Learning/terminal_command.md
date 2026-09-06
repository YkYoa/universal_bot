# Terminal Commands Log

Lịch sử các lệnh quan trọng đã dùng để setup/fix Isaac Sim + Isaac Lab + Reinforce_Learning trên laptop (RTX 4050 6GB).

---

## Setup môi trường

Isaac Sim build từ source (repo `~/isaacsim`, version 6.0.1):
```bash
cd ~/isaacsim
./build.sh
```

Chạy Isaac Sim (GUI):
```bash
cd ~/isaacsim/_build/linux-x86_64/release
./isaac-sim.sh
```

Isaac Lab (repo `~/IsaacLab`, venv `env_isaaclab`, Python 3.12):
```bash
cd ~/IsaacLab
uv venv --python 3.12 --seed env_isaaclab
source env_isaaclab/bin/activate
uv pip install --upgrade pip
./isaaclab.sh -i 'newton,rl[rsl-rl],visualizer[newton]'
./isaaclab.sh -i 'rl[sb3]'   # cần thêm cho Reinforce_Learning (dùng Stable-Baselines3)
```

---

## Chạy Reinforce_Learning (test/train/demo)

```bash
cd ~/universal_bot/Reinforce_Learning
cp openarm.env.example openarm.env   # chỉ cần lần đầu
```

Test nhanh training (4 env, 200 steps, có progress bar):
```bash
./rl.sh local-train --envs 4 --steps 200 --progress
```

Train foreground/background dài hơn:
```bash
./rl.sh local-train --envs 16 --steps 2000 --progress
```

Demo có GUI (dùng `--visualizer kit` tự động):
```bash
./rl.sh demo --model ./logs/active_policy.pt --phase 2 --stage all
```

Deploy + train trên server 4090:
```bash
./rl.sh deploy
./rl.sh train train_osc_phase2 --task_phase 2 --assist-schedule --descent-assist --stage all --num-envs 1024 --timesteps 50000000
./rl.sh fetch train_osc_phase2
```

---

## Test import thủ công (debug, không qua rl.sh)

```bash
export PYTHONEXE=/home/hans/IsaacLab/env_isaaclab/bin/python
export PYTHONPATH="$HOME/universal_bot/Reinforce_Learning"
/home/hans/isaacsim/_build/linux-x86_64/release/python.sh -c "
from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(['--visualizer', 'none'])
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
from isaaclab_openarm_env.env import ApplePickPlaceEnv
print('OK')
simulation_app.close()
"
```

---

## Port sang Isaac Lab 3.0 (Newton) — các lỗi đã fix

Sau khi chuyển sang nhánh develop của IsaacLab (Newton experimental), gặp 3 lớp lỗi, đã fix hết:

1. **`AppLauncher` bỏ `--headless`** → dùng `--visualizer none` (headless) / `--visualizer kit` (GUI). Đã sửa trong `rl.sh`, `patch_robot_usd.py`, `patch_qvic_usd.py`.
2. **`isaaclab.utils` không re-export `configclass`** → phải `from isaaclab.utils.configclass import configclass` (thay vì `from isaaclab.utils import configclass`). Đã sửa trong `mdp/actions.py`, `config.py`.
3. **Quaternion đổi quy ước `(w,x,y,z)` → `(x,y,z,w)`, và `.data.*` giờ trả về `ProxyArray` thay vì `torch.Tensor` trực tiếp**:
   - `config.py`: `rot=(1.0,0,0,0)` (wxyz identity) → `rot=(0,0,0,1.0)` (xyzw identity) cho Bottle/Bowl.
   - `helpers.py`: các chỗ lấy `.data.root_pos_w` / `.data.root_quat_w` / `.data.root_lin_vel_w` / `.data.root_ang_vel_w` **không đi kèm index `[...]`** phải thêm `.torch` tường minh (vd `env._bottle.data.root_quat_w.torch`). Các chỗ CÓ index ngay (`.data.body_quat_w[:, id, :]`) thì an toàn, không cần sửa (indexing tự trả về torch.Tensor qua deprecation bridge của `ProxyArray`).
   - Công cụ hữu ích: `python3 ~/IsaacLab/scripts/tools/find_quaternions.py --path <dir> --show-all` để dò quaternion literal cần đổi convention.

Test xác nhận chạy full training loop OK (env build → reset → step → PPO/SB3 → save checkpoint), không lỗi:
```bash
./rl.sh local-train --envs 4 --steps 200 --progress
```

---

## Server 4090 (huyhoang-4090)

SSH alias thêm vào `~/.ssh/config` (key ed25519 đã được server chấp nhận sẵn, không cần password):
```
Host huyhoang-4090
    HostName 192.168.1.122
    User huyhoang
```

Xem inventory / fetch policy đã train trên server:
```bash
./rl.sh list
./rl.sh fetch train_osc_phase2          # ~99.9M steps — gc:74%, exhaust:Y, không lift được (xem mục debug bên dưới)
./rl.sh fetch train_osc_phase2_assist9  # ~79.9M steps — đang so sánh xem có lift ổn hơn không
```

---

## Debug: grasp OK nhưng không lift được (2026-08-25)

Triệu chứng: robot REACH + GRASP tốt (dist~0.024, sát chai), nhưng `grip:0.47` (đóng kẹp chỉ 47%), `gc:74%` (grasp closure không đạt 100%), `exhaust:Y` (hết lượt retry đóng kẹp trong `actions.py`), `tilt` tăng dần 11°→15° trong khi `lift` gần như 0m → chai bị nghiêng/trượt thay vì được nhấc thẳng lên. Ngoài ra 2 ngón gripper (pad trái/phải của CÙNG gripper tay trái — `openarm_left_finger_joint1` driven + `openarm_left_finger_joint2` mimic follower) đóng lệch nhau khi có tải (kẹp chai), dù đóng cân đối khi test không tải.

Test cô lập cơ chế mimic joint (không qua policy, không có vật cản):
```bash
export PYTHONEXE=/home/hans/IsaacLab/env_isaaclab/bin/python
export PYTHONPATH="$HOME/universal_bot/Reinforce_Learning"
/home/hans/isaacsim/_build/linux-x86_64/release/python.sh -c "
from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(['--visualizer', 'none'])
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
import torch
from isaaclab_openarm_env.env import ApplePickPlaceEnv
from isaaclab_openarm_env.config import ApplePickPlaceEnvCfg
cfg = ApplePickPlaceEnvCfg(); cfg.scene.num_envs = 1; cfg.sim.render_interval = cfg.decimation
env = ApplePickPlaceEnv(cfg=cfg); env.reset()
gripper_ids, names = env._robot.find_joints(['openarm_left_finger_joint1', 'openarm_left_finger_joint2'])
action = torch.zeros(env.num_envs, 7, device=env.device); action[:, -1] = -1.0
for i in range(60):
    env.step(action)
    jp = env._robot.data.joint_pos[:, gripper_ids]
    if i % 10 == 0:
        print(f'step {i}: joint1={jp[0,0].item():.4f} joint2={jp[0,1].item():.4f}')
env.close(); simulation_app.close()
"
```
Kết quả: joint1/joint2 hội tụ về 0 đồng bộ (không tải) → **mimic constraint hoạt động đúng ở tầng vật lý cơ bản, không phải bug do port sang Isaac Sim 6.0.1**. Vấn đề chỉ xuất hiện khi có lực tiếp xúc thật (kẹp chai) — nghi ngờ do policy chưa học đủ tốt, hoặc contact/friction resolution khác giữa PhysX lúc train trên server vs PhysX Isaac Sim 6.0.1 local.

**So sánh checkpoint (2026-08-25):**
- `train_osc_phase2` (~99.9M steps): grip kẹt ở 0.47, `gc:74%`, `exhaust:Y` (hết lượt retry đóng kẹp), critic value âm liên tục, tilt tăng dần 11°→15° (chai trượt/nghiêng), không lift được.
- `train_osc_phase2_assist9` (~79.9M steps): **tốt hơn hẳn** — grip tăng mượt lên 0.60-0.64, đạt state `[Success] gripped — need lift`, critic value dương cao (50-57), vào giai đoạn `lift↑`. NHƯNG qua nhiều episode lặp lại vẫn toàn bộ `TIMEOUT`, `lift` không bao giờ vượt quá 0 (luôn âm ~-0.011 đến -0.017m) — **grasp tốt nhưng vẫn chưa từng lift thành công**.

Kết luận: không còn nghi ngờ physics/port nữa (mimic joint xác nhận OK, `assist9` cho thấy grip có thể tốt). Vấn đề giờ nằm ở **policy chưa học xong giai đoạn LIFT** (dù grasp tốt) — đây là vấn đề training/RL tuning, không phải bug code cần fix từ việc port Isaac Sim.

**⚠️ Bug đã fix**: lúc đầu sửa `rl.sh` dùng `--visualizer none` cho CẢ local lẫn remote (`cmd_train`) — làm training job đầu tiên trên server crash ngay (`unrecognized arguments: --visualizer none`) vì server chạy IsaacLab v2.3.2 cũ vẫn cần `--headless`. Đã revert `cmd_train` (remote) về `--headless`; chỉ `cmd_local_train`/`cmd_demo` (chạy trên máy local, IsaacLab develop) mới dùng `--visualizer`. Luôn kiểm tra `pgrep -u huyhoang -af 'isaaclab_train.py'` trên server sau khi launch để chắc chắn process thật sự sống, không chỉ tin log "PID: xxx" của `rl.sh train`.

**Train tiếp trên server (2026-08-25)** — resume từ `assist9`, KHÔNG chạy `./rl.sh deploy` trước (server dùng IsaacLab v2.3.2 ổn định, code local đã sửa chỉ đúng cho nhánh `develop` — deploy sẽ làm hỏng server):
```bash
./rl.sh train train_osc_phase2_assist10 \
  --task_phase 2 --assist-schedule --descent-assist --stage all \
  --num-envs 512 --timesteps 20000000 \
  --checkpoint /data21tb/users/huyhoang/openarm_train_ws/logs_openarm/train_osc_phase2_assist9/best_policy.pt
```
Theo dõi log:
```bash
ssh huyhoang-4090 "tail -f /data21tb/users/huyhoang/openarm_train_ws/logs_openarm/train_osc_phase2_assist10.log"
```
Dừng nếu cần (đôi khi tiến trình con `kit/python/bin/python3` không chết theo parent, phải kill riêng):
```bash
ssh huyhoang-4090 "pgrep -u huyhoang -f 'isaaclab_train.py' | xargs -r kill"
sleep 3
ssh huyhoang-4090 "pgrep -u huyhoang -af 'isaaclab_train.py'"   # nếu còn dòng nào thì kill -9 PID đó
```

**Đã dừng (2026-08-25 14:xx)** — teammate cần fix server, đã kill `train_osc_phase2_assist10` giữa chừng (mới chạy ~vài phút). GPU đã giải phóng về baseline (~12.4GB, chỉ còn job zipformer_asr không liên quan). Kế hoạch nâng cấp Isaac Sim/Isaac Lab server (sync với local) tạm hoãn — chờ server ổn định trở lại.

**⚠️ RAID degraded trên server (2026-08-25 15:xx)** — `train_osc_phase2_assist11` bị kẹt ngay từ đầu (log 0 byte, process ở trạng thái `D`/disk-sleep nhiều phút, CPU chỉ ~3-4%). Chẩn đoán: `cat /proc/mdstat` cho thấy `md21` (chứa `/data21tb`) chạy **degraded — chỉ 1/2 ổ đĩa hoạt động** (`[2/1] [U_]`), `iostat -x` xác nhận `md21`/`sda` bão hòa 100% util, w_await ~145ms. Rất có thể đây chính là vấn đề teammate đang fix, chưa xong hẳn dù server đã reboot. Đã kill job (`pgrep -u huyhoang -f isaaclab_train.py | xargs -r kill`), KHÔNG tự sửa RAID (việc hạ tầng, để teammate xử lý). Checkpoint 2M-step (`rl_model_2000000_steps.zip`) đã được scp lên server tại `logs_openarm/train_osc_phase2_assist11/checkpoints/` — vẫn dùng được để resume sau khi RAID ổn định trở lại (dùng lại đúng lệnh `--resume` bên dưới).

Lệnh check RAID nhanh:
```bash
ssh huyhoang-4090 "cat /proc/mdstat; iostat -x 1 2 | grep -E 'md21|sda'"
```

**Train tạm trên laptop local (RTX 4050 6GB) trong lúc chờ server** — resume từ `assist9`, envs thấp để an toàn VRAM:
```bash
cd ~/universal_bot/Reinforce_Learning
./rl.sh local-train \
  --envs 16 --steps 5000000 --progress \
  --phase 2 --stage all --assist-schedule \
  --checkpoint ./logs/best_policy_train_osc_phase2_assist9.pt
```
VRAM thực tế dùng ~2.5GB/6.1GB với 16 envs — an toàn, có thể tăng envs nếu muốn nhanh hơn nhưng theo dõi `nvidia-smi` tránh OOM.

**Resume sau khi tạm dừng** (server bị RAID degraded, quay lại train local) — dùng `--resume` (tự tìm checkpoint mới nhất trong `logs/train/checkpoints/`, giữ nguyên optimizer state, chính xác hơn `--checkpoint`):
```bash
./rl.sh local-train --envs 16 --steps 5000000 --progress --resume --phase 2 --stage all --assist-schedule
```

Chạy nền + theo dõi:
```bash
tail -f ~/local_train_assist10.log
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

**✅ Hoàn tất (2026-08-26 01:40)** — đủ 5,000,000/5,000,000 bước (2M assist9 gốc + 3M train thêm local), không lỗi. `ep_rew_mean` tăng từ ~255 → **785**, `explained_variance:0.832`. Kết quả: `./logs/train/best_policy.pt`, `./logs/train/ppo_openarm_pick_place.zip`. Cần test demo xem có lift thành công không (mục tiêu ban đầu của đợt train thêm này).

**Lưu ý version**: `~/IsaacLab` local đang ở nhánh `develop` (bleeding-edge, Newton experimental) — KHÔNG phải bản ổn định. Server 4090 dùng `v2.3.2` (ổn định). Đây là lý do cần các fix API ở mục trên. Không deploy code local lên server để tránh xung đột version.

---

## Ghi chú quan trọng

- **`ISAAC_SIM_PYTHON`** (trong `openarm.env`) trỏ vào `~/isaacsim/_build/linux-x86_64/release/python.sh` (source build 6.0.1), **không phải** standalone binary 5.1.0 cũ (đã không còn tồn tại).
- **`PYTHONEXE`** (trong `openarm.env`) trỏ vào `~/IsaacLab/env_isaaclab/bin/python` — `python.sh` của Isaac Sim đọc biến này để dùng venv có `isaaclab`/`isaaclab_rl`/`newton`/`stable-baselines3` thay vì python nội bộ của Kit.
- Isaac Lab hiện tại ở nhánh **develop/main** (bleeding-edge, có Newton experimental) — **API `AppLauncher` đã đổi**: `--headless` bị bỏ, thay bằng `--visualizer none` (headless) / `--visualizer kit` (có GUI). Nếu sau này pull code mới từ IsaacLab mà lỗi lạ liên quan `AppLauncher`/`configclass`, kiểm tra lại các thay đổi này trước.
- `isaaclab.utils` không còn re-export `configclass` — phải import từ submodule: `from isaaclab.utils.configclass import configclass`.

---

## Sửa pipeline LIFT (2026-08-27, nhánh `fix/lift-pipeline`)

Chẩn đoán bằng 4 agent đọc mã nguồn: robot grasp tốt nhưng **chưa bao giờ nhấc được chai**, train >5M bước vô ích. Kế hoạch đầy đủ: `~/.claude/plans/expressive-honking-possum.md`.

### Công cụ đo (bắt buộc dùng thay vì nhìn demo bằng mắt)

`eval_lift_metrics.py` đã được nâng cấp: đa env, nhiễu vị trí chai, `--assist-scale` để tách "script nhấc" khỏi "policy nhấc".

```bash
cd ~/universal_bot/Reinforce_Learning
export PYTHONEXE=/home/hans/IsaacLab/env_isaaclab/bin/python
export PYTHONPATH="$(pwd)"
/home/hans/isaacsim/_build/linux-x86_64/release/python.sh ./eval_lift_metrics.py \
  --model-path ./logs/train/best_policy.pt \
  --episodes 24 --num-envs 8 --bottle-noise 0.05 --assist-scale 1.0 --stage all --seed 0
```
Luôn eval ở `--assist-scale 1.0 / 0.5 / 0.0`. Khoảng cách giữa 1.0 và 0.0 chính là con số cho biết policy học được bao nhiêu.

**Baseline trước khi sửa** (24 ep): `success 0.00 | grasp 0.92 | latch 0.46 | lift_start 0.46 | lift_cmd_steps 178`

### Đo trực tiếp bằng script chẩn đoán

Khi cần biết "tay có đi lên không, hay tay lên mà chai tuột", log `ee_z` / `bottle_z` / action trong lúc RISING — script mẫu ở `scratchpad/liftdiag.py`. Đây là cách tìm ra 2 nguyên nhân thật (xem dưới), mà đọc code thuần không thấy được.

Đo khối lượng chai / lực OSC tối đa:
```python
m = env._bottle.root_physx_view.get_masses()   # wp.array → dùng wp.to_torch(m).sum()
# lực nhấc = osc_stiffness * grasp_lift_world_m  (pose_rel: sai số KHÔNG tích luỹ)
```

### Nguyên nhân thật (khác với suy đoán ban đầu)

1. **Mốc `bottle_lift` lệch 12mm** — `_bottle_rest_z` lấy từ spawn z=0.638 ghi *trước* khi physics chạy; chai rơi xuống mặt bàn 0.626. Đã sửa: đo lại z thật trong 15 bước đầu episode (`_update_bottle_rest_baseline` trong `rewards.py`). Log xác nhận: `[RestZ] chai nằm ở 0.6259, spawn 0.6380, rơi 12.1mm`.
2. **Lực nhấc quá yếu → tay bò 2.4mm/s.** `pose_rel` cộng delta vào pose *hiện tại* mỗi bước nên sai số không tích luỹ → lực = `stiffness × delta` = `90 × 0.018` = 1.62N. Chai chỉ 0.0955kg (0.937N) nên vẫn nhấc được, nhưng phần dư 0.68N bị damping ăn hết. Tăng `grasp_lift_world_m` 0.018 → 0.055 làm nhanh **17 lần**.
3. **Gripper tự mở giữa không trung.** Khi nhấc, chai đung đưa → khoảng cách 2 pad lệch → `severe_asym` → đường reopen mở kẹp → mất latch → hủy nhấc. Đã sửa: cấm reopen khi `_lift_phase != IDLE` (`bad_close` và `mid_ramp` trong `actions.py`).
4. **`_grip_close_done()` bất khả thi** — đòi gc ≥ 0.96 nhưng ramp chặn cứng ở 0.75. Đã sửa: lấy `min(done_progress, cap)`.
5. **Cơ chế gain thích ứng `follow` chết** — `_update_lift_slip` ghi đè `_prev_lift_bottle` trước khi `_osc_world_up_lift` đọc → luôn kẹt ở sàn 0.45. Đã xoá, dùng magnitude hằng số.
6. **State machine nhấc mới** IDLE→RISING→HOLDING (`grasp_assist.py`). Điểm mấu chốt: RISING **không** đánh giá lại điều kiện khởi động (logic cũ reset cứng bộ đếm khi lỗi 1 bước). HOLDING đóng băng tay để chai giảm tốc — `grasp_lift_success_ready` đòi `‖vel‖ < 0.35` trong 5 bước liên tiếp.

### Lưu ý

- Đang làm trên nhánh `fix/lift-pipeline` (tách khỏi `master`).
- Regression gate sau **mỗi** lần sửa: `grasp_rate ≥ 0.90` và `latch_rate ≥ 0.85` @ scale 1.0. Vi phạm → revert ngay.
- Server 4090 `192.168.1.122` hiện **No route to host** (mất kết nối, không chỉ RAID degraded như hôm trước).

### Cập nhật 2026-08-27 (chiều) — đo trực tiếp lật ngược nhiều kết luận

Chạy `scratchpad/liftdiag.py` log `ee_z`/`bottle_z`/`dL`/`dR`/`span` trong lúc RISING. Kết quả:

1. **State machine mới hoạt động đúng**: RISING duy trì 400+ bước, không hủy oan. Cánh tay đi lên mượt 0.81 → 1.22m (40cm).
2. **Nhưng chai bị bỏ lại trên bàn** (`bottle_z` đứng im 0.6259 suốt). Kết luận trước đó ("chai bám theo tay tốt") là **sai** — lúc đó tay chỉ nhích 0.8mm nên không phân biệt được.
3. **`finger_span_xy` là chỉ số SAI để đánh giá "có đang giữ chai không"** — nó được SUY RA từ góc khớp. Khi khớp bị chai chặn lại, span vẫn báo 0.0583 (rộng hơn chai 0.0433) dù kẹp đang ép 18.9N. Đo được: khớp kẹt ở 0.0231 trong khi lệnh 0.0105.
   → Thêm `_grip_pressing()` trong `grasp_assist.py`: dùng **độ chênh khớp-so-với-lệnh** (`joint - 0.044*(1-close_progress)`). Kẹp không khí ~0.0015m, kẹp chai ~0.0126m → ngưỡng 0.005 tách sạch. Config: `grasp_press_min_stall_m`, `grasp_press_max_dist_f`.
   → Đã thử siết `grasp_phys_close_max_span` xuống 0.040 (dưới đường kính chai) nhưng **sai hướng**: lift không bao giờ arm nữa vì span không bao giờ xuống dưới 0.040 khi đang ép chai. Đã revert.
4. **Lỗi tự gây ra ở S1.2**: sửa `min_gc` nhận biết cap đã hồi sinh đường reopen vốn chết → gripper tự mở giữa không trung (`span` 0.058 → 0.083, `grip` 0.47 → 0.20, abort `unlatched`). Sửa đúng bản chất: cấm reopen khi `_lift_phase != IDLE`.
5. **Vấn đề còn lại: kẹp lệch tâm.** `dL=0.035 dR=0.029` lúc bắt đầu, tới rise#20 thành `dL=0.055 dR=0.0155` — pad trái rời xa, pad phải ép vào → một pad đẩy chai trượt đi thay vì hai pad kẹp. Chai lên được tối đa ~14mm rồi tuột.
   Nghi ngờ: mô hình hình học ngón trong `helpers.py` (`finger_tip_hand_y_closed_m=0.006`) lệch ~7mm so với collision mesh thật trong USD. Cần hiệu chuẩn lại từ USD.

**Trạng thái**: cơ chế nhấc đã thông (state machine + lực + không còn tự mở kẹp), nhưng hình học kẹp chưa đúng nên chưa nhấc thành công. Đây là việc tiếp theo.

### Bài học về đo lường (2026-08-27)

**Chỉ số eval rất nhiễu — đừng kết luận từ ít episode.** Cùng một code, chỉ đổi seed:

| seed | grasp | latch | lift_start | success |
|---|---|---|---|---|
| 0 | 0.77 | 0.46 | 0.46 | 0.00 |
| 1 | 0.67 | 0.38 | 0.38 | 0.00 |

Chênh 10pp ở 48 episode. Ở 24 episode còn tệ hơn (baseline cho 0.92, các lượt sau 0.77-0.79 — tôi đã tưởng là hồi quy, thực ra chỉ là nhiễu). **Quy tắc: mọi quyết định phải dựa trên ≥48 episode và ít nhất 2 seed.**

Chỉ số ỔN ĐỊNH duy nhất: `success_rate = 0.00` qua mọi lần đo.

**Không đo được baseline "thuần gốc"**: code trước khi port có `from isaaclab.utils import configclass`, không chạy nổi trên Isaac Lab 3.0. Muốn so sánh phải giữ port fix và chỉ revert phần lift.

Cách tạm stash để đo (nhớ `git stash pop` sau):
```bash
git stash push -m "WIP" -- Reinforce_Learning/isaaclab_openarm_env Reinforce_Learning/eval_lift_metrics.py
git checkout stash@{0} -- Reinforce_Learning/eval_lift_metrics.py   # giữ script đo mới
# ... chạy eval ...
git stash pop
```

### Phát hiện lớn: policy hiện tại grasp rất kém

`latch_rate` chỉ **0.38-0.46** — hơn nửa số episode gripper không bao giờ latch được. Đây là vấn đề ở CHÍNH POLICY, không phải ở assist. Policy này được train trong môi trường mà lift bất khả thi và reward trả công cho việc đứng yên ôm chai, nên hành vi của nó đã thích nghi với môi trường hỏng đó. Tinh chỉnh assist thêm nữa khó cứu được một policy vốn không đặt kẹp đúng chỗ.

---

## Giai đoạn 2 — sửa reward + train lại (2026-08-27)

Lý do chuyển sang GĐ2 dù GĐ1 chưa nhấc được: cơ chế nhấc đã thông về vật lý (tay lên mượt 40cm), nhưng `latch_rate` chỉ 0.38-0.46 → vấn đề nằm ở **chính policy**, vốn được train trong môi trường mà nhấc bất khả thi và reward trả công cho việc đứng yên gấp 5.8 lần thành công.

### Thay đổi

1. **Terminal bonus + phạt lật chai** (`rewards.py`: `terminal_success_bonus`, `terminal_tipped_penalty`; đăng ký thành RewardTermCfg riêng trong `config.py` để thấy tổng per-term trong TensorBoard).
   - `success` đăng ký KHÔNG có `time_out=True` → SB3 coi là termination thật → `V(s_cuối)=0`. Trước đây không có bonus nào nên hoàn thành nhiệm vụ tự xoá sạch giá trị tương lai.
   - Phạt lật chai **bắt buộc**: khi camping có giá trị âm mà `V(lật)=0` thì cố tình gạt đổ chai thành hành động tốt nhất.
   - Chia `step_dt` trong hàm để RewardManager nhân lại → config đọc thẳng là "cộng bấy nhiêu vào return".
2. **Camp decay** — shaping tĩnh (`r_align + r_descend + r_hover + r_grip_close + r_grip_cmd + r_track + r_orient`) suy giảm về 0 trong 60 bước SAU LATCH, cộng phạt thời gian 6.0 raw/bước. Shaping TRƯỚC latch giữ nguyên từng bit → không đụng hành vi hạ/khép đang chạy tốt.
3. **Residual control** thay vì ghi đè (`grasp_assist.py`) — PPO lưu log-prob của action đã sample rồi gán advantage cho nó; ghi đè → gradient bằng 0, blend lồi → gradient lệch. Giờ cộng bias vào trục Z và giảm chấn các trục khác theo `w`. `w=1` đi đúng đường ghi đè cũ (GĐ1 bất biến, kiểm chứng được).
4. **Lịch assist** — clamp `[0,1]` (với `--resume`, `progress` cũ có thể âm → `_assist_blend_scale < 0` → early-out tắt sạch assist âm thầm) và anneal về 0 trong 40% đầu rồi giữ, để **60% cuối on-policy sạch**. Lịch cũ giữ assist khác 0 tới bước cuối nên credit assignment cho việc nhấc chưa bao giờ hợp lệ.
5. **`TrainMetricsCallback`** — log `train/success_rate`, `grasp_rate`, `latch_rate`, `lift_start_rate`, `mean_lift_m`.
6. **Flags fine-tune** `--lr-start/--lr-end/--clip-range/--ent-coef` (reward đổi thang ~10× nên LR mặc định 3e-4 sẽ phá reach/grasp trong vài update đầu).

### Kinh tế học phần thưởng sau khi sửa (γ=0.99, step_dt=1/60)

| Trạng thái | Giá trị |
|---|---|
| Camping (>60 bước sau latch) | −11.7 |
| Lật chai | −30.0 |
| **Nhấc thành công** | **+72.8** |

Lợi thế của nhấc so với camping: từ **−134** thành **+84.5** (đổi dấu).

### Lệnh train đang chạy

```bash
./rl.sh local-train --envs 16 --steps 5000000 --progress \
  --phase 2 --stage all --assist-schedule \
  --checkpoint ./logs/train/best_policy.pt \
  --lr-start 1e-4 --lr-end 1e-5 --clip-range 0.1 --ent-coef 0.002
```
Log: `~/train_stage2.log`. VRAM 2.5GB/6.1GB.

**Train LOCAL chứ không deploy lên server** vì code đang là Isaac Lab 3.0 còn server là v2.3.2 — đặc biệt `rot=(0,0,0,1)` xyzw trên hệ wxyz của 2.3.2 là xoay 180°, làm lệch mục tiêu gắp 164mm mà **không crash**, rất khó phát hiện. Cần viết compat shim trước khi deploy.

### ⚠️ Kiểm tra tỉnh táo quan trọng nhất

`train/success_rate` và `ep_rew_mean` giờ phải đi **CÙNG** chiều. Nếu reward tăng mà success_rate phẳng → reward redesign chưa ăn, **dừng run ngay**.

## Run 5M bước trên server (train_stage2_lift) — KẾT QUẢ: exploit thứ 5, đã dừng

Lệnh đã chạy:
```bash
./rl.sh train train_stage2_lift \
  --task_phase 2 --assist-schedule --stage all \
  --num-envs 1024 --timesteps 5000000 \
  --checkpoint /data21tb/users/huyhoang/openarm_train_ws/policy_5M_good.pt \
  --lr-start 1e-4 --lr-end 1e-5 --clip-range 0.1 --ent-coef 0.002
```
Chạy xong 5,046,272 bước, không crash. Kết quả cuối: `success_rate=0`, `latch_rate=0`, `lift_start_rate=0`, `grasp_rate=0.275` (từ 1.0), `ep_rew_mean=812` (từ ~30).

**Diễn biến theo thời gian** (số liệu đầy đủ trong tin nhắn assistant, không chép lại ở đây): assist về 0 tại step ~2,000,896. `grasp_rate` giữ ~0.9-1.0 tới step ~2,490,368 rồi sụp liên tục xuống 0.24-0.28 tại 4.3M-5M. `ep_rew_mean` tăng ngược chiều từ ~100 lên 812 trong đúng cửa sổ đó.

**Chẩn đoán — exploit #5 (họ hàng với 4 exploit trước)**: `_compute_reach_reward` (rewards.py:205-293) **không có suy giảm nào** (quyết định có chủ ý ở dòng 282-287, dựa trên "chưa có bằng chứng REACH bị lợi dụng" — bằng chứng giờ đã có). Giá trị tĩnh tối đa gần bottle (`r_reach`+`r_lateral`+`milestones`+`r_close`+`r_top_down`+`finger_level`+`descent`+`r_hold`) cộng dồn có thể **> 180/bước**, không giới hạn thời gian, miễn là agent **không** thoả điều kiện chuyển stage `advance = (_stage==REACH) & (_steps_in_contact >= reach_hold) & aligned` (rewards.py:175). Một khi assist=0 và policy có toàn quyền, 3M bước on-policy đủ để nó học "lảng vảng sát ngưỡng, không bao giờ giữ contact đủ `reach_hold` bước liên tục" → farm REACH shaping suốt ~1200 bước/episode, lợi hơn nhiều so với vào GRASP (nơi shaping suy giảm theo `_steps_in_grasp`).

**Phát hiện phụ, không phải nguyên nhân chính của lần sụp này**: `log_std` của action-dim thứ 7 (index 6, nhiều khả năng là gripper) đã đóng băng ở ~16.0 (std≈9 triệu) từ checkpoint 1M bước, xuyên suốt mọi checkpoint tới 5M — tồn tại **từ trước** cả việc phục hồi `policy_5M_good.pt`. Không giải thích được sự sụp đổ lần này (grasp_rate vẫn ~1.0 dù std này đã tồn tại từ đầu), nhưng là nợ kỹ thuật cần dọn: entropy bonus đẩy log_std của 1 chiều hành động lên vô hạn khi chiều đó không có gradient sửa sai (nhiều khả năng do bị assist/scripted override lấn át gần như toàn bộ episode trong lịch sử train cũ). Cần thêm chặn `log_std` (vd. `policy_kwargs=dict(log_std_init=..., ...)` hoặc clamp thủ công) trước lần fine-tune kế tiếp.

**Đã chọn hướng sửa**: đếm bước "đứng ở cửa" (`env._steps_at_grasp_door`, đơn điệu trong vùng hysteresis contact zone lúc STAGE_REACH, reset khi rời zone hoặc advance/reset episode). Suy giảm `milestones+r_close+r_top_down+r_finger_level+r_descent+r_hold` về 0 sau `reach_door_decay_steps=60` bước KỂ TỪ KHI đã đủ `reach_stage_hold_steps` mà vẫn chưa advance (tức đúng lúc alignment đang bị né tránh). Reach hợp lệ (~700-1000 bước tới lúc advance) không bị đụng vì decay chỉ tính từ mốc "đã đủ hold, còn thiếu align".

File sửa: `mdp/rewards.py` (`_update_contact_and_stages` thêm đếm `_steps_at_grasp_door`; `_compute_reach_reward` áp decay), `mdp/helpers.py` (khởi tạo buffer), `mdp/terminations.py` (reset buffer khi env reset), `config.py` (key mới `reach_door_decay_steps=60`).

**Phát hiện phụ ghi nhận, CHƯA sửa**: `log_std` dim cuối (gripper, index 6) đóng băng ~16.0 (std≈9 triệu) từ trước — không phải nguyên nhân chính của lần sụp #5 (grasp_rate đo bằng stage-transition, chỉ phụ thuộc control tay, không phụ thuộc gripper). Để dành sửa sau khi verify exploit #5 xong (cần `policy_kwargs` clamp log_std hoặc train lại từ checkpoint sạch hơn).

### Verify local — 600k bước, log_dir riêng để không đè `best_policy.pt`

```bash
source ./openarm.env   # export ISAAC_SIM_PYTHON + PYTHONEXE (đọc python.sh + venv isaaclab)
nohup env PYTHONPATH="$(pwd)" PYTHONEXE="$PYTHONEXE" "$ISAAC_SIM_PYTHON" ./isaaclab_train.py \
  --num_envs 16 --timesteps 600000 --log_dir ./logs/verify_exploit5 \
  --visualizer none --progress \
  --task_phase 2 --stage all --assist-schedule \
  --checkpoint ./logs/train/policy_5M_good.pt \
  --lr-start 1e-4 --lr-end 1e-5 --clip-range 0.1 --ent-coef 0.002 \
  > /home/hans/verify_exploit5.log 2>&1 &
```
PID 77163 (wrapper) / 77170 (python thật). Log: `~/verify_exploit5.log`.

anneal_frac mặc định 0.4 × 600k = handoff assist→0 tại ~240k bước, còn lại ~360k bước on-policy thuần để lộ exploit nếu vẫn còn — cùng quy mô với các lần verify trước đó (300-340k bước đã đủ lộ exploit #1-#4 mỗi lần).

**Tiêu chí đọc kết quả**: nếu `grasp_rate` giữ được ~0.85-1.0 sau mốc 240k (thay vì sụp về 0.24-0.28 như run 5M vừa rồi) VÀ `ep_rew_mean` không tăng vọt bất thường trong lúc `success/latch/grasp` phẳng/giảm → exploit #5 đã đóng, có thể cam kết chạy dài trên server (khi GPU rảnh thật — hiện đang bị `naiscorp`/zipformer chiếm ~13GB/43%).

### KẾT QUẢ verify local 600k (lần 1) — exploit #5 ĐÃ ĐÓNG, lộ vấn đề mới

Tiến trình lần đầu (PID 77170) bị dừng đột ngột ở step 172k **không phải do lỗi** (không traceback/OOM) — do bị dọn theo vòng đời phiên làm việc khi phiên trước kết thúc (nohup/disown không đủ, tiến trình vẫn cùng process group). Chạy lại bằng `setsid ... </dev/null >log 2>&1 &` để tách hẳn session — sống sót qua nhiều lần gián đoạn phiên.

**Kết quả đầy đủ 600,064 bước**:
- `grasp_rate`: giữ vững **0.98-1.0 suốt toàn bộ run**, kể cả rất lâu sau khi assist về 0 (step 470k-600k vẫn 0.98-1.0) → **exploit #5 đã đóng, không còn "đứng ở cửa" farm reward REACH**.
- `latch_rate`: trôi từ ~0.75 (early, assist còn cao) xuống đáy dao động **0.02-0.08** từ ~step 300k trở đi, không hồi phục nhưng cũng không sụp về 0 tuyệt đối như grasp_rate từng làm ở run 5M cũ.
- `success_rate = 0` suốt toàn bộ 600k bước.
- `ep_rew_mean` KHÔNG còn kiểu tăng vọt trong khi task sụp (dấu hiệu exploit) — dao động thấp, có đoạn rất âm (~-640 cuối run, chưa rõ nguyên nhân, không critical vì latch/success đã phẳng thấp sẵn từ trước đó).

**Chẩn đoán latch kẹt thấp**: đúng như phát hiện phụ đã ghi — `log_std` dim cuối (index 6 = `gripper_action`, xác nhận qua `OSCActionsCfg`: `arm_action` 6 chiều rồi `gripper_action` 1 chiều) đóng băng ở ~16.0 (std≈9 triệu) từ checkpoint 1M của lịch sử train cũ, tồn tại xuyên suốt `policy_5M_good.pt`. Cơ chế: lúc assist_scale>0, gripper bị scripted override gần như toàn bộ episode → dim đó không có gradient sửa sai → entropy bonus (`ent_coef`) cứ đẩy log_std lên vô hạn không có phản lực. Khi assist về 0, gripper action thực thi gần như nhiễu thuần (std cực lớn) → không giữ latch được dù tay đã vào đúng vị trí (grasp_rate vẫn tốt vì đó là dim tay, log_std lành).

**Đã sửa**: thêm `LogStdClampCallback` (`isaaclab_train.py`) — clamp `model.policy.log_std` vào `[-3.0, 1.0]` (std ∈ [0.05, 2.72]) mỗi bước, đăng ký vào cả 2 `CallbackList` (fresh-train và assist-schedule). Sửa ngay dim đã hỏng khi nạp checkpoint cũ (clamp về 1.0 ngay từ bước đầu) VÀ ngăn tái phát cho mọi dim ở các lần train sau (không cần biết trước dim nào sẽ bị lãng quên).

### KẾT QUẢ verify local 600k (lần 2, log_dir `verify_exploit5b`) — clamp hiệu chỉnh sai trần

`std` về hẳn 2.24, KHÔNG còn 1.27e6 — clamp hoạt động. Nhưng `latch_rate` **tệ hơn lần 1**: đáy 0.01 (so với 0.02-0.08 lần 1 chưa có clamp), `success_rate=0` suốt.

Kiểm tra checkpoint cuối (`logs/verify_exploit5b/best_policy.pt`) lộ nguyên nhân: `log_std` = `[0.023, 0.572, 0.734, 0.996, 0.993, 0.999, 0.996]` — **4 dim** (index 3,4,5,6, không chỉ dim gripper hỏng) bị ép dính đúng trần `max_log_std=1.0` suốt cả 600k bước (đứng yên tuyệt đối, không nhích — dấu hiệu rõ ràng của việc bị clamp ép mỗi bước chứ không phải hội tụ tự nhiên). Dim 3,4,5 (orientation/xoay cổ tay) tự nhiên đã có log_std 1.6-2.3 (std 5-10) từ checkpoint 1M-5M **mà KHÔNG hỏng** (grasp_rate vẫn tốt lúc đó) — trần 1.0 xoá mất calibration đã học của 3 dim lành, không chỉ sửa dim bệnh (index 6).

**Đã sửa**: nâng `max_log_std` từ 1.0 lên **3.0** (std≈20) trong `LogStdClampCallback` (`isaaclab_train.py`) — chỉ chặn đúng giá trị bệnh lý (16.0), không đụng dim nào đang trong phạm vi bình thường (≤2.3). `min_log_std=-3.0` giữ nguyên (chưa thấy dim nào chạm sàn).

### KẾT QUẢ verify local 600k (lần 3, log_dir `verify_exploit5c`) — trần cao hơn CÀNG TỆ

Nâng trần lên 3.0 (như dự định sửa lần 2), kết quả: `latch_rate=0.005` (tệ hơn cả lần 2's 0.01), `std` cuối = 6.59 (cao hơn 2.24 — xác nhận entropy vẫn đẩy nhiều dim lên sát TRẦN MỚI bất kể trần đặt ở đâu).

**Xu hướng qua 3 lần verify (cùng checkpoint, cùng 600k bước)**:
| Lần | Can thiệp | `latch_rate` cuối |
|---|---|---|
| 1 | Không clamp (std=1.27e6 nguyên trạng) | 0.025 |
| 2 | Clamp mỗi bước, trần 1.0 | 0.01 |
| 3 | Clamp mỗi bước, trần 3.0 | 0.005 |

**Kết luận đúng**: KHÔNG phải "trần bao nhiêu là đúng" — bất kỳ clamp MỖI BƯỚC nào cũng làm latch tệ hơn, bất kể trần đặt ở đâu, và trần càng cao càng tệ vì entropy tiếp tục đẩy sát trần mới. Cơ chế: `.clamp_()` ghi đè trực tiếp `.data` mỗi bước ngay sau khi Adam optimizer đã update dựa trên momentum tích luỹ — liên tục "đấu" với trạng thái nội bộ của optimizer cho đúng tham số đó, tạo động lực học bệnh lý mới còn tệ hơn corruption gốc.

**Đã sửa đúng hướng**: bỏ hẳn callback clamp-mỗi-bước. Thay bằng **reset MỘT LẦN DUY NHẤT** ngay sau `model.policy.load_state_dict(state_dict)` khi fine-tune (`isaaclab_train.py`) — chỉ dim nào `log_std > 4.0` (ngưỡng rõ ràng bệnh lý, không đụng dải bình thường ≤2.3) được đặt về `0.0` (std=1, ngang khởi tạo mặc định). Sau đó Adam bắt đầu sạch từ giá trị hợp lý, không có can thiệp nào nữa suốt quá trình train — để cơ chế PPO tự điều chỉnh log_std bình thường (giảm nếu ổn định, tăng nếu cần khám phá) mà không bị ai "đấu tay đôi" với optimizer.

### KẾT QUẢ verify local 600k (lần 4, log_dir `verify_exploit5d`) — reset một lần vẫn TỆ, exploit #5 tái phát

Reset một lần đúng dim [6] (log về 0.0, xác nhận qua print `log_std bệnh lý ở dim [6], reset về 0.0: [15.983...]`). Kết quả: `latch_rate=0` (tệ nhất trong 4 lần), và lần đầu tiên **`grasp_rate` cũng sụp** — từ 0.98 (early) xuống dần còn **0.45** ở step cuối, đúng lúc `ep_rew_mean` tăng dần ngược chiều (−1.4 → 122 trong 250k bước cuối) — **CHÍNH XÁC chữ ký exploit #5 tái phát**, dù code reward chống exploit #5 không hề đổi giữa 4 lần verify.

**Bảng tổng hợp 4 lần verify (cùng checkpoint, cùng 600k bước, cùng seed ngẫu nhiên mặc định)**:
| Lần | Can thiệp log_std | `grasp_rate` cuối | `latch_rate` cuối |
|---|---|---|---|
| 1 | Không đụng gì (std=1.27e6 nguyên trạng) | **0.99** | 0.025 |
| 2 | Clamp mỗi bước, trần 1.0 | 0.99 | 0.01 |
| 3 | Clamp mỗi bước, trần 3.0 | 0.985 | 0.005 |
| 4 | Reset MỘT LẦN dim bệnh về 0.0 | **0.45** | 0 |

**Kết luận đúng (đảo ngược hoàn toàn giả thuyết ban đầu)**: log_std khổng lồ (16.0 ≈ std 9 triệu) ở dim gripper **KHÔNG phải lỗi cần sửa** — nó đang vô tình "im lặng hoá" gradient của một dim mà lịch sử train cũ chưa từng cho học thật (do assist đè hết episode): số hạng log-prob của dim đó chia cho std khổng lồ nên đóng góp gần như 0 vào loss, tức KHÔNG lan truyền gradient nhiễu vào mạng dùng chung (shared trunk tính mean cho cả 7 dim). Bất kỳ can thiệp nào bơm lại gradient cho dim "trống rỗng" này (dù nhẹ hay mạnh) đều làm nhiễu loạn luôn các dim tay đang học tốt qua chính mạng dùng chung đó — càng can thiệp mạnh (reset > clamp nhẹ > clamp mạnh > không đụng), latch/grasp càng tệ.

**Đã revert hoàn toàn**: bỏ mọi can thiệp log_std trong `isaaclab_train.py` (đã xoá `LogStdClampCallback` và khối reset một lần), quay về trạng thái của verify lần 1 — chỉ giữ fix exploit #5 (reward). Đây là cấu hình đã xác nhận tốt nhất qua toàn bộ 4 lần thử.

**Vấn đề còn lại (thật, không phải bug)**: `latch_rate` floor ~0.025 và `success_rate=0` sau 360k bước on-policy (verify lần 1) — nhiều khả năng đơn thuần cần NHIỀU bước on-policy hơn để gripper thật sự học kỹ năng đóng kẹp chính xác (trước giờ luôn bị assist làm hộ), không phải một bug cụ thể nào. Đây đúng là câu hỏi mở đã đặt ra trước khi cam kết chạy 5M bước server — giờ với exploit #5 đã đóng, câu hỏi này có thể trả lời sạch bằng một run dài hơn (5M, ~3M bước on-policy) mà không bị nhiễu bởi exploit REACH-camping nữa.

## Exploit #5b ("bơm" qua mặt fix #5) — phát hiện ở run 5M LOCAL thật, không phải verify ngắn

User yêu cầu train ngay trên local PC (thay vì chờ GPU server) khi GPU server tiếp tục bị chiếm. VRAM local đủ dùng: 16 envs chỉ tốn **2.5GB/6.1GB**. Lệnh:
```bash
source ./openarm.env
setsid env PYTHONPATH="$(pwd)" PYTHONEXE="$PYTHONEXE" "$ISAAC_SIM_PYTHON" ./isaaclab_train.py \
  --num_envs 16 --timesteps 5000000 --log_dir ./logs/train_stage2_lift_v2_local \
  --visualizer none --progress \
  --task_phase 2 --stage all --assist-schedule \
  --checkpoint ./logs/train/policy_5M_good.pt \
  --lr-start 1e-4 --lr-end 1e-5 --clip-range 0.1 --ent-coef 0.002 \
  < /dev/null > /home/hans/train_stage2_lift_v2_local.log 2>&1 &
```
(luôn dùng `setsid ... </dev/null` — nohup/disown KHÔNG đủ, tiến trình vẫn bị dọn theo phiên nếu không tách session riêng, đã gặp 2 lần).

**Kết quả trước khi phát hiện lỗi**: tốt tới step ~1.5M (grasp_rate 0.99, latch_rate tăng dần theo assist giảm) — TỐT HƠN hẳn 4 lần verify 600k trước đó (lift_start_rate/latch_rate cao hơn nhiều ở cùng mức assist, vì có nhiều bước pre-handoff hơn để ổn định). Nhưng bắt đầu từ step ~1,708,032 (assist scale ~0.146, CHƯA về 0 hẳn), `grasp_rate` sụp liên tục 0.52→0.10-0.23 trong khi `ep_rew_mean` tăng liên tục 469→862 — chữ ký exploit y hệt exploit #5, dù code chống exploit #5 không đổi.

**Chẩn đoán — exploit #5b, lỗ hổng trong chính fix #5**: `_steps_at_grasp_door` (rewards.py) RESET VỀ 0 mỗi khi rời vùng hysteresis contact zone (dist>`success_contact_exit_threshold`=0.10). Agent học được chiến thuật "bơm" (pump): tiến vào vùng contact farm milestones/r_close/r_top_down vài bước (luôn ít hơn `reach_stage_hold_steps`=5 để không lỡ advance), rút ra ngoài 0.10 (mất rất ít reward vì r_reach/r_lateral vẫn cao ở dist~0.10-0.12), rồi lặp lại — mỗi chu kỳ refresh counter về 0, né decay vĩnh viễn. **Cần ~1.7M bước train mới đủ tinh vi để tìm ra pump** — giải thích tại sao verify ngắn 600k (chỉ tới 360k bước post-handoff, tổng cộng 600k bước training) không lộ ra: mạng chưa đủ "kinh nghiệm" để khám phá chiến thuật 2 bước tinh vi này, dù về mặt logic lỗ hổng đã có sẵn từ đầu.

**Đã sửa**: đổi `_steps_at_grasp_door` từ "đếm liên tiếp, reset khi rời zone" sang "**tích luỹ cả episode, không reset khi rời zone**" (chỉ reset khi advance sang GRASP hoặc episode reset thật — terminations.py đã có sẵn). Bơm ra/vào giờ không giúp gì: tổng thời gian ở cửa vẫn cộng dồn bất kể bao nhiêu chu kỳ, decay vẫn kích hoạt đúng lúc dựa trên TỔNG thời gian loitering, không phải thời gian liên tục.

**Bài học đo lường quan trọng**: verify 600k KHÔNG đủ để bắt các exploit cần nhiều bước train để "chín" (khám phá). Từ giờ, exploit chỉ được coi là ĐÃ ĐÓNG sau khi sống sót qua toàn bộ 5M bước thật, không phải sau 600k verify — verify ngắn chỉ dùng để loại nhanh các exploit THÔ (như exploit #5 gốc, lộ ngay từ ~300k bước).

## Exploit #5c — fix "surgical" (door-based) thất bại lần 2, chuyển sang blanket decay

Máy local **tự reboot 2 lần** trong quá trình chạy 5M bước (lần 1 do session teardown dù đã `setsid`; lần 2 là reboot thật của máy, 2026-09-03 13:36 — cần kiểm tra vì sao, có thể update tự động hoặc mất điện). Cả 2 lần đều làm mất tiến trình training giữa chừng — laptop KHÔNG đáng tin cho job dài nhiều giờ.

**Run thứ 2 (trước khi mất ở step 3.47M/5M) lộ ra exploit MỚI, sâu hơn #5b**: `grasp_rate` xuống **đúng 0** và đứng yên ở đó suốt **~1 triệu bước liên tục** (step 2.31M→3.25M, hoàn toàn trong vùng post-handoff on-policy thuần), trong khi `ep_rew_mean` tăng đều đặn từ 963 lên hơn **1080** — mức camping cao nhất từng thấy qua tất cả các exploit.

**Chẩn đoán — exploit #5c**: fix #5b (`_steps_at_grasp_door` tích luỹ) chỉ tăng khi `in_contact` = đã từng dip xuống dưới `success_dist_threshold`=0.07 (ngưỡng VÀO vùng hysteresis). Agent tìm ra: đứng ở khoảng cách **0.07 < dist < 0.10** (ngoài vùng hysteresis, trong ngưỡng milestone đầu tiên) — KHÔNG BAO GIỜ trigger `_in_contact_zone` nên `_steps_at_grasp_door` mãi mãi = 0, né hoàn toàn cơ chế decay dù nó đã tích luỹ-cả-episode. `r_reach`, `r_lateral`, `r_top_down` (close_mask dùng dist<0.15, không phải 0.07/0.10), milestone tier đầu (<0.10) vẫn cho reward đáng kể ở khoảng cách này, hoàn toàn không bị decay chạm tới.

**Bài học cốt lõi (đã hỏi lại user, đổi hướng)**: cách tiếp cận "surgical" — chỉ suy giảm khi agent ở một VỊ TRÍ/TRẠNG THÁI cụ thể (ở cửa, trong vùng contact, v.v.) — về cấu trúc KHÔNG THỂ đóng hết họ exploit REACH-camping, vì luôn còn trạng thái khác chưa bị khoá mà agent đủ thời gian train sẽ tìm ra (exploit #5 gốc: ~vài trăm k bước; #5b pump: ~1.7M bước; #5c đứng ngoài vùng: đâu đó trước 2.3M bước). Quyết định: **bỏ hẳn cách tiếp cận theo vị trí, chuyển sang blanket decay theo TỔNG THỜI GIAN Ở STAGE_REACH** — bất biến với MỌI chiêu trò né vị trí vì không quan tâm agent đứng ở đâu.

**Đã sửa** (`rewards.py`, `helpers.py`, `terminations.py`, `config.py`):
- Xoá `_steps_at_grasp_door` (buffer + logic cập nhật trong `_update_contact_and_stages`).
- Thêm `env._steps_in_reach`: tăng đơn điệu mỗi bước còn `_stage==STAGE_REACH` (không phụ thuộc khoảng cách/contact gì cả), reset về 0 khi advance sang GRASP hoặc episode reset thật.
- `_compute_reach_reward` giờ nhân **TOÀN BỘ** tổng reward (kể cả các penalty: `r_z_floor`, `r_stall`, `r_tilt_penalty`, `table_penalty`, `vel_penalty`) với `reach_decay` — suy giảm tuyến tính từ 1.0 về 0.0 bắt đầu sau `reach_reward_decay_onset_steps=1000` bước (khớp REACH hợp lệ đo được ~700-1000 bước), về hẳn 0 sau thêm `reach_reward_decay_steps=200` bước. Trong vòng 1 độ dài episode (~1200 bước), camping bằng BẤT KỲ chiêu trò vị trí nào cũng bị triệt tiêu hoàn toàn.
- Config key mới: `reach_reward_decay_onset_steps=1000`, `reach_reward_decay_steps=200` (thay `reach_door_decay_steps=60` đã xoá).

**Đang chạy lại từ đầu trên local** (đã kill run cũ ở step ~3.47M, đang bị exploit #5c) với fix #5c, cùng lệnh cấu hình như các lần trước, log_dir `train_stage2_lift_v2_local` (ghi đè).

## Exploit #5c fix THẤT BẠI — bug hiệu chỉnh nghiêm trọng, đã sửa bằng đo thật

Run `train_stage2_lift_v3_local` (chạy tối 3/9, dùng `systemd-inhibit --what=sleep:idle` để máy không tự ngủ/tắt — 2 lần trước máy TỰ TẮT SẠCH lúc 00:02 và 13:36 cùng ngày, xác nhận qua `journalctl -k -b -1`: đây là **shutdown sạch có chủ đích** (`systemd-shutdown: Syncing filesystems...`), tức người dùng tự tắt máy khi về/đóng máy — KHÔNG PHẢI lỗi phần cứng, chỉ là chu kỳ dùng máy hàng ngày. Bài học: train local dài giờ cần `systemd-inhibit` VÀ máy phải ở lại mở, không thể "fire and forget" như server).

**Fix #5c (blanket decay onset=1000/decay=200) CŨNG THẤT BẠI**: tại step 3,048,448, `grasp_rate=0.3` (từ ~1.0), `latch_rate=0`, `ep_rew_mean=810` — đúng chữ ký exploit lần thứ 4.

**Chẩn đoán bug hiệu chỉnh**: `env.max_episode_length = episode_length_s(20.0) / step_dt(1/60) = 1200` bước. Onset=1000 + decay=200 → về 0 đúng **tại step 1200 = CHÍNH XÁC lúc episode timeout**. Nghĩa là suốt gần như toàn bộ episode (1000/1200 = 83%), camping nhận đủ 100% reward, và ngay cả 200 bước "decay" cuối cũng chỉ tuyến tính giảm chứ không về 0 ngay — decay được thiết kế nhưng **chưa bao giờ thực sự có cơ hội cắn vào giữa episode**. Con số 1000 dựa trên comment cũ trong code ("REACH tự nhiên cần ~700-1000 bước") — **chưa từng được đo thật**.

**Đã đo thật**: thêm log `episode_length_buf[0]` (số bước TRONG episode, không phải `step_counter` toàn cục) vào đúng chỗ in `[Stage] REACH → GRASP`, chạy 8 env với **assist đầy đủ scale=1.0** (script luôn làm đúng REACH, không có exploit) trong ~30s:
```
[Stage] REACH → GRASP @ step 21 ep_step:21 | ...
[Stage] REACH → GRASP @ step 1259 ep_step:59 | ...
[Stage] REACH → GRASP @ step 2421 ep_step:21 | ...
```
**REACH thật chỉ mất 21-59 bước** — sai lệch **15-30 LẦN** so với con số 700-1000 dùng để hiệu chỉnh onset. Thời gian REACH do động lực học cánh tay quyết định (tốc độ tối đa OSC), hoàn toàn không liên quan tới `episode_length_s`.

**Bài học cốt lõi**: đừng bao giờ hiệu chỉnh tham số reward dựa trên comment/giả định trong code mà chưa verify bằng số đo thật — một con số sai 1 lần (700-1000 bước) bị copy sang comment mới, rồi dùng làm cơ sở cho fix tiếp theo, khiến sai số nhân lên qua nhiều lần sửa mà không ai phát hiện cho tới khi đo trực tiếp.

**Đã sửa**: `reach_reward_decay_onset_steps` 1000→**200** (biên độ an toàn ~3.5x so với 59 đo được), `reach_reward_decay_steps` 200→**100** (về 0 ở step 300, còn 900/1200 bước = 75% episode hoàn toàn không reward nếu vẫn camp).

**KẾT QUẢ verify 2M** (`verify_5c_v2`, chạy xong đủ 2,000,896 bước, không crash): `ep_rew_mean` KHÔNG còn tăng vọt (đỉnh ~180 ở step 789k rồi giữ phẳng quanh 110-150 tới hết run) — decay đã thực sự cắn, **không còn chữ ký exploit runaway**. Nhưng `grasp_rate` vẫn sụp từ ~1.0 xuống 0.05-0.2 ngay sau handoff (~step 800k, 0.4×2M) và KHÔNG hồi phục suốt 1.17M bước còn lại. Vì reward không còn tăng vọt (khác hẳn 4 exploit trước), nghi ngờ đây là vấn đề THẬT (độ chính xác tay khi không có assist) chứ không phải reward-hacking. User chọn: cứ chạy dài 5M để xem có tự hồi phục không (đúng câu hỏi gốc trước khi phát hiện các exploit).

**Đang chạy 5M đầy đủ** (`train_stage2_lift_v4_local`, cùng cấu hình, `systemd-inhibit` chống ngủ). Lệnh đã dùng cho verify 2M (đổi lại `--timesteps 2000000` nếu cần verify lại):
```bash
source ./openarm.env
setsid systemd-inhibit --what=sleep:idle --why="RL verify 2M" \
  env PYTHONPATH="$(pwd)" PYTHONEXE="$PYTHONEXE" "$ISAAC_SIM_PYTHON" ./isaaclab_train.py \
  --num_envs 16 --timesteps 2000000 --log_dir ./logs/verify_5c_v2 \
  --visualizer none --progress \
  --task_phase 2 --stage all --assist-schedule \
  --checkpoint ./logs/train/policy_5M_good.pt \
  --lr-start 1e-4 --lr-end 1e-5 --clip-range 0.1 --ent-coef 0.002 \
  < /dev/null > /home/hans/verify_5c_v2.log 2>&1 &
```

## Đổi công cụ theo dõi: tmux thay vì redirect thuần

`--progress` (tqdm) không tạo ra byte `\r` nào khi output bị redirect vào file (`grep -c $'\r'` = 0) — tqdm tự tắt phần vẽ trực quan khi phát hiện không phải TTY thật. Cài `tmux` (`sudo apt-get install -y tmux`) và chạy training bên trong session tmux để có TTY thật:
```bash
tmux new-session -d -s rl_train "systemd-inhibit --what=sleep:idle --why='...' \
  env PYTHONPATH=... PYTHONEXE=... ISAAC_SIM_PYTHON ./isaaclab_train.py ... \
  2>&1 | tee /path/to/log"
tmux attach -t rl_train   # Ctrl+B rồi D để thoát không dừng tiến trình
```
Lưu ý: kể cả trong tmux, thanh % của SB3 (dựa trên `rich`) vẫn xung đột với bảng log logger (cả hai cùng ghi stdout) nên không hiện rõ — nhưng bảng chỉ số (`total_timesteps` v.v.) vẫn cập nhật sống mỗi ~15-20s, đủ dùng làm thước đo tiến độ qua `tail -f`/`tmux attach`.

## Nguyên nhân gốc `std` gripper tăng vô hạn — tìm ra bằng grep, không phải đoán

Run `train_stage2_lift_v4_local` (chạy trong tmux, dùng recalibrated onset=200/decay=100): session tmux tự mất lúc 03:24 (máy có thể ngủ lại — cần kiểm tra `systemd-inhibit --list` có còn hoạt động không mỗi lần trước khi rời máy). Trước khi mất, log cho thấy **KHÔNG PHẢI exploit mới**: `ep_rew_mean` dao động 85-240 (không leo thang như 3 exploit trước), nhưng `grasp_rate` dao động thất thường 0.99→0.5→0.79→0.53→0.79→0.135 ngay sau handoff (~2M) rồi trôi dần xuống — dấu hiệu **bất ổn định huấn luyện**, không phải reward-hacking.

**Tìm nguyên nhân**: `std` (metric SB3, trung bình std qua 7 dim action) tăng CHẬM NHƯNG LIÊN TỤC suốt cửa sổ quan sát (1.35e6→1.40e6), kể cả rất lâu sau handoff — nghĩa là dim gripper hỏng chưa từng dừng phình to, không chỉ trong giai đoạn assist còn hoạt động.

`grep -n "_assist_blend_scale" actions.py` → **0 kết quả**. Xác nhận: logic đóng/mở kẹp (`actions.py`) luôn chạy kịch bản (scripted) **độc lập hoàn toàn với lịch assist** — override raw action của `gripper_action` ở MỌI thời điểm training, kể cả sau khi `_assist_blend_scale=0`. Dim này KHÔNG BAO GIỜ nhận gradient thật tương quan với reward ở bất kỳ giai đoạn nào của toàn bộ quá trình train (không chỉ trong "giai đoạn assist" như giả định trước) — entropy bonus đẩy `log_std` của nó tăng vô hạn, không có điểm dừng tự nhiên.

**Lý do 3 lần sửa log_std trước đều thất bại** (verify_exploit5b/5c/5d): chỉ RESET giá trị (`.data[mask] = 0.0` hoặc `.clamp_()` mỗi bước) nhưng để gradient tiếp tục tự do chảy qua param đó — dim vừa "yên tĩnh" (std khổng lồ làm mẫu số log-prob triệt tiêu gradient) bị bơm gradient mạnh trở lại đột ngột, gây nhiễu loạn shared trunk (mean network dùng chung cho cả 7 dim). Cơ chế "đấu tay đôi với Adam optimizer": ghi đè `.data` trực tiếp không ngăn được optimizer tiếp tục tính gradient và cập nhật lại ở bước sau.

**Fix mới (chưa thử trước đây)**: reset MỘT LẦN (không neo ở 16.0 vĩnh viễn) **kèm đóng băng gradient vĩnh viễn** bằng `register_hook` trên chính param `log_std` — chặn gradient TRƯỚC khi tới optimizer (không phải ghi đè `.data` sau khi optimizer đã update), nên Adam không bao giờ "thấy" có gì để đấu, tránh đúng cơ chế đã làm 3 lần trước thất bại. Code trong `isaaclab_train.py`, ngay sau `model.policy.load_state_dict(state_dict)`.

**Đang verify 2.5M bước** trong tmux để kiểm chứng — theo dõi đặc biệt: (1) `std` có đứng yên ở mức hợp lý (không còn triệu) suốt run không, (2) `grasp_rate` có hết dao động/trôi dần sau handoff không.

## TÓM TẮT TOÀN BỘ — tạm dừng để cân nhắc chiến lược (2026-09-04)

### Đã sửa và xác nhận đúng (không cần làm lại)
1. **Exploit #5/#5b/#5c** (REACH-camping họ 3 biến thể) — đã đóng bằng blanket decay theo tổng thời gian ở STAGE_REACH (`_steps_in_reach`), hiệu chỉnh bằng số đo thật (`onset=200, decay=100` bước, không phải đoán). Xác nhận: `grasp_rate` không còn sụp kiểu "đứng yên farm reward" nữa.
2. **log_std gripper đóng băng sai cách** — nguyên nhân gốc: `actions.py` không hề đọc `_assist_blend_scale` (grep xác nhận 0 kết quả), nghĩa là logic đóng/mở kẹp LUÔN chạy kịch bản độc lập với lịch assist, nên dim gripper không bao giờ nhận gradient thật → entropy đẩy log_std lên vô hạn (16.0 ≈ std 9 triệu). Sửa bằng reset một lần + `register_hook` đóng băng gradient vĩnh viễn (không phải clamp `.data` mỗi bước — cách đó đấu với Adam optimizer, đã thử 3 lần đều làm tệ hơn). Xác nhận: `std` giữ ổn định ~4.0 suốt toàn bộ run dài, không còn triệu.
3. **Giảm chấn REACH bị tắt nhầm theo lịch assist** — `apply_grasp_arm_assist` có early-return `scale<=1e-6` chặn đứt luôn phần giảm chấn hover (vốn chỉ lọc nhiễu, không phải làm hộ) cùng lúc với lift-bias. Đo trực tiếp bằng `assist_scale=0.0`: `top_down_align` kẹt ở 0.47-0.53 (cần ≥0.58) và CÒN GIẢM DẦN theo thời gian. Sửa: tách giảm chấn hover ra khỏi lịch assist, luôn chạy. Xác nhận: `top_down_align` hết giảm dần, ổn định ~0.52 (dù chưa đạt 0.58).

### Vấn đề CHƯA giải quyết
Run verify 2.5M bước cuối cùng (`verify_hoverdamp`, có đủ 3 fix trên) cho kết quả:
- `grasp_rate`: hồi phục được 1 lần (0.77→1.0 quanh step 1.04M-1.22M, đúng lúc assist về hẳn 0) — CHỨNG MINH các fix có tác dụng thật.
- Nhưng sau đó sụp lần 2 (1.22M→1.9M) và **không hồi phục**, giữ nguyên `grasp_rate=0` tới hết 2.5M bước.
- `ep_rew_mean` trong giai đoạn sụp lần 2 **giảm dần** (203→95), KHÔNG tăng — loại trừ khả năng exploit/lợi dụng reward. Đây là mất năng lực thật (capability gap), không phải reward hacking.
- `std` giữ ổn định ~4.0-4.07 suốt (log_std fix hoạt động đúng, không phải nguyên nhân).

### Giả thuyết còn bỏ ngỏ
- Decline lần 1 bắt đầu ở step ~650k, TRƯỚC KHI assist về hẳn 0 (mới ~35%) — gợi ý CHÍNH QUÁ TRÌNH ANNEAL (thay đổi liên tục) gây bất ổn hơn là điểm cuối (assist=0 cố định). Hồi phục xảy ra ngay khi scale ổn định ở 0.
- Decline lần 2 (1.22M→1.9M) xảy ra rất lâu SAU KHI assist đã ổn định ở 0 — không liên quan gì tới cơ chế anneal/handoff nữa. Đây là hiện tượng RIÊNG BIỆT, nguyên nhân chưa rõ (có thể: value function overfit, policy trôi dạt vào vùng xấu do nhiễu tích luỹ, hoặc `r_top_down` chưa đủ mạnh để chống lại các gradient khác một khi đã lệch).
- Chưa thử: `anneal_frac` dài hơn nhiều (0.7-0.8 thay vì 0.4), tăng trọng số `r_top_down`, hoặc xem xét lại có nên giữ giảm chấn/hỗ trợ orientation VĨNH VIỄN (không chỉ tách khỏi lift-schedule mà còn tăng cường thêm) thay vì kỳ vọng policy tự học điều khiển hướng chính xác hoàn toàn độc lập.

### Trạng thái code hiện tại
Tất cả 3 fix trên đã có trong working tree (`isaaclab_openarm_env/mdp/rewards.py`, `helpers.py`, `terminations.py`, `config.py`, `isaaclab_train.py`, `isaaclab_openarm_env/mdp/grasp_assist.py`). Checkpoint dùng để fine-tune vẫn là `logs/train/policy_5M_good.pt`. Chưa commit git (nhánh `fix/lift-pipeline` đang có nhiều file M chưa add).

## ĐỘT PHÁ THẬT: vá mimic gripper (2026-09-04/05) — success_rate 0 → 3.3%

### Nguyên nhân gốc tìm ra bằng quan sát trực tiếp trong Isaac Sim

User quan sát: "hai ngón tay trái/phải không đối xứng, lệch về một bên" khi kẹp. Kiểm tra USD:
```
openarm_left_finger_joint1  PhysicsPrismaticJoint, CÓ drive (targetPosition)
openarm_left_finger_joint2  PhysicsPrismaticJoint, KHÔNG drive — bám theo joint1
    qua PhysxMimicJoint: naturalFrequency=25.0, dampingRatio=0.005
```
`naturalFrequency=25` (lò xo rất mềm) + `dampingRatio=0.005` (gần như không giảm chấn), trong khi joint1 có `stiffness=1500`. Đo bằng script (`griplag.py`, tạo trong scratchpad): lệnh đóng theo các kiểu ramp khác nhau, đo lệch đỉnh giữa 2 ngón:
- Step input (đóng ngay): lệch đỉnh **11.5mm**
- Ramp 60 bước (~1s): lệch đỉnh **52.4mm** (!) — một ngón đóng hẳn, ngón kia kẹt ở giới hạn MỞ
- Ramp 150 bước (~2.5s): lệch đỉnh **7.1mm** — tốt hơn hẳn

Bottle diameter = 43.3mm — lệch 52mm nghĩa là hoàn toàn không có kẹp đối xứng thật, một ngón đơn độc hất chai sang bên. **Đây là nhiệm vụ bất khả thi về vật lý từ đầu**, không phải lỗi reward.

### Fix: vá vĩnh viễn vào v10.usd (không phải runtime patch)

Viết `Reinforce_Learning/patch_gripper_mimic.py` (giống pattern `patch_robot_usd.py`/`patch_qvic_usd.py` đã có sẵn) — mở `v10.usd`, set `naturalFrequency=200.0, dampingRatio=1.0` cho cả `left/right_finger_joint2`, lưu trực tiếp vào file. Backup gốc: `v10.usd.before_mimic_fix.bak`. Chạy MỘT LẦN:
```bash
$PYTHONEXE Reinforce_Learning/patch_gripper_mimic.py
```
Xác nhận qua `Usd.Stage.Open` + đọc attribute: file thực tế đã đổi 1456→1759 bytes, giá trị mimic mới = 200.0/1.0 ở cả 2 tay.

### Kết quả ngay sau khi vá (trước khi sửa gì khác)

`eval_lift_metrics.py --assist-scale 1.0 --episodes 30 --seed 0`: **`success_rate=0.0333`** (1/30) — LẦN ĐẦU TIÊN khác 0 trong toàn bộ 2 tuần. `grasp_rate=0.77-0.90`, `latch_rate=0.40-0.50` (kẹp đối xứng thật). Fail mode chính: `lift_too_low` (kẹp được, nhấc được vài mm, không đủ 30mm).

## Chuỗi sửa tiếp theo: `_grip_secure` cho phép nhấc dù kẹp yếu

### Bug 1: `_grip_secure` dùng OR, cho phép nhấc chỉ cần hình học (không cần lực)
```python
# CŨ: return _grip_latched(env) & (_grip_close_done | _grip_pressing | _grip_physically_closed)
# _grip_physically_closed CHỈ kiểm tra span/joint gần nhau — KHÔNG lực.
```
Đo bằng `DEBUG_LIFT=1`/`DEBUG_STALL=1` (thêm 2 biến môi trường debug tạm vào `grasp_assist.py`/`actions.py`, gated, an toàn để giữ lại): lift bắt đầu khi stall chỉ **2.69mm** (yếu). Theo dõi env đang RISING: chai theo tay đúng **1.5mm** rồi **stall tụt về 0** — ngón trượt hoàn toàn khỏi thân chai, tay bay lên khoảng không ~886mm (z_error_finger) trong hơn 400 bước mà không ai phát hiện (bộ dò `_update_lift_slip` chỉ so sánh delta-1-bước ~1-2mm, ngưỡng dò 9.9mm — trượt "êm" không bao giờ vượt ngưỡng).

**Fix**: `_grip_secure` giờ bắt buộc `_grip_pressing` (kiểm tra lực thật qua stall), không còn tuỳ chọn:
```python
return _grip_latched(env) & _grip_pressing(env, s) & (_grip_close_done(env, s) | _grip_physically_closed(env, s))
```

### Bug 2: ramp đóng tiếp tục siết SAU điểm chạm, tự đẩy văng chai

Đo full trace (env1, `DEBUG_STALL`): tại `gc≈0.712` (span≈44mm ≈ đúng đường kính chai — thời điểm chạm đầu tiên), stall đạt đỉnh **3.39mm**. Ramp KHÔNG dừng ở đó (không có cơ chế lực-feedback), tiếp tục siết theo `grasp_close_freeze_at_progress` (0.75 hoặc thử 0.90) — và trong quá trình siết tiếp, **đẩy văng chai ra khỏi kẹp**: span cứ giảm dần (vật lý không thể, trừ khi chai đã bị đẩy đi), stall tụt hẳn về 0.00mm ở gc cuối cùng.

Thử 3 giá trị `freeze_at` (0.75/0.90/0.70) đều cho `success_rate=0.033` y hệt (kẹt đúng 1 episode may mắn) — chứng tỏ **chỉnh ngưỡng % không giải quyết được gì**, vấn đề nằm ở CƠ CHẾ (không có phản hồi lực), không phải THAM SỐ.

**Fix đúng**: thêm cơ chế dừng ramp theo lực chạm thật trong `actions.py::apply_actions()` (action term `AssistedBinaryGripperAction`):
- Buffer mới `self._press_hold_steps` (đếm bước liên tiếp có lực, giống pattern `slip_steps`).
- Mỗi bước: tính `stall_now = joint_now - target_now` (cần `_joint_pos_mean_safe()` helper mới — xem bug tương thích bên dưới).
- `pressing_now = near_bottle & (stall_now > grasp_press_freeze_stall_m)`.
- `firm_contact = press_hold_steps >= grasp_press_freeze_hold_steps (3)` → loại khỏi `can_advance`, ramp đóng băng vĩnh viễn tại đó.

### Bug 3 (tự gây ra, đã sửa): 2 ngưỡng lực không khớp nhau

Lần đầu đặt `grasp_press_freeze_stall_m=0.003` THẤP HƠN `grasp_press_min_stall_m=0.005` (lý do sai: "hai mục đích khác nhau") — nhưng ramp ĐÃ ĐÓNG BĂNG ở 3mm nên KHÔNG BAO GIỜ leo lên tới 5mm được nữa → `lift_start_rate` tụt xuống 0.03 (kẹt, `_grip_pressing` không bao giờ pass). Bài học: ngưỡng "dừng siết" và ngưỡng "cho phép nhấc" PHẢI bằng nhau, không được ngưỡng dừng thấp hơn ngưỡng cho phép.

**Fix**: cả hai = **0.0028m** (dựa theo đỉnh lực thật đo được ~3.4mm, trừ hao chút để ổn định 3 bước liên tiếp; noise baseline actuator ~2.4mm khi chưa chạm gì).

### Bug tương thích: `self._joint_ids` trong `actions.py` là `wp.array`, không phải list

Khác với `env._gripper_joint_ids` (từ `find_joints()`, list Python bình thường, dùng ổn trong `grasp_assist.py`), `self._joint_ids` của action term kế thừa từ IsaacLab 3.0 base class là **wp.array** — in ra giống `[24]` nhưng dùng làm index vào torch.Tensor gây lỗi nội bộ warp ("Item indexing is not supported on wp.array objects"), dù chính `joint_pos.torch` đã convert đúng. Debug bằng cách tách riêng từng bước (`type(joint_ids)` → `warp._src.types.array`). Fix: `_joint_pos_mean_safe()` helper mới trong `actions.py`, chuyển `joint_ids` sang list qua `.numpy().tolist()` trước khi index, và tự dò `.torch` property (ProxyArray, IsaacLab 3.0) vs Tensor thường (server 2.3.2) qua `getattr(raw, "torch", raw)`.

### KẾT QUẢ SAU CẢ 3 FIX (cấu hình tốt nhất tới giờ)
```
success=0.03 grasp=0.83 latch=0.43-0.47 lift_start=0.40 lift_cmd_steps=162.2
fail_modes: {lift_too_low: 11, reach: 5, grasp: 12, no_lift_command: 1}
```
Đã thử thêm "đóng chậm lại khi gần chai" (`grasp_press_slow_factor=0.15`, kích hoạt khi `near_bottle` dist<6cm) dựa trên gợi ý từ griplag.py (ramp chậm → lệch ít hơn) — **THẤT BẠI**, `lift_start_rate` tụt về 0.03: ngưỡng 6cm kích hoạt quá sớm so với điểm chạm thật (~44mm span ở gc≈0.71), ramp chậm suốt quãng dài không cần thiết, không kịp đóng đủ trong 1 episode (1199 bước). Đã revert `grasp_press_slow_factor` về 1.0 (tắt), xác nhận khôi phục đúng kết quả tốt nhất ở trên.

### Vấn đề còn lại — đã tách bạch rõ, khác hẳn 2 bug đã sửa
`lift_too_low` (11/30, fail mode chính): kẹp đối xứng, chạm thật, ramp dừng đúng lúc, lift được phát lệnh hàng trăm bước (`lift_cmd_steps=162`) — nhưng lực kẹp ổn định tại điểm chạm đầu (~2.8-3.4mm stall) **tự nó không đủ mạnh** để giữ chai qua toàn bộ hành trình 30mm. Đây KHÔNG còn là bug máy trạng thái/logic nữa — là giới hạn lực/tốc độ đóng vật lý thật. Hướng chưa thử thành công: đóng chậm CÓ CHỦ ĐÍCH đúng vùng gần điểm chạm (không phải toàn bộ vùng near_bottle 6cm) — cần định nghĩa vùng "gần điểm chạm" chính xác hơn (theo span/progress, không theo khoảng cách tuyệt đối).

### File đã sửa trong chuỗi fix này
`isaaclab_openarm_env/mdp/grasp_assist.py` (`_grip_secure`, `_grip_pressing` debug print, `DEBUG_LIFT`/`DEBUG_STALL` instrumentation — giữ lại, gated, hữu ích cho debug sau này), `isaaclab_openarm_env/mdp/actions.py` (`_joint_pos_mean_safe`, `_press_hold_steps` buffer, force-freeze + slow-zone logic trong `apply_actions`), `isaaclab_openarm_env/config.py` (`grasp_press_freeze_stall_m`, `grasp_press_freeze_hold_steps`, `grasp_press_slow_factor`, `grasp_press_min_stall_m` đổi 0.005→0.0028), `Open_arm_a1_ws/.../v10.usd` (vá vĩnh viễn, KHÔNG cần chạy lại patch script), `Reinforce_Learning/patch_gripper_mimic.py` (script vá, giữ lại làm tài liệu/nếu cần vá file khác).

## Kết luận: cơ chế đóng kẹp KHÔNG PHẢI nút thắt còn lại

Sau ~10 lần thử khác nhau (freeze_at 0.75/0.90/0.70/1.0, bắt buộc `_grip_pressing`, đồng bộ 2 ngưỡng lực, slow-zone theo khoảng cách, ramp lực nhấc theo thời gian, cổng hình học span) — `success_rate` **bất biến tuyệt đối ở 0.0333 (1/30, luôn cùng env0)**.

**Phát hiện quyết định** (đo bằng `DEBUG_FREEZE`, log stall mỗi bước từ gc=0): `stall` (chênh khớp-so-với-lệnh) **không phải tín hiệu chạm sạch** — nó là độ trễ bám theo TỐC ĐỘ ĐÓNG KHÔNG ĐỔI của ramp, tăng dần liên tục ngay từ `gc=0` (span=100mm, cách xa chai hàng chục cm) lên tới ~2.4mm và DAO ĐỘNG quanh mức đó suốt phần còn lại — không phải "nhiễu baseline phẳng" như giả định ban đầu. Tín hiệu chạm thật chỉ là phần tăng thêm nhỏ (~0.3-0.6mm) chồng lên, và nó dao động lên xuống quanh ngưỡng 2.8mm (2.95→2.54→2.66→2.30→2.45→2.11mm) chứ không tăng ổn định — không bao giờ giữ đủ 3 bước liên tiếp.

**Xác nhận bằng thực nghiệm dứt điểm**: tắt hẳn cap % cũ (`grasp_close_freeze_at_progress=1.0`) → `max_gc` chỉ còn `[0.0, 1.0]` (cơ chế lực chưa từng kích hoạt được, kể cả khi có toàn quyền) — NHƯNG `success_rate` không đổi. Đóng 76% hay đóng 100% cho kết quả giống hệt nhau. **Cơ chế đóng kẹp — dù tinh vi tới đâu — không phải nguyên nhân của 29/30 thất bại còn lại.**

**Quyết định**: giữ nguyên `grasp_close_freeze_at_progress=0.75` (đơn giản, đã biết đủ tốt). Giữ lại toàn bộ mã lực-thật (`_grip_pressing` bắt buộc, `_press_hold_steps`, `grasp_press_freeze_stall_m/span_margin_m`, ramp lực nhấc `grasp_lift_onset_ramp_steps`) làm defensive backstop — vô hại khi cap % đã đủ chặt, và hữu ích nếu sau này cap % bị nới lỏng vì lý do khác. Instrumentation debug (`DEBUG_LIFT`/`DEBUG_STALL`/`DEBUG_FREEZE`) giữ lại, gated bằng env var, không ảnh hưởng chạy bình thường.

**Kết quả ổn định cuối cùng của toàn bộ chuỗi sửa lỗi mimic + grip_secure**:
```
success=0.0333  grasp=0.83  latch=0.43-0.50  lift_start=0.40  lift_cmd_steps≈160
fail_modes: lift_too_low≈10-11, grasp≈11-12, reach=5, tilt≈1-2, no_lift_command=1
```

**Hướng điều tra tiếp theo (chưa làm)**: vì `success_rate` bất biến với MỌI thay đổi ở đóng kẹp, và LUÔN CÙNG env0 thành công (seed cố định) — nghi vấn chuyển sang **vị trí/góc tiếp cận trước khi đóng** (điều kiện reset ngẫu nhiên của env0 tình cờ thuận lợi hơn — ví dụ độ cao ngón so với thân chai, góc top-down, hoặc lệch ngang tại thời điểm bắt đầu đóng). Cần so sánh trực tiếp state (z_error_finger, top_down_align, lateral_finger_xy, symmetric) của env0 lúc latch với một env thất bại điển hình (env1) tại CÙNG thời điểm, để xác định biến số nào khác biệt.

---

## Phase 7 — Train 5M bước plateau ở success≈3-5%, user không tin vật lý đã đúng → tìm ra root cause thật (actuator group thiếu joint2)

Sau train 5M bước thật (không phải chỉ eval), `latch_rate` cải thiện thật (43-50%→60-65%) nhưng `success_rate` đứng yên ở ~3-5% bất kể train thêm — dấu hiệu mạnh của giới hạn VẬT LÝ chứ không phải thiếu training. Đề xuất "chấp nhận đây là trần vật lý, dừng lại" bị user bác bỏ dứt khoát: *"help me check in the urdf file I still think gripper physic wrong some how / I don't believe that I can't lift a bottle"*. Tiếp tục đào sâu URDF/USD theo đúng yêu cầu này.

### Loại trừ các giả thuyết sai (theo thứ tự)
1. **Backend Newton vs PhysX**: `checkbackend.py` xác nhận local đang chạy PhysX thật (Isaac Lab 3.0 chọn theo `FactoryBase._get_backend()` — kiểm tra physics manager sống, không mặc định Newton nếu sim context đã tồn tại). Không phải nguyên nhân.
2. **`referenceJointAxis="rotX"/"rotY"` trên khớp prismatic**: kiểm tra schema `PhysxMimicJointAPI` xác nhận field này **bị NGƠ LƠ hoàn toàn** với khớp 1-DOF (Revolute/Prismatic) — chỉ là token đặt tên tuỳ ý, không mang ý nghĩa trục thật. Không phải bug.
3. **`finger_joint2` Effort Limit = 0.000`** (thấy trong bảng "Simulation Joint Information" lúc load env, `checkmass.py`): giả thuyết là PhysX giới hạn lực khớp này = 0 bất kể mimic chỉnh gì. Vá bằng `patch_gripper_mimic2.py` — thêm `UsdPhysics.DriveAPI` (linear, `type=force, maxForce=333, stiffness=0, damping=0`) vào `finger_joint2` cả 2 tay, lưu thẳng vào `v10.usd` (backup `v10.usd.before_drive_fix.bak`). Đo lại bằng `loadtest.py` (ramp đóng 80 bước thực tế, CHỈ set target cho joint1) → **KHÔNG cải thiện đáng kể** (lệch đỉnh ~15mm so với ~17-19mm trước vá, lệch cuối 199 bước gần như y hệt ~10mm). Giả thuyết effort-limit=0 chỉ đúng một phần, không đủ giải thích.

### Phát hiện quyết định: `loadtest.py` có lỗi phương pháp — không tái tạo đúng cách code thật điều khiển gripper
Đọc `actions.py:521-580` (code SẢN XUẤT thật, dùng trong train/eval): mỗi bước ghi target **tường minh cho CẢ 2 khớp cùng lúc** — `finger_targets = finger_pos.repeat(1, 2)` rồi `set_joint_position_target(finger_targets, joint_ids=self._gripper_joint_ids)` (gồm cả joint1 VÀ joint2). `loadtest.py` chỉ set target cho joint1, để joint2 "trôi" theo target cũ (=vị trí hiện tại) — không giống thật.

Viết `fingerpos.py` để kiểm tra world position 2 ngón ở joint=0.044 (open) và joint=0.0 (closed) khi set target tường minh cho CẢ 2 khớp → hội tụ gần như hoàn hảo (lệch <10 micromet) — **trái ngược hẳn** với `loadtest.py`. Xác nhận: vấn đề không nằm ở tốc độ đóng ramp.

Viết `loadtest2.py` — tái tạo ĐÚNG cách sản xuất (cả 2 khớp có target tường minh mỗi bước) NHƯNG thêm bài test tải trọng: đóng tới điểm "coi như chạm chai" (target=13mm ≈ gc=0.71), giữ ổn định, rồi đẩy NGƯỢC cả 2 khớp +5mm đều nhau (giả lập phản lực chai ép ngược) — đo khớp nào phục hồi mạnh hơn.

**Kết quả TRƯỚC fix** (joint2 chỉ là mimic-follower thuần, không có actuator riêng):
```
Ổn định (không tải): lệch 4.075mm (không phải 0 dù cả 2 có cùng target!)
Dưới tải +5mm: j1 phục hồi 4.0mm, j2 CHỈ phục hồi 2.3mm (yếu hơn 43%)
Sau 60 bước: lệch cuối = 5.797mm — KHÔNG bao giờ hội tụ lại dưới tải
```
→ Đây chính là hiện tượng user báo cáo từ đầu ("tay đi lệch 1 bên") — nhưng xảy ra dưới TẢI (khi chạm/kẹp chai), không phải lúc đóng trong không khí.

### Root cause thật: `finger_joint2` KHÔNG NẰM TRONG BẤT KỲ ActuatorCfg NÀO
`config.py:154-158` (code TỪ TRƯỚC segment này, không phải bug mới):
```python
"gripper": ImplicitActuatorCfg(
    joint_names_expr=["openarm_left_finger_joint1"],   # ← CHỈ joint1!
    stiffness=1500.0, damping=60.0,
),
```
Comment cũ giải thích đây là **cố ý**: "Driving j2 with its own implicit actuator FIGHTS the mimic constraint → j1 slams shut instantly while j2 lags" — quyết định này được đưa ra khi mimic còn RẤT MỀM (`naturalFrequency=25, dampingRatio=0.005`, đã vá ở đầu session này thành `200/1.0`). Lý do cũ (mimic quá yếu nên actuator riêng lấn át, gây "đánh nhau") **không còn đúng nữa** sau khi mimic đã được vá cứng/tắt dao động (critically damped).

**Fix**: thêm `openarm_left_finger_joint2` vào cùng `joint_names_expr` với joint1, cho nó PD gain thật giống hệt joint1 (`stiffness=1500, damping=60`) thay vì để nó dựa hoàn toàn vào ràng buộc mimic (vốn yếu hơn actuator PD thật, nhất là dưới tải).

**Kết quả SAU fix** (đo lại bằng đúng `loadtest2.py`):
```
Ổn định (không tải): lệch 0.166mm (giảm 24 lần so với 4.075mm)
Dưới tải +5mm: j1 phục hồi 8.49mm, j2 phục hồi 8.62mm — GẦN NHƯ GIỐNG HỆT
Sau 60 bước: lệch cuối = 0.032mm (giảm 180 lần so với 5.797mm) — hội tụ sạch
```
**KHÔNG có hiện tượng "j1 slam / j2 lag"** như comment cũ lo ngại — vì mimic giờ đủ cứng để 2 actuator PD độc lập CỘNG HƯỞNG thay vì đánh nhau.

### File đã sửa
- `isaaclab_openarm_env/config.py:154-169` — thêm `"openarm_left_finger_joint2"` vào `joint_names_expr` của actuator `"gripper"`, viết lại comment giải thích lý do + số đo thực nghiệm.
- (Không cần revert `patch_gripper_mimic2.py`'s raw USD drive trên joint2 — `ImplicitActuatorCfg` sẽ ghi đè stiffness/damping/effort_limit của nó ở runtime cho các khớp nằm trong `joint_names_expr`, nên bản vá USD lần 2 giờ vô hại/thừa nhưng không gây hại.)

### Thử fix #1 (SAI, đã revert): thêm joint2 vào actuator group nhưng KHÔNG sửa gripper_action
Chỉ thêm `openarm_left_finger_joint2` vào `ImplicitActuatorCfg` (giữ nguyên `gripper_action` chỉ ghi target joint1) → PD mới của joint2 chống lại chính vận tốc mimic kéo nó (vì target luôn tự tham chiếu = vị trí hiện tại → damping=60 chặn mọi chuyển động). Xác nhận REGRESSION bằng 2 cách:
- `loadtest.py` (chỉ set target joint1, đúng cách sản xuất cũ): joint2 kẹt cứng ở ~7.5mm, không bao giờ đóng tiếp qua bước ~140 (trong khi joint1 tiếp tục đóng gần về 0).
- `eval_lift_metrics.py` thật: `lift_start_rate` tụt từ 0.40 → **0.08**, `grasp_rate` 0.83→0.76. Revert ngay.

### Fix #2 (ĐÚNG — đối chiếu repo tham khảo chính thức enactic/openarm_isaac_lab)
User gợi ý kiểm tra repo GitHub chính thức của OpenArm (`github.com/enactic/openarm_isaac_lab`) — hoá ra `isaaclab_openarm_env` này vốn được fork/dựa trên chính repo đó (`grasp_assist.py` vẫn còn header "Copyright 2026 Enactic, Inc."). Trong `source/.../assets/openarm_unimanual.py` của họ:
```python
"openarm_gripper": ImplicitActuatorCfg(
    joint_names_expr=["openarm_finger_joint.*"],   # CẢ 2 khớp
    stiffness=2e3, damping=1e2, effort_limit_sim=333.33,
),
```
và trong `unimanual/lift/config/joint_pos_env_cfg.py`:
```python
self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
    joint_names=["openarm_finger_joint.*"],  # CẢ 2 khớp, class CHUẨN không custom
    open_command_expr={"openarm_finger_joint.*": 0.044},
    close_command_expr={"openarm_finger_joint.*": 0.0},
)
```
→ Họ ghi **target tường minh cho CẢ 2 khớp mỗi bước** (không để khớp nào "trôi" theo mimic thuần tuý), KẾT HỢP với actuator PD thật cho cả 2. Đây là 2 mảnh ghép phải đi cùng nhau — thiếu 1 trong 2 (như fix #1) sẽ gây đúng hiện tượng "đánh nhau" mà comment cũ trong code lo ngại.

**Áp dụng cả 2 thay đổi cùng lúc** (`config.py`):
1. `"gripper"` `ImplicitActuatorCfg`: `joint_names_expr=["openarm_left_finger_joint1", "openarm_left_finger_joint2"]`, `stiffness=2000.0, damping=100.0` (khớp gain tham khảo).
2. `AssistedBinaryGripperActionCfg`: `joint_names=["openarm_left_finger_joint1", "openarm_left_finger_joint2"]`, `open_command_expr`/`close_command_expr` map CÙNG giá trị cho cả 2 tên khớp → `AssistedBinaryGripperAction.apply_actions()` (không cần sửa code, chỉ cần cfg) tự nhiên ghi `targets` giống hệt nhau cho cả 2 khớp mỗi bước vì `self._open_command`/`self._close_command` giờ có 2 phần tử bằng nhau.

Xác nhận `raw_actions[:,0]` (đọc ở `rewards.py:378`) không đổi shape — action space vẫn 1 chiều nhị phân (open/close), chỉ khác SỐ KHỚP nhận lệnh đó → **không phá checkpoint cũ** (đã tự xác nhận qua việc `policy_5M_good.pt` load và chạy được ngay).

### KẾT QUẢ EVAL THẬT (50 episode, `policy_5M_good.pt`, CHƯA train lại trên dynamics mới)
```
                    TRƯỚC (baseline)   SAU (chỉ sửa physics)
grasp_rate          0.83               0.86
latch_rate          0.43-0.50          0.56
lift_start_rate     0.40               0.50
p50_max_lift_m      +0.0000            +0.0016   ← lần đầu tiên > 0!
success_rate        0.033              0.04
fail_modes chính    grasp/no_lift_cmd  lift_too_low (23/50 — kẹp được, chạm được, nhưng lực giữ chưa đủ suốt 30mm)
```
**Ý nghĩa**: bottleneck đã dịch chuyển hẳn từ "không kẹp/không dám nhấc" sang "kẹp được nhưng lực chưa đủ bền suốt hành trình nhấc" — đúng loại vấn đề mà kế hoạch Giai đoạn 1 (S1.3 — sửa lực nhấc, RC5) đã dự đoán từ đầu. Cải thiện này đạt được THUẦN TUÝ từ sửa vật lý, chưa train lại — nghĩa là train tiếp trên dynamics mới (giờ đối xứng thật) nhiều khả năng sẽ đẩy các số này lên cao hơn nữa vì policy chưa từng thấy gripper hoạt động đúng.

### File đã sửa (fix cuối cùng, đang active)
`isaaclab_openarm_env/config.py`: actuator `"gripper"` (dòng ~145-169) và `AssistedBinaryGripperActionCfg` (`gripper_action=`, dòng ~264-282).

Script chẩn đoán mới (trong scratchpad, không phải trong repo): `checkbackend.py`, `findprim.py`, `checkmass.py`, `loadtest.py` (chỉ set target 1 khớp — hoá ra ĐÚNG với sản xuất TRƯỚC fix, SAI sau fix), `fingerpos.py` (đo world position 2 đầu ngón, set cả 2 target), `loadtest2.py` (bài test tải trọng dual-target — ĐÚNG với sản xuất SAU fix, dùng để verify).

### Hướng tiếp theo (chưa làm)
Train tiếp (fine-tune từ `policy_5M_good.pt` hoặc train mới) trên dynamics đã sửa, rồi đo lại `lift_too_low` có giảm không — đây mới là bottleneck thật còn lại (lực/thời lượng giữ kẹp trong lúc nhấc, không phải cơ chế đóng/logic trạng thái).

---

## Phase 8 — Fine-tune 5M bước SẬP: tìm ra bug reward decay bị neo sai mốc (2026-09-06)

### Fine-tune thất bại: latch_rate/lift_start_rate sập về 0 vĩnh viễn, KHÔNG hồi phục

Chạy `train_gripper_symfix_v1` (5M bước, `--checkpoint policy_5M_good.pt --lr-start 1e-4 --lr-end 1e-5 --clip-range 0.1 --ent-coef 0.002 --task_phase 2 --stage all --assist-schedule`, tmux+systemd-inhibit). Diễn biến thật (trích theo `total_timesteps`):

```
ts=169K   grasp=0.99  latch=0.50  lift_start=0.45  rew=42
ts=1.66M  grasp=1.00  latch=0.09  lift_start=0.08  rew=227
ts=1.99M  grasp=0.62  latch=0.00  lift_start=0.00  rew=207
ts=5.00M  grasp=0.58  latch=0.00  lift_start=0.00  rew=163
```
`best_policy.pt` hoá ra chỉ là snapshot CUỐI CÙNG (không có `EvalCallback` trong callback list → nhánh `best_model.zip` không tồn tại → code rơi vào else "Saved final policy weights"), tức là checkpoint tệ nhất trong toàn bộ run. Checkpoint 1M bước (`rl_model_1000000_steps.zip`) cũng không cứu được (`grasp=0.60, latch=0.36, lift_start=0.36` — đã suy giảm ngay từ đây, `reach` fail mode tăng vọt 20/50).

### Chẩn đoán: nhìn kỹ theo từng update (không chỉ theo mốc lớn)

Trong cửa sổ 1.4M→2.0M: `latch_rate` giảm ĐỀU 0.28→0.00 suốt ~500K bước trong khi `grasp_rate` giữ nguyên 1.0 (chỉ sập SAU khi latch đã về 0) và **`ep_rew_mean` TĂNG 202→228**. Reward tăng trong khi năng lực giảm = dấu hiệu kinh điển của reward-hacking/lệch động lực (không phải catastrophic forgetting ngẫu nhiên như đoán ban đầu).

### Root cause: `grasp_camp_decay_steps=200` neo vào SAI mốc thời gian

`_compute_grasp_reward` (rewards.py) suy giảm `static_shaping` (= `r_align+r_descend+r_hover+r_grip_close+r_grip_cmd+r_track+r_orient` — TOÀN BỘ phần thưởng định hướng/vị trí) về 0 tuyến tính theo `_steps_in_grasp` (số bước kể từ lúc VÀO stage GRASP), `decay_n=200`, KHÔNG có thời gian ân hạn (onset=0, khác với REACH decay có onset=200). Vì "lift" chỉ là sub-mode của STAGE_GRASP (không phải stage riêng), `_steps_in_grasp` **tiếp tục tăng xuyên suốt cả quá trình nhấc**.

Đo THẬT bằng debug mới (`DEBUG_DECAY=1`, thêm tạm vào `_compute_grasp_reward`): latch thật xảy ra ở `steps_in_grasp = 136-371` (mẫu nhỏ, 4 lần latch trong 15 episode) — nghĩa là **decay đã gần hoặc bằng 0 NGAY TẠI thời điểm latch** (0.0-0.32/1.0), TRƯỚC CẢ KHI quá trình nhấc bắt đầu. Toàn bộ ~150-190 bước tiếp theo (giữ lệnh nhấc, chờ xác định thành công/fail) diễn ra với `static_shaping=0` hoàn toàn — không còn tín hiệu thưởng nào cho việc giữ vị trí/hướng đúng đúng lúc quan trọng nhất, trong khi `vel_penalty`/`table_penalty` vẫn âm liên tục. Vì `lift_too_low` là fail mode chính (chai thường không nhấc đủ), commit trọn vẹn tới latch+thử nhấc mang lại giá trị kỳ vọng gần 0 hoặc âm — PPO dần học cách KHÔNG commit hết (latch_rate trôi dần xuống 0) vì "dở dang" giờ lời hơn "đi trọn".

**Đây là bug khác về BẢN CHẤT so với 2 bug đã sửa trước đó** (gripper mimic damping, actuator thiếu joint2) — bug trước là VẬT LÝ (không kẹp/nhấc được), bug này là REWARD (kẹp/nhấc được nhưng bị dạy là không nên làm). Bug này chỉ LỘ RA sau khi 2 bug vật lý được sửa — trước đó latch/lift gần như không bao giờ xảy ra nên khoảng thời gian sau-latch không đủ dài để decay kịp gây hại.

So khớp với đúng ý định gốc ghi trong comment cũ (dòng ~434, đã có TỪ TRƯỚC): "phần thưởng...được trả MÃI MÃI **SAU KHI ĐÃ KẸP XONG**" — ý định là suy giảm áp dụng cho thời gian đứng yên SAU LATCH, nhưng code lại lỡ neo vào thời gian kể từ lúc VÀO GRASP (bao gồm cả thời gian tiếp cận/đóng kẹp cần thiết, vốn KHÔNG nên bị phạt).

### Fix: thêm bộ đếm `_steps_since_latch`, neo decay vào đó thay vì `_steps_in_grasp`

- `helpers.py`: khởi tạo `env._steps_since_latch = torch.zeros(...)`.
- `terminations.py`: reset `env._steps_since_latch[env_ids] = 0` khi episode reset.
- `rewards.py::_update_contact_and_stages`: mỗi bước, đọc `gripper_action` term's `_grasp_latched` (bộ tích luỹ OR, chỉ False khi reopen thật sự xảy ra — có `grasp_reopen_max_count` giới hạn số lần, nên KHÔNG dao động nhanh như trạng thái tức thời, an toàn để neo trực tiếp mà không tái tạo lỗ hổng "dao động biên" đã gặp ở v1/v2 lịch sử): `_steps_since_latch[latched] += 1; [~latched] = 0`.
- `rewards.py::_compute_grasp_reward`: đổi `grasp_steps = env._steps_in_grasp.float()` → `env._steps_since_latch.float()`. Giữ nguyên `decay_n=200`.

**Xác nhận bằng DEBUG_DECAY sau fix**: latch giờ luôn xảy ra ở `steps_since_latch=1` (đúng logic — vừa latch xong), `decay=0.995` (≈1.0, full reward) — thay vì 0.0-0.32 như trước. Toàn bộ thời gian tiếp cận/đóng kẹp giờ KHÔNG bị decay (vì `_steps_since_latch=0` suốt khi chưa latch), decay chỉ bắt đầu đếm THẬT SỰ từ lúc latch — đúng ý định gốc.

### File đã sửa
`isaaclab_openarm_env/mdp/rewards.py` (thêm `import os`, bộ đếm `_steps_since_latch` trong `_update_contact_and_stages`, đổi anchor decay trong `_compute_grasp_reward`, debug instrumentation `DEBUG_DECAY` gated — giữ lại, hữu ích cho lần sau), `isaaclab_openarm_env/mdp/helpers.py` (khởi tạo buffer), `isaaclab_openarm_env/mdp/terminations.py` (reset buffer).

### Chưa làm — cần train lại để xác nhận fix có giải quyết được collapse không
Chưa chạy lại 5M bước với fix này. Kỳ vọng: `latch_rate` không còn trôi dần về 0 trong lúc train, vì tín hiệu thưởng cho việc "đi trọn tới latch+nhấc" giờ còn nguyên vẹn trong ít nhất 200 bước ĐẦU TIÊN sau latch (đủ phủ hết khoảng ~150-190 bước nhấc thật đã đo được), thay vì đã chết từ trước khi latch xảy ra.

---

## Phase 9 — Train song song local + server 128 env, XÁC NHẬN fix decay + tìm ra checkpoint tốt nhất từ trước tới giờ (success=57%)

### Setup: bật WireGuard, deploy code, train song song
User yêu cầu chuyển sang train trên server 4090 vì local quá chậm. Bật WireGuard (`resolvectl domain <if> ""` sau đó để sửa lỗi DNS toàn cục bị chiếm bởi domain `~.` của interface VPN — xem ghi chú riêng, không liên quan RL). Deploy code mới (`./rl.sh deploy`) — xác nhận sync đủ `config.py, actions.py, grasp_assist.py, helpers.py, rewards.py, terminations.py` + `v10.usd` đã vá. GPU 4090 lúc đó đang bận job khác (16.5GB/24.5GB, 92%) nên chọn `--num-envs 128` (đo được: chỉ tốn 2.9GB, an toàn).

Chạy đồng thời 2 job cùng code/checkpoint gốc (`policy_5M_good.pt`) + cùng hyperparam (`--lr-start 1e-4 --lr-end 1e-5 --clip-range 0.1 --ent-coef 0.002 --task_phase 2 --stage all --assist-schedule`):
- Local: 16 env, 2.5M bước (`verify_decayfix_v1`)
- Server: 128 env, 5M bước (`train_gripper_decayfix_server1`, ~936-1164 fps — nhanh hơn local ~2x dù chạy chung GPU với job khác)

### Kết quả PHA 1 (assist còn > 0): fix decay hoạt động đúng, XÁC NHẬN bằng số
Cả 2 run: KHÔNG còn hiện tượng `latch_rate` trôi dần một chiều về 0 kiểu cũ (bug đã sửa). Server đạt đỉnh `success=0.36-0.60`, `latch` lên tới 0.51-1.0, `mean_max_lift_m` thường xuyên 2-5cm — cao hơn hẳn baseline trước fix (0.03-0.06).

### Phát hiện MỚI: collapse muộn hơn — nhưng nguyên nhân khác hẳn, không phải bug reward
Cả 2 run đều sập về `success=0, lift_start_rate=0.000` tuyệt đối kéo dài hàng triệu bước — nhưng lần này thời điểm sập **trùng khít chính xác** với lúc `--assist-schedule` giảm scale về hẳn 0:
```
Server (total=5M, anneal_frac=0.4): [Assist] scale=0.000 @ 2,000,000 steps  ← collapse bắt đầu ngay đây
Local  (total=2.5M, anneal_frac=0.4): [Assist] scale=0.000 @ 1,000,000 steps ← collapse bắt đầu ngay đây
```
Đây KHÔNG phải bug mới — là giới hạn năng lực đã biết từ trước (policy chưa đủ khả năng tự chủ hoàn toàn khi mất hỗ trợ kịch bản, đúng rủi ro đã ghi trong kế hoạch Giai đoạn 2 / đúng hiện tượng đã gặp ở `verify_hoverdamp` trước đây trong session này). Khác với bug decay (đã sửa): bug đó gây sập NGAY CẢ KHI còn assist; đây chỉ sập SAU KHI assist về 0.

### Quyết định thực dụng: vì eval luôn chạy ở `--assist-scale 1.0`, lấy checkpoint TRƯỚC lúc assist về 0
Vì cách đánh giá/dùng thực tế bấy lâu nay đều ở `assist_scale=1.0`, không cần đợi cả 5M bước — checkpoint giữa chừng (lúc assist còn cao, đã học được hành vi tốt) mới là artifact hữu ích, bất kể đuôi run có sập sau đó.

**Eval `rl_model_999936_steps.zip` (server, ~1M bước, assist lúc đó ~50%) ở `--assist-scale 1.0`, 30 episode:**
```
success=0.57  grasp=0.90  latch=0.67  lift_start=0.67  p50_lift=+3.06cm  lift_cmd_steps=41.7
fail_modes: reach=3, grasp=6, hold_unstable=2, lift_too_low=1, tilt=1  (phân bố đều, không còn 1 nút thắt áp đảo)
```
**ĐÂY LÀ KẾT QUẢ TỐT NHẤT TRONG TOÀN BỘ QUÁ TRÌNH ĐIỀU TRA** (so với baseline 0.033-0.06 trước đó — cải thiện ~10-17 lần). Nhiều episode đạt `lift_hold=5/5`, `lift_m≈0.030-0.032m`, tilt hợp lý (~4-7°).

### Đã lưu checkpoint này an toàn
Trích xuất state_dict từ `.zip` (qua `PPO.load(...).policy.state_dict()`) → lưu thành `policy_1M_success57.pt`, tải về local tại `Reinforce_Learning/logs/train/policy_1M_success57.pt`. **Đây là checkpoint nên dùng làm baseline mới cho mọi việc tiếp theo** (demo, eval, fine-tune tiếp), thay cho `policy_5M_good.pt` cũ.

### Sự cố nhỏ cuối run server (không ảnh hưởng)
Server training chạy hết ~5.0M bước (log cuối cùng: `total_timesteps=5,005,312, success=0, latch=0.025` — khớp đúng pattern collapse-sau-assist=0 đã biết), sau đó Kit/Isaac Sim crash lúc dọn dẹp/lưu cuối (log cho thấy 1 phiên Kit mới tự khởi động, kèm dòng "[previous crash] preventing upload of minidump" — crash-recovery tự động của Omniverse Kit, không phải lỗi trong code RL). Không mất gì quan trọng vì checkpoint tốt nhất đã được trích xuất riêng TRƯỚC đó.

### Hướng tiếp theo (chưa làm)
1. Dùng `policy_1M_success57.pt` làm checkpoint chính, chạy demo/eval kỹ hơn (100 episode, nhiều seed) để xác nhận 57% không phải may mắn thống kê.
2. Vấn đề "policy không tự chủ được khi assist=0" vẫn CHƯA giải quyết — đây là công việc Giai đoạn 2 riêng (residual action, xem kế hoạch cũ S2.3), không nằm trong phạm vi 2 fix đã làm hôm nay (vật lý gripper + reward decay). Có thể thử: giữ `assist` không về hẳn 0 (đặt `grasp_assist_blend_end` > 0, ví dụ 0.2-0.3) thay vì ép policy tự chủ 100% ngay, hoặc tăng `anneal_frac` để quá trình chuyển giao chậm hơn/êm hơn.
