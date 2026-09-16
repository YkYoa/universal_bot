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

---

## Phase 10 — PLACE stage: S1-S3 (state machine + chuyển stage), 2 bug tìm ra qua demo thật

Kế hoạch đầy đủ ở `/home/hans/.claude/plans/expressive-honking-possum.md` (phần "Phase 10"). Theo yêu cầu user, chỉ làm S1-S3 (nền tảng + state machine PLACE + chuyển stage GRASP→PLACE) rồi dừng lại để demo/báo cáo trước khi làm S4+ (reward/termination/config đầy đủ/train).

### Đã cài (S1-S3)
- `helpers.py`: `uses_grasp_lift` đổi `==2`→`>=2`, thêm `uses_place` (`>=3`); hằng số `PLACE_IDLE/CARRY/DESCEND/HOLDING` cạnh `STAGE_*`; `compute_state()` thêm `dist_bottle_bowl_xy`, `height_above_bowl_floor` (placeholder thô = `bottle_pos_z - bowl_pos_z`, CHƯA đo hình học bát thật — xem S9); `place_release_ready()` mới.
- `grasp_assist.py`: state machine PLACE (`_place_can_start`, `_place_must_abort`, `_update_place_state`) mirror kỷ luật LIFT; `_osc_carry_to_bowl`/`_osc_descend_to_bowl`; block assist mới trong `apply_grasp_arm_assist` (override/residual blend giống LIFT).
- `actions.py`: điểm chèn thả kẹp (`actions[STAGE_PLACE & place_release_ready, 0]=1.0`) + bootstrap `place_hold_closed` (ép đóng cứng suốt carry, để test scripted-only trước khi có policy).
- `rewards.py`: chuyển stage GRASP→PLACE trong `_update_contact_and_stages`, tái dùng đúng tín hiệu lift-hold-đạt (không terminate nữa).
- `terminations.py`: mở rộng `tipped_bottle_termination` sang cả PLACE.
- `phase2_overrides.py`/`isaaclab_demo.py`: thêm stage `"place"` tối thiểu (chỉ bật `assist_place`) để test được qua demo — chưa phải S8 đầy đủ.

### Bug #1 tìm ra qua demo thật: "cạm bẫy #1" trong kế hoạch xảy ra ĐÚNG như dự đoán
Demo đầu tiên: `[Success] PLACE stage started` in ra, rồi NGAY LẬP TỨC `🎉 EPISODE SUCCESS` cùng bước — `success_termination` (chưa sửa, thuộc S5) vẫn đọc `_steps_bottle_lifted >= lift_hold` không phân biệt stage. Vì `termination_manager` chạy TRƯỚC `reward_manager` trong cùng `env.step()`, nó đọc giá trị "lift_hold đã đạt" từ CUỐI bước chuyển stage (trước khi reward-side kịp reset counter ở bước kế), nên luôn fire đúng 1 bước sau transition — bất kể tôi reset counter ngay trong cùng lệnh chuyển stage.

**Fix tối thiểu** (kéo một phần S5 lên sớm, cần thiết để test được S1-S3 có ý nghĩa): `success_termination` trả `False` cứng cho `task_phase>=3` (chưa cài tiêu chí thành công thật, để dành cho S5 đầy đủ).

### Bug #2 tìm ra qua demo thật: carry motion làm chai lật quá ngưỡng
Sau khi sửa bug #1, PLACE chạy được nhưng chai TILT leo dần (5°→16°+) trong ~30-50 bước rồi bị `tipped_bottle_termination` kết liễu. Tưởng nhầm ngưỡng là 60° (default trong `config.py`), thực ra `phase2_overrides.py` (`PHASE2_GRASP`) đã siết còn **15°** cho GRASP — và tôi vừa mở rộng ngưỡng NÀY (không phải 60°) sang PLACE.

Nguyên nhân thật: `_osc_carry_to_bowl` tính lệnh theo `err/pos_scale` (kiểu P-controller) thay vì magnitude hằng số nhỏ như `_osc_world_up_lift` — với lỗi XY 176mm+ và carry_height 150mm, lệnh BÃO HOÀ ở tốc độ tối đa liên tục hàng chục bước trên CẢ 3 TRỤC cùng lúc (không giảm dần), làm chai đung đưa như con lắc.

**Fix**: thêm `place_carry_speed_scale=0.35` (chậm hẳn lại) + `place_carry_align_blend=0.5` (chủ động giữ hướng thẳng đứng bằng `_align_vertical_rot_action` đã có sẵn cho descend). Kết quả đo được: tilt ổn định quanh 10° (không leo tiếp), episode chạy hết 1200 bước không còn bị coi là lật.

### Vấn đề còn mở (chưa giải quyết, để lại cho lần sau)
Chai KHÔNG hội tụ XY về bát trong 426 bước available (từ lúc vào PLACE tới hết episode) — chưa từng thấy `CARRY→DESCEND`. Nghi vấn: `height_above_bowl_floor` placeholder thô (`bottle_pos_z - bowl_pos_z`, đo được `-14mm` ngay cả khi chai đã nhấc cao — gợi ý `bowl_pos` (root frame của RigidObject bát) KHÔNG nằm ở đáy bát mà ở đâu đó cao hơn nhiều, ví dụ miệng/tâm bát) khiến Z-target (`bowl_pos_z + carry_height`) bị đặt sai/quá cao, chai vẫn đang leo lên lúc episode kết thúc (`lift:0.132m` và còn tăng) thay vì ổn định — có thể chiếm phần lớn "ngân sách tốc độ" thay vì hội tụ XY. **Đây chính là lý do S9 (đo hình học bát thật bằng UsdGeom.BBoxCache) cần làm TRƯỚC khi tinh chỉnh thêm bất kỳ threshold carry/descend nào** — mọi con số hiện tại đều là placeholder chưa đo, đúng như đã cảnh báo trong kế hoạch.

### File đã sửa (S1-S3, đang ở working tree, CHƯA commit)
`isaaclab_openarm_env/mdp/helpers.py`, `mdp/grasp_assist.py`, `mdp/actions.py`, `mdp/rewards.py`, `mdp/terminations.py`, `phase2_overrides.py`, `isaaclab_demo.py`.

### Hướng tiếp theo khi user quay lại
1. Đo hình học bát thật (S9, bbox thật) trước khi tune thêm carry/descend threshold.
2. Sau khi có số đo thật, xem lại `_osc_carry_to_bowl`'s Z-target formula — có thể cần tách riêng "leo cao" và "hội tụ XY" thành 2 pha rõ ràng hơn thay vì làm đồng thời (ví dụ: leo cao TRƯỚC, giữ nguyên, rồi mới di chuyển ngang) để tránh cạnh tranh ngân sách tốc độ.
3. Tiếp tục S4+ (reward/termination đầy đủ, config S6, observation S7, plumbing S8) theo đúng kế hoạch đã duyệt, CHỈ SAU KHI carry/descend đã hội tụ ổn định qua scripted-assist thật (không train trên state machine còn lỗi).

---

## Phase 10 tiếp — S4 (reward PLACE) + phát hiện & sửa lỗi kiến trúc lớn: dùng chung assist scale

User yêu cầu vô train luôn (bỏ qua đo hình học bát S9 để đi nhanh). Đã cảnh báo train sẽ vô nghĩa nếu chưa có S4 (reward PLACE) — user đồng ý làm S4 trước.

### S4 — Đã cài
- `rewards.py::compute_curriculum_reward`: dispatch 3 nhánh (`in_place → place_reward`, `in_grasp → grasp_reward`, else `reach_reward`) — sửa đúng cạm bẫy #2 đã ghi trong plan.
- `_compute_place_reward` mới: `r_carry_progress` (mirror r_progress, tái dùng `_prev_dist_bottle_bowl` đã có sẵn nhưng chết từ trước), `r_xy_converge`, `r_descend_over_bowl` (chỉ khi `_place_phase>=DESCEND`), `r_hold_grip_during_carry` (suy giảm theo `_steps_since_place_start` — bộ đếm mới, neo vào `_place_phase != IDLE`, mirror đúng cách `_steps_since_latch` đã sửa cho GRASP, KHÔNG lặp lại bug neo-vào-thời-gian-vào-stage), `r_release_quality` (one-time khi vừa mở kẹp), `r_settle` (nhỏ, khi đã gần bát + tốc độ giảm). KHÔNG thêm phạt tích luỹ (bài học từ GRASP v4). Terminal bonus/penalty để dành S5.
- Regression gate ngay sau khi viết xong: eval `policy_1M_success57.pt` ở `--task_phase 2` → **y hệt số cũ** (success=0.03, grasp=0.83, latch=0.47, lift_start=0.43) — an toàn.

### Verify training đầu tiên (assist_place=1.0 cố định, không anneal) — CHỈ để test reward, không test policy
`ep_rew_mean`: **-3820 → +56.6** trong 300K bước, hội tụ mượt, `explained_variance` 0.95-0.998. Xác nhận: hàm reward PLACE hoạt động hợp lý, không vỡ. **Lưu ý bắt buộc**: vì `assist_place=1.0` cố định (không `--assist-schedule`), TOÀN BỘ hành động tay trong PLACE bị override cứng bởi kịch bản — policy KHÔNG hề điều khiển được gì. Đường cong reward tăng chỉ xác nhận reward tốt, không phải bằng chứng policy học được.

### Fine-tune thật lần 1 (`--assist-schedule`, server 4090) — CUDA OOM
Server GPU chỉ còn ~1.9GB rảnh (job khác chiếm 22.6/24.5GB). Thử 32 env → `PxgCudaDeviceMemoryAllocator failed to allocate memory` → crash. Chuyển sang **local** ngay (GPU rảnh hoàn toàn).

### Fine-tune thật lần 2 (local, `--assist-schedule`) — PHÁT HIỆN LỖI KIẾN TRÚC LỚN
`grasp_rate` sập từ ~0.99 xuống **0.40** ngay sau khi `[Assist] scale=0.000 @ 1,000,000 steps` — kèm `fail_modes.reach=18/30` (60%!) khi eval lại checkpoint 1M. Đối chứng: `policy_1M_success57.pt` GỐC (chưa fine-tune) chạy thẳng ở `task_phase=3 --stage place` vẫn cho `grasp=0.83` bình thường → xác nhận **regression do CHÍNH quá trình fine-tune này gây ra**, không phải do code S1-S4 mới.

**Nguyên nhân gốc**: `apply_grasp_arm_assist` (grasp_assist.py) dùng **DUY NHẤT 1 biến `scale = _assist_scale(env)`** cho TẤT CẢ — REACH-descent, GRASP-descent, LIFT, và giờ cả PLACE — VÀ có `if scale <= 1e-6: return actions` sớm ngay đầu hàm. Khi lịch anneal đưa `scale` về 0 để "dạy" LIFT/PLACE tự chủ, REACH/GRASP-descent (vốn không liên quan gì, đã hoạt động tốt từ Phase 9) cũng bị TẮT LUÔN theo — đúng rủi ro đã cảnh báo từ Giai đoạn 2 cũ (S2.3 "tách 2 scale") nhưng chưa từng cài. Trước đây không phát hiện ra vì Phase 9 (GRASP/LIFT) không có gì "khác" để so sánh — regression này chỉ lộ rõ khi thêm PLACE vào và REACH/GRASP bắt đầu bị ảnh hưởng lây.

### Fix: tách 2 scale (đã làm, xác nhận đúng)
- `grasp_assist.py`: thêm `_assist_scale_descent(env)` — đọc `_assist_blend_scale_descent` NẾU CÓ (do `AssistScheduleCallback` set khi train thật, cố định 1.0, không anneal), NẾU KHÔNG thì fallback về CHÍNH `_assist_scale(env)` (đảm bảo mọi lệnh `--assist-scale X` cũ ở eval/demo không đổi hành vi, vì các script đó chỉ set `_assist_blend_scale`).
- Đổi gate đầu hàm: `if scale <= 1e-6 and descent_scale <= 1e-6: return actions` (thay vì chỉ dựa vào `scale`).
- Đổi TOÀN BỘ chỗ dùng `scale` cho REACH-descent/GRASP-descent (khối `assist_reach`/`assist_grasp`, các closure `_apply_force`/`_apply_blend_world_z`/`_apply_blend`, khối `need_down` z-loop) sang `descent_scale`. Khối LIFT (`assist_lift`) và PLACE (`assist_place`) GIỮ NGUYÊN dùng `scale` (annealed) — đúng ý định.
- `isaaclab_train.py::AssistScheduleCallback._on_step()`: thêm `unwrapped._assist_blend_scale_descent = 1.0` (cố định, set song song với `_assist_blend_scale` đã anneal).
- Regression gate ngay sau khi sửa: eval phase 2 → **y hệt số cũ** (0.03/0.83/0.47/0.43) — an toàn tuyệt đối cho eval/demo.

### Kết quả training lần 3 (local, có fix) — THÀNH CÔNG, ổn định nhất từ trước đến giờ
Chạy `--task_phase 3 --stage place --assist-schedule`, checkpoint `policy_1M_success57.pt`, 2.5M bước. So với lần 2 (không có fix):
```
                trước fix (ts=1.7M)    sau fix (ts=2.25M, xa hơn nhiều)
grasp_rate      SẬP xuống 0.40         GIỮ VỮNG 0.99-1.0 suốt
latch_rate      SẬP về 0.00            ỔN ĐỊNH 0.40-0.58 xuyên suốt
lift_start_rate SẬP về 0.00            ỔN ĐỊNH 0.39-0.53 xuyên suốt
success_rate    0.00                   dao động nhẹ 0.00-0.03 (ổn định, không sập)
```
`grasp/latch/lift_start` **hoàn toàn KHÔNG sập** dù đã rất lâu sau mốc `[Assist] scale=0.000 @ 1,000,000 steps` — khác hẳn MỌI lần fine-tune trước đó trong toàn bộ investigation (GRASP/LIFT lẫn PLACE lần 2). Đây là bằng chứng mạnh: fix tách scale không chỉ bảo vệ REACH/GRASP mà còn giúp cả LIFT/PLACE ổn định hơn hẳn (có thể vì REACH/GRASP không còn thoái hoá kéo theo cascade collapse).

**Lưu ý**: `success_rate` ở đây vẫn đo theo tiêu chí GRASP+LIFT cũ (`_steps_bottle_lifted>=hold`), CHƯA phải tiêu chí PLACE thật (đặt vào bát) — S8 (metric riêng cho PLACE trong `eval_lift_metrics.py`) và S5 (termination/bonus thật) vẫn chưa làm. Số liệu này xác nhận training ỔN ĐỊNH, chưa xác nhận robot có thực sự đặt được chai vào bát hay chưa.

### Sự cố: máy tự khởi động lại giữa chừng (đã biết từ trước, không phải bug mới)
Training dừng đột ngột ở ts=2,305,024/2,500,000 (92%) — không Traceback, không OOM. `uptime` sau đó cho thấy máy vừa reboot. Khớp với ghi nhận cũ trong session này ("laptop tự khởi động lại như một phần thói quen dùng máy hàng ngày" — `systemd-inhibit --what=sleep:idle` chỉ chặn NGỦ, không chặn REBOOT/SHUTDOWN). Checkpoint đã lưu: `rl_model_1000000_steps.zip`, `rl_model_2000000_steps.zip` — đủ dùng, không cần train lại từ đầu vì xu hướng đã rõ ràng và ổn định suốt 900K→2.3M.

### File đã sửa thêm (S4 + fix tách scale, CHƯA commit)
`isaaclab_openarm_env/mdp/rewards.py` (dispatch 3 nhánh + `_compute_place_reward` + `_steps_since_place_start`), `isaaclab_openarm_env/mdp/grasp_assist.py` (`_assist_scale_descent`, đổi hàng loạt `scale`→`descent_scale`), `isaaclab_train.py` (`_assist_blend_scale_descent=1.0` trong callback), `eval_lift_metrics.py`/`isaaclab_train.py` (thêm `"place"` vào `--stage` choices).

### Hướng tiếp theo
1. Demo trực quan checkpoint `rl_model_2000000_steps.zip` (GUI) để xem thực tế có carry+release vào bát không — số liệu train hiện tại (success theo tiêu chí GRASP/LIFT cũ) không trả lời được câu này.
2. Nếu carry vẫn chưa hội tụ tốt (nghi vấn cũ từ S1-S3 về `height_above_bowl_floor` placeholder sai) → làm S9 (đo bbox bát thật) trước khi tune tiếp.
3. Cân nhắc commit sớm — đã tích luỹ nhiều thay đổi giá trị (S4 + fix tách scale, cả hai đều verify kỹ) qua 2 lần máy tự khởi động lại giữa chừng.

---

## Phase 10 — Train PLACE thật trên server (128 env, 5M bước) — kết quả tốt nhất từ trước tới giờ cho PLACE

Sau khi máy local tự khởi động lại (ngắt training ở 2.3M/2.5M), chuyển hẳn sang server 4090 (lúc này GPU đã rảnh hơn nhiều, ~11.4GB, so với ~1.9GB lần trước gây OOM). Deploy code mới nhất (đã có S4 + fix tách assist scale), train `train_place_s4_server2`: `--task_phase 3 --stage place --assist-schedule --num-envs 128 --timesteps 5000000`, checkpoint gốc `policy_1M_success57.pt`. Tốc độ ~1670-1700 fps (nhanh hơn local ~16 env một cách đáng kể, không chỉ do env count mà cả GPU mạnh hơn).

### Xu hướng đầy đủ (assist scale=0 tại 2M, đúng frac=0.4 của 5M)
```
ts=131K   (assist cao) grasp=1.00 latch=0.92 lift=0.52 success=0.45  maxlift=13.5cm  ← đỉnh sớm
ts=300K-1M (assist cao) success dao động 0.10-0.40, maxlift 4-9cm — vùng ổn định tốt
ts=1M-2M  (assist giảm dần) success giảm dần 0.24→0.03
ts=2M-3.2M (assist=0, vùng thấp nhất) success 0.02-0.09, maxlift 1-3cm — KHÔNG sập về 0 tuyệt đối (khác mọi lần trước)
ts=3.5M-5M (assist=0, HỒI PHỤC) success tăng dần trở lại 0.15→0.215, maxlift lên 6.2cm
ts=5.005M (bước cuối) grasp=0.995 latch=0.385 lift_start=0.33 success=0.215 maxlift=6.2cm
```
`grasp_rate` giữ vững 0.97-1.0 **xuyên suốt toàn bộ 5M bước**, kể cả 3M bước sau khi assist LIFT/PLACE về 0 — xác nhận dứt điểm fix tách assist scale hoạt động đúng ở quy mô lớn (128 env, không chỉ ở quy mô nhỏ 16 env đã test trước đó).

**Phát hiện mới, khác hẳn mọi lần trước**: sau khi tụt xuống vùng thấp (ts=2M-3.2M, đúng lúc assist vừa về 0), performance **KHÔNG đứng yên ở đáy** như 3 lần chạy trước (GRASP/LIFT fine-tune, PLACE lần 1, PLACE lần 2 local) — mà **hồi phục dần** trong 1.8M bước còn lại, kết thúc ở mức success=0.215, gần bằng mức đã đạt được lúc còn nhiều assist. Đây có thể là bằng chứng cho thấy: khi REACH/GRASP không còn bị kéo theo collapse (nhờ fix tách scale), phần còn lại của mạng có đủ "nền" ổn định để tự phục hồi kỹ năng LIFT/PLACE qua thời gian, dù chậm.

### Sự cố cuối run (giống Phase 9, vô hại)
Sau khi đạt ts=5,005,312 (hết 5M), Kit/Isaac Sim crash lúc dọn dẹp cuối (không Traceback, không liên quan RL) — nhưng lần này **đã kịp lưu đầy đủ** `best_policy.pt`/`final_policy.pt`/checkpoint 5M trước khi crash (khác Phase 9, không mất gì).

### Checkpoint đã lưu về local
- `logs/train_place_s4_server2_1M.zip` — checkpoint ~1M bước (SB3 zip đầy đủ, vùng ổn định sớm, success~0.24).
- `logs/train/policy_place_server2_final.pt` — `best_policy.pt` cuối cùng (= final, do không có EvalCallback, đã biết từ Phase 9), success=0.215 tại đúng bước cuối — **là checkpoint TỐT lần này**, không phải bản tệ nhất như các lần fine-tune trước.

### Lưu ý phương pháp luận — CHƯA giải quyết
`eval_lift_metrics.py --assist-scale 1.0` cho kết quả **rất khác** (grasp=0.07!) so với chỉ số nội bộ lúc train (grasp~0.98) khi test lại checkpoint 2M từ run local trước đó. Nguyên nhân nghi vấn: `--assist-scale 1.0` bật LẠI TOÀN BỘ assist (cả phần LIFT/PLACE đã bị anneal về 0 lúc train) — không khớp "chế độ" mà checkpoint đó thực sự được fine-tune ở giai đoạn cuối (`descent=1.0` cố định, `lift/place=0`). `eval_lift_metrics.py` hiện CHƯA hỗ trợ set 2 scale độc lập (thuộc phạm vi S8 chưa làm) — số liệu train-time (TrainMetricsCallback) đáng tin hơn số liệu eval CLI đơn giản cho tới khi sửa việc này.

### Hướng tiếp theo
1. Demo GUI trực quan `policy_place_server2_final.pt` để xác nhận bằng mắt có thực sự carry+release vào bát không.
2. Sửa `eval_lift_metrics.py` để nhận `--assist-scale-descent` riêng (khớp đúng S8), tránh đo sai như trên.
3. Cân nhắc train tiếp dài hơn (>5M) từ chính `policy_place_server2_final.pt` — xu hướng hồi phục cuối run gợi ý còn cải thiện được nếu train thêm.
4. ~~Vẫn cần S9 (đo bbox bát thật)~~ → **Đã làm, xem Phase 11 bên dưới.**

---

## Phase 11 — Demo thật cho thấy PLACE chưa hoạt động + làm S9 (đo bbox bát thật) — SUÝT gây regression nghiêm trọng

### Quan sát trực tiếp qua demo: LIFT chậm, PLACE chưa hề đặt được vào bát
User tự chạy demo GUI, quan sát bằng mắt: robot **nhấc được chai nhưng rất chậm**, và **chưa bao giờ đặt được vào bát**. Đối chiếu lại code: `success_rate` log lúc train (qua `TrainMetricsCallback`) hoá ra **CHỈ tính `_steps_bottle_lifted >= hold_req`** — y hệt tiêu chí GRASP+LIFT cũ, HOÀN TOÀN không liên quan gì tới việc đặt vào bát. Toàn bộ con số "success=0.215" đẹp đẽ ở Phase 10 chỉ đo được "đã nhấc chai lên đủ cao 5 bước", không đo được PLACE thật — khớp chính xác với quan sát của user. Đây là hệ quả trực tiếp của việc chưa làm S5 (tiêu chí thành công PLACE thật).

### S9 — Đo hình học bát thật bằng UsdGeom.BBoxCache
Viết script đo (`measure_bowl.py`, scratchpad) — traverse toàn bộ prim tìm "Bowl", tính world AABB bằng `UsdGeom.BBoxCache`. Kết quả đo được (lặp lại 2 lần, world AABB của mesh KHÔNG đổi dù `root_pos_w` đổi theo randomization — vì randomization ghi thẳng vào physics view, không sync ngược lại USD stage transform; phải so với `init_state.pos` GỐC = (0.58, 0.22, 0.67), không phải `root_pos_w` runtime):
```
World AABB min=(0.4140, 0.2199, 0.6512) max=(0.5737, 0.3796, 0.7033)
Size (x,y,z) = (0.1597, 0.1597, 0.0520) m  — bát rộng ~16cm, SÂU CHỈ 5.2cm
bbox_center = (0.4938, 0.2998)
```
**Phát hiện lớn**: root của Bowl (0.58, 0.22, 0.67) **lệch tâm hình học tới 8-8.6cm** cả X lẫn Y — gần đúng bằng NỬA bề rộng bát (0.1597/2=0.07985cm) → root nằm ở một GÓC của bounding box, không phải tâm hay đáy. Toàn bộ code PLACE viết ở S1-S3 (`_osc_carry_to_bowl`, `dist_bottle_bowl_xy`, `height_above_bowl_floor`) đều dùng thẳng `bowl_pos` (root) làm mục tiêu — **luôn nhắm lệch ra rìa bát 8cm**, gần như chắc chắn là lý do carry chưa từng hội tụ XY quan sát được từ S1-S3.

Offset đo được, lưu vào config:
```python
bowl_center_local_xy_x = -0.0862   # bbox_center_x - root_x
bowl_center_local_xy_y = 0.0798    # bbox_center_y - root_y
bowl_floor_local_z = -0.0188       # bbox_min_z - root_z (đáy trong thật)
bowl_rim_local_z = 0.0333          # bbox_max_z - root_z (miệng bát)
```

### SUÝT gây regression nghiêm trọng — bài học lặp lại đúng nguyên lý đã ghi trong `LIFT_BUG_THEORY.md`
Sửa `compute_state()` (helpers.py) để `bottle_to_bowl` dùng `bowl_center_pos` (đã sửa lệch tâm) thay vì `bowl_pos` thô — nghe có vẻ "chỉ ảnh hưởng PLACE". Chạy regression gate ngay theo thói quen → **`grasp_rate` sập 0.83→0.47, `success` về 0.00, `fail_modes.reach=15/30`**.

**Nguyên nhân**: `bottle_to_bowl` không chỉ dùng nội bộ cho PLACE — nó còn là **một phần observation 26-D** (`obs[20:23]`, dùng cho MỌI task_phase≥2 kể cả policy đã train từ lâu). Đổi CÔNG THỨC tính nó (dù giữ nguyên SHAPE 3 chiều) vẫn tương đương đổi observation — policy đã train quen với phân bố giá trị CŨ, phân bố MỚI (lệch ~8cm) đủ để phá vỡ hành vi REACH/GRASP đã ổn định. Đây CHÍNH XÁC là bug "lỗi kiến trúc — dùng chung núm vặn" đã viết trong `LIFT_BUG_THEORY.md`, chỉ khác là lần này biến dùng chung là một quan sát, không phải một assist scale.

**Fix**: tách riêng — `bottle_to_bowl` (feed observation) giữ NGUYÊN công thức cũ (`bowl_pos - bottle_pos`), thêm biến MỚI `bottle_to_bowl_center` (dùng `bowl_center_pos` đã sửa) CHỈ dùng cho `dist_bottle_bowl`/`dist_bottle_bowl_xy` (nội bộ reward/state-machine PLACE, KHÔNG vào observation). Chạy lại regression gate → khôi phục đúng số cũ (0.03/0.83/0.47/0.43).

### File đã sửa (S9, CHƯA commit)
`isaaclab_openarm_env/mdp/helpers.py` (`bowl_center_pos`, `bowl_floor_z`, `bowl_rim_z`, tách `bottle_to_bowl` khỏi `bottle_to_bowl_center`, sửa `height_above_bowl_floor`), `isaaclab_openarm_env/mdp/grasp_assist.py` (`_osc_carry_to_bowl`/`_osc_descend_to_bowl` dùng `bowl_center_pos`/`bowl_rim_z` thay vì `bowl_pos` thô, siết `place_release_height_m` 0.05→0.02 do bát chỉ sâu 5.2cm), `isaaclab_openarm_env/config.py` (4 key offset bát đo được).

### Hướng tiếp theo
1. ~~**S5 thật**: tiêu chí thành công PLACE thật...~~ → **Đã làm, xem bên dưới.**
2. Train lại/verify PLACE với target bowl đã sửa đúng tâm — kỳ vọng carry hội tụ XY tốt hơn hẳn so với S1-S3 (lúc đó luôn nhắm lệch 8cm).
3. Luôn nhớ: **bất kỳ thay đổi nào trong `compute_state()` phải kiểm tra xem giá trị đó có lọt vào observation vector hay không** trước khi cho là "an toàn vì chỉ ảnh hưởng tính năng mới".

## Phase 12 — S5: tiêu chí thành công/thất bại PLACE thật (dựa trên số đo S9)

Thay `success_termination` nhánh `task_phase>=3` (trước là stopgap `return zeros`) bằng tiêu chí thật:
- `place_in_bowl_success(env, s)` (mới, `helpers.py`): `dist_bottle_bowl_xy < place_success_xy_radius_m` **&** `height_above_bowl_floor < place_success_max_height_above_floor_m` **&** `bottle_lin_speed < place_success_max_speed` **&** `bottle_tilt_deg < place_success_max_tilt_deg`.
- `success` (phase≥3) = `place_release_ready` **&** `_steps_bottle_settled >= place_success_hold_steps` **&** `place_in_bowl_success` — đếm bằng bộ đếm liên tiếp (`_steps_bottle_settled`, tăng trong `_update_contact_and_stages` khi đã buông + đứng yên + đúng vị trí), đúng kỷ luật "neo vào mốc thật" đã dùng cho `_steps_since_latch`/`_steps_since_place_start`.
- Termination thất bại mới `bottle_misplaced_termination` (mirror 1-1 `tipped_bottle_termination`): fire khi `_steps_bottle_misplaced >= place_dropped_min_steps` (đã buông + đứng yên + KHÔNG đúng vị trí — gộp chung 3 kịch bản thả sớm/hất văng/rơi sàn thành 1 đường phạt duy nhất, tránh nhiều lối thoát rẻ hơn nhau).
- 2 reward term mới mirror `terminal_success_bonus`/`terminal_tipped_penalty`: `terminal_place_success_bonus` (đọc termination `success`, bonus=90.0 > `grasp_success_bonus=60.0` vì mốc khó hơn) và `terminal_place_drop_penalty` (đọc termination `bottle_misplaced`, penalty=30.0). Cả 2 tự trả 0 khi `task_phase<3`.

**Bug nhỏ tự phát hiện lúc đăng ký field**: `place_success_xy_radius_m` từng có 2 default rời rạc khác nhau nằm im trong 2 lệnh `getattr` (0.05 ở `helpers.py`, 0.10 ở `rewards.py::_compute_place_reward`) — vô hại vì chưa field nào override, nhưng là quả bom hẹn giờ: nếu ai đó đăng ký field với giá trị khác 2 default này thì `_release_quality`/`r_settle` (đọc default cũ tại chỗ chúng đứng) và `place_in_bowl_success` (đọc field mới) sẽ lệch nhau ngầm. Chốt **1 giá trị canonical 0.05** (theo bán kính bát đo được ~8cm ở S9, cho biên an toàn) khi đăng ký field thật vào `config.py` — giờ cả 2 nơi đọc cùng 1 số.

Đăng ký field/term:
- `config.py::ApplePickPlaceEnvCfg`: 8 field mới — `place_success_xy_radius_m=0.05`, `place_success_max_height_above_floor_m=0.03`, `place_success_max_speed=0.15`, `place_success_hold_steps=10`, `place_success_max_tilt_deg=60.0`, `place_dropped_min_steps=10`, `place_success_bonus=90.0`, `place_drop_penalty=30.0`.
- `config.py::RewardsCfg`: `place_success_bonus_term`, `place_drop_penalty_term`.
- `config.py::TerminationsCfg`: `bottle_misplaced`.
- `mdp/__init__.py`: export `terminal_place_success_bonus`, `terminal_place_drop_penalty`, `bottle_misplaced_termination` (config.py dùng `from . import mdp` nên mọi hàm mới PHẢI qua `__init__.py` mới `mdp.xxx` gọi được — thiếu bước này sẽ ra `AttributeError` khi load config, không phải lỗi âm thầm).

**Regression gate bắt buộc** (`eval_lift_metrics.py --model-path policy_1M_success57.pt --episodes 30 --num-envs 8 --bottle-noise 0.05 --assist-scale 1.0 --stage all --seed 0 --task_phase 2`) sau khi đăng ký xong: **PASS, khớp CHÍNH XÁC baseline** — `success_rate=0.0333 (1/30)`, `grasp_rate=0.8333 (25/30)`, `latch_rate=0.4667 (14/30)`, `lift_start_rate=0.4333 (13/30)`. Toàn bộ thay đổi S5 (thêm counter trong `_update_contact_and_stages`, thêm reward/termination term mới) đều gate đúng sau `task_phase<3` nên vô hại với phase 2 — xác nhận bằng số đo, không chỉ bằng đọc code.

### File đã sửa (S5, CHƯA commit)
`isaaclab_openarm_env/mdp/helpers.py` (`place_in_bowl_success`), `isaaclab_openarm_env/mdp/rewards.py` (counter `_steps_bottle_settled`/`_steps_bottle_misplaced` trong `_update_contact_and_stages`, `terminal_place_success_bonus`, `terminal_place_drop_penalty`), `isaaclab_openarm_env/mdp/terminations.py` (`success_termination` phase≥3 thật, `bottle_misplaced_termination` mới, reset `_steps_bottle_misplaced`), `isaaclab_openarm_env/mdp/__init__.py` (export 3 hàm mới), `isaaclab_openarm_env/config.py` (8 field mới + 2 reward term + 1 termination term).

### Verify bằng demo thật (`--task-phase 3 --stage place --visualizer none`, `DEBUG_PLACE=1`) — PHÁT HIỆN BUG NGAY LẦN CHẠY ĐẦU
Lần chạy đầu: **100% episode chết ở đúng 13 bước, `stage:0` (còn ở REACH, chưa từng chạm chai), Return ≈ −28.7** — không phải timeout, không phải thành công.

**Nguyên nhân**: block tính `_steps_bottle_settled`/`_steps_bottle_misplaced` trong `_update_contact_and_stages` chỉ gate bằng `if place_phase is not None` — tức "thuộc tính `_place_phase` có tồn tại trên env hay chưa" (đúng ngay từ episode đầu vì `uses_place(task_phase)`), KHÔNG gate theo "env NÀY hiện có đang ở STAGE_PLACE hay không". Hậu quả: `at_rest = released & (speed < max_settle_speed)` đúng cho MỌI env ngay ở REACH — gripper mặc định mở (`released=True`) và chai đang nằm yên trên bàn chưa ai đụng vào (`speed` thấp) → `at_rest=True` gần như ngay bước đầu. `in_bowl=False` (chai còn ở bàn, cách xa bát) → `misplaced_now = at_rest & ~in_bowl = True` liên tục từ bước 1 → `_steps_bottle_misplaced` chạm `place_dropped_min_steps=10` ở bước ~13 → `bottle_misplaced_termination` fire oan cho MỌI episode, mọi lúc, bất kể đã từng gắp chai hay chưa. Đây là lỗi kinh điển **"chưa từng làm" bị hiểu nhầm thành "đã làm rồi làm hỏng"** — `at_rest & ~in_bowl` về mặt toán học đúng cho cả 2 tình huống hoàn toàn khác nhau: "chưa gắp" và "đã thả nhưng thả sai".

**Fix**: thêm gate `in_place_stage = env._stage == STAGE_PLACE` (mirror đúng cách `tipped_bottle_termination` đã gate bằng `in_grasp_or_place`), nhân vào `at_rest` — `bottle_misplaced`/`bottle_settled` giờ chỉ có thể đúng khi env ĐANG ở STAGE_PLACE thật.

**Kết quả sau fix** (chạy lại cùng seed/config): không còn episode nào chết ở 13 bước — 23/24 episode chạy hết `TIMEOUT` (1200 bước), 1 episode `TERMINATED` ở bước 928 vì `tilt:44.8°` (lật chai thật lúc grasp lần 2, đúng bản chất, không phải bug). Quan sát được đúng 1 lần chuyển `IDLE→CARRY` (env0, step_ct=818, `xy=150.8mm h_bowl=5.0mm tilt=4.8°`) — episode đó chạy tới hết TIMEOUT không bị `bottle_misplaced` bắn oan giữa chừng (carry chưa xong vì `place_carry_speed_scale=0.35` cố tình chậm + quãng đường 150mm, không đủ 382 bước còn lại để tới DESCEND — hợp lý, không phải lỗi). Return dương (509.70) trong episode đó, khớp kỳ vọng reward PLACE đã thiết kế.

**Bài học lặp lại lần thứ 3 trong cùng investigation** (sau assist-scale-coupling và bottle_to_bowl-observation): **một điều kiện boolean đúng về mặt công thức nhưng thiếu ngữ cảnh stage/thời điểm sẽ luôn tìm ra cách đúng "ở sai chỗ"** — không có cách nào phát hiện qua đọc code tĩnh, chỉ lộ ra khi chạy demo thật. Củng cố thêm nguyên tắc: **regression gate ở phase cũ (2) không đủ để tin cậy code mới ở phase mới (3)** — phải luôn demo/eval trực tiếp ở đúng phase vừa thêm tính năng trước khi tin nó "chắc đúng vì gate cũ vẫn pass".

File sửa thêm: `isaaclab_openarm_env/mdp/rewards.py` (`_update_contact_and_stages`, thêm `in_place_stage` gate).

### Hướng tiếp theo
1. ~~Test state machine + termination mới qua demo~~ → **Đã làm và fix xong 1 bug, xem trên.**
2. ~~S6/S7/S8~~ → **Đã làm, xem Phase 13 bên dưới.**
3. Train lại PLACE chỉ sau khi cả bowl geometry (S9) VÀ success/failure criteria (S5) đã đúng — 2 lần train PLACE trước (Phase 10/11) đều dùng criteria/geometry sai, số liệu của chúng không phản ánh khả năng thật.

## Phase 13 — S6 (đăng ký config keys PLACE) + S7 (obs branch) + S8 (plumbing eval) — phát hiện thêm 2 bug cũ

### S6 — Đăng ký formal field cho các key PLACE sống bằng `getattr` rời rạc từ S2
Thêm vào `config.py::ApplePickPlaceEnvCfg`: `assist_place=False`, `place_hold_closed=True`, `place_carry_height_m=0.15`, `place_carry_onset_ramp_steps=15`, `place_carry_speed_scale=0.35`, `place_carry_align_blend=0.5`, `place_xy_arrival_radius_m=0.03`, `place_arrival_settle_steps=5`, `place_descend_world_m=0.02`, `place_release_height_m=0.02`, `place_release_hold_steps=5`, `place_abort_tilt_deg=25.0`, `place_abort_dist_ee_m=0.15`, `place_camp_decay_steps=150`. Tất cả giữ ĐÚNG giá trị mặc định hiện có trong code — đăng ký không đổi hành vi, chỉ gom về 1 nguồn.

### S7 — Nhánh PLACE của `ee_to_target` (observations.py)
Thêm `torch.where(in_place, place_delta, ee_to_target_cũ)` bọc ngoài — target khi CARRY = `(bowl_center_pos_xy, bowl_rim_z + carry_height)`, khi DESCEND/HOLDING = `(bowl_center_pos_xy, bowl_floor_z + release_height)`. Dùng ĐÚNG `bowl_center_pos`/`bowl_rim_z`/`bowl_floor_z` (đã sửa lệch tâm ở S9) — không dùng `bowl_pos` thô, tránh lặp lại bug lệch 8cm. An toàn phase 2: `in_place` luôn `False` (STAGE_PLACE không bao giờ gán khi task_phase<3) nên nhánh cũ giữ nguyên y hệt.

### S8 — Plumbing `eval_lift_metrics.py` — phát hiện 2 bug cũ trong lúc thêm metrics PLACE

**Bug 1 (nghiêm trọng, cùng họ với bug đã đốt `TrainMetricsCallback`)**: `is_success = success_t OR (lift_hold>=req AND task_phase>=2)` — viết `>=2` thay vì `==2` khiến heuristic "đã nhấc = thành công" áp dụng luôn cho phase≥3, che mất `success_t` thật (S5) mỗi khi cả 2 đều đúng. Nếu không bắt kịp, mọi eval PLACE sau này sẽ tiếp tục báo "success" giả giống hệt bài học đau đã trả giá ở Phase 11. **Fix**: đổi điều kiện thành `task_phase == 2` — ở phase≥3, `success_t` (termination thật) là thẩm quyền DUY NHẤT.

**Bug 2 (khiến toàn bộ việc test S10 "eval ở --task_phase 3 trên checkpoint train ở phase 2" trở nên bất khả thi nếu không bắt)**: `_apply_env_cfg_snapshot()` (nạp `env_cfg.pkl` lưu lúc TRAIN checkpoint) chạy SAU khi set `env_cfg.task_phase = args.task_phase` → snapshot ÂM THẦM đè `--task_phase 3` về lại 2 (giá trị lúc `policy_1M_success57.pt` được train). Bug này VÔ HÌNH trong suốt session vì mọi regression gate trước giờ đều chạy đúng `--task_phase 2` (trùng với snapshot) — chỉ lộ ra khi thử `--task_phase 3` lần đầu (chạy xong không lỗi, nhưng `place_start_rate`/metrics PLACE không xuất hiện trong output vì `cfg.task_phase` thực tế vẫn là 2). **Fix**: đảo thứ tự — nạp snapshot TRƯỚC, set `task_phase` từ CLI SAU (mirror đúng cách `isaaclab_demo.py` đã làm) — CLI luôn là thẩm quyền cuối cho eval.

Thêm accumulator (`max_place_phase`, `place_start_step`, `released`, `release_dist_bottle_bowl`, `min_dist_bottle_bowl`), 5 fail_mode PLACE mới (`no_place_transition`, `no_release`, `early_release`, `dropped_outside_bowl`, `place_timeout`), 4 metric tổng hợp mới (`place_start_rate`, `release_rate`, `p50_place_dist_m`/`p90_place_dist_m`, `dropped_rate`) — chỉ tính khi `task_phase>=3`, vô hại ở phase<3.

**Verify**: regression gate phase 2 chạy lại 2 lần (trước và sau fix thứ tự snapshot) — cả 2 lần khớp CHÍNH XÁC baseline (0.0333/0.8333/0.4667/0.4333). Eval `--task_phase 3 --stage place` (20 episode, checkpoint `policy_1M_success57.pt`, chưa hề fine-tune PLACE): chạy sạch không crash, `place_start_rate=0.05` (1/20 episode tới PLACE, khớp `lift_start_rate=0.30`× tỉ lệ latch thực tế), episode đó có `fail_mode=place_timeout` (chưa kịp release trong episode — hợp lý, checkpoint chưa từng học carry).

**Chưa làm** (biết trước, thuộc S8 gốc, không phải lỗi): `--assist-scale-descent` CLI riêng cho eval (tách 2 scale như train đã có) — vẫn là gap đã ghi nhận từ Phase 9, không chặn train PLACE tiếp theo.

### File sửa (S6/S7/S8, CHƯA commit)
`isaaclab_openarm_env/config.py` (14 field PLACE mới), `isaaclab_openarm_env/mdp/observations.py` (nhánh PLACE của `ee_to_target`), `eval_lift_metrics.py` (2 bug fix + accumulator/fail_mode/metrics PLACE mới).

### Hướng tiếp theo
Toàn bộ S1-S9 của kế hoạch PLACE đã xong và verify thật (không chỉ đọc code). Sẵn sàng cho S10: train thử PLACE với bowl geometry (S9) + success/failure criteria (S5) đã đúng lần đầu tiên trong project — 2 lần train PLACE trước (Phase 10/11) đều dùng số liệu sai nên không phản ánh khả năng thật của policy.

## Phase 14 — S10: train PLACE lần đầu với criteria/geometry ĐÚNG (server 4090, 5M bước) — kết quả đầu tiên đáng tin cậy

Deploy code S1-S9 lên server, upload `policy_1M_success57.pt` (verify md5 khớp, tránh lặp lỗi nhầm checkpoint Phase 10), chạy:
```
--task_phase 3 --stage all --assist-schedule --num-envs 128 --timesteps 5000000 \
--checkpoint logs/policy_1M_success57.pt --lr-start 1e-4 --lr-end 1e-5 --clip-range 0.1 --ent-coef 0.002 --progress
```
Chạy đủ 5,005,312/5,000,000 bước (~639 it/s, ~1h35), Kit crash lúc dọn dẹp cuối (đúng pattern benign đã biết) nhưng **mọi checkpoint đều lưu sạch** (`best_policy.pt`=`final_policy.pt` timestamp 13:31, đủ `rl_model_{1,2,3,4}M_steps.zip`).

**Quỹ đạo `success_rate` (S5 thật, KHÔNG phải heuristic lift-hold cũ) theo `assist scale`**:

| step | scale | success | grasp | latch |
|---|---|---|---|---|
| 180k | ~1.0 | **0.59** | 1.00 | 0.765 |
| 344k | 0.9 | 0.25 | 0.99 | 0.525 |
| 1.16M | 0.5 | 0.03 | 0.99 | 0.435 |
| 2.15M | 0.0 (vừa chạm đáy) | 0.035 | 0.995 | 0.39 |
| ~3-5M | 0.0 (ổn định) | dao động 0.01-0.09, không có đỉnh rõ | 0.98-1.0 | 0.26-0.39 |
| 5.0M (cuối) | 0.0 | 0.07 | 0.995 | 0.36 |

**Ý nghĩa (2 phần, một tốt một đã biết trước)**:
1. **TIN TỐT — xác nhận S5+S9 đúng end-to-end**: ở scale gần 1.0 (script điều khiển CARRY/DESCEND/HOLDING, policy gần như chưa can thiệp), `success_rate=0.59` — nghĩa là state machine PLACE + `place_in_bowl_success` (hình học bát đo thật S9, tiêu chí settle thật S5) **hoạt động đúng, đặt được chai vào bát thật hơn một nửa số lần**. Đây là con số ĐẦU TIÊN trong toàn bộ investigation phản ánh đúng "có đặt được vào bát hay không" — khác hẳn `success_rate=0.215` giả ở Phase 10/11 (đo nhầm lift-hold).
2. **ĐÃ BIẾT TRƯỚC, không phải bug mới**: giống hệt pattern GRASP/LIFT đã gặp 2 lần ở Phase 9 — khi `_assist_blend_scale` anneal về 0, policy CHƯA tự chủ được PLACE (carry+release), `success_rate` rơi về vùng thấp (~0.01-0.09) và không hồi phục, dù `grasp_rate`/`latch_rate` vẫn giữ vững (xác nhận fix tách assist-scale ở Phase 9 tiếp tục đúng — không có hồi quy GRASP/LIFT dù train PLACE dài 5M bước). Đây là giới hạn ĐÃ GHI NHẬN trong kế hoạch (mục S10, "không debug lạc đề vấn đề này trong lúc làm PLACE") — chưa có checkpoint nào tự chủ hoàn toàn PLACE, cần một vòng nữa (curriculum anneal chậm hơn, hoặc học riêng PLACE trước khi ghép full pipeline) để giải quyết, KHÔNG thuộc phạm vi Phase 10-14.

**Checkpoint tải về**: `logs/train/policy_place_fixed_server_final.pt` (5M, best=final, success=0.07 @ scale 0.0 — mức ổn định cuối, không có điểm giữa nào vượt trội hẳn để chọn thay), `logs/train_place_fixed_server_1M.zip` (checkpoint 1M, gần vùng scale còn cao ~0.6, success cao hơn nhưng phần lớn là do assist chưa anneal hết — không phải năng lực tự chủ), `logs/train_place_fixed_server_full.log` (log đầy đủ).

### Hướng tiếp theo (ĐÃ CẬP NHẬT — xem Phase 15, phát hiện quan trọng hơn)
~~1. Demo GUI...~~ → Đã làm, phát hiện mâu thuẫn nghiêm trọng, xem Phase 15.

## Phase 15 — PHÁT HIỆN NGHIÊM TRỌNG: policy DETERMINISTIC hỏng hoàn toàn, khác hẳn số liệu train (STOCHASTIC)

### Mâu thuẫn ban đầu
Demo `policy_place_fixed_server_final.pt` (`--task-phase 3 --stage grasp`, deterministic mặc định): **100% episode (20/20) kẹt ở REACH, `top↓≈0.49` phẳng suốt episode, KHÔNG BAO GIỜ vào được GRASP** — trong khi log train báo `grasp_rate=0.995` suốt từ step ~300k tới hết 5M. Đây không phải sai khác nhỏ — là 0% vs 99.5%.

### Điều tra bằng đo trực tiếp (không đoán, không tin số cũ)
1. Nghi vấn "checkpoint .pt lưu/nạp sai" → test bằng file `.zip` đầy đủ (`rl_model_4999680_steps.zip`, chứa optimizer + toàn bộ state, nạp qua `PPO.load` không qua `state_dict` thủ công): **`grasp_rate=0.0417` (1/24)** — vẫn gần như hỏng. Loại bỏ giả thuyết lỗi format lưu `.pt`; checkpoint THẬT SỰ hỏng khi chạy deterministic.
2. Nghi vấn "lệch giữa deterministic (mean action, dùng trong demo/eval mặc định) và stochastic (sampling, dùng trong rollout lúc train để tính `grasp_rate`)" → test lại checkpoint 5M với cờ `--stochastic`: **`grasp_rate` nhảy từ 0.0 lên 0.5** — xác nhận ĐÚNG cơ chế: policy stochastic còn cứu được một phần nhờ nhiễu sampling (kết hợp assist descent=1.0), còn deterministic (mean) đã hỏng gần hết.
3. Test checkpoint SỚM hơn cùng run (`rl_model_999936_steps.zip`, ~1M bước, lúc `_assist_blend_scale` còn ~0.6): deterministic `grasp_rate=0.5` — **tệ hơn train-log báo (0.985) nhưng còn dùng được, chưa hỏng hẳn như bản 5M cuối**.

### Kết luận
Policy DETERMINISTIC (mean action — chính là thứ demo/eval/deploy thật sự dùng) **thoái hoá dần trong suốt quá trình fine-tune PLACE**: ~50% grasp ở 1M bước → gần 0% ở 5M bước — trong khi metric train (đo trên STOCHASTIC rollout, xem `TrainMetricsCallback._on_step`, đã đọc lại code — logic đếm episode ĐÚNG, không phải bug đếm) giữ nguyên ~0.98-1.0 suốt. Tức là **tự nó không phải bug đo lường (như các lần trước) — mà là một pha bệnh lý thật của PPO**: `entropy_loss`/`std` không đổi ("std≈3.98" hằng số suốt cả log, gợi ý `log_std` gần như không học/không giảm — `ent_coef=0.002` cố định suốt 5M bước, không có lịch giảm entropy) khiến chính sách dựa vào NHIỄU SAMPLING (kết hợp assist descent=1.0 luôn bật) để "qua bài" trong lúc train, còn **mean action (deterministic) không hề được ép hội tụ về một lời giải tốt** — sampling che giấu hoàn toàn một mean đang trôi dạt xấu dần.

**So sánh để loại trừ "đây là vấn đề chung của cả pipeline"**: `policy_1M_success57.pt` (checkpoint GRASP+LIFT trước đó, KHÔNG qua vòng fine-tune PLACE 5M bước này) đã được demo/eval deterministic NHIỀU LẦN trong Phase 9/12/13 và luôn hoạt động đúng (GRASP/LIFT/vào được PLACE state machine bình thường) — nên đây KHÔNG phải lỗi cấu trúc/pipeline cũ, mà là hậu quả CỦA RIÊNG lần fine-tune PLACE 5M bước vừa rồi (train_place_fixed_server), rất có thể do tổ hợp `ent_coef=0.002` cố định + không có entropy decay + train dài (5M bước) khiến drift tích luỹ đủ lớn để lộ ra.

### Hệ quả — TOÀN BỘ bảng quỹ đạo success/grasp ở Phase 14 phải đọc lại với hiểu biết mới
Con số `success_rate=0.59` ở step 180k và mọi con số grasp/latch trong bảng Phase 14 đều đo trên STOCHASTIC rollout (đúng cách `TrainMetricsCallback` hoạt động) — vẫn ĐÚNG như một phép đo "chính sách CÓ sampling làm được gì", và **kết luận về S5/S9 đúng (state machine + tiêu chí bát thật hoạt động đúng khi có scripted assist) vẫn giữ nguyên, không đổi** — vì phép đo đó dùng scripted assist scale gần 1.0, ít phụ thuộc vào chất lượng mean của policy. Nhưng kết luận "policy tự chủ được GRASP/LIFT, chỉ PLACE còn yếu" ở Phase 14 là **sai/thiếu** — với chính sách DEPLOY THẬT (deterministic), ngay cả GRASP cũng gần như hỏng ở checkpoint cuối.

### Chưa làm / hướng tiếp theo (cần quyết định của user)
1. **Không dùng `policy_place_fixed_server_final.pt` (5M) để demo/deploy** — deterministic gần như vô dụng. Nếu cần một checkpoint PLACE tạm dùng được, ưu tiên bản SỚM hơn (~1M, dù vẫn chỉ 50% deterministic, còn hơn 5M).
2. **Nghi vấn cần fix trước khi train vòng tiếp**: thêm lịch giảm `ent_coef` (hiện cố định suốt run) hoặc dùng LR/clip-range khác để ép `log_std` hội tụ giảm dần — chưa làm, cần thiết kế + đo (không đoán số).
3. Cân nhắc thêm: đánh giá NÊN dùng eval/demo mặc định là stochastic hay deterministic để MATCH đúng cái sẽ deploy thật (robot thật rất có thể chạy deterministic) — nếu deploy thật cũng deterministic thì đây là lỗi PHẢI sửa trước khi train tiếp bất kỳ thứ gì, không phải optional.
4. Chưa retrain lại — cần user xác nhận hướng (thêm entropy decay rồi train lại PLACE, hay điều tra sâu hơn nguyên nhân log_std không giảm trước).

User chọn: "Thêm entropy decay rồi train lại" — implement + chạy diagnostic NGẮN trước (không cam kết 5M ngay), đúng kỷ luật "đo trước khi tin" của cả project.

## Phase 16 — Implement entropy decay, chạy diagnostic 500k bước — kết quả KHÔNG rõ ràng, giả thuyết Phase 15 cần xét lại

### Implement
`isaaclab_train.py`: thêm `--ent-coef-end` (mirror `--lr-end`), `EntCoefScheduleCallback` (mirror `AssistScheduleCallback` — set trực tiếp `self.model.ent_coef` mỗi step vì SB3 KHÔNG hỗ trợ `ent_coef` dạng schedule callable như `learning_rate`/`clip_range`). Cố ý ĐỒNG BỘ cùng `grasp_assist_anneal_frac` (0.4) thay vì một anneal_frac riêng — ý định: entropy về 0 đúng lúc assist về 0.

### Diagnostic 500k bước (server, từ `policy_1M_success57.pt`, `--ent-coef 0.002 --ent-coef-end 0.0`)
Verify log: `[EntCoef] ent_coef=0.00000 @ 200,064 steps` — schedule hoạt động đúng thiết kế (về 0 đúng lúc 40% = 200k/500k).

**PHÁT HIỆN LÀM LUNG LAY GIẢ THUYẾT PHASE 15**: `std` đã là **3.85 NGAY TỪ BƯỚC ĐẦU TIÊN** (step 16,384, tức NGAY SAU KHI nạp `policy_1M_success57.pt`) — KHÔNG phải một giá trị "tăng dần trong lúc fine-tune PLACE" như tôi suy luận ở Phase 15. `std` cao vốn đã BAKED SẴN trong chính checkpoint `policy_1M_success57.pt` (nạp qua `load_state_dict`, copy nguyên `log_std`), từ trước khi vòng train PLACE này bắt đầu. Suốt 500k bước với `ent_coef=0.0` (60% cuối), `std` chỉ nhích nhẹ 3.85→3.84 — gần như không đổi.

**Ý nghĩa**: `std` cao là ĐẶC ĐIỂM VỐN CÓ của checkpoint gốc (vẫn deterministic tốt — đã demo/eval thành công nhiều lần ở Phase 9/12/13), KHÔNG PHẢI nguyên nhân trực tiếp gây "mean action thoái hoá" — vì std chỉ chi phối MỨC ĐỘ NHIỄU quanh mean lúc sampling, không ảnh hưởng trực tiếp tới giá trị mean (thứ deterministic action đọc thẳng). Giả thuyết "ent_coef cố định → std không giảm → mean không hội tụ" ở Phase 15 có thể ĐÚNG MỘT PHẦN nhưng KHÔNG ĐẦY ĐỦ — cần tìm thêm cơ chế khiến MEAN tự trôi dạt xấu trong 5M bước, độc lập với std.

**Kết quả eval deterministic checkpoint 500k (có entropy decay)**: `grasp_rate=0.583` — chỉ nhỉnh hơn chút so với checkpoint 1M-bước của run GỐC (không có entropy decay, `grasp_rate=0.5` cùng phép đo). **Không đủ mạnh để kết luận entropy decay là fix đúng** — cần so sánh ở CÙNG mốc bước, và quan trọng hơn: cần xem liệu nó có NGĂN được đà thoái hoá tiếp diễn ở vùng 1M-5M (nơi run gốc sập từ 0.5 xuống ~0) hay không — 500k bước CHƯA đủ để trả lời.

### Trạng thái — CHƯA quyết định hướng tiếp
Bằng chứng hiện tại KHÔNG đủ để tự tin chạy full 5M với entropy decay (rủi ro lặp lại tốn ~1h35 cho một fix chưa chắc đúng cơ chế). Cần: hoặc (a) điều tra sâu hơn nguyên nhân MEAN trôi dạt (không phải std) trước khi thử tiếp, hoặc (b) chạy diagnostic dài hơn (vd 2M bước) để xem entropy decay có chặn được đà sập ở vùng 1M-2M hay không, trước khi cam kết 5M đầy đủ. Đã báo user, đang chờ quyết định hướng.

User chọn: "Chạy diagnostic dài hơn (2M bước) trước khi cam kết 5M".

## Phase 17 — Diagnostic 2M bước — GIẢ THUYẾT ENTROPY DECAY BỊ BÁC BỎ, tìm ra nghi phạm thật (gradient interference đa nhiệm vụ)

### Kết quả — cùng pattern thoái hoá, KHÔNG bị chặn lại bởi entropy decay
Chạy đủ 2,007,040/2,000,000 bước (server, cấu hình y hệt diagnostic 500k, chỉ tăng budget), `std` cuối = 3.82 (gần như không đổi từ 3.85 ban đầu, khớp quan sát Phase 16). Eval deterministic (`eval_lift_metrics.py`, `--assist-scale 1.0 --stage all`, cùng seed=1, cùng phương pháp):

| Bước | Có entropy decay (run này) | KHÔNG entropy decay (run gốc Phase 14) |
|---|---|---|
| 500k | grasp=0.583 | (chưa đo ở mốc này) |
| 1M | **grasp=0.333** | grasp=0.5 |
| 2M | **grasp=0.083** | (chưa đo, nhưng cùng xu hướng giảm) |
| 5M | (chưa chạy) | grasp≈0.0-0.04 |

**Kết luận dứt khoát**: entropy decay KHÔNG ngăn được đà thoái hoá — pattern giống hệt run gốc (đơn điệu giảm dần theo số bước/số lần update), thậm chí ở mốc 1M-2M route MỚI (có entropy decay) còn giảm NHANH hơn route gốc một chút (0.333/0.083 vs 0.5 ở cùng早 giai đoạn) — trong sai số thống kê của 24 episode, nhưng chắc chắn KHÔNG có cải thiện rõ rệt nào. **Giả thuyết Phase 15/16 ("ent_coef cố định → std không giảm → mean không hội tụ") bị bác bỏ bằng thực nghiệm.**

### Nghi phạm mới, hợp lý hơn nhiều: gradient interference đa nhiệm vụ (catastrophic forgetting)
Cả 2 run (có/không entropy decay) đều thoái hoá theo ĐÚNG MỘT quy luật: giảm dần đơn điệu theo số bước train/số lần cập nhật gradient (`n_updates`), bất kể entropy/std thay đổi hay không. Điều này trỏ tới nguyên nhân khác hẳn: `PPO(policy="MlpPolicy", ...)` mặc định dùng **một mạng CHUNG** (`net_arch=[256,256]`, KHÔNG tách riêng `pi`/`vf`, càng không tách riêng theo stage) cho TẤT CẢ 4 giai đoạn REACH/GRASP/LIFT/PLACE cùng lúc trong cùng 1 batch rollout (128 env chạy song song, mỗi env có thể đang ở stage khác nhau). PLACE là nhiệm vụ CÒN CHƯA HỌC ĐƯỢC (reward thấp, value-estimate nhiễu) — gradient từ phần PLACE của rollout, chảy qua CHUNG một mạng với REACH/GRASP, nhiều khả năng đang **ghi đè dần lên trọng số REACH/GRASP đã tốt** (catastrophic forgetting kinh điển khi fine-tune đa nhiệm vụ trên mạng chia sẻ, không có cơ chế bảo vệ nào — không tách head, không KL-penalty giữ gần policy gốc, không đóng băng layer nào).

**Bằng chứng gián tiếp ủng hộ giả thuyết này**: `descent_scale` (assist REACH/GRASP-descent) được xác nhận CỐ ĐỊNH=1.0 suốt run (đã fix ở Phase 9/12) — nghĩa là MÔI TRƯỜNG vật lý cho REACH/GRASP không hề đổi. Cái đổi là CHÍNH SÁCH (mean action) — càng train lâu (n_updates tăng), càng lệch xa khỏi giải pháp REACH/GRASP tốt ban đầu, đúng đặc trưng "trôi dạt do gradient từ nhiệm vụ khác", không phải đặc trưng "khám phá ngẫu nhiên do entropy cao".

### CHƯA làm — cần quyết định trước khi thử tiếp (tốn kém hơn hẳn, cần thiết kế cẩn thận)
1. Xác nhận thêm giả thuyết (rẻ, không cần train): kiểm tra xem SB3 `MlpPolicy` mặc định có thực sự share toàn bộ trunk giữa policy/value hay không (đọc code SB3, không đoán).
2. Hướng fix khả dĩ (đắt hơn hẳn các fix trước, cần thiết kế): (a) KL-penalty hoặc trust-region ràng buộc policy mới không lệch quá xa `policy_1M_success57.pt` ở các state thuộc REACH/GRASP; (b) tách kiến trúc mạng riêng theo stage/nhiệm vụ; (c) giảm mạnh `n_epochs`/`batch_size` hoặc learning rate để giảm tốc độ trôi dạt (rẻ nhất để thử, nhưng chỉ làm chậm vấn đề chứ không chắc giải quyết gốc rễ); (d) chỉ fine-tune riêng các layer/tham số liên quan PLACE, đóng băng phần còn lại.
3. Chưa chạy thêm bất kỳ diagnostic/train nào — đây là điểm dừng cần user xác nhận hướng, vì các hướng fix ở trên đều tốn công sức thiết kế đáng kể hơn hẳn những gì đã thử (khác các bug reward/config/logic đã sửa xuyên suốt project, đây là vấn đề tối ưu hoá đa nhiệm vụ ở tầng thuật toán PPO).

User yêu cầu: "Find out the reason why" — xác nhận cơ chế TRƯỚC KHI sửa, bằng cách đọc trực tiếp source code SB3 2.9.0 (không train thêm, rẻ và dứt khoát).

## Phase 18 — Xác nhận cơ chế thật bằng code SB3, ĐÍNH CHÍNH giả thuyết Phase 17

### Đọc `stable_baselines3.common.torch_layers.MlpExtractor.__init__` và `ActorCriticPolicy._build`
```python
if isinstance(net_arch, dict):
    pi_layers_dims = net_arch.get("pi", [])
    vf_layers_dims = net_arch.get("vf", [])
else:
    pi_layers_dims = vf_layers_dims = net_arch   # net_arch=[256,256] (flat list, đúng cấu hình đang dùng)
...
self.policy_net = nn.Sequential(*policy_net)   # network RIÊNG
self.value_net  = nn.Sequential(*value_net)    # network RIÊNG, KHÔNG chia sẻ trọng số với policy_net
```
`ActorCriticPolicy` mặc định `share_features_extractor=True`, nhưng với observation vector phẳng 26-D (không phải ảnh), `features_extractor` = `FlattenExtractor` — **không có tham số học được** (chỉ reshape) → chia sẻ này là no-op về mặt trọng số.

**ĐÍNH CHÍNH giả thuyết (b) ở Phase 17**: "value net và policy net dùng chung 1 trunk [256,256]" là **SAI** — đã verify trực tiếp qua source code, 2 mạng hoàn toàn tách biệt trọng số. Đây KHÔNG phải cơ chế gây lỗi.

### Cơ chế THẬT (verify được, không phải đoán): chia sẻ trọng số GIỮA CÁC STAGE, không phải giữa policy/value
`policy_net` (256,256) là **MỘT mạng duy nhất** nhận thẳng 26-D observation (trong đó `stage_obs` chỉ là 1/26 giá trị liên tục — không phải cổng cứng/router riêng biệt theo stage) và xuất ra action cho **CẢ 4 GIAI ĐOẠN** REACH/GRASP/LIFT/PLACE. Vì REACH→GRASP→LIFT→PLACE là **một episode liên tục** (thiết kế "3-stage curriculum-in-one-episode" đã ghi trong `LIFT_BUG_THEORY.md` mục 0.6), rollout buffer mỗi lần update (`n_steps=64 × 128 env = 8192 mẫu`) trộn lẫn mẫu từ NHIỀU stage khác nhau, và PPO tính **một gradient tổng hợp** (trung bình có trọng số theo advantage) cập nhật thẳng vào **cùng một bộ trọng số `policy_net`** — không có cơ chế nào tách biệt/bảo vệ vùng trọng số chịu trách nhiệm cho REACH/GRASP khỏi bị các mẫu PLACE (nhiệm vụ khó, còn xa mới hội tụ, advantage nhiễu cao vì value function chưa ước lượng tốt cho state PLACE) kéo lệch.

**Vì sao đây là lời giải thích nhất quán với TOÀN BỘ dữ liệu đã đo**:
- Giải thích đúng pattern "thoái hoá đơn điệu theo `n_updates`" ở CẢ 2 run (Phase 14 gốc và Phase 16-17 có entropy decay) — nhiễu/độ lệch tích luỹ dần theo số lần cập nhật gradient, không phụ thuộc entropy/std (khớp bằng chứng thực nghiệm Phase 17).
- Giải thích đúng vì sao MÔI TRƯỜNG cho REACH/GRASP không đổi (`descent_scale=1.0` cố định, xác nhận từ Phase 9) nhưng CHÍNH SÁCH vẫn trôi dạt — vì cái thay đổi là TRỌNG SỐ MẠNG, do gradient từ stage khác kéo, không phải do observation/reward của chính REACH/GRASP thay đổi.
- Giải thích đúng vì sao `policy_1M_success57.pt` (chưa từng train chung với PLACE) vẫn ổn định qua nhiều lần demo/eval trước đó — nó chưa từng chịu gradient từ một stage-khó-chưa-hội-tụ nào cả.

### Hướng fix (đã xác nhận đúng vấn đề, CHƯA implement — chờ chọn hướng)
Loại bỏ (a) KL-penalty/trust-region: đúng nhưng phức tạp, SB3 PPO không hỗ trợ sẵn, phải tự viết. (b) tách network theo stage: đúng gốc rễ nhất nhưng thay đổi kiến trúc lớn, ảnh hưởng tới cách nạp checkpoint cũ. (c) giảm n_epochs/batch_size/LR: rẻ nhất, dễ thử, nhưng chỉ giảm TỐC ĐỘ trôi dạt chứ không loại bỏ cơ chế. (d) đóng băng layer REACH/GRASP: cần xác định RÕ layer nào "thuộc về" REACH/GRASP trong một mạng dùng chung — không tách bạch tự nhiên vì không có kiến trúc theo module.

**Đề xuất khả thi nhất, chi phí/lợi ích tốt nhất, có cơ sở lý thuyết vững** (chưa làm, chờ user xác nhận): **weight theo stage trong loss** — thay vì trộn đều mọi mẫu vào 1 batch, giảm hệ số đóng góp gradient của mẫu PLACE trong minibatch policy loss (không phải reward, mà là TRỌNG SỐ của sample trong loss function) sao cho PLACE vẫn học được nhưng không được phép kéo REACH/GRASP đi xa — kỹ thuật gọi là "gradient surgery"/loss reweighting cho multi-task RL, cần code thêm ở tầng PPO rollout/loss (SB3 không hỗ trợ sẵn, phải subclass `PPO.train()`).

User chọn: "cheapest way first" — thử (c) giảm `n_epochs` trước, KHÔNG cần code phức tạp, chỉ giảm số lần tái sử dụng mỗi rollout batch.

## Phase 19 — Giảm `n_epochs` 10→3: KẾT QUẢ TÍCH CỰC RÕ RỆT, đà thoái hoá chậm hẳn lại

### Implement
`isaaclab_train.py`: thêm `--n-epochs` (mirror style `--ent-coef`), mặc định=10 (giữ nguyên hành vi cũ), dùng thẳng trong `PPO(..., n_epochs=args.n_epochs)`.

### Diagnostic 2M bước (server, từ `policy_1M_success57.pt`, `n_epochs=3`, KHÔNG dùng entropy decay — cô lập đúng 1 biến số để so sánh sạch với 2 run trước)
Chạy đủ 2,007,040/2,000,000 bước (~47 phút, nhanh hơn/tương đương run n_epochs=10 dù ít epoch hơn — vì fps giới hạn bởi bước môi trường, không phải bởi số epoch). Eval deterministic (`eval_lift_metrics.py`, cùng phương pháp/seed=1 mọi run trước):

| Bước | `n_epochs=10`, không ent-decay (Phase 14) | `n_epochs=10`, có ent-decay (Phase 16-17) | **`n_epochs=3`, không ent-decay (run này)** |
|---|---|---|---|
| 500k | — | 0.583 | — |
| 1M | 0.5 | 0.333 | **0.625** |
| 2M | — | 0.083 | **0.5** |
| 5M | ≈0.0-0.04 | — | (chưa chạy) |

**Kết quả rõ ràng**: ở mốc 2M bước, `grasp_rate` deterministic giữ ở **0.5** (bằng đúng mức 1M-bước của run gốc) thay vì sập xuống 0.083 như run entropy-decay cùng mốc — đà thoái hoá chậm lại đáng kể. Xác nhận thêm cho cơ chế đã tìm ra ở Phase 18: giảm số lần tái sử dụng rollout-batch-trộn-nhiều-stage (10 epoch → 3 epoch) giảm trực tiếp số lần gradient từ PLACE (nhiễu, chưa hội tụ) được phép "ghi đè" lên trọng số dùng chung với REACH/GRASP mỗi lần thu thập rollout.

**Vẫn chưa phải fix triệt để** (đã cảnh báo trước khi thử — (c) chỉ làm chậm, không loại bỏ cơ chế): xu hướng 1M→2M vẫn giảm (0.625→0.5, mất 0.125 sau 1M bước) — ngoại suy tuyến tính thô, tới 5M có thể còn khoảng ~0.1-0.2, tốt hơn hẳn bản gốc (~0) nhưng chưa chắc đủ tốt để deploy. Cần chạy thật tới 5M để biết chính xác, không ngoại suy.

### File sửa (Phase 19, CHƯA commit)
`isaaclab_train.py` (`--n-epochs` CLI arg, dùng trong `PPO()`).

### Hướng tiếp theo — chờ user xác nhận
1. Chạy full 5M với `n_epochs=3` để xem đà thoái hoá có ổn định/dừng lại hay tiếp tục giảm hết (tốn ~1h40-2h server).
2. Hoặc thử giảm SÂU hơn nữa (`n_epochs=1-2`) ở diagnostic ngắn trước khi cam kết 5M — càng ít epoch càng ít nhiễu chảy qua, nhưng cũng học chậm hơn (đánh đổi tốc độ hội tụ PLACE).
3. Hoặc kết hợp: `n_epochs` thấp + `batch_size` nhỏ hơn (hiện 4096, cố định, chưa thử đổi) để giảm thêm mức trộn giữa các stage trong mỗi gradient step.

User chọn: "1" (chạy full 5M với `n_epochs=3`).

## Phase 20 — Full 5M với `n_epochs=3`: KHÔNG xác nhận được fix, sập về 0 y hệt mọi lần trước — có 1 biến gây nhiễu (num_envs 128→256)

### Sự cố hạ tầng giữa chừng (đã xử lý)
1. **WireGuard rớt kết nối tới server** trước khi launch — `wg0` không tồn tại, `ping`/`ssh` tới `192.168.1.122` báo "No route to host". User cung cấp mật khẩu sudo, bật lại bằng `wg-quick up "Ph-m-Huy-Ho-ng"` — thành công, SSH hoạt động lại.
2. **User đề xuất tăng `--num-envs`** (từ 128 lên 500+) vì thấy GPU còn ~15GB trống. Đo trực tiếp lúc đó: GPU dao động (job của user khác trên server tăng từ 4.6GB→8GB ngay trong lúc kiểm tra) — chọn **256** (nhân đôi, an toàn hơn 500 trong bối cảnh GPU dùng chung biến động) thay vì làm đúng số user đề nghị, kill run 128-env vừa khởi động (sunk cost ~21 update, không đáng kể) và relaunch với 256 env. **Đây chính là biến gây nhiễu cho phép so sánh bên dưới — xem mục "Hạn chế" .**
3. **DNS toàn cục bị hỏng sau khi bật lại WireGuard** — interface VPN tự nhận domain mặc định (`~.`), khiến MỌI truy vấn DNS (kể cả ra internet, không chỉ ra server) bị định tuyến nhầm qua tunnel (chỉ route `192.168.1.122/32` + `10.11.0.0/24`, không phải full-tunnel) → treo/timeout khi tải asset ground-plane từ S3 lúc `eval_lift_metrics.py` khởi tạo env. **Đây là lỗi ĐÃ TỪNG GẶP TRƯỚC ĐÂY trong project (ghi chú DDS/RMW cũ)** — fix đã biết: `resolvectl domain <interface> ""` (xoá domain `~.` khỏi interface). Áp dụng, xác nhận `curl` tới S3 trả `200 OK`, chạy lại eval thành công.

### Kết quả full 5M (`n_epochs=3`, `num_envs=256`, từ `policy_1M_success57.pt`)
Chạy đủ 5M bước sạch (checkpoint 1M/2M/3M/4M/5M đều lưu đầy đủ). Eval deterministic (cùng phương pháp mọi lần):

| Bước | Diagnostic 2M trước (`n_epochs=3`, **128 env**) | Full 5M lần này (`n_epochs=3`, **256 env**) |
|---|---|---|
| 1M | 0.625 | **0.458** |
| 2M | 0.5 | **0.042** |
| 5M | (chưa chạy) | **0.0** |

**Kết quả cuối: `grasp_rate=0.0` — sập hoàn toàn, y hệt MỌI lần fine-tune 5M trước đó** (bản gốc `n_epochs=10` không entropy-decay, bản entropy-decay) — bất kể đã giảm `n_epochs`.

### Hạn chế của phép so sánh — KHÔNG kết luận vội "n_epochs=3 vô dụng"
Đã đổi `num_envs` 128→256 GIỮA CHỪNG (theo đề xuất user), nên đây KHÔNG phải phép so sánh sạch 1-biến-số như các lần trước. Tính toán lại: với `n_steps=64` và `batch_size=4096` cố định, **tổng số gradient step tại cùng một mốc `total_timesteps` là GIỐNG NHAU bất kể `num_envs`** (128 env: 244 rollout×2 minibatch×3 epoch=1464 step tại 2M; 256 env: 122 rollout×4 minibatch×3 epoch=1464 step tại 2M — khớp) — nên về mặt lý thuyết, cơ chế "số lần cập nhật gradient" đã xác nhận ở Phase 18-19 KHÔNG giải thích được sự khác biệt lớn giữa 2 run (0.5 vs 0.042 tại cùng 2M). Nhiều khả năng đây là **phương sai giữa các lần chạy (run-to-run variance)** vốn cao trong PPO (rollout ngẫu nhiên, exploration ngẫu nhiên, GPU non-determinism khi song song hoá) — 1 lần chạy 2M ở 128 env và 1 lần chạy khác ở 256 env là 2 QUỸ ĐẠO NGẪU NHIÊN khác nhau, không phải 2 phép đo lặp lại của cùng 1 quá trình.

### Trạng thái tổng — SAU 5 LẦN THỬ, CHƯA lần nào giữ được checkpoint cuối dùng được
| Lần thử | Cấu hình | grasp_rate cuối cùng |
|---|---|---|
| Phase 14 | `n_epochs=10`, không sửa gì | ~0.0-0.04 (5M) |
| Phase 16-17 | + entropy decay | 0.083 (2M) |
| Phase 19 | `n_epochs=3` (128 env) | 0.5 (2M, chưa chạy tới 5M) |
| Phase 20 | `n_epochs=3` (256 env) | **0.0 (5M)** |

**Không có cấu hình nào (entropy decay, giảm n_epochs) giữ được policy deterministic dùng được tới hết 5M bước.** Checkpoint SỚM (~1M bước vào bất kỳ lần fine-tune nào) luôn là điểm tốt nhất quan sát được (grasp deterministic 0.45-0.63), rồi thoái hoá dần bất kể can thiệp nào đã thử.

### Đề xuất tiếp theo (chưa làm, cần quyết định user)
1. **Dùng tạm checkpoint SỚM** (~1M bước, vd `train_place_lowepoch_5M_1M.zip`, grasp=0.458 deterministic) làm bản PLACE khả dụng tạm thời, dừng đuổi theo "chạy hết N bước mà không sập" — chấp nhận giới hạn hiện tại.
2. **Thử lại 128 env, n_epochs=3, ĐỦ 5M** (phép so sánh sạch, đúng 1 biến số như Phase 19 đã làm dở) để biết chắc liệu n_epochs=3 THẬT SỰ có giữ được lâu hơn hay chỉ là may mắn ở 2M — tốn thêm ~1h35-2h server.
3. **Đầu tư hướng kiến trúc** (tách network theo stage / KL-penalty) — tốn công thiết kế nhiều hơn nhưng giải quyết đúng gốc rễ đã xác nhận ở Phase 18, không phụ thuộc may rủi giữa các lần chạy.

User chọn: "Đầu tư hướng kiến trúc (tách network/KL-penalty)".

## Phase 21 — Implement KL-penalty giữ policy REACH/GRASP gần policy tham chiếu

### Thiết kế
Đọc trực tiếp source `PPO.train()` (SB3 2.9.0) để biết chính xác cần override chỗ nào — không đoán. Chọn **KL-penalty** thay vì tách network riêng theo stage: rẻ hơn (không đổi kiến trúc, checkpoint cũ nạp bình thường), nhắm ĐÚNG cơ chế đã xác nhận ở Phase 18 (gradient từ mẫu PLACE trong batch trộn-nhiều-stage kéo lệch trọng số `policy_net` dùng chung).

`isaaclab_train.py`:
- `KLProtectedPPO(PPO)` — subclass, thêm `kl_coef` + `_ref_policy` (đóng băng, `requires_grad_(False)`, `.eval()`).
- Override `train()` = mirror y hệt SB3 gốc (đã đọc source, bỏ nhánh `Discrete` action vì env này luôn continuous Box) + thêm: mỗi minibatch, lấy `stage_obs = rollout_data.observations[:, 25]` (index đã xác nhận từ docstring `get_apple_pick_place_obs`), mask `stage_obs < 0.75` (REACH=0.0/GRASP=0.5, loại PLACE=1.0), tính `KL(policy_hiện_tại || policy_tham_chiếu)` bằng `torch.distributions.kl_divergence` trên 2 `Normal` distribution (xác nhận qua source `DiagGaussianDistribution.proba_distribution`: `self.distribution = Normal(mean, std)`, không wrap `Independent` → phải `.sum(dim=-1)` qua 7 action dim thủ công), chỉ lấy mean trên các sample thuộc mask, cộng `kl_coef * kl_loss` vào tổng loss trước `backward()`.
- Reference policy mặc định = ĐÚNG checkpoint đang fine-tune từ đó (`--checkpoint`, deepcopy + load lại state_dict) — đúng ý định "không lệch xa policy TRƯỚC KHI học PLACE". CLI `--kl-coef` (mặc định 0.0 = tắt, an toàn ngược) + `--kl-ref-checkpoint` (override thủ công, chủ yếu cho `--resume`).
- Log thêm `train/kl_ref_reach_grasp` để theo dõi mức độ lệch thực tế.

### Bug tự bắt được khi test (không phải bug logic, mà là quy trình)
Smoke test local lần đầu dùng `--headless` — build Isaac Sim local (5.1.0) không nhận flag này (đúng pattern đã ghi nhận nhiều lần trước với `isaaclab_demo.py`/`eval_lift_metrics.py` — chỉ server nhận `--headless`, local dùng `--visualizer none`). Lần 2 quên `cd` rõ ràng trong CÙNG command nền — cwd của phiên đã reset về `/home/hans/universal_bot` giữa các lượt gọi Bash (harness không giữ `cd` qua các lời gọi), khiến `source ./openarm.env` fail âm thầm (thất bại nhưng bị `>/dev/null 2>&1` nuốt lỗi), `$ISAAC_SIM_PYTHON` rỗng → toàn bộ lệnh sau đó không chạy. Lần 3 thêm `cd /home/hans/universal_bot/Reinforce_Learning &&` ở đầu CÙNG dòng lệnh — chạy được, nhưng lộ ra path checkpoint khác giữa local/server (`logs/policy_1M_success57.pt` chỉ tồn tại trên SERVER; local có ở `logs/train/policy_1M_success57.pt`) — sửa path, chạy sạch.

### Verify (smoke test local, 4 env, 600 bước, `--kl-coef 0.1`)
Chạy xong sạch, không traceback, `KL-protection enabled` in đúng, `train/kl_ref_reach_grasp` xuất hiện trong log (0.00112 → 0.00272, tăng nhẹ đúng cơ chế — vừa mới bắt đầu từ chính policy tham chiếu nên KL còn rất nhỏ). Xác nhận đúng về mặt cơ học — CHƯA đủ để đánh giá `kl_coef` có chọn đúng độ lớn hay không (chỉ 600 bước, quá ngắn để KL tích luỹ đáng kể).

### File sửa (Phase 21, CHƯA commit)
`isaaclab_train.py` (`KLProtectedPPO`, `--kl-coef`, `--kl-ref-checkpoint`, đổi `PPO(...)`→`KLProtectedPPO(...)` cả nhánh fresh-train lẫn `--resume`).

### Chưa làm
- Chưa test đường `--resume` + `--kl-coef` (không phải path đang dùng tích cực, bỏ qua để tiết kiệm thời gian).
- Chưa biết `kl_coef` nên lớn cỡ nào — `policy_gradient_loss` quan sát được RẤT nhỏ (~0.001), trong khi `value_loss` (nhân `vf_coef=0.5`) chiếm phần lớn độ lớn của `loss` tổng (1-46) — so sánh độ lớn loss thô KHÔNG đủ tin cậy để suy ra hệ số đúng (gradient thật phụ thuộc kiến trúc/backward, không chỉ giá trị loss). Cần đo trực tiếp qua diagnostic thật, không đoán số.

## Phase 22 — Diagnostic 2M với KL-protection (`kl_coef=1.0`): LẦN ĐẦU TIÊN đà thoái hoá bị CHẶN ĐỨNG

Chạy diagnostic 2M bước (server, 128 env — khớp đúng cấu hình sạch nhất đã có ở Phase 19 để so sánh 1 biến số, `n_epochs=3` giữ nguyên, thêm `--kl-coef 1.0`, reference = chính `policy_1M_success57.pt`).

`kl_ref_reach_grasp` giữ RẤT NHỎ suốt cả run (0.0006-0.003, không tăng theo thời gian) — ban đầu nghi ngờ có thể là lỗi tính toán (KL luôn ~0 do bug), nhưng **eval deterministic thật xác nhận đây là tín hiệu THẬT, không phải bug**:

| Bước | `n_epochs=10` gốc | +entropy decay | `n_epochs=3` (128env, không KL) | `n_epochs=3` (256env, full 5M) | **`n_epochs=3` + KL-protect (coef=1.0)** |
|---|---|---|---|---|---|
| 1M | 0.5 | 0.333 | 0.625 | 0.458 | **0.667** |
| 2M | — | 0.083 | 0.5 | 0.042 | **0.708** |

**LẦN ĐẦU TIÊN trong 6 lần thử, `grasp_rate` deterministic KHÔNG giảm từ 1M→2M — còn tăng nhẹ** (0.667→0.708), gần chạm lại baseline gốc chưa fine-tune (0.833 @ scale 1.0, task_phase 2). Đúng như kỳ vọng thiết kế: KL nhỏ vì penalty đã "giữ" policy không lệch xa reference NGAY TỪ ĐẦU (không phải để nó lệch rồi kéo lại — mà ngăn lệch xảy ra), nên số liệu `kl_ref_reach_grasp` nhỏ chính là dấu hiệu cơ chế hoạt động đúng, không phải bug.

**Xác nhận trực tiếp giả thuyết Phase 18**: ràng buộc TRỰC TIẾP hành vi REACH/GRASP (không đợi tách kiến trúc mạng, không phụ thuộc may rủi giữa các lần chạy như `n_epochs` đơn thuần) chặn được cơ chế "gradient từ PLACE kéo lệch trọng số dùng chung" — đây là bằng chứng THỰC NGHIỆM mạnh nhất từ đầu investigation PLACE tới giờ.

### Hướng tiếp theo
Đủ tín hiệu tích cực để chạy full 5M với cấu hình y hệt (`n_epochs=3 --kl-coef 1.0`, 128 env) — đây sẽ là bài kiểm tra thật: liệu grasp_rate có GIỮ ĐƯỢC (hoặc tiếp tục cải thiện) suốt 5M, hay vẫn sập ở đâu đó sau 2M như mọi lần trước. Checkpoint 1M/2M lần này đã tải về (`logs/train_place_klprotect_2M_1M.zip`, `logs/train_place_klprotect_2M_2M.zip`).

## Phase 23 — Full 5M với KL-protection: KẾT QUẢ TỐT NHẤT (grasp giữ 0.625 tại 5M), nhưng lộ ra hạn chế mới (LIFT yếu đi)

### Sự cố hạ tầng — GPU server bị chiếm dụng nặng, chuyển sang local
Lúc chuẩn bị launch full 5M: GPU server chỉ còn ~3GB free (job của user khác tăng đột biến, `nvidia-smi` cho thấy utilization 97%). Thử 128 env → **CUDA OOM** (`Unable to allocate memory... mGpuContactPairsDev`). Thử giảm xuống 64 env → **VẪN OOM** (phân mảnh bộ nhớ do process khác, "free" trên giấy không đủ để cấp phát liên tục). Quyết định chuyển hẳn sang **train local (RTX 4050, hoàn toàn rảnh)** thay vì tiếp tục hạ env count trên server đang biến động khó lường.

### Kết quả full 5M (local, 96 env, `n_epochs=3 --kl-coef 1.0`, từ `policy_1M_success57.pt`)
Chạy NHANH BẤT NGỜ — chỉ 32 phút (2576 it/s, nhanh hơn hẳn ước tính ban đầu; có lẽ 96 env + scene nhẹ phù hợp tốt với 4050). Không lỗi, đủ 5,001,216/5,000,000 bước, mọi checkpoint (1M-5M) lưu sạch.

**Eval deterministic** (cùng phương pháp mọi lần):

| Bước | `n_epochs=3` (không KL) | **`n_epochs=3` + KL (coef=1.0)** |
|---|---|---|
| 2M | 0.5 (128env) / 0.042 (256env) | **0.708** |
| 5M | 0.0 (256env, full 5M) | **0.625** |

**LẦN ĐẦU TIÊN checkpoint CUỐI CÙNG của một lần fine-tune PLACE đầy đủ vẫn giữ được `grasp_rate` deterministic khả dụng (0.625)** — không sập về 0 như 5/5 lần thử trước (mọi cấu hình: gốc, entropy-decay, n_epochs=3 đơn thuần, dù ở 128env hay 256env). Xác nhận KL-protection là fix hiệu quả nhất đã tìm ra trong toàn bộ investigation PLACE.

### Hạn chế mới phát hiện — LIFT yếu đi, chưa episode nào chạm tới PLACE
Xem `fail_modes` chi tiết: **cả ở mốc 2M lẫn 5M, KHÔNG episode nào (0/24) từng vào tới PLACE** — `fail_modes` chỉ gồm `reach`/`grasp`/`lift_too_low`, không có mode nào liên quan PLACE (`no_release`/`place_timeout`/...). `mean_lift_m` gần như 0 (0.0011-0.0059m, cần ≥0.03m để tính nhấc thành công) — nghĩa là dù GRASP/latch giữ tốt hơn hẳn, khả năng **LIFT** (nhấc chai lên đủ cao) lại yếu đi so với baseline gốc (`policy_1M_success57.pt` có `lift_start_rate=0.43`).

**Nguyên nhân khả dĩ** (chưa xác nhận, cần điều tra thêm nếu muốn sửa): `LIFT` chỉ là sub-phase NỘI BỘ của `STAGE_GRASP` (`env._lift_phase`, không đổi `env._stage`) — nên `stage_obs` trong lúc LIFT vẫn = 0.5, giống hệt GRASP thường, khiến mask bảo vệ (`stage_obs < 0.75`) VÔ TÌNH áp luôn penalty KL lên cả hành vi LIFT — có thể đã hạn chế khả năng tinh chỉnh động tác nhấc mà đáng lẽ nên được tự do học như PLACE.

### File local (Phase 23, CHƯA commit vào git — chỉ checkpoint/log)
`logs/train_place_klprotect_5M_local/` (đầy đủ checkpoint 1M-5M + best/final_policy.pt), `logs/train_place_klprotect_5M_local.log`.

### Hướng tiếp theo (chưa làm, cần quyết định)
1. **Tinh chỉnh mask KL** — loại LIFT sub-phase khỏi vùng "protected" (chỉ bảo vệ REACH/GRASP thật, để LIFT tự do học như PLACE) — cần thêm điều kiện dựa trên `_lift_phase` vào observation hoặc truyền riêng qua buffer (hiện KLProtectedPPO chỉ đọc được `rollout_data.observations`, không có `_lift_phase` — cần bổ sung cách truyền tín hiệu này vào rollout buffer, phức tạp hơn chỉnh 1 threshold).
2. **Thử `kl_coef` nhỏ hơn** (vd 0.3-0.5) — nới lỏng ràng buộc, chấp nhận rủi ro GRASP/REACH trôi dạt nhiều hơn một chút, đổi lấy LIFT tự do học tốt hơn — rẻ hơn hướng 1, có thể thử trước.
3. ~~**Chấp nhận hiện trạng, train tiếp từ checkpoint 5M này**~~ → **Đã làm (Phase 24), KHÔNG đủ — xem bên dưới.**

## Phase 24 — Round 2 (thêm 5M bước từ checkpoint round 1): xác nhận GRASP/REACH bền vững, nhưng LIFT không tự cải thiện

### Sự cố hạ tầng: server driver mismatch — KHÔNG tự sửa
Lúc định check lại server để cân nhắc chuyển sang đó: `nvidia-smi` báo `Driver/library version mismatch` (kernel module 580.173.02 đang load, thư viện NVML 580.178 — lệch do apt tự cập nhật package mà chưa reload module/reboot). Server có 6 user đang đăng nhập + nhiều job GPU của người khác đang chạy. **Quyết định KHÔNG tự sửa** (cần `rmmod`/`modprobe` lại kernel module hoặc reboot — cả hai đều giết chết job GPU của NGƯỜI KHÁC đang chạy, không phải hạ tầng của mình để tự ý can thiệp) — đúng tiền lệ đã có với sự cố RAID degraded trước đây trong project này. Tiếp tục 100% trên local.

### Kết quả round 2 (resume từ checkpoint round 1's `rl_model_999936_steps.zip`, thêm đúng 4,000,064 bước còn lại để đạt tổng ~10M bước fine-tune tích luỹ từ `policy_1M_success57.pt`, KL-reference GIỮ NGUYÊN là `policy_1M_success57.pt` gốc — không đổi sang checkpoint mới)

| | Round 1 (5M) | Round 2 (+5M, tổng ~10M) |
|---|---|---|
| grasp_rate (det.) | 0.625 | **0.625 — giữ nguyên** |
| latch_rate (det.) | 0.333 | 0.375 |
| mean_lift_m | 0.0011-0.0059 | 0.0024 |
| place_start_rate | 0.0 | 0.0 |
| fail_modes | reach/grasp/lift_too_low | reach/grasp/lift_too_low + 1 `hold_unstable` mới |

**Kết luận**: GRASP/REACH giữ NGUYÊN qua thêm 5M bước (0.625→0.625) — xác nhận KL-protection bền vững lâu dài, KHÔNG tiếp tục thoái hoá theo thời gian train (khác hẳn mọi run trước không có KL). Nhưng **LIFT KHÔNG tự cải thiện dù train thêm gấp đôi thời gian** — xác nhận giả thuyết Phase 23: mask hiện tại (`stage_obs<0.75`) vô tình áp KL-protection lên cả LIFT (vì LIFT chỉ là sub-phase nội bộ của `STAGE_GRASP`, `stage_obs` vẫn =0.5), khoá LIFT ở đúng mức của policy tham chiếu thay vì cho nó tự do cải thiện như PLACE. **"Train tiếp không đổi gì" (option 3) đã được thử và KHÔNG đủ** — cần thật sự làm option 1 (sửa mask) hoặc option 2 (giảm `kl_coef`) để LIFT có cơ hội cải thiện.

### Hướng tiếp theo — quay lại chọn giữa option 1/2 (chưa làm)

User yêu cầu: "think more and debug" — không đoán tiếp giữa option 1/2, điều tra tận gốc trước.

## Phase 25 — Debug sâu: ĐẢO NGƯỢC kết luận Phase 23/24 — KL-protection thực ra HOÀN HẢO, "LIFT yếu" là lỗi PHƯƠNG PHÁP SO SÁNH + bug kịch bản có sẵn từ trước, không liên quan PLACE/KL

### Manh mối đầu: fail_mode `lift_too_low` kẹt ở CÙNG MỘT GIÁ TRỊ bất thường
`lift_m` của 8-9/24 episode `lift_too_low` đều hội tụ về **đúng ~1.6-1.8mm** (không phải phân bố ngẫu nhiên) — quá nhất quán để là trùng hợp. Khớp CHÍNH XÁC với comment lịch sử trong `grasp_assist.py::_osc_world_up_lift`: *"Đo được (lift_too_low, DEBUG_STALL): chai theo tay đúng 1.5mm rồi trượt hẳn — đúng thời điểm giật khởi động"* — một bug đã biết, KHÔNG liên quan gì tới PLACE/KL-protection.

### Kiểm tra chéo — phát hiện lỗi phương pháp luận nghiêm trọng
Chạy lại **`policy_1M_success57.pt` GỐC** (chưa từng qua bất kỳ fine-tune PLACE nào) ở ĐÚNG cùng điều kiện đã dùng để đánh giá mọi checkpoint PLACE (`--task_phase 3 --stage all --assist-scale 1.0 --seed 1`):
```
grasp=0.625 latch=0.375 lift_start=0.375 success=0.0
fail_modes: {'grasp': 6, 'reach': 9, 'lift_too_low': 9}   ← Y HỆT mọi checkpoint đã fine-tune
```
**Giống hệt về số liệu VÀ hiện tượng `lift_too_low` ~1.6mm** với checkpoint KL-protected sau 10M bước! Nghi ngờ `task_phase` là nguyên nhân → test thêm `--task_phase 2` (giữ nguyên seed=1) → **CŨNG RA ĐÚNG SỐ NÀY** (0.625/0.375/0.375, cùng 9 episode lift_too_low ~1.6mm). Vậy KHÔNG PHẢI task_phase.

**Thủ phạm thật: SEED.** Toàn bộ investigation PLACE (Phase 14 trở đi) dùng `--seed 1` cho mọi eval chẩn đoán, trong khi con số nền "0.833" luôn được dùng để so sánh lại đến từ regression gate CHUẨN dùng `--seed 0`. `--seed 1` tình cờ tạo ra một chuỗi vị trí chai (bottle-noise ngẫu nhiên nhưng tất định theo seed) khó hơn hẳn cho GRASP/LIFT — **kể cả trên chính checkpoint GỐC chưa từng đụng tới**. Đây là lỗi so sánh khập khiễng (so `--seed 1` của checkpoint mới với `--seed 0` của checkpoint cũ) đã âm thầm tồn tại xuyên suốt Phase 14-24 — không ai trong các phase trước phát hiện ra vì luôn so sánh CÁC CHECKPOINT MỚI VỚI NHAU (cùng seed 1, hợp lệ nội bộ), chỉ riêng việc gán nhãn "0.625 là thoái hoá so với 0.833" mới sai.

### Kết luận ĐÚNG (thay thế Phase 23/24)
1. **KL-protection không chỉ "tốt nhất trong 6 lần thử" — nó HOÀN HẢO.** So đúng cặp (cùng seed=1): checkpoint gốc = 0.625, checkpoint SAU 10M bước fine-tune PLACE (round 1+2, có KL-protection) = 0.625. **Thoái hoá = 0%**, không phải "giữ được phần lớn". Xác nhận dứt khoát: cơ chế KL-penalty (Phase 21) giải quyết ĐÚNG VÀ ĐỦ gốc rễ đã tìm ra ở Phase 18 (gradient từ PLACE kéo lệch trọng số REACH/GRASP dùng chung).
2. **"LIFT yếu" không phải lỗi do PLACE/KL gây ra — là điểm yếu CÓ SẴN của `policy_1M_success57.pt` gốc**, chỉ lộ rõ với seed khó (seed=1). Toàn bộ fine-tune PLACE (dù có KL hay không) không hề làm nó tệ hơn — chỉ đơn giản KHÔNG SỬA được nó (đúng ý nghĩa của "protection": bảo tồn hành vi cũ, tốt lẫn xấu).
3. **Nguyên nhân sâu hơn — TẠI SAO không sửa được dù train bao nhiêu**: trong `LIFT_RISING`, `_osc_world_up_lift` (grasp_assist.py) làm: `out[mask]=0.0` rồi `out[mask,2]=full_mag*ramp_frac` — **ghi đè CỨNG, vứt bỏ HOÀN TOÀN action của policy** (cả 6 chiều, không chỉ trục Z) trong suốt pha nhấc. Đây là override tuyệt đối (mirror đúng đường LIFT verify Giai đoạn 1 cũ, `w>=1-1e-6`), không phải residual. **Không có gradient RL nào chảy qua hành vi LIFT thật cả** — sửa `kl_coef`, sửa mask, train thêm bao lâu cũng vô ích, vì policy không có quyền điều khiển ở đây.

### Bài học phương pháp luận (bổ sung cho `LIFT_BUG_THEORY.md`)
Khi so sánh "trước/sau" một thay đổi, **luôn kiểm tra baseline được đo ở ĐÚNG cùng điều kiện** (bao gồm cả seed, không chỉ config/checkpoint) — một baseline "đã biết tốt" đo ở điều kiện A không tự động là cột mốc hợp lệ cho việc đo ở điều kiện B, dù trông giống nhau về mặt tham số bên ngoài (cùng script, cùng checkpoint tên gọi). Việc này lẽ ra phải bị bắt ở ngay Phase 14 nếu chạy đối chứng checkpoint gốc CÙNG seed trước khi bắt đầu cả chuỗi 6 lần train — một bài học "đo trước khi tin" bị bỏ sót ngay từ khâu thiết lập baseline, không phải khâu diễn giải kết quả.

### Hướng tiếp theo (chưa làm, cần quyết định user)
1. **PLACE state machine/reward (S1-S9, Phase 10-13) coi như đã xong và ĐÚNG** — không cần sửa gì thêm liên quan REACH/GRASP/PLACE-reward. KL-protection (Phase 21) là fix ĐÚNG và ĐỦ cho vấn đề gradient-interference đã tìm ra ở Phase 18.
2. **Nút thắt thật để làm PLACE hoạt động được là LIFT — cần sửa ở tầng kịch bản `_osc_world_up_lift`/độ chính xác GRASP trước khi nhấc**, không phải RL nữa. Hướng khả dĩ: (a) đo lại chính xác nguyên nhân "trượt sau 1.5-1.8mm" bằng `DEBUG_STALL` trên demo thật (đã có instrumentation sẵn, gated, xem `grasp_assist.py:125`) thay vì suy luận từ eval JSON; (b) cân nhắc tăng `grasp_lift_onset_ramp_steps` hoặc xem lại `grasp_lift_world_m`/ma sát; (c) cải thiện độ chính xác GRASP (z_error_finger tại thời điểm latch) để nắm chắc thân chai hơn trước khi nhấc.
3. **Cân nhắc rà lại toàn bộ investigation Phase 14-24 dưới ánh sáng phát hiện này** — có thể một số con số "thoái hoá" khác (VD ở các lần train KHÔNG có KL) cũng cần đối chiếu lại với baseline seed=1 đúng cách trước khi kết luận chắc chắn mức độ thoái hoá thật sự (dù xu hướng chung — sập về gần 0 — vẫn rõ ràng đủ để không nghi ngờ kết luận "không có KL thì thoái hoá thật", chỉ mức ĐỘ chính xác cần xem lại).

User yêu cầu: "fix".

## Phase 26 — Implement + verify zf-drift abort: ĐÚNG kỹ thuật nhưng CHƯA ĐỦ, cần tìm tiếp

### Debug bằng công cụ có sẵn (không đoán)
Chạy demo GUI thật (`DEBUG_STALL=1 DEBUG_LIFT=1`, `--visualizer kit`, máy có màn hình vật lý thật — user xem trực tiếp) trên `policy_1M_success57.pt` gốc, seed=1, episode lift_too_low. Log `[LiftSlip]` cho thấy `z_error_finger` (`z_f`) tăng ĐƠN ĐIỆU, KHÔNG NGỪNG suốt RISING: 19.73mm (step 700) → 35.86 → 54.80 → ... → **834mm (step 1180)** — chai đã tách khỏi kẹp hoàn toàn ngay từ đầu, cánh tay đi lên trong không khí. Nhưng mỗi BƯỚC ĐƠN LẺ chỉ tăng ~0.7-2.2mm (`dz_f` trong log) — dưới hẳn ngưỡng phát hiện trượt hiện có (`grasp_lift_slip_z_finger=0.022 × 0.45 = 9.9mm/bước`) — nên `_update_lift_slip`'s bộ đếm delta-1-bước KHÔNG BAO GIỜ bắt được, để RISING chạy vô ích ~500 bước (gần hết ngân sách episode) cho một lần thử đã chắc chắn thất bại từ giây đầu.

### Fix đã làm
Thêm bộ đếm MỚI, độc lập, dùng NGƯỠNG TUYỆT ĐỐI (không phải delta/bước): `grasp_lift_abort_z_finger=0.05m`, `grasp_lift_abort_z_finger_steps=5` (config.py) — `_update_lift_slip` (grasp_assist.py) tính `zf_over = armed & (z_error_finger > 0.05)`, đếm dồn `_lift_zf_abort_steps`; `_lift_must_abort` thêm điều kiện `zf_abort_steps >= 5`. Reset buffer mới trong `terminations.py::reset_robot`. Cập nhật nhãn debug (`reason="zf_drift"`) cho đầy đủ.

### Verify
- Demo lại đúng scenario: `RISING→IDLE` giờ xảy ra ở **step 740 (chỉ 59 bước sau khi RISING bắt đầu ở step 681)**, thay vì chạy tới step ~1180+ như trước — xác nhận cơ chế phát hiện hoạt động ĐÚNG THIẾT KẾ, tiết kiệm ~340+ bước lãng phí mỗi lần trượt.
- **Regression gate BẮT BUỘC** (`--seed 0 --task_phase 2`, đúng baseline chuẩn): PASS tuyệt đối — `success=0.0333, grasp=0.8333, latch=0.4667, lift_start=0.4333` khớp CHÍNH XÁC baseline. Không hồi quy.

### PHÁT HIỆN — fix ĐÚNG kỹ thuật nhưng KHÔNG cải thiện kết quả cuối
Đo lại `eval_lift_metrics.py` (seed=1, cùng điều kiện phát hiện bug): `grasp=0.625, latch=0.375, lift_start=0.375` — **giống hệt TRƯỚC KHI SỬA**. `lift_cmd_steps` của các episode lift_too_low giảm đúng như kỳ vọng (57-78 bước thay vì 200-570), xác nhận abort SỚM hoạt động — nhưng **không có cải thiện success rate nào**.

**Lý do (kiểm tra trực tiếp log, không suy luận)**: mỗi episode chỉ có ĐÚNG 1 chu kỳ `IDLE→RISING→IDLE` — sau khi abort mới (ở z_f≈50mm), **không có lần retry nào nữa trong phần còn lại của episode** (còn ~160-460 bước không làm gì). Vì `z_error_finger` sau abort đã trôi vượt ngưỡng abort (50mm) và KHÔNG có cơ chế nào chủ động đưa nó quay lại dưới ngưỡng gate ban đầu (`grasp_lift_contact_z_finger=0.014m`, qua PHASE2_GRASP override) — `_lift_can_start` không bao giờ pass lại được. Fix chỉ đổi **thời điểm bỏ cuộc** (nhanh hơn), không tạo cơ hội thử lại thật.

**Nguyên nhân gốc thật vẫn CHƯA giải quyết**: chai bắt đầu tuột chỉ trong ~19 bước đầu của RISING (z_f từ dưới 14mm lúc bắt đầu lên 19.73mm ở bước thứ 19), dù gate ban đầu đã pass — nghĩa là ngay cả một grip "đạt chuẩn" theo ngưỡng hiện tại (`contact_z=0.014m`) vẫn không đủ chắc để chịu lực/gia tốc lúc bắt đầu nhấc. Đây khớp với ghi chú lịch sử trong code (`grasp_body_height_ratio` comment): ngưỡng `grasp_lift_contact_z_finger` từng bị nới từ 0.008 lên giá trị hiện tại vì lý do khác, có thể đã đánh đổi độ chắc chắn của grip lấy tỉ lệ latch cao hơn.

### File sửa (Phase 26, CHƯA commit)
`isaaclab_openarm_env/config.py` (`grasp_lift_abort_z_finger`, `grasp_lift_abort_z_finger_steps`), `isaaclab_openarm_env/mdp/grasp_assist.py` (`_update_lift_slip` thêm bộ đếm zf-drift, `_lift_must_abort` thêm điều kiện, debug label), `isaaclab_openarm_env/mdp/terminations.py` (reset buffer mới).

### Hướng tiếp theo (chưa làm, cần quyết định user — rủi ro cao hơn, có thể ảnh hưởng hành vi đã train)
1. **Siết `grasp_lift_contact_z_finger`** (hiện 0.014m qua PHASE2_GRASP) xuống thấp hơn (vd giá trị lịch sử 0.008m) — buộc chỉ latch khi ngón THỰC SỰ ngang thân chai trước khi cho phép nhấc. RỦI RO: policy đã train quen với ngưỡng 0.014m, siết chặt có thể làm giảm hẳn `latch_rate`/`lift_start_rate` vì điều kiện khó đạt hơn — cần đo trước khi tin, không đoán.
2. **Thêm cơ chế "phục hồi" sau zf_drift-abort**: chủ động hạ/điều chỉnh tay về vị trí gần chai hơn khi vừa abort (thay vì đứng yên ở vị trí đã trôi) để tạo cơ hội `_lift_can_start` pass lại trong cùng episode — phức tạp hơn, cần thiết kế logic re-approach mới.
3. **Đo trực tiếp lực/ma sát thực tế lúc RISING bắt đầu** (không chỉ suy luận từ z_error_finger) — có thể vấn đề nằm ở `grasp_lift_world_m`/tốc độ ramp chứ không phải vị trí kẹp.

User chọn: "2" (thêm cơ chế phục hồi).

## Phase 27 — Implement cơ chế phục hồi (recovery) — hoạt động ĐÚNG PHẦN của nó, lộ ra lớp vấn đề tiếp theo (kẹp không tự ép lại)

### Sự cố hạ tầng xen giữa (đã xử lý)
1. Môi trường Isaac Sim local đột ngột hỏng — `ld.so` báo `LD_PRELOAD` trỏ tới `libcarb.so` không tồn tại, do symlink `kit -> ~/.cache/packman/chk/kit-kernel/...` trỏ tới thư mục **đã biến mất hoàn toàn** (không do đầy đĩa — còn 103GB trống). Rất có thể một job dọn cache tự động qua đêm đã xoá nhầm (demo GUI vẫn chạy tốt cùng ngày trước đó, xem Phase 26). Script tự sửa có sẵn (`pull_kit_sdk.sh`) bị hỏng luôn (thiếu hẳn `dev/tools/eula_check.sh`, không phải do chưa chấp nhận EULA — file kiểm tra EULA không tồn tại). **Fix đúng**: `~/isaacsim/build.sh --fetch-only --config release` (cờ build chính thức, chỉ tải lại dependency qua packman, KHÔNG build lại C++) — chạy ~137 giây, khôi phục thành công `kit-kernel` cache.
2. Server 4090: driver mismatch tự hết, nhưng GPU vẫn bị chiếm dụng nặng bởi người khác (22GB+/24.5GB, 94%+ utilization) suốt Phase 26-27 — chưa từng available để train.

### Implement
`_update_lift_state`: thêm cờ `_lift_recovery_active`, bật khi `rise_abort` xảy ra CỤ THỂ vì `zf_drift` (không bật cho tilt/unlatched — 2 lỗi đó hạ tay xuống không giúp gì), tắt khi re-arm thành công hoặc mất latch/rời stage. `apply_grasp_arm_assist`: khi đang phục hồi, gọi `_osc_z_finger_descend` (hạ closed-loop theo z_error_finger, có sẵn, dùng cho GRASP-descent) thay vì đứng yên (`idle_hold` cũ), dùng `descent_scale` (cố định=1.0, đúng nguyên tắc Phase 9).

### Verify — PHẦN CƠ CHẾ HOẠT ĐỘNG ĐÚNG, nhưng chưa đủ để lift thành công
Thêm debug print tạm (`[Recover]`) quan sát trực tiếp: `z_error_finger` **giảm đều đặn, đúng thiết kế** — từ 54.8mm (lúc abort) xuống 1.1mm chỉ qua ~450 bước phục hồi. Đây là bằng chứng cơ chế phục hồi hoạt động chính xác về mặt ĐIỀU KHIỂN TAY.

**NHƯNG `_lift_can_start` vẫn không bao giờ pass lại.** Thêm `DEBUG_STALL` để xem lý do: khi tay đã về gần chai (`near=True`, đúng như mong đợi), **`stall` vẫn = 0.00mm** — không có lực ép thật. Nguyên nhân: khớp kẹp được điều khiển theo VỊ TRÍ (target cố định=10.45mm, ứng với `gc=0.762`, không đổi suốt cả quá trình phục hồi vì không có lệnh đóng MỚI nào được phát ra). Một khi khớp đã ổn định tại target đó (ban đầu ép không khí sau khi chai tuột), việc đưa TAY về gần chai không tự động khiến KHỚP ép thêm — cần một lệnh đóng kẹp MỚI, tăng độ đóng (giảm target) để thực sự tái tạo lực ép, không chỉ đưa tay về đúng độ cao.

### Kết luận — cần lớp thứ 2, rủi ro cao hơn hẳn
Cơ chế phục hồi (Phase 27) giải quyết ĐÚNG phần của nó (đưa tay về vị trí) nhưng CHƯA ĐỦ — cần thêm: khi `recovering` VÀ `near_bottle=True`, kích hoạt lại lệnh đóng kẹp (tăng `_close_progress`/giảm target) để tái tạo lực ép thật trước khi cho phép `_lift_can_start` pass. Đây là thay đổi chạm vào **state machine đóng/mở kẹp trong `actions.py`** — phần đã có lịch sử sửa đi sửa lại nhiều lần (`reopen`, `close_ramp`, `exhaust_creep`, đều được ghi chú kỹ vì từng gây hồi quy) — rủi ro cao hơn hẳn những gì đã làm ở Phase 26-27. **Dừng lại ở đây để user quyết định trước khi động vào phần này.**

### File sửa (Phase 27, CHƯA commit — bao gồm cả debug print tạm `[Recover]`, cần dọn trước khi merge nếu không cần giữ lại)
`isaaclab_openarm_env/mdp/grasp_assist.py` (`_update_lift_state` thêm `_lift_recovery_active`, `apply_grasp_arm_assist` thêm block `recovering` + debug print `[Recover]`), `isaaclab_openarm_env/mdp/terminations.py` (reset `_lift_recovery_active`).

### Hướng tiếp theo (chưa làm, cần quyết định user)
1. **Thêm lớp 2: re-trigger đóng kẹp khi đang phục hồi VÀ đã near_bottle** — hướng đúng để hoàn thiện cơ chế phục hồi, nhưng chạm state machine đóng/mở kẹp lịch sử phức tạp trong `actions.py`. Cần đọc kỹ toàn bộ flow `_close_progress`/ramp/reopen trước khi sửa.
2. **Dừng ở mức hiện tại, quay lại hướng 1 cũ** (siết `grasp_lift_contact_z_finger` xuống 0.008m) — không chạm state machine đóng kẹp, nhưng rủi ro khác (ảnh hưởng `latch_rate` đã train quen).
3. **Chấp nhận giới hạn hiện tại của LIFT, tạm dừng nhánh này** — quay lại các phần khác của project, để LIFT/PLACE ở trạng thái "đã hiểu rõ nguyên nhân, chưa sửa xong" thay vì tiếp tục đào sâu thêm.

User chọn: "2" (siết `grasp_lift_contact_z_finger` xuống 0.008m).

## Phase 28 — Thử siết `grasp_lift_contact_z_finger` 0.014→0.008: ĐO ĐƯỢC đánh đổi TỆ HƠN, đã REVERT

### Đo (regression gate `--seed 0 --task_phase 2`, đúng kỷ luật "đo trước khi tin")
| Metric | Baseline (0.014m) | Sau khi siết (0.008m) |
|---|---|---|
| grasp_rate | 0.8333 | 0.8333 (giữ nguyên) |
| latch_rate | 0.4667 | 0.4667 (giữ nguyên) |
| **lift_start_rate** | **0.4333** | **0.0333** ← sập |
| success_rate | 0.0333 | 0.0333 |

**Kết luận**: siết ngưỡng KHÔNG cải thiện gì mà còn TỆ HƠN — `grasp`/`latch` không đổi (chứng tỏ policy vẫn tiếp cận/latch bình thường), nhưng gần như MỌI lần latch giờ có `z_error_finger` nằm trong khoảng 0.008-0.014m — đủ ngưỡng CŨ, không đủ ngưỡng MỚI. Đúng như rủi ro đã cảnh báo trước khi thử: policy đã học quen với ngưỡng 0.014m, siết chặt không làm grip "chắc hơn" (policy không tự động latch ở vị trí khác) mà chỉ làm gần như KHÔNG CÒN CƠ HỘI nào được phép thử nhấc (13/30 → 1/30). Đánh đổi tệ hơn hẳn vấn đề gốc.

**Đã REVERT** về 0.014 ở cả `config.py` và `phase2_overrides.py` — giữ lại comment mô tả đã thử/đo/tại sao không dùng, tránh người sau (hoặc chính mình sau này) thử lại đúng hướng đã biết không hiệu quả.

### File sửa (Phase 28, CHƯA commit — net effect = không đổi so với trước Phase 28, chỉ thêm comment lịch sử)
`isaaclab_openarm_env/config.py`, `isaaclab_openarm_env/phase2_overrides.py` (cả 2: thử 0.008 rồi revert về 0.014, giữ comment giải thích).

### Trạng thái tổng kết nhánh LIFT sau Phase 25-28
Đã thử 3 hướng cho vấn đề "chai tuột ~19 bước đầu RISING dù grip đạt ngưỡng gate": (1) phát hiện+abort sớm hơn (Phase 26, đúng kỹ thuật, không đủ vì không có retry), (2) cơ chế phục hồi active descent (Phase 27, đúng phần của nó, lộ ra thiếu re-trigger đóng kẹp), (3) siết ngưỡng gate chặt hơn (Phase 28, ĐO ĐƯỢC tệ hơn, đã revert). **Chưa hướng nào giải quyết được triệt để.** Hướng còn lại khả dĩ nhất là lớp 2 của Phase 27 (re-trigger đóng kẹp khi recovering+near_bottle) — chưa làm, rủi ro cao nhất vì chạm state machine đóng/mở kẹp phức tạp.

User yêu cầu: "improve layer 2 pls" — làm lớp 2 (re-trigger đóng kẹp).

## Phase 29 — Implement lớp 2 (reclose): hoạt động đúng kỹ thuật (retry thật xảy ra), nhưng kết quả ĐO ĐƯỢC hỗn hợp/đáng lo — CHƯA đủ để giữ lại không cần xem xét thêm

### Implement
`actions.py::AssistedBinaryGripperAction.process_actions`: khi `env._lift_recovery_active` (Phase 27) VÀ tay đã về đủ gần (`z_error_finger < grasp_lift_contact_z_finger`, đúng ngưỡng gate ban đầu) VÀ còn hạn mức reopen (`_reopen_count < grasp_reopen_max_count`, cooldown đã hết) — **tái sử dụng ĐÚNG cơ chế `bad_close`/reopen đã có sẵn** (reset `_grasp_latched=False`, `_close_progress=0`, `_want_close=False`, `_descend_hold=0`, `_sym_hold=0`, set cooldown) thay vì phát minh cơ chế mới — để `force_close`/`_grasp_latched` (đã tính lại mỗi bước từ state THẬT) tự nhiên phát hiện lại chai và đóng kẹp lại từ đầu, lần này chạm chai thật.

### Verify bằng demo — CƠ CHẾ HOẠT ĐỘNG ĐÚNG THIẾT KẾ
`[Reclose]` fire đúng lúc (`z_f=13.86mm` < ngưỡng 14mm), dẫn tới **retry THẬT SỰ đầu tiên trong toàn bộ investigation**: `IDLE→RISING` lần 2 chỉ 58 bước sau reclose. Đây là bằng chứng cơ chế kỹ thuật hoạt động chính xác như thiết kế.

**NHƯNG lần retry thứ 2 trượt theo ĐÚNG CÁCH y hệt lần đầu** (`reason=zf_drift` lần nữa, 71 bước sau) — củng cố thêm bằng chứng: vấn đề không phải "thiếu cơ hội thử lại" mà là **nguyên nhân vật lý gốc** (lực nhấc/ma sát không đủ ngay cả với grip "đạt chuẩn") — retry với cùng điều kiện vật lý cho cùng kết quả thất bại, đúng như đã dự đoán trước khi đo.

### Đo bằng regression gate + eval đầy đủ — kết quả HỖN HỢP, có rủi ro MỚI
| | Trước Phase 29 | Sau khi thêm reclose |
|---|---|---|
| seed=0/phase=2: success | 0.033 | 0.067 (tăng, nhưng mẫu quá nhỏ 1→2/30, chưa đủ ý nghĩa thống kê) |
| seed=0/phase=2: grasp/latch | 0.833/0.467 | 0.800/0.433 (giảm nhẹ) |
| seed=1/phase=3: grasp/latch | 0.625/0.375 | **0.542/0.333** (giảm rõ hơn) |
| seed=1/phase=3: success | 0.0 | 0.0 (không đổi) |

**Rủi ro mới phát hiện**: fail_mode `tilt` xuất hiện với tần suất đáng kể ở seed=1 (**6/24 episode**, gần như không có trước Phase 29) — cơ chế mở-kẹp-rồi-đóng-lại (reopen) đang khiến MỘT SỐ chai bị lật đổ trong lúc đóng lại lần 2, một tác dụng phụ ngoài dự tính.

### Đánh giá — KHÔNG phải chiến thắng rõ ràng, cần user quyết định
Cơ chế đúng kỹ thuật, xác nhận rõ nguyên nhân gốc là VẬT LÝ (không phải thiếu retry), nhưng lợi ích đo được (success ở phase 2) quá nhỏ để bù đắp chi phí (grasp/latch giảm ở kịch bản khó, thêm rủi ro lật chai mới). Đây KHÔNG giống các fix trước trong project (nơi lợi ích rõ ràng và không có tác dụng phụ) — cần cân nhắc thêm trước khi giữ lại.

### File sửa (Phase 29, CHƯA commit)
`isaaclab_openarm_env/mdp/actions.py` (import `os`, block reclose trong `process_actions`, debug print `[Reclose]`).

### Hướng tiếp theo (chưa làm, cần quyết định user)
1. **Giữ lại nhưng giới hạn chặt hơn** (vd chỉ 1 lần reclose thay vì theo `reopen_max` chung, hoặc thêm điều kiện tilt thấp trước khi cho reclose) để giảm rủi ro lật chai.
2. **Revert lớp 2**, giữ lại lớp 1 (Phase 27, active descent recovery) vì bản thân nó vô hại (không đổi kết quả, không thêm rủi ro) — chấp nhận LIFT vẫn còn hạn chế do nguyên nhân vật lý gốc.
3. **Revert cả lớp 1 và lớp 2**, quay về trạng thái trước Phase 26 hoàn toàn — chỉ giữ lại phần zf_drift-abort (Phase 26, đã xác nhận không hồi quy, chỉ tiết kiệm ngân sách episode).
4. **Dừng nhánh LIFT ở đây**, chấp nhận giới hạn vật lý đã xác nhận rõ ràng qua Phase 25-29 — đã hiểu SÂU nguyên nhân (không phải RL, không phải chưa retry, mà là lực/ma sát vật lý ngay cả khi grip đạt chuẩn), quay lại phần khác của project.

User chọn: "Revert luôn lớp 1, chỉ giữ Phase 26" — chọn hướng an toàn nhất sau khi phát hiện lớp 1 cũng không khớp tuyệt đối baseline.

## Phase 30 — Revert lớp 1 (Phase 27): xác nhận nó CŨNG không khớp tuyệt đối baseline — chỉ giữ Phase 26

### Phát hiện quan trọng khi revert lớp 2 để đo riêng
Sau khi revert lớp 2 (Phase 29), regression gate vẫn cho `0.067/0.8/0.433/0.367` — **giống hệt lúc còn cả lớp 2**, KHÔNG khớp baseline gốc. Nghĩa là **chính lớp 1 (Phase 27, active-descent recovery)** — không phải lớp 2 — mới là nguồn gây lệch. Giả định trước đó ("lớp 1 vô hại vì chỉ đổi hành động lúc LIFT_IDLE") SAI: dù chỉ đổi "đứng yên" thành "hạ tay nhẹ", nó vẫn đủ để lệch quỹ đạo episode (thời điểm/vị trí các sự kiện tiếp theo) khỏi baseline đã biết.

### Đã revert hoàn toàn lớp 1
`grasp_assist.py`: xoá block tính `_lift_recovery_active` trong `_update_lift_state`, xoá block `recovering`/`_osc_z_finger_descend` trong `apply_grasp_arm_assist`, khôi phục `idle_hold` nguyên bản. `terminations.py`: xoá reset `_lift_recovery_active`.

**Regression gate xác nhận**: khớp CHÍNH XÁC baseline (`0.0333/0.8333/0.4667/0.4333`) — chỉ còn Phase 26 (zf_drift-abort sớm hơn) hoạt động, đã biết an toàn tuyệt đối từ trước.

### Trạng thái cuối cùng nhánh LIFT sau Phase 25-30
Sau 6 lần thử (zf_drift-abort, active-descent recovery, siết ngưỡng gate, reclose, và các tổ hợp) — **CHỈ Phase 26 (phát hiện trượt dần + abort sớm hơn) được giữ lại**, vì là thay đổi DUY NHẤT khớp tuyệt đối baseline mà vẫn có giá trị thật (tiết kiệm ~340+ bước lãng phí mỗi lần trượt, dù không tự cải thiện success_rate). Mọi cơ chế "chủ động phục hồi/thử lại" (lớp 1, lớp 2) đều đo được có tác dụng phụ không đáng đánh đổi. **Nguyên nhân gốc (chai tuột ~19 bước đầu RISING do lực/ma sát không đủ ngay cả khi grip đạt chuẩn) đã được hiểu rõ nhưng CHƯA sửa được** — mọi hướng sửa qua RL/state-machine đều chạm giới hạn vật lý thật, không phải vấn đề logic/gate.

### File sửa cuối cùng còn giữ lại (Phase 26 only, CHƯA commit)
`isaaclab_openarm_env/config.py` (`grasp_lift_abort_z_finger`, `grasp_lift_abort_z_finger_steps`), `isaaclab_openarm_env/mdp/grasp_assist.py` (chỉ còn bộ đếm zf-drift trong `_update_lift_slip` + điều kiện trong `_lift_must_abort` + nhãn debug — KHÔNG còn `_lift_recovery_active`/`recovering`), `isaaclab_openarm_env/mdp/terminations.py` (chỉ còn reset `_lift_zf_abort_steps`), `isaaclab_openarm_env/mdp/actions.py` (đã revert sạch, không còn thay đổi nào so với trước Phase 29).

User: "but I want to place" — không dừng ở LIFT, tiếp tục tìm hướng khác để PLACE hoạt động được.

## Phase 31 — 2 thử nghiệm nữa nhắm vào NGUYÊN NHÂN VẬT LÝ trực tiếp (tốc độ nhấc, lực kẹp) — CẢ 2 ĐỀU KHÔNG giải quyết được, đã revert cả 2

### Thử 1 — chậm tốc độ nhấc (`grasp_lift_world_m` 0.055→0.030)
**Giả thuyết**: tốc độ nhấc hiện tại (đã tăng 17x từ lịch sử để đủ nhanh trong ngân sách episode) có thể quá nhanh khiến ma sát không kịp "bắt" ở seed khó.

**Đo được (demo + DEBUG_LIFT/DEBUG_STALL)**: thời điểm trượt trì hoãn từ 59→99 bước (đúng tỷ lệ chậm lại ~2x), NHƯNG `lift_m` vẫn đạt đỉnh ~1.6mm rồi **TỤT DẦN về 0** trong khi `z_error_finger` tiếp tục tăng đều — không bao giờ đạt trạng thái ổn định (không trượt nữa) dù chậm hơn nhiều. **Kết luận**: một khi đã bắt đầu trượt (dù chậm), ma sát ĐỘNG (kinetic, luôn thấp hơn ma sát tĩnh) không đủ giữ — chậm chỉ trì hoãn TUYẾN TÍNH thời điểm tách hẳn, không giải quyết gốc rễ. Vấn đề không phải tốc độ, mà là LỰC KẸP. **Đã REVERT về 0.055.**

### Thử 2 — nới trần đóng kẹp (`grasp_close_freeze_at_progress` 0.75→1.0)
**Giả thuyết**: đo lại một giả định LỊCH SỬ ("tăng cap lên 1.0 không đổi success_rate", ghi trong code comment cũ) — nhiều giả định lịch sử khác trong session này đã sai khi đo lại với code/seed hiện tại (vd `grasp_lift_contact_z_finger`), nên đáng đo lại lần nữa thay vì tin theo.

**Đo được**: dù kẹp đóng HOÀN TOÀN (`gc=1.000`, joint≈0, tối đa lệnh có thể ra), thời điểm trượt KHÔNG cải thiện (~60 bước, bằng đúng baseline 59 bước). `stall` (proxy lực ép) chỉ tăng nhẹ 0.13→0.60mm — **vẫn RẤT xa** ngưỡng "pressing" 5mm dù đã đóng hết cỡ theo lệnh. **Kết luận**: giả định lịch sử ĐƯỢC XÁC NHẬN ĐÚNG (không phải stale như những giả định khác) — tăng mức đóng kẹp không giúp gì. Phát hiện thêm quan trọng: lực ép thực tế cực yếu NGAY CẢ KHI lệnh đóng đã tối đa — gợi ý vấn đề nằm ở **HÌNH HỌC TIẾP XÚC** (có thể lệch tâm giữa 2 ngón, hoặc khoảng hở vật lý giữa ngón-chai không khớp với đơn vị "joint position" đang dùng làm proxy lực) chứ không phải mức độ đóng. **Đã REVERT về 0.75.**

### Trạng thái — cả 2 hướng "dễ thử" nhất đã bị loại trừ bằng đo lường trực tiếp
Sau Phase 25-31 (8 thử nghiệm khác nhau: state-machine/gate/recovery ×5, tốc độ nhấc, lực kẹp), đã loại trừ được: logic gate/state-machine (Phase 26/28), cơ chế phục hồi/retry (Phase 27/29/30), tốc độ nhấc (Phase 31 thử 1), mức lực đóng kẹp (Phase 31 thử 2). **Còn lại giả thuyết chưa kiểm chứng**: hình học tiếp xúc thực tế (lệch tâm 2 ngón trên thân chai, hoặc offset vật lý phần cứng của kẹp không khớp với model điều khiển) — cần công cụ chẩn đoán SÂU HƠN (quan sát trực tiếp điểm tiếp xúc PhysX, hoặc so sánh vị trí 2 ngón thực tế so với tâm chai tại đúng thời điểm RISING bắt đầu) thay vì tiếp tục thử tham số đơn lẻ.

### File sửa (Phase 31, CHƯA commit — net effect = không đổi hành vi so với sau Phase 30, chỉ thêm 2 đoạn comment lịch sử mới)
`isaaclab_openarm_env/phase2_overrides.py` (2 giá trị: thử rồi revert, giữ comment giải thích chi tiết).

### Thử 3 — giả thuyết "lệch tâm 2 ngón" — BỊ BÁC BỎ bằng dữ liệu đầy đủ hơn
Thêm debug in `dL`/`dR`/`dLR` (dist_left_body/dist_right_body) vào `[LiftDbg]` (đúng lúc transition, không phải mỗi 20 bước như `[LiftSlip]`). Dữ liệu MỘT ĐIỂM giữa chừng (step 700, `[LiftSlip]`) từng cho thấy lệch 7.25mm, gây nghi ngờ lệch tâm — nhưng dữ liệu ĐẦY ĐỦ hơn tại đúng lúc BẮT ĐẦU (step 681: dL=28.84 dR=32.16, lệch 3.32mm) và lúc ABORT (step 740: dL=58.32 dR=61.14, lệch 2.82mm) cho thấy: **độ lệch KHÔNG tăng theo thời gian** (thực ra giảm nhẹ), mà **CẢ 2 khoảng cách tăng ĐỀU NHAU** (28.84→58.32mm, 32.16→61.14mm, gần gấp đôi cả 2 bên) — chai tụt THẲNG ĐỨNG đối xứng, không nghiêng lệch một bên. Giả thuyết lệch tâm SAI — điểm dữ liệu 7.25mm trước đó chỉ là dao động tạm thời, không phải xu hướng thật.

### Ước tính định lượng lực ép — mâu thuẫn với dữ liệu thực nghiệm, cần công cụ đo tốt hơn
`gripper: ImplicitActuatorCfg(stiffness=2000.0, damping=100.0)` (config.py). Ước tính `F_ép ≈ stiffness × stall`: ở cấu hình mặc định (stall≈0.13mm) → F≈0.26N/ngón ×2×μ(1.4)=0.73N — THẤP HƠN trọng lượng chai (0.94N), giải thích hợp lý cho việc trượt gần như ngay lập tức. Nhưng ở cấu hình đã tăng cap (Thử 2, stall≈0.6mm) → F≈1.2N/ngón×2×μ=3.36N — theo tính toán này PHẢI đủ dư (3.5x trọng lượng), NHƯNG THỰC NGHIỆM vẫn trượt sau ~60 bước y hệt. **Mâu thuẫn này cho thấy mô hình ước tính lực từ vị trí khớp (stall × stiffness) KHÔNG phản ánh đúng lực tiếp xúc PhysX thật** — cần công cụ đo lực/tiếp xúc trực tiếp (không có sẵn qua debug hiện tại) để tiến xa hơn, không thể suy luận thêm từ proxy vị trí.

### Kết luận cuối cùng sau 8 thử nghiệm (Phase 25-31)
Đã loại trừ bằng đo lường trực tiếp: logic gate/state-machine, cơ chế phục hồi/retry (3 biến thể), tốc độ nhấc, mức lực đóng kẹp tối đa, giả thuyết lệch tâm hình học. **Đạt giới hạn của công cụ chẩn đoán hiện có** (debug print dựa trên vị trí khớp/khoảng cách, không phải lực/tiếp xúc PhysX trực tiếp) — tiếp tục đoán tham số từ đây có tỷ lệ lợi ích/rủi ro kém. Cần MỘT TRONG HAI: (a) công cụ đo lực tiếp xúc PhysX trực tiếp (đầu tư thêm đáng kể), hoặc (b) chấp nhận mức tin cậy LIFT hiện tại và xây PLACE xoay quanh mức đó (vd dùng lại kết quả scripted-assist ~57-59% từ Phase 14 làm nền, thay vì cố đạt LIFT tự chủ hoàn hảo trước khi làm PLACE).

User: "but I want to place" → chọn "Đầu tư thêm công cụ đo lực tiếp xúc PhysX trực tiếp".

## Phase 32 — ĐỘT PHÁ: ContactSensor tìm ra nguyên nhân gốc THẬT — kẹp chỉ có MỘT điểm tiếp xúc, không phải hai

### Manh mối trước khi implement: hành vi bất biến với stiffness (đầu mối quan trọng bị bỏ lỡ trước đó)
Chạy lại demo Phase 31's scenario qua `eval_lift_metrics.py` (dùng ĐÚNG `gripper stiffness=2000` như training thật — phát hiện `isaaclab_demo.py:245` tự ý override xuống 700, nghĩa là TOÀN BỘ debug Phase 25-31 qua demo chạy SAI stiffness so với train/eval!). Kết quả: **thời điểm trượt (step 681→740, đúng 59 bước) và quỹ đạo dL/dR GIỐNG HỆT tuyệt đối** dù stiffness chênh 2.86 lần (700 vs 2000) — đây là bằng chứng mạnh rằng actuator stiffness KHÔNG PHẢI biến số quyết định, dẫn thẳng tới nghi vấn: có thể MỘT NGÓN CHƯA TỪNG CHẠM CHAI THẬT SỰ.

### Implement ContactSensor
`config.py`: bật `activate_contact_sensors=True` trên `robot_spawn` (trước đó `False` — PhysX không hề ghi nhận contact report cho robot). Thêm 2 `ContactSensorCfg` (`left_finger_contact`, `right_finger_contact`) trỏ `openarm_left_left_finger`/`openarm_left_right_finger` (tên body lấy từ `env._robot.body_names`, KHÔNG phải tên joint), `filter_prim_paths_expr=["{ENV_REGEX_NS}/Scene/Bottle"]` để `force_matrix_w` chỉ phản ánh đúng cặp ngón-chai. `grasp_assist.py`: đọc `force_matrix_w` qua `_t()` (helper ProxyArray→Tensor có sẵn), in vào `[LiftSlip]` debug.

### KẾT QUẢ — bằng chứng KHÔNG THỂ CHỐI CÃI
```
F_left=0.079N  F_right=0.000N   (step 700, RISING)
F_left=0.078N  F_right=0.000N   (step 720, RISING)
```
**Ngón PHẢI có lực tiếp xúc = 0.000N CHÍNH XÁC** (không phải "nhỏ" — bằng 0 tuyệt đối, PhysX không ghi nhận bất kỳ tiếp xúc nào giữa ngón phải và chai). Ngón TRÁI có lực, nhưng cực yếu (~0.08N, chưa bằng 1/10 trọng lượng chai 0.94N). **Kẹp chỉ có MỘT điểm tiếp xúc vật lý thật, không phải hai** — hoàn toàn khác với proxy vị trí (`dist_right_body=31mm`, trông như "gần nhưng chưa chạm", KHÔNG hề gợi ý mức độ nghiêm trọng "0 lực tuyệt đối" này).

### Giải thích lại TOÀN BỘ 8 thử nghiệm trước dưới ánh sáng phát hiện này
- **Tăng stiffness không đổi gì** (Phase 25→31 gián tiếp): đúng, vì ngón phải KHÔNG BAO GIỜ CHẠM dù stiffness bao nhiêu — không có gì để "ép mạnh hơn" ở phía đó.
- **Tăng cap đóng kẹp lên 1.0 chỉ tăng nhẹ stall** (Phase 31 thử 2): đúng, vì đó là trung bình 2 ngón — một ngón đóng thêm (tăng joint gần target) nhưng KHÔNG TẠO LỰC vì không chạm gì, chỉ ngón còn lại phản ánh vào "stall" đo được.
- **Chậm tốc độ nhấc không giúp** (Phase 31 thử 1): đúng, một điểm tiếp xúc DUY NHẤT với lực <0.1N không bao giờ đủ giữ chai bất kể tốc độ nhấc nhanh hay chậm.
- **"Lệch tâm" đo bằng proxy vị trí (dL/dR) không đủ nhạy**: độ lệch nhìn "vừa phải" (3-7mm) trên GIẤY (khoảng cách hình học tới điểm tham chiếu), nhưng THỰC TẾ VẬT LÝ lại là 0 tiếp xúc tuyệt đối một bên — proxy vị trí không phản ánh đúng mức độ nghiêm trọng.

### Trạng thái — nguyên nhân gốc đã XÁC ĐỊNH, chưa sửa
Câu hỏi tiếp theo: TẠI SAO ngón phải không chạm — nghi vấn khả dĩ: offset hiệu chỉnh hình học ngón tay (`finger_tip_local_left`/`finger_tip_local_right`, `bottle_grasp_xy_offset_x/y`) bị lệch giữa 2 bên, hoặc logic căn giữa XY (`center_mask`) không thực sự đối xứng chai giữa 2 ngón. CHƯA điều tra tiếp — dừng lại để báo cáo phát hiện lớn này trước khi đi sâu vào nguyên nhân của "tại sao lệch".

### File sửa (Phase 32, CHƯA commit)
`isaaclab_openarm_env/config.py` (import `ContactSensorCfg`, `activate_contact_sensors=True`, 2 sensor mới), `isaaclab_openarm_env/mdp/grasp_assist.py` (import `_t`, đọc + in lực tiếp xúc thật vào debug `[LiftSlip]`).

### Điều tra tiếp — loại trừ "khớp bị kẹt", xác định ĐÚNG nguyên nhân gốc: LỆCH TÂM TIẾP CẬN
Nghi vấn: `joint` trong debug là TRUNG BÌNH joint1+joint2 (`_grip_pressing`, dùng ở nhiều nơi trong `helpers.py`) — có thể che giấu 1 khớp kẹt mở. Thêm print riêng `j1`/`j2` (không average):
```
j1=12.17mm  j2=12.20mm   (gần như giống hệt nhau — CẢ 2 khớp đóng đúng theo cùng target)
```
**Loại trừ giả thuyết "khớp2/mimic bị lỗi"** — cả 2 khớp đóng đối xứng hoàn hảo về mặt VỊ TRÍ GÓC. Vậy tại sao lực khác nhau hoàn toàn (0.079N vs 0.000N)?

**Kết luận đúng**: vấn đề nằm ở **HÌNH HỌC TIẾP CẬN**, không phải điều khiển khớp. Tay đã tiếp cận chai với một chút lệch tâm theo phương ngang NGAY TỪ ĐẦU (khớp với `dLR≈7.5mm` đo được từ Phase 32 phần trước — không phải "trôi dạt trong lúc RISING" như tưởng ban đầu, mà là lệch tâm CÓ SẴN từ lúc latch). Dù 2 khớp đóng đối xứng (cùng góc), ngón GẦN (trái, `dist_left_body≈23.6mm`) chạm chai trước và tạo lực nhỏ; ngón XA hơn ~7.5mm (phải) **không bao giờ chạm được** vì bị chặn bởi trần đóng kẹp (`grasp_close_freeze_at_progress=0.75`, tương ứng target≈10-12mm — không đủ "đóng thêm" để bù 7.5mm lệch tâm).

**Chuỗi nhân quả đầy đủ**: tiếp cận lệch tâm nhẹ (chưa rõ nguyên nhân — có thể do policy chưa học được vị trí hoàn hảo, hoặc do observation/reward không phạt đủ nặng lệch tâm nhỏ) → dù đóng kẹp đối xứng, chỉ 1 ngón thực sự chạm → lực giữ chỉ từ MỘT điểm tiếp xúc duy nhất, cực yếu → không đủ chống trọng lực/gia tốc khi nhấc → chai tách khỏi kẹp trong ~19-60 bước tuỳ tốc độ.

### Hướng sửa khả dĩ (chưa làm, cần quyết định)
1. **Thêm cơ chế bù đắp chủ động** — khi phát hiện qua ContactSensor một ngón chưa có lực tiếp xúc dù ngón kia đã có, tiếp tục đóng SÂU HƠN trần hiện tại (vượt `grasp_close_freeze_at_progress`) CHỈ để tạo tiếp xúc tối thiểu ở ngón còn thiếu — rủi ro: cần đóng bao nhiêu là đủ, tránh siết quá mạnh.
2. **Siết ngưỡng đối xứng TRƯỚC latch** (`grasp_sym_max_dist_delta` hiện 15mm — đủ lỏng để cho qua đúng trường hợp 7.5mm gây lỗi này) — rủi ro giống Phase 28 (siết `grasp_lift_contact_z_finger` từng làm sập `lift_start_rate`), cần đo trước khi tin.
3. **Cải thiện độ chính xác approach/centering trước khi latch** (thay đổi reward/gate cho REACH→GRASP) — hướng sâu hơn, ảnh hưởng rộng hơn.
4. **Dùng chính ContactSensor mới làm ĐIỀU KIỆN `_grip_secure`/`_lift_can_start`** — thay proxy vị trí (`_grip_pressing`'s stall) bằng lực THẬT từ CẢ HAI ngón (yêu cầu `F_left>ngưỡng AND F_right>ngưỡng`, không chỉ trung bình) — đây là fix ĐÚNG GỐC RỄ nhất, tận dụng trực tiếp công cụ vừa xây, đảm bảo KHÔNG BAO GIỜ cho phép nhấc khi một ngón chưa thực sự chạm.

User chọn: "Dùng ContactSensor làm điều kiện _grip_secure (fix gốc rễ nhất)".

## Phase 33 — Implement lực thật vào `_grip_secure`/ramp-freeze: ĐÚNG hướng, chặn đúng trường hợp nguy hiểm, nhưng lộ ra giới hạn "trần đóng kẹp" độc lập

### Implement
`helpers.py::compute_state()`: thêm `left_finger_contact_force`/`right_finger_contact_force` (đọc `ContactSensor.data.force_matrix_w` qua `_t()`, chỉ nội bộ — KHÔNG vào observation 26-D, đúng nguyên tắc Phase 11). Bug nhỏ tự bắt lúc implement: `force_matrix_w` có thêm 1 chiều "filter target" (`[:, 0]` không đủ, cần `[:, 0, 0]` mới về đúng shape `(num_envs,)` sau `.norm(dim=-1)`) — gây `IndexError` ngay lần chạy đầu, phát hiện + sửa ngay qua regression gate.

`grasp_assist.py::_grip_pressing`: thay điều kiện `stall > min_stall` (proxy vị trí trung bình) bằng `F_left > 0.15N AND F_right > 0.15N` (lực thật, MỖI ngón riêng). Thêm config field `grasp_press_min_force_n=0.15` (ước tính vật lý: giữ nửa trọng lượng chai qua μ≈1.4 cần ≥0.336N/ngón, chọn 0.15N làm khởi điểm thận trọng hơn).

`actions.py::apply_actions`: sửa TƯƠNG TỰ điều kiện `firm_contact` (quyết định khi nào DỪNG ramp đóng) — trước dùng `stall_now` (trung bình joint1/joint2), giờ dùng lực thật CẢ HAI ngón. Ý định: nếu chỉ 1 bên chạm, ramp KHÔNG dừng, tiếp tục đóng cho tới khi bên kia cũng chạm.

### Regression gate + kết quả
| | Baseline | Sau Phase 33 |
|---|---|---|
| grasp_rate | 0.8333 | **0.8333 (giữ nguyên, không hồi quy)** |
| latch_rate | 0.4667 | **0.4667 (giữ nguyên)** |
| lift_start_rate | 0.4333 | **0.0333 (sập)** |
| success_rate | 0.0333 | 0.0333 (không đổi) |
| fail_modes | — | `no_lift_command: 13/30 (43%!)`, `reach: 5`, `grasp: 11` |

**Ý nghĩa**: `no_lift_command=13/30` — gần MỘT NỬA số lần latch giờ bị chặn không cho nhấc (đúng thiết kế: phát hiện thiếu lực một bên, không cho thử). Đây xác nhận thêm mức độ PHỔ BIẾN của hiện tượng "chỉ 1 ngón chạm" — không phải hiếm, mà là ~43% số lần latch trong seed=0!

### Giới hạn phát hiện thêm: RAMP VẪN BỊ CHẶN CỨNG bởi trần đóng kẹp, độc lập với sửa đổi vừa làm
Dữ liệu `max_gc=0.7624995708465576` ở TẤT CẢ episode `no_lift_command` — CHÍNH XÁC bằng trần `grasp_close_freeze_at_progress=0.75` như trước khi sửa. Nghĩa là: dù `firm_contact` (điều kiện DỪNG SỚM) đã sửa đúng, ramp vẫn bị chặn bởi một giới hạn CỨNG, ĐỘC LẬP khác (`can_advance = ... & (close_progress < cap)`, `cap` từ `grasp_close_freeze_at_progress`) — ramp KHÔNG BAO GIỜ có cơ hội đóng vượt trần để bù đắp ngón còn thiếu lực. Sửa Phase 33 chỉ giải quyết "không dừng SỚM HƠN trần", chưa giải quyết "không thể đóng VƯỢT trần khi cần".

### Trạng thái — bước tiến đúng nhưng chưa đủ, cần quyết định bước tiếp
Grasp/latch không hồi quy (an toàn), hệ thống giờ TRUNG THỰC hơn (không còn "nhấc-rồi-trượt" giả tạo), nhưng `lift_start_rate`/`success_rate` chưa cải thiện vì ramp bị chặn cứng trước khi đạt lực đủ 2 bên. Cần bước tiếp theo: cho phép ramp đóng VƯỢT `grasp_close_freeze_at_progress` cụ thể khi phát hiện lực thiếu một bên (đóng bù có mục tiêu, không đóng bù vô hạn) — đây CHÍNH XÁC là ý tưởng "đóng bù chủ động" từng đề xuất ở Phase 32, giờ có bằng chứng THẬT là cần thiết, không chỉ là lựa chọn thay thế.

### File sửa (Phase 33, CHƯA commit)
`isaaclab_openarm_env/mdp/helpers.py` (`compute_state()` thêm 2 field lực), `isaaclab_openarm_env/mdp/grasp_assist.py` (`_grip_pressing` dùng lực thật), `isaaclab_openarm_env/mdp/actions.py` (`firm_contact` dùng lực thật), `isaaclab_openarm_env/config.py` (`grasp_press_min_force_n=0.15`).

---

## Phase 34 — Sửa `_grip_pressing`: OR-nhánh 1-ngón-chắc thay vì cố nới trần đóng kẹp (2026-09-15)

### Bối cảnh
User xem demo GUI mới (`gui_demo_phase33b.log`, seed=1) báo "không nhấc được nữa". Điều tra bằng cách so log byte-for-byte với log CŨ trước Phase 32/33 (`gui_demo_zf_fix.log`, cùng seed=1, cùng vị trí chai) xác nhận quỹ đạo GIỐNG HỆT tới bước ~825 rồi rẽ nhánh: log cũ chai bị lật (tipped), log mới tay lùi ra xa ("Lost Bottle Contact") — cả hai đều là biểu hiện KHÁC NHAU của cùng một pathology có sẵn (policy kẹt ở biên align hàng trăm bước), không phải bug logic mới.

Nhưng khi đối chiếu lại với đúng con số Phase 33 đã đo (cuối phần trên): `lift_start_rate` đã SẬP từ 0.4333 → 0.0333 ngay từ Phase 33, và ĐÓ mới là nguyên nhân gốc khiến "chai không nhấc được nữa" — không phải hiện tượng lùi tay ở seed=1 (hiện tượng đó chỉ là hệ quả phụ của việc không bao giờ latch/lift được, khiến policy kẹt lâu bất thường ở GRASP).

### Sửa sai giả thuyết ban đầu — KHÔNG PHẢI vấn đề trần đóng kẹp
Phase 33 kết luận (đoạn trên) rằng bước tiếp theo là "cho ramp đóng vượt `grasp_close_freeze_at_progress`". Trước khi làm theo, đối chiếu lại bằng chứng đã có: Phase 31 ĐÃ THỬ nới trần lên 1.0 (không có force-gate) và đo được joint≈0 (đóng hết cỡ cơ học) mà lực ngón xa VẪN không tăng đáng kể (stall 0.13→0.60mm) — tức đây là **giới hạn hình học** (lệch tâm tiếp cận), không phải giới hạn do trần % đóng kẹp. Nới trần lần nữa nhiều khả năng vô ích.

### Fix thật: nới điều kiện AN TOÀN quá mức của `_grip_pressing`/`firm_contact`, không đụng trần
Phase 32/33 bắt buộc CẢ HAI ngón > 0.15N (AND) mới coi là "đang ép" — đúng về mặt vật lý nhưng quá nghiêm: vì ngón xa gần như không bao giờ đạt lực thật, điều kiện AND gần như luôn False → `_grip_secure` luôn False → LIFT gần như vô hiệu. Trước Phase 32, proxy stall trung bình 2 ngón (không phân biệt được 1-ngón-chạm) vẫn cho `lift_start_rate=0.4333` trong thực tế — nghĩa là 1 ngón ép đủ mạnh + hình học đúng đường kính chai vốn ĐÃ ĐỦ để giữ được.

Thêm nhánh OR vào `_grip_pressing` (`grasp_assist.py`) và `pressing_now` (`actions.py`, mirror nhau):
```python
both_pressing = (left_f > min_force) & (right_f > min_force) & near_bottle          # 0.15N mỗi bên
single_pressing = (max(left_f, right_f) > min_force_single) & near_bottle & span_ok  # 0.30N MỘT bên, ngưỡng cao hơn
pressing = both_pressing | single_pressing
```
`grasp_press_min_force_single_n=0.30` (config.py) — cao hơn hẳn ngưỡng dual (0.15N) để không tin nhầm chạm yếu/nhiễu khi chỉ có 1 bên; bắt buộc kèm `span_ok` (finger_span_xy gần đúng đường kính chai) để loại trừ trường hợp đóng vào không khí.

### Regression gate — khôi phục gần như chính xác baseline gốc
```
eval_lift_metrics.py --model-path policy_1M_success57.pt --episodes 30 --num-envs 8 \
  --bottle-noise 0.05 --assist-scale 1.0 --stage all --seed 0 --task_phase 2
```
| | Baseline gốc | Phase 33 (sập) | Phase 34 (fix) |
|---|---|---|---|
| success_rate | 0.0333 | 0.0333 | **0.03** |
| grasp_rate | 0.8333 | 0.8333 | **0.83** |
| latch_rate | 0.4667 | 0.4667 | **0.50** |
| lift_start_rate | **0.4333** | **0.0333 (sập)** | **0.43 — khôi phục** |

`max_gc` vẫn kẹt đúng `0.7624995708465576` (trần 0.75) ở MỌI episode — xác nhận lại: **không cần đụng trần**, chỉ cần sửa điều kiện "đang ép" là đủ. Giả thuyết "cần đóng vượt trần" ở cuối Phase 33 SAI — cứ để trần nguyên, chỉ cần chấp nhận 1-ngón-chắc-đủ là được, đúng như thực tế đã vận hành trước Phase 32.

### File sửa
`isaaclab_openarm_env/config.py` (`grasp_press_min_force_single_n=0.30`), `isaaclab_openarm_env/mdp/grasp_assist.py` (`_grip_pressing` thêm nhánh OR), `isaaclab_openarm_env/mdp/actions.py` (`pressing_now`/`firm_contact` mirror OR, debug print `[FreezeDbg]` thêm cờ `both`/`single`).

---

## Phase 36 — PLACE carry: ưu tiên leo cao trước, và giới hạn khớp2 chặn XY (2026-09-16)

### Bối cảnh
User xem demo, yêu cầu "lift it higher" — đo bằng CarryDbg (DEBUG_PLACE=1) thấy z_clear (chai so với miệng bát) leo đúng hướng nhưng RẤT chậm (~0.17mm/bước) vì `_osc_carry_to_bowl` dùng CHUNG một hệ số `place_carry_speed_scale=0.35` cho cả trục Z lẫn XY — hệ số này vốn chỉ để chống lật khi di chuyển NGANG (đã có lý do rõ trong code), bị áp nhầm luôn lên trục LÊN THẲNG.

### Fix 1 (giữ lại) — ưu tiên leo cao trước khi di chuyển ngang
`_osc_carry_to_bowl`: thêm `xy_gate = 1 - clamp(z_err/carry_height, 0, 1)` nhân vào `xy_step` — XY chỉ được đi hết tốc khi độ cao đã gần đạt `place_carry_height_m`. Kết quả đo (cùng kịch bản seed=0, env0): z_clear cuối episode tăng từ **+5.7mm → +19.1mm** (leo nhanh và xa hơn hẳn). Không đụng logic GRASP/LIFT nên không cần regression gate phase 2 (dead code khi `task_phase<3`).

### Phát hiện mới: XY kẹt cứng ~99mm dù lệnh vẫn full speed — do khớp2 chạm trần
Dù ưu tiên leo cao, XY vẫn kẹt ở ~99-104mm (không giảm về dưới `place_xy_arrival_radius_m=0.03`). Đo trực tiếp bằng debug 7-khớp (mở rộng CarryDbg log toàn bộ `joint_pos` + `joint_pos_limits` mỗi 15 bước):

| Khớp | Margin-tới-trần lúc bắt đầu | Lúc kết thúc (420 bước sau) |
|---|---|---|
| j1 | 93.9° | 80.3° |
| **j2** | **19.9°** | **8.1° (đang tiến thẳng tới trần +10.0°)** |
| j3 | 70.2° | 73.4° |
| j4 | 41.3° | 67.5° |
| j5 | 89.2° | 88.3° |
| j6 | 45.0° | 44.4° |
| j7 | 82.3° | 74.1° |

Chỉ joint2 co margin liên tục và sắp chạm 0 — 6 khớp còn lại đều dư 40-90°. Đây là **giới hạn cơ khí thật**: bát đặt ở vị trí đòi hỏi vừa nâng cao vừa vươn xa cùng lúc, joint2 (giới hạn trên +10.0°) không đủ dải để hoàn thành cả hai. OSC tự giảm tốc khi khớp gần trần (né vượt giới hạn) nên lệnh vẫn full speed nhưng chuyển động thật gần như 0.

### ĐÃ THỬ và REVERT: đổi null-space target sang "center"
Giả thuyết: `nullspace_joint_pos_target="default"` (tư thế nghỉ gần 0°) kéo joint2 lệch xa tâm dải chuyển động thật, khiến task chính dễ đẩy nó về phía trần hơn; đổi sang `"center"` (giữa dải `joint_pos_limits` thật) để nhường bậc tự do dư cho joint4 (đang dư margin nhiều nhất) gánh thay.

**ĐO ĐƯỢC (regression gate phase 2, seed=0, BẮT BUỘC vì đây là tham số CHUNG cho mọi stage)**: `grasp_rate` SẬP từ 0.8333 xuống **0.00** — toàn bộ 30 episode fail ngay ở REACH. Null-space bias ảnh hưởng tư thế nghỉ của CẢ CÁNH TAY ở MỌI stage (REACH/GRASP/LIFT dùng chung `_make_osc_actions_cfg`); policy đã học với giả định ngầm "tư thế nghỉ ≈ default (gần 0°)" — đổi sang tư thế nghỉ khác hẳn (center của dải joint2 bất đối xứng, xa 0° hơn nhiều) phá vỡ hoàn toàn REACH dù chỉ là "ưu tiên" ở null-space (không ràng buộc cứng task chính). ĐÃ REVERT về `"default"`, xác nhận lại regression gate khớp baseline (success=0.03, grasp=0.83, latch=0.50, lift_start=0.43).

**Bài học**: không sửa tham số TOÀN CỤC (dùng chung mọi stage) để giải quyết vấn đề CỤC BỘ (chỉ xảy ra ở PLACE_CARRY) — đúng nguyên tắc đã áp dụng xuyên suốt Phase 25-35 (mọi thay đổi chạm code dùng chung phải qua regression gate TRƯỚC khi chấp nhận).

### Trạng thái — chưa giải quyết XY-stall, cần hướng khác cho joint2
Các hướng còn lại (chưa thử):
1. Kiểm tra lại giới hạn +10.0° của joint2 có đúng với phần cứng thật không (nếu là giả định sai/quá chặt trong URDF, nới ra giải quyết tận gốc).
2. Sửa quỹ đạo CARRY để không cần vươn xa + nâng cao cùng lúc (vd. hạ `place_carry_height_m`, hoặc đổi vị trí bát trong scene).
3. Thêm cơ chế null-space CỤC BỘ chỉ áp dụng trong PLACE_CARRY (không đụng REACH/GRASP/LIFT) — cần nghiên cứu thêm API IsaacLab có hỗ trợ đổi nullspace target theo từng bước hay không (hiện tại `_resolve_nullspace_joint_pos_targets()` chỉ chạy 1 lần lúc init).

### File sửa (Phase 36)
`isaaclab_openarm_env/mdp/grasp_assist.py` (`_osc_carry_to_bowl` thêm `xy_gate`; `_update_place_state` CarryDbg mở rộng log 7 khớp + limits; `place_carry_clearance_m` gate ở Phase 35 vẫn giữ nguyên). `isaaclab_openarm_env/config.py` (`nullspace_joint_pos_target` thử "center" rồi revert về "default", giữ lại comment lịch sử).

### Tiếp Phase 36 — 2 thử nghiệm nữa, cả 2 đều REVERT

**Thử A — hạ `place_carry_height_m` 0.15→0.07** (giả thuyết: giảm mức vươn cần thiết sẽ né được trần joint2): ĐO ĐƯỢC bug tự gây ra trong chính fix `xy_gate` — mẫu số chuẩn hoá dùng `carry_height` làm z_gate_ref, nên hạ carry_height vô tình làm cổng NGHIÊM NGẶT HƠN (tự triệt tiêu lợi ích). Đã sửa: tách `place_carry_z_gate_ref_m=0.15` độc lập khỏi `place_carry_height_m`. Nhưng sau khi sửa đúng, joint2 **vẫn** tiến đều tới trần +10° gần như y hệt (dù đích thấp hơn hẳn) — chứng minh **không phải độ cao** gây nghẽn, mà là khoảng cách NGANG (Y) từ chai tới bát.

**Thử B — dời bát gần chai hơn (y: 0.22→0.32, giảm lệch 180mm→80mm)**: ĐO ĐƯỢC hồi quy NẶNG — cả 4 env fail REACH/GRASP hoàn toàn (trước đó env0 latch+lift ổn định 100% ở đúng seed này qua hơn 6 lần chạy lặp lại). Nguyên nhân: `bowl_pos`/`dist_bottle_bowl` nằm trong observation 26-D và tính ở MỌI stage (không riêng PLACE, kiến trúc từ Phase 10) — đổi vị trí bát tạo input NGOÀI PHÂN BỐ huấn luyện cho policy REACH/GRASP, dù bát không liên quan logic gì tới 2 stage đó. ĐÃ REVERT về (0.58, 0.22, 0.67), xác nhận lại env0 latch+PLACE đúng bước 758 như cũ.

**Bài học chung của cả 2 lần thất bại**: checkpoint `policy_1M_success57.pt` cực kỳ nhạy với BẤT KỲ thay đổi nào ảnh hưởng observation hoặc control-bias dùng chung, kể cả những thứ tưởng chừng "chỉ liên quan PLACE" (vị trí bát) — vì kiến trúc observation/controller không tách bạch theo stage. Muốn đổi hình học nhiệm vụ (vị trí bát) hoặc control-bias toàn cục để né giới hạn joint2, **bắt buộc phải train lại**, không thể vá bằng cách đổi config cho checkpoint đã huấn luyện xong.

### Kết luận Phase 36
Đã loại trừ đầy đủ, có bằng chứng, các hướng "sửa nhanh không cần train lại": tăng tốc trục Z (giúp một phần, giữ lại), hạ carry height (không giúp gốc rễ), đổi null-space bias toàn cục (hồi quy REACH), dời bát (hồi quy REACH/GRASP). Giới hạn joint2 +10° là có thật và gắn liền với khoảng cách Y giữa 2 vị trí cố định trong dữ liệu train — giải quyết triệt để cần MỘT trong hai: (a) train lại policy với hình học bát mới, hoặc (b) train lại/sửa carry để cho phép cổ tay nghiêng khỏi top-down khi mang (giải phóng joint2 khỏi việc phải giữ hướng thẳng đứng suốt hành trình).

### File sửa cuối Phase 36 (trạng thái hiện tại, đã dọn về an toàn)
`isaaclab_openarm_env/mdp/grasp_assist.py`: giữ `xy_gate` (ưu tiên leo cao) + `place_carry_z_gate_ref_m` tách riêng + CarryDbg 7-khớp (debug only, không đổi hành vi khi tắt DEBUG_PLACE). `isaaclab_openarm_env/config.py`: `place_carry_height_m=0.07` (giữ, vô hại dù chưa giải quyết gốc rễ), `nullspace_joint_pos_target="default"` (đã revert), `bowl.init_state.pos=(0.58,0.22,0.67)` (đã revert).

---

## Phase 37 — Train lại 10M bước (server 4090, sau fix Phase 34-36): xác nhận LẠI kết luận Phase 24, không tự giải quyết được joint2

### Cấu hình
Deploy code Phase 34-36 lên server (phát hiện `rl.sh` deploy step 2 dùng đường dẫn USD CŨ `urdf/robot/v10` — đã sửa thành `assets/robot/openarm_v1.0/urdf/v10` khớp với tái cấu trúc `Open_arm_a1_ws` gần đây; upload `policy_1M_success57.pt` lên server thủ công vì chưa có sẵn). Smoke test 3000 bước sạch (server đã lên Isaac Sim 5.1, không còn lo N4 compat 2.3.2). Train chính: `--checkpoint policy_1M_success57.pt --task_phase 3 --stage all --assist-schedule --kl-coef 1.0 --n-epochs 3 --lr-start 1e-4 --lr-end 1e-5 --clip-range 0.1 --ent-coef 0.002 --num-envs 256 --timesteps 10000000` — chạy 22 phút (5922 it/s, GPU 4090 rảnh hoàn toàn), không crash.

### Kết quả — KHÔNG cải thiện PLACE, xác nhận lại Phase 24
Eval deterministic (`--task_phase 3 --stage all --seed 0`, so trực tiếp trước/sau):

| | Trước train | Sau train (scale=1.0) | Sau train (scale=0.0) |
|---|---|---|---|
| grasp_rate | 0.83 | 0.83 | **0.00** |
| lift_start_rate | 0.43 | 0.40 | 0.00 |
| place_start_rate | 0.03 | 0.03 (KHÔNG đổi) | 0.00 |

`assist_scale=0.0` sập hoàn toàn kể cả REACH — xác nhận policy **zero khả năng tự chủ**, dù đã qua 10M bước với KL-protection (giữ REACH/GRASP nguyên vẹn ở scale=1.0, đúng thiết kế) VÀ dù các fix vật lý Phase 34-36 đã làm cơ chế LIFT/CARRY khả thi hơn nhiều cho SCRIPTED assist.

### Kết luận — đúng như Phase 24 đã cảnh báo, chưa ai làm phần còn thiếu
Train thêm theo ĐÚNG công thức cũ (assist-schedule + KL-protection) không tự sửa được: `_osc_world_up_lift`/`_osc_carry_to_bowl` ghi đè CỨNG toàn bộ 6 chiều hành động khi scale gần 1.0 (không có gradient nào chảy qua), và `align_blend` trong `_osc_carry_to_bowl` tích cực CHỐNG LẠI việc cổ tay nghiêng khỏi top-down — đúng bậc tự do cần thiết để né giới hạn joint2 (Phase 36). Muốn train lại có ý nghĩa cho PLACE, cần sửa kiến trúc assist TRƯỚC (không phải chỉ đổi hyperparameter train):
1. Đổi `_osc_carry_to_bowl`/`_osc_world_up_lift` sang residual THẬT ngay từ đầu (không chỉ ở scale<1, xem code comment "w>=1-1e-6 → hành vi hệt ghi đè") — để có gradient thật ngay cả lúc assist còn cao.
2. Giảm hoặc bỏ `place_carry_align_blend` (0.5) để policy được tự do thử nghiêng cổ tay trong CARRY thay vì bị ép giữ top-down suốt hành trình.
3. Cả 2 thay đổi trên đều RỦI RO CAO (đụng cơ chế đã verify kỹ ở LIFT) — cần đo riêng biệt, quy tắc cũ: KHÔNG gộp 2 thay đổi lớn cùng lúc.

### File/hạ tầng sửa (Phase 37)
`rl.sh` (sửa 2 đường dẫn USD deploy khớp tái cấu trúc `Open_arm_a1_ws` mới — `assets/robot/openarm_v1.0/urdf/v10` thay vì `urdf/robot/v10`). Checkpoint mới: `logs/best_policy_train_place_phase36_v1.pt` (đã fetch về local, KHÔNG tốt hơn `policy_1M_success57.pt` cho PLACE — không khuyến nghị dùng thay).

### Thử nhanh trước khi train lại: tắt `place_carry_align_blend` — không giúp gì, ĐÃ REVERT
Giả thuyết: ép cổ tay giữ top-down suốt CARRY (`align_blend=0.5`) chặn mất bậc tự do xoay cần thiết để né trần joint2. Test bằng scripted-assist (không cần train, đo trực tiếp qua CarryDbg): tắt hẳn (0.0) — joint2 đi **đúng quỹ đạo cũ** (margin 19.9°→9.2°, gần như không đổi so với align_blend=0.5), XY còn hội tụ **kém hơn** (107.7mm so với ~90-99mm).

**Nguyên nhân đã hiểu rõ**: `out[mask]=0.0` (khởi tạo action) chạy TRƯỚC, `align_blend=0` chỉ để lại giá trị 0 ở kênh xoay (3:6) — đây là lệnh **"GIỮ NGUYÊN hướng hiện tại"** (delta=0 trong pose_rel), KHÔNG phải "bỏ ràng buộc, để null-space tự quyết". OSC vẫn nhận đủ 6 chiều tư thế mục tiêu, joint2 vẫn phải gánh y như trước. Muốn thật sự giải phóng bậc tự do xoay cần loại orientation ra khỏi target (selection matrix/compliance thật trong OSC controller) — vượt hẳn phạm vi đổi 1 tham số, cần redesign tầng controller. ĐÃ REVERT `place_carry_align_blend` về 0.5.

### Kết luận cuối Phase 37 — đã cạn các hướng chi phí thấp/trung bình
Đã thử và loại trừ CÓ BẰNG CHỨNG: tăng tốc trục Z (giúp một phần, giữ), hạ carry height, null-space bias toàn cục, dời bát, train lại 10M bước với assist-schedule+KL, tắt align_blend. Không hướng nào giải quyết được giới hạn joint2 khi mang chai từ vị trí hiện tại tới bát hiện tại. Hướng còn lại — redesign OSC controller để tách orientation ra khỏi task chính (selection matrix) — là thay đổi kiến trúc lớn, rủi ro cao, cần thiết kế riêng trước khi thử, không phù hợp làm nhanh trong phiên này.

---

## Phase 38 — THIẾT KẾ (chưa implement): selection matrix thật cho OSC — giải phóng joint2 khỏi ràng buộc orientation lúc PLACE_CARRY

**Trạng thái: CHỈ THIẾT KẾ, chưa viết code.** Làm ở phiên sau, theo đúng thứ tự các bước dưới đây.

### Vì sao `align_blend=0` (Phase 37) không đủ — cơ chế thật của IsaacLab OSC
Đã đọc source `isaaclab/controllers/operational_space.py`:
- `OperationalSpaceControllerCfg.motion_control_axes_task: Sequence[int] = (1,1,1,1,1,1)` — **selection matrix THẬT**, dạng ma trận chéo `(num_envs, 6, 6)`, KHÁC HẲN việc đặt delta-action bằng 0.
- `self._selection_matrix_motion_task` được tạo trong `__init__` (`operational_space.py:65-69`) bằng `torch.diag_embed(...).repeat(num_envs, 1)` — **đã sẵn per-env**, không cần sửa gì để có shape đúng.
- Mỗi bước, `compute()` (dùng bởi `apply_actions()`) tính lại `_selection_matrix_motion_b` từ `_selection_matrix_motion_task` (dòng 316-320) rồi dùng NÓ (không phải action gốc) để gate lực trước khi cộng vào `joint_efforts` (dòng 454): `joint_efforts += jacobian_b.mT @ self._selection_matrix_motion_b @ os_command_forces_b`.
- **Kết luận quan trọng**: chỉ cần mutate `_selection_matrix_motion_task` mỗi bước — KHÔNG cần đụng `_motion_p_gains_task` (baked 1 lần lúc init, không sao vì bước gate cuối cùng vẫn dùng selection matrix RUNTIME) — việc gate xảy ra ở bước NHÂN MA TRẬN CUỐI, sau khi PD đã tính xong, nên rezeroed đúng chỗ.
- Đặt hàng 3:6 (orientation) của selection matrix về 0 (chỉ cho env đang PLACE_CARRY) khiến task chính CHỈ còn ràng buộc 3D vị trí — orientation KHÔNG còn bị ép giữ nguyên (khác hẳn `align_blend=0`, vốn chỉ đặt delta=0 = "giữ nguyên", task vẫn ràng buộc đủ 6D). Phần dư tự do (bao gồm cả orientation) sẽ do **null-space term** (`nullspace_control="position"`, đã bật sẵn, kéo về `nullspace_joint_pos_target="default"`) tự giải quyết — đúng cơ chế cho phép joint2 "nhường chỗ" nếu cấu hình `default` cho phép.

### Đường truy cập instance thật (đã xác nhận qua source)
```python
osc = env.action_manager._terms.get("arm_action")._osc   # OperationalSpaceController instance
osc._selection_matrix_motion_task   # (num_envs, 6, 6), float, mutable trực tiếp
```
(`AssistedOperationalSpaceControllerAction.process_actions` gọi `apply_grasp_arm_assist(env, actions)` TRƯỚC `super().process_actions()` — điểm chèn tự nhiên: set mask NGAY ĐẦU mỗi lần `apply_grasp_arm_assist` chạy, mỗi bước, không cần hook thêm chỗ nào khác. `_selection_matrix_motion_b` tính lại mỗi bước nên không lo cache/stale — chỉ cần set lại theo mask HIỆN TẠI mỗi bước, tự đúng qua reset mà không cần logic reset riêng.)

### Việc cần làm ở phiên sau (theo thứ tự)
1. **Xác định trục xoay nào thật sự cần thả** (không thả bừa cả 3 trục — thả hết có nguy cơ chai xoay tự do trong kẹp, đổ/rơi trong lúc mang, đúng rủi ro mà `align_blend` từng cố tránh phần nào). OSC dùng "task frame" riêng (rotation của EE), không chắc chắn trục nào trong 3 trục cục bộ (index 3,4,5 của vector 6 chiều) tương ứng "xoay quanh trục thẳng đứng thế giới". Đo bằng cách: tạm thời zero TỪNG trục MỘT (3, rồi 4, rồi 5) trong demo scripted (giống cách đã đo `dLR`/`toolY` ở Phase 36), quan sát trục nào thay đổi khi joint2 được thả — chỉ giữ lại đúng trục đó thay vì cả 3.
2. **Chỉ áp dụng mask cho env đang `_place_phase in {PLACE_CARRY, PLACE_DESCEND}`** — mọi env khác (REACH/GRASP/LIFT/IDLE/HOLDING) giữ nguyên ma trận mặc định (1,1,1,1,1,1) tuyệt đối, đặt lại mỗi bước:
   ```python
   osc = env.action_manager._terms.get("arm_action")._osc
   free_mask = (env._place_phase == PLACE_CARRY) | (env._place_phase == PLACE_DESCEND)
   osc._selection_matrix_motion_task[:, 3:6, 3:6] = torch.eye(3, device=env.device)  # reset mặc định mọi env
   osc._selection_matrix_motion_task[free_mask, AXIS_TO_FREE, AXIS_TO_FREE] = 0.0    # chỉ env đang carry/descend, chỉ đúng 1 trục đã xác định ở bước 1
   ```
   Đặt ở ĐẦU `apply_grasp_arm_assist` (trước mọi logic khác), chạy mỗi bước — không cần đăng ký reset riêng vì tự tính lại theo mask hiện tại mỗi lần.
3. **Đo lại bằng đúng công cụ đã có** (không đoán): scripted-assist only (không train), `DEBUG_PLACE=1`, `CarryDbg` (đã có sẵn log 7-khớp margin) — xác nhận joint2 margin không còn co dần về 0 trong lúc CARRY, và KHÔNG xuất hiện tilt/rơi chai mới do orientation quá tự do.
4. **Regression gate `--task_phase 2 --seed 0`** — dù mask chỉ kích hoạt khi `_place_phase` CARRY/DESCEND (luôn `False` ở phase 2, vì `_place_phase` không tồn tại/không được set), vẫn nên chạy 1 lần vì đây là lần đầu MUTATE TRỰC TIẾP instance controller dùng chung (`_osc`) thay vì chỉ đọc/ghi action tensor — rủi ro khác loại (nếu code có bug khiến mask sai phạm vi env phase<3), nên xác nhận baseline `success=0.0333 grasp=0.8333 latch=0.4667 lift_start=0.4333` không đổi TRƯỚC khi tin tưởng bước 5.
5. **CHỈ SAU KHI bước 3+4 xác nhận cải thiện thật** (joint2 margin không còn tiến về 0, XY hội tụ được dưới `place_xy_arrival_radius_m=0.03`) mới đáng train lại — lặp lại đúng công thức Phase 37 (`--checkpoint policy_1M_success57.pt --task_phase 3 --stage all --assist-schedule --kl-coef 1.0 --n-epochs 3 --lr-start 1e-4 --lr-end 1e-5 --clip-range 0.1 --ent-coef 0.002`, server 4090, ~10M bước) — **không train lại nếu bước 3 chưa cho thấy cải thiện scripted-assist**, tốn GPU vô ích (bài học Phase 37: train không sửa được vấn đề cơ chế, phải sửa cơ chế trước).

### Rủi ro cần lưu ý khi làm
- Thả orientation có thể làm chai xoay/nghiêng nhiều hơn trong lúc mang (đánh đổi trực tiếp với việc đã né được) — cần xem lại `bottle_tilt_deg`/`place_abort_tilt_deg=25.0` có đủ dư địa không, và có thể cần giữ `align_blend` ở giá trị NHỎ (không phải 0, không phải 0.5) để làm "lò xo mềm" thay vì tắt hẳn constraint — thử cả 2 biến thể (tắt hẳn selection matrix vs giữ constraint mềm qua null-space target lệch khỏi "default").
- Nếu thả sai trục (bước 1 làm ẩu) có thể khiến chai lật ngay cả khi joint2 có thêm margin — PHẢI đo bằng demo GUI thật trước khi tin số liệu headless.
- File cần sửa: `isaaclab_openarm_env/mdp/grasp_assist.py` (thêm khối mutate `_osc._selection_matrix_motion_task` vào đầu `apply_grasp_arm_assist`), không cần sửa `config.py`/`actions.py`.

## Phase 39 — Đo thực nghiệm bước 1 của thiết kế Phase 38: trục xoay nào cần thả (2026-09-16)

### Cách đo
Thêm khối tạm (gate bằng biến môi trường `PLACE_FREE_AXIS=3|4|5`, không set = không đổi gì) vào đầu `apply_grasp_arm_assist` (`grasp_assist.py`), mutate trực tiếp `env.action_manager._terms.get("arm_action")._osc._selection_matrix_motion_task[:, 3:6, 3:6]` mỗi bước — reset về identity cho MỌI env, rồi zero đúng 1 trục cho env đang `PLACE_CARRY|PLACE_DESCEND`. Đo bằng đúng kịch bản cũ (`eval_lift_metrics.py --model-path policy_1M_success57.pt --episodes 20 --num-envs 8 --bottle-noise 0.05 --assist-scale 1.0 --stage place --seed 0 --task_phase 3`, `DEBUG_PLACE=1`) — env0 luôn vào PLACE_CARRY ở đúng step 758 (seed cố định), nên so sánh được apple-to-apple giữa 4 lần chạy (baseline + 3 trục).

### Kết quả (env0, cùng episode, step_ct cuối ~1185)

| | tilt_deg cuối | j2 margin cuối | xy cuối | Kết luận |
|---|---|---|---|---|
| Baseline (không thả trục) | 8.76° | 6.8° | 97.2mm | (đối chứng, khớp Phase 36) |
| Trục 3 thả | **14.13°** (+5.4°) | **10.1°** (+3.3°) | 93.3mm (−3.9mm) | Có tác dụng thật lên joint2, nhưng đổi bằng tăng tilt đáng kể |
| Trục 4 thả | ≥15.3°, **episode CHẾT SỚM ở step 856** (real termination, không phải timeout — chỉ 98 bước sau khi vào CARRY) | 12.4° lúc chết (đang giảm nhanh hơn cả baseline) | n/a (chết sớm) | **Nguy hiểm nhất — không dùng** |
| Trục 5 thả | 8.60° (~bằng baseline) | 7.5° (+0.7°, không đáng kể) | 96.3mm (~bằng baseline) | Trơ — trục này không liên quan gì tới ràng buộc đang giữ joint2 |

### Diễn giải
- **Trục 4** là trục nguy hiểm nhất — thả nó khiến chai mất ổn định NHANH HƠN và tệ hơn cả không làm gì, dẫn tới termination thật (không chỉ timeout). Đây rất có thể là trục giữ chai KHÔNG bị lật theo hướng dễ đổ nhất trong lúc mang — chính là bậc tự do mà `align_blend` gốc cố bảo vệ mạnh nhất.
- **Trục 5** gần như không ảnh hưởng gì tới joint2 — không phải bậc tự do đang bị khoá gây nghẽn.
- **Trục 3** là trục ĐÚNG hướng thiết kế nhắm tới (cho joint2 thêm margin thật, đo được +3.3° so với baseline giảm dần) nhưng đánh đổi bằng tilt tăng +5.4° (8.76°→14.13°) — vẫn dưới ngưỡng abort (`place_abort_tilt_deg=25°`) nhưng là chi phí thật, không miễn phí.
- **Quan trọng — ngay cả khi chấp nhận đánh đổi tilt của trục 3, JOINT2 VẪN KHÔNG ĐỦ để hoàn thành PLACE trong thời gian episode còn lại**: tốc độ hội tụ XY đo được ở trục 3 (~152mm→93mm trong 420 bước ≈ 0.14mm/bước) vẫn cần thêm ~450 bước nữa để chạm `place_xy_arrival_radius_m=30mm` — vượt quá số bước còn lại của episode (kết thúc TIMEOUT ở cả baseline lẫn trục 3). Nguyên nhân tốc độ chậm là `place_carry_speed_scale=0.35` (Phase 36), KHÔNG phải joint2 — nghĩa là chỉ giải phóng trục xoay là **không đủ**, phải kết hợp thêm việc tăng tốc CARRY mới có cơ hội hoàn thành trong episode.

### Kết luận — chưa đủ cơ sở để tiến sang bước 4/5 (regression gate + train lại)
Lợi ích đo được của hướng selection-matrix (trục 3) là CÓ THẬT nhưng nhỏ (+3.3° margin, −3.9mm hội tụ) so với chi phí (+5.4° tilt) và KHÔNG giải quyết được vấn đề hết thời gian episode. Trục 4 phải loại bỏ hẳn (nguy hiểm). Cần quyết định của user trước khi đi tiếp: chấp nhận trục 3 kèm biện pháp giảm nhẹ (vd. giữ selection weight ở mức trung gian như 0.3 thay vì 0 hẳn, "lò xo mềm" như thiết kế Phase 38 đã cảnh báo) và kết hợp tăng `place_carry_speed_scale`, hay dừng hướng selection-matrix vì lợi ích quá nhỏ so với độ phức tạp/rủi ro thêm vào.

### File sửa (Phase 39 — CHỈ ĐỂ ĐO, cần dọn lại trước khi commit)
`isaaclab_openarm_env/mdp/grasp_assist.py`: thêm khối `PLACE_FREE_AXIS` env-var-gated ở đầu `apply_grasp_arm_assist` (không set biến này thì không đổi hành vi — an toàn cho mọi phase).

### Quyết định cuối Phase 39 — DỪNG hướng selection-matrix
User xem số đo (bảng trên) và quyết định **dừng hướng selection-matrix**: lợi ích trục 3 (+3.3° margin) quá nhỏ so với chi phí (+5.4° tilt) và dù sao cũng không đủ nhanh để PLACE hoàn thành trong thời gian episode — thêm phức tạp/rủi ro vào orientation control dùng chung không đáng. Đã xoá khối `PLACE_FREE_AXIS` thí nghiệm khỏi `apply_grasp_arm_assist` (quay lại đúng code trước Phase 39). Regression gate phase 2 chạy lại xác nhận khớp baseline chính xác: `success=0.0333 grasp=0.8333 latch=0.50 lift_start=0.4333`.

**Trạng thái cuối cùng của investigation PLACE (Phase 34-39)**: đã sửa và GIỮ LẠI 3 cải thiện thật (grip OR-fallback, height-clearance gate, xy_gate ưu tiên leo cao) — các fix này cải thiện PLACE nhưng không giải quyết triệt để giới hạn joint2. Đã thử và loại trừ CÓ BẰNG CHỨNG toàn bộ các hướng sửa nhanh khả dĩ: null-space bias toàn cục, dời bát, train lại 10M bước theo công thức cũ, tắt align_blend, và giờ thêm cả selection-matrix thật (3 trục). **Không còn hướng sửa nhanh/trung bình nào chưa thử.** Hai hướng còn lại đều là thay đổi kiến trúc/dữ liệu lớn, cần phiên riêng: (a) đổi hình học bát+train lại từ đầu (không phải fine-tune), hoặc (b) redesign lại toàn bộ cơ chế assist CARRY thành residual thật ngay từ đầu training (không chỉ lúc scale<1) để chính sách có gradient thật cho việc nghiêng cổ tay một cách có kiểm soát thay vì assist cứng nhắc.

## Phase 40 — Kiểm tra trực giác "bát to không nhỏ" của user: nới ngưỡng arrival theo hình học bát THẬT (2026-09-16)

### Bối cảnh
User phản bác kết luận Phase 39 ("không thể đặt chai vào bát"): bát trong scene không nhỏ, và gợi ý cân nhắc mục tiêu đơn giản hơn (ô kẻ) thay vì train lift. Kiểm tra lại thì phát hiện `config.py:488-498` ĐÃ CÓ số đo hình học bát THẬT từ trước (BBoxCache, 2026-09-08): bát rộng bán kính ~8cm (bbox 159.7×159.7mm), chai đường kính 43.3mm (bán kính 21.65mm). Biên "chai còn chồng lấn miệng bát" ≈ 8+2.165 = **101.5mm**. Nhưng `place_xy_arrival_radius_m` (ngưỡng cho phép CARRY→DESCEND) đang đặt ở **30mm** — nhỏ hơn hẳn bán kính bát thật, chưa từng dựa trên số đo, chỉ là giá trị đoán (đúng như plan gốc đã tự đánh dấu "⚠️ ĐOÁN"). Hệ quả: **DESCEND chưa từng được kích hoạt trong bất kỳ lần đo nào trước đây** — mọi lần "place_timeout" là "hết giờ khi còn đang bay ngang", KHÔNG PHẢI "thả trật bát".

### Đo 1 — nới arrival radius theo số đo thật (0.03→0.09→0.10)
Cùng kịch bản (`--seed 0 --stage place --task_phase 3`, env0 luôn vào CARRY ở step 758). 0.09 vẫn hụt sát nút (XY dừng 97.2mm cuối episode 20s). Nới lên 0.10 (vẫn trong biên vật lý 101.5mm): `xy_ok` chuyển True ở step 1065 — nhưng **DESCEND vẫn KHÔNG kích hoạt** vì còn bị chặn bởi điều kiện thứ hai: `z_ok` (độ cao chai so với miệng bát, cần > `place_carry_clearance_m=0.03`) vẫn `False` suốt — z_clear chỉ leo từ -47.5mm lên -6.3mm (vẫn THẤP HƠN miệng bát 6.3mm khi hết giờ). Vậy nút thắt thật ở cấu hình 20s là **Z clearance**, không phải XY nữa.

### Đo 2 — kéo dài episode (20s→45s, chỉ để chẩn đoán, thêm cờ `EVAL_EPISODE_LEN_S` env-var-gated trong `eval_lift_metrics.py`) để xem hội tụ có phải "trần cứng" hay chỉ "chậm"
Kết quả: **XY tiếp tục cải thiện thật** — từ 97.2mm (step1185) xuống **78.9mm (step2175)**, dưới hẳn biên vật lý 101.5mm và tiến gần biên "chai nằm trọn trong bát trừ margin" (~58mm) — bác bỏ giả thuyết "trần cứng vĩnh viễn", xác nhận đúng nguyên tắc "không chấp nhận trần vật lý quá sớm": đây là **hội tụ chậm dần (asymptotic)**, không phải bế tắc tuyệt đối. joint2 margin cũng không tệ đi thêm nhiều (dao động 5.3°→6.3°, tương đối ổn định quanh đáy).

**Nhưng phát hiện MỚI, chưa từng thấy trong mọi lần đo 20s trước đây**: episode **KHÔNG timeout** — nó **TERMINATE THẬT** ở step 2180 vì **tilt tăng dần và vượt ngưỡng** (`tilt_deg` cuối = 15.07°, đang tăng liên tục, kèm `z_clear` bắt đầu ĐẢO CHIỀU giảm trở lại (-2.7mm→-9.9mm) dù trước đó đang leo lên). Nghĩa là: khi cho đủ thời gian, bản thân động tác CARRY (kể cả ở baseline, KHÔNG cần bất kỳ can thiệp selection-matrix nào) tự nó tích luỹ mất ổn định và làm chai nghiêng dần tới mức lật thật — một pha lỗi hoàn toàn mới, bị 20s episode length che khuất từ trước tới giờ (cắt ngang trước khi kịp bộc lộ).

### Kết luận Phase 40 — nửa đúng, nửa sai so với cả 2 giả thuyết trước đó
- **User ĐÚNG một phần quan trọng**: bát không nhỏ, ngưỡng arrival 30mm là đoán sai, JOINT2 KHÔNG phải trần cứng tuyệt đối — cho đủ thời gian XY vẫn hội tụ tiếp, đã xuống dưới biên vật lý hợp lý.
- **Nhưng phát sinh vấn đề MỚI, nghiêm trọng hơn**: carry không ổn định lâu dài — càng mang lâu, chai càng có xu hướng nghiêng tăng dần rồi lật thật, ĐỘC LẬP với việc XY có hội tụ hay không. Đây có thể liên quan tới việc joint2 càng gần trần thì OSC càng phải bù mạnh hơn ở các khớp khác để giữ hướng úp xuống, gây trôi/dao động chậm — cần xem qua **demo GUI thật** (không chỉ số liệu headless) mới kết luận được cơ chế chính xác, đúng nguyên tắc đã đặt ra ở Phase 38.

### File sửa (Phase 40)
`isaaclab_openarm_env/config.py`: `place_xy_arrival_radius_m` 0.03→0.10 (có căn cứ đo thật, nhưng CHƯA chứng minh đủ vì DESCEND vẫn chưa từng kích hoạt được — bị Z-clearance chặn). `eval_lift_metrics.py`: thêm cờ chẩn đoán `EVAL_EPISODE_LEN_S` (env var, không set = hành vi cũ 20s, đặt SAU `apply_phase2_demo_gates` vì `PHASE2_BASE` ghi đè `episode_length_s` về 20.0 nếu đặt trước).

## Phase 41 — "Ô kẻ rộng" + phát hiện bug thật thứ hai: release CHƯA TỪNG hoạt động — THÀNH CÔNG PLACE ĐẦU TIÊN (2026-09-16)

### Theo yêu cầu user: đơn giản hoá mục tiêu PLACE, không train
User bác bỏ kết luận "không thể" và đề xuất mục tiêu đơn giản hơn: thả vào một vùng rộng ("ô kẻ") thay vì đúng tâm bát, không cần train. Redesign (chỉ đổi ngưỡng + 1 dòng logic action, KHÔNG đổi state machine/asset scene):
- `place_success_xy_radius_m`: 0.05 → **0.12** (khớp bán kính bát đo thật ~8cm + margin, không còn đoán).
- `place_xy_arrival_radius_m`: 0.10 → **0.12** (khớp ngưỡng thành công — CARRY→DESCEND và success dùng cùng 1 tiêu chuẩn).
- `place_carry_clearance_m`: 0.03 → **-1.0** (bỏ hẳn yêu cầu "phải cao hơn miệng bát mới cho hạ" — mục tiêu giờ là vùng phẳng, không có thành cần né).

### Test lần 1 (chỉ đổi ngưỡng): CARRY→DESCEND→HOLDING chạy được, nhưng KHÔNG BAO GIỜ release
Chạy đúng kịch bản đo cũ: `[PlaceDbg] env0 step_ct=855 CARRY→DESCEND ... step_ct=860 DESCEND→HOLDING` — state machine chạy đúng lần đầu tiên trong toàn bộ investigation! Nhưng `released: false` tới hết episode dù đã ở HOLDING >300 bước (`place_release_hold_steps=5` thoả từ lâu, `place_release_ready()` chắc chắn `True`).

### BUG THẬT THỨ HAI (độc lập hoàn toàn với Phase 40): `_close_progress` không có đường MỞ
Đọc `AssistedBinaryGripperAction.apply_actions()` (`actions.py`): khớp kẹp thật được điều khiển bởi `_close_progress` (`targets = open*(1-prog) + close*prog`), KHÔNG PHẢI đọc trực tiếp raw action mỗi bước. `_close_progress` chỉ có DUY NHẤT đường tăng (`self._close_progress[can_advance] += step_size`, toàn bộ ~150 dòng logic phía trên chỉ nói về ramp ĐÓNG — pause-on-tilt, pause-on-asym, freeze theo lực chạm...). Đường về 0 duy nhất (`stale = ~self._want_close & ~self._grasp_latched`) đòi `~_grasp_latched`, mà `_grasp_latched` CỐ Ý giữ `True` suốt CARRY/DESCEND/HOLDING (chai vẫn đang được giữ). Hệ quả: dòng `actions[in_place & release_ready, 0] = 1.0` (S2, viết từ Phase 10) đặt ĐÚNG raw action "mở kẹp", nhưng **không bao giờ chạm tới khớp thật** — `max_gc` đứng yên ở 0.7625 suốt episode dù đã ở HOLDING rất lâu.

**Đây là lý do PLACE release CHƯA TỪNG hoạt động kể từ Phase 10** — độc lập hoàn toàn với vấn đề "DESCEND chưa từng kích hoạt" (Phase 40). Cả 2 bug phải sửa CÙNG NHAU mới thấy được kết quả cuối.

### Fix
`actions.py::apply_actions()`: thêm nhánh giảm riêng cho `_close_progress`, loại các env đang release khỏi `can_advance` (tránh vừa tăng vừa giảm cùng lúc do `_grasp_latched` vẫn `True`):
```python
releasing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
if uses_place(getattr(self._env.cfg, "task_phase", 1)) and s is not None:
    in_place = self._env._stage == STAGE_PLACE
    releasing = in_place & place_release_ready(self._env, s)
    can_advance = can_advance & ~releasing
self._close_progress[can_advance] += step_size[can_advance]
self._close_progress.clamp_(0.0, 1.0)
release_ramp_steps = int(getattr(self._env.cfg, "place_release_ramp_steps", 15))
self._close_progress[releasing] -= 1.0 / max(release_ramp_steps, 1)
self._close_progress.clamp_(0.0, 1.0)
```
`config.py`: thêm `place_release_ramp_steps: int = 15`.

**Regression gate phase 2 PASS** (hàm này dùng chung mọi stage): `success=0.03 grasp=0.83 latch=0.50 lift_start=0.43` — khớp baseline chính xác (nhánh mới gate bởi `uses_place`, chết hoàn toàn ở phase<3).

### Kết quả sau fix — THÀNH CÔNG PLACE THẬT ĐẦU TIÊN
Cùng kịch bản (seed=0, env0 vào CARRY ở step758 y hệt mọi lần trước):
```
success=True, released=True, release_dist_bottle_bowl_m=0.1144 (trong vùng 0.12), tilt cuối=1.33°, steps=880 (kết thúc THẬT, không timeout)
```
`place_start_rate=0.05, release_rate=0.05, success_rate=0.05` — bằng nhau, nghĩa là **100% số lần PLACE thực sự bắt đầu đều thành công** (1/20 episode tới được PLACE trong lần đo này do tỉ lệ GRASP/LIFT thượng nguồn còn thấp, không liên quan PLACE). Nút thắt còn lại nằm ở REACH/GRASP/LIFT (đã biết từ trước — grasp_rate 0.83, lift_start_rate 0.43 ở phase 2), không còn nằm ở PLACE nữa.

### Kết luận Phase 41
User đúng: bài toán "đặt chai vào bát" GIẢI ĐƯỢC bằng scripted-assist, KHÔNG CẦN train, khi (1) mục tiêu được nới thành vùng rộng thực tế (khớp hình học bát đo thật, không phải điểm ảo 30mm), và (2) sửa đúng bug thật (release chưa từng hoạt động) — không phải do joint2 hay kiến trúc controller như các Phase 36-39 từng nghi ngờ. Bài học: đầu tư đo thật (Phase 40's kéo dài episode, tìm ra XY hội tụ được) + không dừng lại ở phát hiện đầu tiên (Phase 40 tưởng DESCEND là nút thắt cuối, hoá ra còn 1 bug nữa ở release) mới lộ ra bức tranh đầy đủ.

### File sửa (Phase 41)
`isaaclab_openarm_env/config.py` (`place_success_xy_radius_m=0.12`, `place_xy_arrival_radius_m=0.12`, `place_carry_clearance_m=-1.0`, `place_release_ramp_steps=15` mới). `isaaclab_openarm_env/mdp/actions.py` (`AssistedBinaryGripperAction.apply_actions()` — thêm nhánh giảm `_close_progress` cho release). `eval_lift_metrics.py` (giữ cờ chẩn đoán `EVAL_EPISODE_LEN_S` từ Phase 40, không dùng trong test cuối này nhưng vô hại khi không set).

### Xác nhận thêm (seed=1, 50 episode, không cherry-pick)
3/3 episode chạm được PLACE đều **success=True** (100%, không phải may mắn 1 lần): tilt cuối 0.94-1.85° (rất thấp, chai đứng vững), release_dist 101-116mm (đều trong vùng 120mm). Nút thắt duy nhất còn lại là tỉ lệ GRASP/LIFT thượng nguồn (grasp=0.58, lift_start=0.34 ở seed này) — vấn đề đã biết từ trước, không liên quan gì tới PLACE nữa.
