# Hướng dẫn thiết lập và huấn luyện OpenArm A1 (Isaac Lab)

Thư mục này chứa mã nguồn để huấn luyện Reinforcement Learning (RL) cho nhiệm vụ gắp quả táo đặt vào bát sử dụng **Isaac Lab** và thuật toán **PPO (Stable-Baselines3)**.

---

## 📁 Cấu trúc thư mục

- [isaaclab_openarm_env.py](file:///home/hans/universal_bot/Open_arm_a1_ws/src/isaacsim_setup/isaaclab_openarm_env.py): Định nghĩa môi trường RL kế thừa từ lớp `DirectRLEnv` của Isaac Lab.
- [isaaclab_train.py](file:///home/hans/universal_bot/Open_arm_a1_ws/src/isaacsim_setup/isaaclab_train.py): File chạy chính để huấn luyện thuật toán PPO.
- [isaaclab_demo.py](file:///home/hans/universal_bot/Open_arm_a1_ws/src/isaacsim_setup/isaaclab_demo.py): Script chạy thử (playback) và kiểm tra trực quan mô hình chính sách đã huấn luyện.
- [deploy_training.sh](file:///home/hans/universal_bot/Open_arm_a1_ws/deploy_training.sh): Script tự động đồng bộ mã nguồn và kích hoạt huấn luyện chạy nền trên server GPU (RTX 4090).
- [fetch_model.sh](file:///home/hans/universal_bot/Open_arm_a1_ws/fetch_model.sh): Script tải mô hình huấn luyện từ server về máy tính cá nhân (Laptop).

---

## 🚀 1. Huấn luyện Reinforcement Learning (RL)

### Cách A: Huấn luyện trên Server GPU (Chạy nền / Headless)
Đồng bộ mã nguồn từ laptop lên server và khởi chạy huấn luyện:
```bash
./deploy_training.sh naiscorp@192.168.1.122
```
*Ghi chú: Lịch sử huấn luyện và các checkpoint sẽ được tự động lưu tại thư mục `/data21tb/huyhoang/openarm_train_ws/logs_openarm/` trên server.*

### Cách B: Huấn luyện trên máy tính cá nhân (Local PC/Laptop)
Để thử nghiệm huấn luyện nhỏ trực tiếp trên máy của bạn có mở giao diện mô phỏng:
```bash
/home/hans/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  src/isaacsim_setup/isaaclab_train.py \
  --num_envs 16 \
  --timesteps 1000000 \
  --log_dir src/isaacsim_setup/logs
```

---

## 📊 2. Theo dõi tiến trình huấn luyện (TensorBoard)

Hệ thống TensorBoard chạy trên server qua cổng **`6008`** (tránh cổng `6007` vì đã bị tiến trình hệ thống khác chiếm dụng):
- URL truy cập: **http://192.168.1.122:6008**

Theo dõi log huấn luyện trực tiếp (real-time) từ Terminal laptop:
```bash
ssh naiscorp@192.168.1.122 'tail -f /data21tb/huyhoang/openarm_train_ws/logs_openarm/train.log'
```

---

## 👁️ 3. Chạy Thử Nghiệm Trực Quan (Playback / Demo)

Trước khi chạy demo hiển thị trên laptop, bạn cần tải mô hình mới nhất từ server về máy:
```bash
./fetch_model.sh naiscorp@192.168.1.122
```

### Cách A: Chạy trực tiếp trên Laptop (Có giao diện đồ họa GUI)
Chạy mô phỏng hiển thị trên chính màn hình laptop của bạn:
```bash
/home/hans/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  src/isaacsim_setup/isaaclab_demo.py \
  --model_path src/isaacsim_setup/logs/best_policy.pt \
  --num_envs 1
```

### Cách B: Chạy trên Server bằng WebRTC (Xem qua Trình duyệt Web)
Chạy mô phỏng trên server từ xa nhưng xem hình ảnh hiển thị trên trình duyệt laptop:
1. Chạy demo ở chế độ headless stream trên server:
   ```bash
   ssh -t naiscorp@192.168.1.122 "
     /data21tb/huyhoang/isaacsim/python.sh \
       /data21tb/huyhoang/openarm_train_ws/src/isaacsim_setup/isaaclab_demo.py \
       --model_path /data21tb/huyhoang/openarm_train_ws/logs_openarm/train/best_policy.pt \
       --num_envs 1 \
       --headless \
       --livestream 1
   "
   ```
2. Mở trình duyệt web trên laptop và truy cập:
   **[http://192.168.1.122:8211/streaming/webrtc-demo/](http://192.168.1.122:8211/streaming/webrtc-demo/)**

### Cách C: Chạy trên Server xuất ra Màn hình Vật lý của Server
Nếu server của bạn có kết nối màn hình riêng và bạn muốn xuất giao diện trực tiếp trên màn hình đó:
```bash
ssh -t naiscorp@192.168.1.122 "DISPLAY=:0 /data21tb/huyhoang/isaacsim/python.sh \
  /data21tb/huyhoang/openarm_train_ws/src/isaacsim_setup/isaaclab_demo.py \
  --model_path /data21tb/huyhoang/openarm_train_ws/logs_openarm/train/best_policy.pt \
  --num_envs 1"
```
*(Nếu màn hình không lên giao diện, thử thay thế `DISPLAY=:0` bằng `DISPLAY=:1`).*

---

## 🛠️ Lưu ý Kỹ thuật

- **Trùng lặp mô hình & Va chạm**: File thiết lập cảnh nền `qvic.usd` nguyên bản chứa sẵn một robot và vật thể tĩnh. Hệ thống đã được lập trình để tự động ẩn các đối tượng này (`openarm`, `Apple`, `Bowl`, `Cup`, `Bottle` tĩnh) khi khởi chạy, chỉ giữ lại các đối tượng động do Isaac Lab quản lý để tránh lỗi va chạm và nổ vật lý.
- **Gọi super trong reset**: Hàm reset của môi trường `ApplePickPlaceEnv` đã được sửa để gọi `super()._reset_idx(env_ids)`. Điều này đảm bảo bộ đếm bước đi (`episode_length_buf`) của lớp cơ sở hoạt động chính xác, tránh việc môi trường bị reset liên tục sau mỗi bước đi.
