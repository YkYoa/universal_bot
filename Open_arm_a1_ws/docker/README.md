# Hướng dẫn chạy OpenArm VLA Bridge bằng Docker

Tài liệu này hướng dẫn cách build, cấu hình mạng CycloneDDS và chạy **VLA Bridge Node** bên trong Docker container trên Server/Laptop để kết nối với giả lập Isaac Sim hoặc robot thật.

---

## 1. Tổng quan kiến trúc mạng
* **VLA Bridge Node (Python)**: Chạy bên trong Docker container ROS 2 Humble. Node này sẽ subscribe các topic camera và khớp từ robot/Isaac Sim, gửi HTTP request tới VLA Prediction Server (Qwen-VL) và nhận chuỗi hành động trả về để gửi lại điều khiển robot.
* **Isaac Sim / Robot thật**: Chạy trực tiếp trên máy host (Laptop/Server).
* **CycloneDDS**: Được sử dụng để kết nối ROS 2 giữa Docker container và máy host (hoặc giữa 2 máy tính khác nhau trong mạng LAN) bằng cách tối ưu hóa kích thước gói tin (MaxMessageSize) để truyền ảnh camera mượt mà không bị ngắt quãng.

---

## 2. Các bước triển khai (Deploy) từ Laptop lên Server

Nếu bạn phát triển code trên Laptop và muốn đẩy lên chạy trên Server (Laptop phụ đóng vai trò Server chạy VLA):

Sử dụng script `deploy_to_server.sh` để tự động đồng bộ mã nguồn và build Docker image trên Server:
```bash
# Đứng tại thư mục Open_arm_a1_ws trên Laptop của bạn:
./docker/deploy_to_server.sh <user_server>@<ip_server>

# Ví dụ:
./docker/deploy_to_server.sh naiscorp@192.168.1.122
```
*Script này sẽ thực hiện:*
1. Tạo thư mục `~/openarm_ws` trên server.
2. Dùng `rsync` đồng bộ thư mục `src/` và `docker/`.
3. SSH vào server và build image `openarm-humble:latest` từ `docker/Dockerfile`.

---

## 3. Cấu hình CycloneDDS (Quan trọng để kết nối mạng)

Để Docker container và Isaac Sim (hoặc robot) nhìn thấy các topic của nhau, cấu hình CycloneDDS trên cả hai phía phải khớp nhau về `ROS_DOMAIN_ID` và cấu hình Peer IP nếu chạy khác máy.

### A. Cấu hình trên Server/Laptop chạy Isaac Sim
1. Mở terminal trên máy chạy Isaac Sim, export các biến môi trường trước khi chạy giả lập:
   ```bash
   export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
   export CYCLONEDDS_URI=$HOME/Desktop/isaacsim/cyclonedds.xml
   ```
2. Mở file `cyclonedds.xml` trong thư mục Isaac Sim và thêm IP của máy chạy Docker vào danh sách `<Peers>` (nếu chạy khác máy):
   ```xml
   <Discovery>
       <Peers>
           <Peer Address="IP_CỦA_MÁY_CHẠY_DOCKER" />
       </Peers>
   </Discovery>
   ```

### B. Cấu hình trên máy chạy Docker Container
Cấu hình CycloneDDS đã được copy vào trong container tại đường dẫn `/openarm_ws/cyclonedds.xml` thông qua `docker-compose.yml`.
* Nếu Docker và Isaac Sim chạy **trên cùng một máy**, file cấu hình mặc định [cyclonedds_server.xml](file:///home/hans/universal_bot/Open_arm_a1_ws/docker/cyclonedds_server.xml) sẽ sử dụng chế độ Multicast (không cần điền Peer IP).
* Nếu chạy **khác máy**, hãy mở file [cyclonedds.xml](file:///home/hans/universal_bot/Open_arm_a1_ws/cyclonedds.xml) ở máy dev, chỉnh sửa IP của máy chạy Isaac Sim trong phần `<Peers>` rồi tiến hành deploy lại.

---

## 4. Hướng dẫn chạy VLA Bridge trên Server

Di chuyển vào thư mục workspace trên Server:
```bash
cd ~/openarm_ws
```

### Chạy VLA Bridge Node bằng Docker Compose:
Sử dụng `docker-compose` để tự động quản lý biến môi trường và chạy container:

```bash
VLA_INSTRUCTION="Push the apple to the block" \
VLA_HOST="localhost" \
VLA_ARM="left" \
docker compose -f docker/docker-compose.yml up vla_bridge
```

**Các tham số tùy chỉnh (Environment Variables):**
* `VLA_INSTRUCTION`: Câu lệnh tiếng Anh hướng dẫn robot thực hiện nhiệm vụ (ví dụ: *"Push the apple to the block"*, *"Pick up the cup"*).
* `VLA_HOST`: Địa chỉ IP của máy đang host VLA Model (mặc định cổng `10000`). Nếu VLA API chạy trên cùng máy với Docker, hãy để là `localhost` hoặc `127.0.0.1`.
* `VLA_PORT`: Cổng dịch vụ của VLA API (mặc định là `10000`).
* `VLA_ARM`: Chọn cánh tay để nhận hành động điều khiển (`left` hoặc `right`).

---

## 5. Hướng dẫn gỡ lỗi (Debug & Troubleshooting)

### Kiểm tra các container đang chạy
```bash
docker ps
```

### Xem log trực tiếp của VLA Bridge
```bash
docker compose -f docker/docker-compose.yml logs -f vla_bridge
```

### Truy cập vào Bash Shell của Container để debug bằng lệnh ROS 2
Nếu bạn muốn dùng các lệnh kiểm tra như `ros2 topic list` hay `ros2 topic echo` để xem ảnh camera từ Isaac Sim có vào được container hay không:

1. **Khởi chạy container debug shell:**
   ```bash
   docker compose -f docker/docker-compose.yml run --rm shell
   ```
2. **Kiểm tra kết nối ROS 2 bên trong shell:**
   ```bash
   # Liệt kê danh sách topic đang hoạt động
   ros2 topic list
   
   # Kiểm tra xem có nhận được ảnh từ camera trái không
   ros2 topic hz /camera_left/image_raw
   ```

### Xử lý lỗi thường gặp
* **Không nhận được dữ liệu ảnh/khớp từ Isaac Sim:**
  * Kiểm tra xem cả hai máy đã thông ping LAN chưa.
  * Đảm bảo biến môi trường `ROS_DOMAIN_ID=42` được set đồng nhất trên cả máy chạy Isaac Sim và trong file `docker-compose.yml`.
  * Đảm bảo firewall (UFW) trên Ubuntu không chặn các cổng của CycloneDDS (bạn có thể thử tắt tạm ufw bằng lệnh `sudo ufw disable` để kiểm tra kết nối).
