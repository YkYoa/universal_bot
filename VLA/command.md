# Chọn model VLA cho IQ-9075 — tra cứu HuggingFace (2026-08-28)

## Kết luận

| Model | Kiến trúc | License | Phù hợp IQ-9075 (Hexagon NPU/QNN)? |
|---|---|---|---|
| **pi-0.5** (`lerobot/pi05_base`) | PaliGemma VLM (~3B) + FAST-tokenizer **autoregressive decode** | Apache 2.0 (qua LeRobot) | **Tốt nhất** — cùng pattern LLM-decode với Qwen3-4B đã chạy Genie/NPU trên chính board này (`README_QWEN3_4B_GENIE_NPU_IQ9075.md`) |
| **GR00T N1.5-3B** | SigLip2 + T5-ish + **DiT flow-matching** action head | NVIDIA License — **non-commercial only** | Khả thi nhưng rủi ro quantize DiT cao hơn, license hạn chế |
| **GR00T N1.7-3B** | SigLip2/Cosmos-Reason2-2B + T5 + **DiT flow-matching** (cùng họ kiến trúc N1.5) | NVIDIA Open Model License (thoáng hơn N1.5, không giới hạn non-commercial) | Cùng rủi ro DiT như N1.5, nhưng license tốt hơn + data pretrain mới hơn (20K giờ EgoScale video) |

**Chọn GR00T N1.7-3B thay vì N1.5** (khớp với thư mục `Gr00t-N1,7` đã có sẵn trên
server) — vì license thoáng hơn và benchmark tốt hơn, KHÔNG phải vì nó dễ chạy
NPU hơn N1.5: cả hai dùng chung action head kiểu DiT/flow-matching, nên độ rủi
ro khi quantize/compile qua QNN là **như nhau**, không đổi theo version.

**pi-0.5 vẫn là target chính cho IQ-9075** vì lý do kiến trúc, không phải vì
đã có sẵn: FAST tokenizer biến action chunk thành autoregressive-decode giống
hệt LLM — đúng pattern Qualcomm Genie SDK đã chứng minh chạy tốt trên
**chính board IQ-9075 này** (Qwen3-4B-Instruct, xem README ở trên). GR00T
dùng cả hai model chạy song song: pi-0.5 làm track chính, GR00T N1.7 chạy dry
run để lấy số liệu operator-coverage QNN trước khi đầu tư sâu (theo plan ở
`/home/hans/.claude/plans/i-just-got-top-streamed-goose.md`).

Không có bản GR00T "nhỏ/edge-optimized" nào khác trên HF — tất cả variant
trong collection N1.7 (SimplerEnv-Bridge, SimplerEnv-Fractal, DROID, LIBERO)
đều là 3B, chỉ khác dữ liệu fine-tune, không khác kích thước.

## Lệnh tải GR00T N1.7-3B (chưa có trên server, cần tải)

Chạy trên `huyhoang-4090` (không phải trên IQ-9075 — GPU converter/quantize
QNN chỉ có bản x86, và fine-tune cần CUDA):

```bash
ssh huyhoang-4090
source /data21tb/users/huyhoang/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

# HF cache đã trỏ đúng chỗ chưa thì kiểm tra trước (tránh rơi vào $HOME nhỏ)
export HF_HOME=/data21tb/users/huyhoang/.cache/huggingface

huggingface-cli download nvidia/GR00T-N1.7-3B \
  --local-dir "/home/huyhoang/VLA/Gr00t-N1,7/weights/gr00t_n1.7_3b"
```

> Model ~3B tham số, BF16 safetensors — ước tính ~6GB tải về. `/data21tb` còn
> 4.6TB, không phải lo dung lượng.

## pi-0.5 — đã có sẵn, không cần tải lại

`/home/huyhoang/VLA/PI_0,5/weights/pi0_base/` (14GB, `lerobot/pi0_base`).
Việc còn thiếu là gỡ CUDA OOM khi fine-tune trên GPU dùng chung (xem
`VLA/outputs/pi05_trial_run4.log`), không phải tải model.

## Nguồn

- https://huggingface.co/nvidia/GR00T-N1.7-3B
- https://huggingface.co/nvidia/GR00T-N1.5-3B
- https://huggingface.co/collections/nvidia/gr00t-n17
- https://huggingface.co/lerobot/pi05_base
- https://huggingface.co/blog/nvidia/gr00t-n1-7

---

## Kết quả dry-run thật trên NPU (2026-08-28)

Đã export **chính khối DiT thật** của GR00T-N1.7-3B (không phải model giả
lập) từ checkpoint đã tải, rút gọn còn 4/32 layer (đủ bao phủ cả 2 loại
block self-attn/cross-attn xen kẽ theo `interleave_self_attention=True`),
compile qua Qualcomm AI Hub, chạy trên đúng Hexagon NPU của board IQ-9075.

**Kết quả: 185/185 layer chạy trên NPU (100%, không rơi CPU), latency 6.18ms
cho 1 forward pass qua 4 layer.** Ước tính tuyến tính cho cả 32 layer × 4
bước denoise (`num_inference_timesteps=4` theo config.json) ≈ 25-50ms/action
chunk — nằm trong ngưỡng chấp nhận được cho replanning rate 1-10Hz.

Đây là tín hiệu tích cực bất ngờ cho GR00T trên IQ-9075 — rủi ro quantize
DiT nêu ở trên vẫn còn (đây là dry-run FP32/random-init, chưa quantize INT8,
chưa dùng checkpoint thật), nhưng ít nhất operator coverage là hoàn hảo,
không có op nào bị QNN từ chối/rơi về CPU.

**Lưu ý kỹ thuật quan trọng phát hiện được (áp dụng cho cả pi-0.5sau này):**
1. Input `timestep` kiểu `torch.long` (int64) → QNN compile fail với lỗi
   "Must use --truncate_64bit_io" — Hexagon không hỗ trợ int64 native, phải
   thêm flag `--truncate_64bit_io` vào compile options.
2. Khi export model >2GB, PyTorch tự tách weight thành nhiều file external
   data riêng lẻ (hàng trăm file) — `qai_hub` xử lý dạng này CỰC CHẬM (~20-
   24KB/s, không phải do mạng — đã đo băng thông thật ~20-28MB/s trên cả 2
   máy). Nếu model dưới 2GB, LUÔN xuất single-file .onnx (không ép
   `convert_model_to_external_data`) để tránh bug này.
3. Route mạng từ `huyhoang-4090` → AI Hub có vấn đề (băng thông chung tốt
   nhưng riêng path này rất chậm dù đã sửa single-file) — chạy job AI Hub
   từ board IQ-9075 (`ubuntu@192.168.1.226`, env `/mnt/ssd/vla/env.sh`) ổn
   định hơn nhiều. Nếu cần build/export trên máy khác, chuyển file qua rồi
   chạy compile job từ chính board IQ-9075.

Script: `/home/huyhoang/VLA/Gr00t-N1,7/export/01b_export_dit_small.py`,
`04_gr00t_dit_compile.py` (trên board, tại `/mnt/ssd/vla/smoke/`).

---

## Đính chính quan trọng: kiến trúc pi0/pi0.5 (2026-08-28)

Nhận định trước đó trong file này ("pi-0.5 dùng FAST-tokenizer autoregressive
decode, giống Qwen3-4B") là **sai**, xác nhận qua đọc trực tiếp code
`lerobot/src/lerobot/policies/pi05/modeling_pi05.py`: checkpoint `pi0_base`
đã tải (`type: pi0`, `paligemma_variant: gemma_2b`,
`action_expert_variant: gemma_300m`, `num_inference_steps: 10`) dùng
**flow-matching** (`sample_noise`, `denoise_step`, `x_t = time*noise +
(1-time)*actions`) — kiến trúc cùng họ với GR00T (VLM backbone + action
expert nhỏ chạy lặp lại qua nhiều bước denoise), KHÔNG phải autoregressive
token decode. FAST tokenizer là một hướng nghiên cứu khác của Physical
Intelligence, không phải cơ chế mặc định của pi0/pi0.5 base checkpoint.

Điều này không đổi kết luận "pi-0.5 khả thi" nhưng đổi *lý do* — không còn
là "giống LLM decode nên dễ" mà cần dry-run riêng để kiểm chứng, y như đã
làm với GR00T.

## Kết quả dry-run pi0/pi0.5 trên NPU thật (2026-08-28)

Export 3/18 layer thật của **action expert** (`gemma_300m`: hidden=1024,
heads=8, kv_heads=1 (GQA), head_dim=256 — block chạy lặp lại ở MỖI bước
denoise, 10 lần/action chunk), dùng đúng `GemmaDecoderLayer` chuẩn từ
`transformers` (pi0.5's `PiGemmaModel` kế thừa trực tiếp `GemmaModel`, không
override attention/MLP). Compile qua AI Hub, target IQ-9075.

**Kết quả: 187/187 layer chạy NPU (100%), 4.99ms cho 3 layer.**

So sánh ước tính (tuyến tính, cùng phương pháp với GR00T ở trên):

| Model | Block lặp lại | Layer thật | Latency đo | Ước tính full stack × số bước |
|---|---|---|---|---|
| GR00T N1.7 DiT | 32 layer × 4 bước denoise | 4 layer | 6.18ms | ~197ms (32L×4 bước) |
| pi0/pi0.5 action expert | 18 layer × 10 bước denoise | 3 layer | 4.99ms | ~300ms (18L×10 bước) |

Cả 2 đều 100% NPU, không op nào rơi CPU — kết luận Section "Kết luận" ở đầu
file (chọn pi-0.5 làm track chính) **cần xem lại**: GR00T thực ra có thể
nhanh hơn về phần action-head thuần túy nhờ ít bước denoise hơn (4 vs 10),
dù chưa tính phần VLM backbone (chạy 1 lần, chưa dry-run). Quyết định cuối
nên dựa vào dry-run VLM backbone của cả 2 (chưa làm) + độ chính xác sau
quantize, không chỉ tốc độ action-head.

Script: `/home/huyhoang/VLA/PI_0,5/export/01_export_gemma_layer.py`,
`05_pi05_gemma_compile.py` (trên board, `/mnt/ssd/vla/smoke/`).

---

## Kết quả quantize INT8 với TRỌNG SỐ THẬT (2026-08-28)

Load đúng 64 tensor trọng số thật (không random) từ checkpoint
GR00T-N1.7-3B cho 4 layer DiT đầu, quantize INT8 qua AI Hub
(`submit_quantize_job`, calibration = 1 sample), compile, profile, rồi
`submit_inference_job` để lấy output thật trên NPU và so với FP32 gốc.

**Tốc độ**: INT8 nhanh gấp ~2x FP32 (2.99ms vs 6.18ms), vẫn 100% layer
trên NPU (197/197).

**Độ chính xác — CẦN LƯU Ý**: cosine similarity FP32 vs INT8 chỉ đạt
**0.9556** (max relative diff 39.56%), dưới ngưỡng >0.99 thường chấp
nhận được. Đúng như plan ban đầu cảnh báo — quantize action head kiểu
DiT/flow-matching khó hơn LLM thường.

**Giới hạn của kết quả này (quan trọng, đừng kết luận vội)**:
calibration data chỉ dùng **1 sample ngẫu nhiên** (không phải dữ liệu
camera/robot thật, không đa dạng) — PTQ với 1 sample gần như chắc chắn
kém, bất kể kiến trúc. Cần lặp lại với: (a) calibration data thật (vài
chục-trăm frame camera + state thật của OpenArm), (b) thử kỹ thuật
scale-calibrated PTQ (QuantVLA) thay vì PTQ mặc định, trước khi kết
luận "GR00T DiT không quantize tốt được trên IQ-9075".

Script: `board_scripts/06_gr00t_quantize_real.py`,
`board_scripts/06b_gr00t_inference_only.py` (bản sửa lỗi dtype int64→int32
cho input `timestep` khi gọi inference sau compile với
`--truncate_64bit_io`).

---

## Camera driver thật — Stage 1 hoàn thành (2026-08-28)

Bối cảnh: 2 cổng Type-C camera của IQ-9075 bị cháy, nên camera thật (1 camera
thường + Orbbec Gemini RGB-D) cắm vào **Jetson NX** riêng, nối với IQ-9075
qua Ethernet trực tiếp (`end0`, subnet riêng `192.168.10.0/24` — IQ-9075 là
`.1`, Jetson là `.3`; Jetson KHÔNG có ROS2 cài sẵn).

**Kiến trúc đã dựng**: Jetson chạy script Python thuần (không cần ROS2) đọc
2 camera (`video0` = thường, `video6` = Orbbec RGB — xác định qua
`/sys/class/video4linux/videoN/name`), encode JPEG, POST liên tục qua HTTP
sang node ROS2 mới `camera_bridge` chạy trên IQ-9075, node này decode và
publish thành `sensor_msgs/Image` chuẩn.

**Đã xác nhận chạy thật, real-time**: `/camera_front/image_raw` (~2.7Hz),
`/camera_left/image_raw` — đúng tên topic convention đã dùng trong Isaac
Sim action graph, nên `vla_bridge` sau này đọc được cả sim lẫn robot thật
mà không cần đổi code.

**Lưu ý phần cứng đã gặp khi debug camera:**
- Camera "thường" (`video0`) lúc đầu đọc về toàn số 0 tuyệt đối
  (`min=max=mean=0`) dù mở được device — hoá ra không phải lỗi driver, chỉ
  cần: (a) ép định dạng `MJPG` thay vì để mặc định `YUYV`, và (b) bỏ qua
  ≥15-20 frame đầu để sensor ổn định (cảm biến rẻ tiền cần warm-up dài hơn
  bình thường).
- Ảnh "cháy sáng" (overexposed) gặp phải không phải lỗi — camera đang chĩa
  sát bề mặt bàn trắng thật.

**File trong repo:**
- `Open_arm_a1_ws/src/communication/camera_bridge/` — ROS2 package (ament_python),
  chạy trên IQ-9075: `ros2 run camera_bridge camera_bridge_node`
  (port HTTP nội bộ 8090, nhận `POST /frame/front`, `POST /frame/left`)
- `VLA/board_scripts/jetson_camera_streamer.py` — chạy trên Jetson
  (`python3 jetson_camera_streamer.py`), không cần ROS2

**Việc còn thiếu để dùng production**: chuyển Flask dev server sang WSGI
thật (hiện đang cảnh báo "development server"), thêm systemd service cho
cả 2 phía để tự khởi động lại khi mất kết nối/reboot, và tăng FPS nếu cần
(hiện dùng chung 1 vòng lặp cho 2 camera nên chỉ ~2.7Hz/camera thay vì 5Hz
cấu hình).

---

## Camera pipeline → systemd services (2026-08-28)

Cả 2 phía chạy qua `systemctl --user` (linger đã bật cho `ubuntu`@IQ-9075 và
`nx`@Jetson — không cần sudo, tự sống lại sau reboot/crash, không cần ai
SSH đăng nhập):

```bash
# IQ-9075 (camera_bridge_node, port 8090)
systemctl --user status|restart|stop camera-bridge.service
journalctl --user -u camera-bridge.service -f

# Jetson qua bastion (jetson_camera_streamer.py)
ssh -J ubuntu@192.168.1.226 nx@192.168.10.3
systemctl --user status|restart|stop camera-streamer.service
journalctl --user -u camera-streamer.service -f
```

Unit file: `~/.config/systemd/user/camera-bridge.service` (IQ-9075),
`~/.config/systemd/user/camera-streamer.service` (Jetson, script tại
`/home/nx/jetson_camera_streamer.py`).

**Lỗi đã gặp khi setup**: lần đầu bật service bị crash loop
("Address already in use" cổng 8090) vì process chạy tay từ trước chưa
kill hết — luôn `fuser -v 8090/tcp` kiểm tra trước khi bật service mới.

---

## Làm việc 100% local trên laptop (2026-09-02)

Bối cảnh: đang nghỉ lễ, IQ-9075 và huyhoang-4090 đều mất kết nối (netbird
peer offline — lỗi mạng vật lý phía văn phòng, không phải lỗi phía máy
laptop). Dựng lại toàn bộ pipeline export **hoàn toàn local**, không phụ
thuộc 2 máy trên, để tiếp tục làm việc tới khi quay lại văn phòng (3/9).

**Đã dựng xong**: venv riêng `VLA/.venv_local/` (Python 3.12, torch 2.9.0+cu128
— GPU CUDA thật của laptop, RTX 4050 Laptop GPU, nhận diện đúng), package
`gr00t` cài từ `github.com/NVIDIA/Isaac-GR00T` clone local, checkpoint
`GR00T-N1.7-3B` tải thẳng từ HuggingFace (6.5GB, ~5 phút).

**Kết quả xác nhận reproducibility**: chạy lại export DiT 4-layer với trọng
số thật (giống hệt cách làm trên huyhoang-4090) → `mean=-0.010799,
std=0.314286` — **khớp tuyệt đối** với số liệu đã đo trên huyhoang-4090
trước đó. Xác nhận setup local đúng 100%, checkpoint tải lại từ HF giống
hệt bản trên huyhoang-4090.

**Lưu ý cài đặt (nếu setup lại lần sau)**: `pip install -e .` đầy đủ của
gr00t bị treo rất lâu ở bước build metadata cho `deepspeed` (không có nvcc
trên máy) — giải pháp: cài `--no-deps` rồi bổ sung thủ công từng dep còn
thiếu theo lỗi ImportError hiện ra (pandas, scipy, dm-tree, albumentations,
torchvision==0.24.0) + pin đúng version `numpy==1.26.4 tyro==0.9.17
transformers==4.57.3 torch==2.9.0` như trong `Isaac-GR00T/pyproject.toml`
(cài phiên bản mới nhất gây lỗi `TypeError: non-default argument follows
default argument` trong dataclass config do tyro/transformers đổi API).

**Còn thiếu để nộp AI Hub từ laptop**: `qai_hub` chưa có API token trên máy
này (lấy qua kênh riêng trước đây từ IQ-9075, giờ offline). Cần chạy:
```bash
/home/hans/universal_bot/VLA/.venv_local/bin/qai-hub configure --api_token <token>
```
(token lấy tại aihub.qualcomm.com/settings — không cần IQ-9075).

Script: `VLA/export/gr00t/05_export_real_weights_local.py` (bản local của
`05_export_real_weights.py`, đổi đường dẫn checkpoint sang
`VLA/weights_local/gr00t_n1.7_3b`).

---

## Vòng lặp ảnh → model → action đầu tiên chạy được (2026-09-02, quay lại VP)

Server `serve_pi05_http.py` trên `huyhoang-4090` (port 9091, đã sửa lỗi
remap prefix `model.` từ trước) **sống sót suốt kỳ nghỉ** (chạy liên tục từ
28/8), và sau khi remap đúng: `missing=1 unexpected=0` (gần hoàn hảo, so với
778 missing lúc chưa remap).

**Test end-to-end đầu tiên**: gửi 1 ảnh mẫu (`open_loop_eval_so100.jpg` có
sẵn trong repo Isaac-GR00T, KHÔNG phải camera OpenArm thật) qua
`POST /predict` → nhận về **action vector 32 chiều thật**, inference
5.77 giây trên CPU. Xác nhận: model load đúng trọng số thật, forward pass
chạy hết pipeline preprocessor → PI05Pytorch.select_action → postprocessor
không lỗi.

**Lưu ý quan trọng — CHƯA dùng được cho robot thật**:
1. Ảnh test là ảnh mẫu ngẫu nhiên, không phải camera OpenArm → action
   không có ý nghĩa vật lý thật.
2. Checkpoint `pi0_base` vẫn là bản GỐC (Franka, action space 32-D
   EE-delta-kiểu), chưa fine-tune trên OpenArm → action space không khớp
   14 DoF joint-space thật của tay. TUYỆT ĐỐI không actuate tay thật với
   action này (đúng cảnh báo cũ trong `vla0_api.md`).
3. 5.77s/inference (CPU) quá chậm cho control loop thật — chỉ dùng để xác
   nhận pipeline đúng, chưa phải benchmark tốc độ.

Bước hợp lý tiếp theo: nối input thật từ `/camera_front/image_raw` (ROS2,
đã publish thật từ Jetson) thay cho ảnh mẫu, để có test đầu-cuối với dữ
liệu camera OpenArm thật (vẫn chỉ log action ra, chưa actuate).

---

## Quantize INT8 pi0.5 Gemma action-expert với TRỌNG SỐ THẬT (2026-09-03)

Bối cảnh: `huyhoang-4090` bị quá tải nặng (load 20-59) suốt buổi, làm cả
việc export nhẹ (chỉ đọc vài chục tensor) cũng gần như đứng hình sau 10+
phút. Chuyển hẳn sang **laptop local** (đã dựng sẵn từ 2/9): tải checkpoint
`pi0_base` (14GB) qua mạng LAN văn phòng (nhanh, ~10-30MB/s dao động do
disk I/O nguồn vẫn bị ảnh hưởng), export 3 layer đầu Gemma action-expert
với TRỌNG SỐ THẬT (27 tensor, khớp 100% — `missing=[]`), quantize+compile+
profile+inference qua AI Hub chạy từ board IQ-9075 (theo đúng lưu ý routing
mạng đã ghi ở trên).

**Tốc độ**: INT8 nhanh gấp ~3.2x FP32 (1.565ms vs 4.99ms đo trước đó cho
cùng 3 layer), 100% layer trên NPU (224/224).

**Độ chính xác**: cosine similarity FP32 vs INT8 = **0.9167** (max relative
diff 42.39%) — **THẤP HƠN** kết quả GR00T DiT đã đo trước (0.9556). Cả 2
model đều dùng chung 1 điểm yếu: calibration chỉ 1 sample tổng hợp (không
phải activation thật từ camera).

**Kết luận quan trọng**: qua cả 2 model (GR00T DiT: 0.9556, pi0.5 Gemma:
0.9167), pattern rõ ràng là **operator coverage KHÔNG phải vấn đề** (cả 2
đều 100% NPU), mà **độ chính xác sau PTQ với calibration yếu mới là nút
thắt thật sự** — đúng như QuantVLA paper đã chỉ ra cho toàn bộ họ model
VLA (flow-matching/diffusion action head). Ưu tiên tiếp theo nên là cải
thiện calibration data (nhiều sample hơn, tốt nhất là activation thật từ
camera OpenArm) trước khi so sánh model nào "quantize tốt hơn".

Script: `VLA/PI_0,5/export/02_export_gemma_layer_real_local.py` (export
local, không phụ thuộc server), `VLA/board_scripts/07_pi05_gemma_quantize_real.py`
(chạy trên IQ-9075).

---

## Quantize INT8 pi0.5 với calibration data THẬT từ camera OpenArm (2026-09-03)

Theo đúng kết luận ở trên, thu calibration data thật thay vì 1 sample
tổng hợp:

1. Bật hook capture trong `serve_pi05_http.py` (`/calib/start`), chạy
   `pi05_camera_test_node` trên IQ-9075 với camera OpenArm thật ~22 lần
   gọi `/predict` (num_inference_steps=10 → 10 sample/lần) → **220 sample
   hidden_states thật**, lưu qua `/calib/save` →
   `real_calib_hidden_states.npy`, shape `(220, 51, 1024)`.
2. **Phát hiện lệch shape**: ONNX cũ (`07_pi05_gemma_quantize_real.py`)
   export với seq_len=64 (giả lập), nhưng hidden_states thật có
   seq_len=51 → không dùng trực tiếp làm calibration được (QNN cần shape
   cố định khớp graph). Export lại ONNX dùng chính 1 sample thật (không
   phải `torch.randn`) làm input trace, seq_len lấy từ shape data thật
   (=51) — script mới: `VLA/PI_0,5/export/03_export_gemma_layer_real_calib.py`.
3. Quantize với đủ 220 sample thật (`calibration_data` truyền list 220
   phần tử thay vì 1) — script: `VLA/board_scripts/08_pi05_gemma_quantize_realcalib.py`.

**Kết quả trên board IQ-9075 thật**:
- INT8 latency: 1930us (so với 1565us của bản calib=1-sample — chênh lệch
  nhỏ, không đáng kể so với lợi ích độ chính xác)
- 224/224 layer vẫn 100% NPU (không đổi so với trước)
- **cosine similarity: 0.9753** (so với 0.9167 của calib=1 sample tổng
  hợp — tăng đáng kể, vượt cả kết quả GR00T DiT 0.9556 đo trước đó)
- max abs diff: 20.11, max relative diff: 26.04% (giảm từ 42.39%)

**Kết luận**: xác nhận đúng giả thuyết — calibration data thật (không
phải random) là yếu tố quyết định độ chính xác PTQ, không phải kiến trúc
model. Việc re-export ONNX theo đúng shape thật (thay vì shape giả định
ban đầu) là bước bắt buộc, dễ bị bỏ sót khi chuyển từ dry-run sang data
thật.

**Việc còn để ngỏ**: chưa áp dụng lại kỹ thuật này cho GR00T DiT (vẫn
đang ở kết quả calib=1-sample, 0.9556) — nếu muốn so sánh công bằng giữa
2 model, cần thu calibration thật tương tự cho GR00T rồi quantize lại.

---

## Fine-tune pi0.5 CPU-local: smoke-test pipeline (2026-09-04)

Bối cảnh: GPU huyhoang-4090 tiếp tục bị chiếm dụng nặng (21.6/24.5GB dùng,
util 95%, job ASR + video-filter của user khác) — không đủ chỗ train.
GPU laptop (RTX 4050, 6GB VRAM) cũng không đủ (model bf16 cần ~6.6GB chỉ
riêng trọng số). Quyết định: train **CPU-local trên laptop** làm
**smoke-test pipeline** (KHÔNG kỳ vọng ra policy dùng được thật).

**Phát hiện quan trọng**: dataset `openarm_fixture_lerobot_v2.1` hiện chỉ
có **1 episode** — dù train ở đâu, đây chỉ đủ để verify pipeline
(fine-tune → checkpoint → export → quantize thông suốt), không đủ dữ liệu
cho policy thật. Cần thu thêm episode thật trước khi fine-tune "thật".

**Setup**: đồng bộ source `lerobot` fork đã sửa (từ
`/data21tb/users/huyhoang/lerobot` trên server, có patch tương thích
transformers mới cho `modeling_pi0.py`/`pi_gemma.py`/`pretrained.py`) về
laptop qua rsync (90MB, loại `.git`), cài `pip install -e . --no-deps`
vào `.venv_local`, rồi cài thêm từng dependency thiếu (đối chiếu
`pip freeze` của conda env `lerobot` trên server để khớp đúng version,
tránh dò từng lỗi): `termcolor`, `draccus==0.10.0`, `gymnasium`,
`jsonlines`, `datasets`, `pyarrow`, `typing_inspect`, `mypy_extensions`,
`mergedeep`, `av==15.1.0` (bản mới nhất 18.1.0 KHÔNG có `av.option`, phải
pin đúng khoảng `av-dep` trong pyproject.toml), `pyyaml-include==1.4.1`
(bản mới `yaml_include` đổi tên module — draccus cần đúng tên cũ
`yamlinclude`), `accelerate==1.14.0`.

**Lệnh chạy**: script mới `VLA/train_pi05_cpu_local.sh [steps]` — biến
thể CPU của `train_pi05.sh`, thêm `--policy.dtype=bfloat16` (bắt buộc,
vì RAM laptop chỉ 14GB/~5.9GB khả dụng, fp32 model ~13GB sẽ swap nặng;
bf16 giảm còn ~6.6GB), `--policy.use_amp=false` (AMP chỉ có ý nghĩa trên
CUDA), giữ `--policy.train_expert_only=true` + `--policy.freeze_vision_encoder=true`
+ `--policy.gradient_checkpointing=true` + `--batch_size=1` như bản GPU
gốc, `--log_freq=1` để thấy từng step (theo quy tắc luôn cho training
visibility).

**Kết quả**: 20/20 step thành công, ~22-48s/step (chậm dần lúc đầu do
warmup, ổn định quanh ~22s/step), tổng 8m44s. Log xác nhận
`Remapped 777 state dict keys` + `All keys loaded successfully!` — pipeline
train chính thức của lerobot fork **tự động xử lý đúng** việc remap
prefix "model." (khác với script `serve_pi05_http.py` tự viết, phải vá
thủ công bug này — xem phần trước). Checkpoint lưu tại
`VLA/outputs/pi0_cpu_smoketest_20260904_085551/checkpoints/000020/pretrained_model/`
(8.9GB, bf16).

**Kết luận**: pipeline fine-tune hoạt động đúng end-to-end trên CPU không
cần GPU — hữu ích làm đường dự phòng khi cả 4090 lẫn GPU laptop đều
không khả dụng. Bước tiếp theo hợp lý: thu thêm episode thật từ OpenArm
(dataset hiện chỉ 1 episode) trước khi chạy fine-tune "thật" (nhiều step
hơn, nhiều episode hơn) — có thể chạy CPU-local qua đêm nếu GPU vẫn bận,
hoặc dùng script `08_pi05_gemma_quantize_realcalib.py`-style để
export/quantize checkpoint fine-tune này (dù chỉ 1 episode) làm bước
kiểm tra thêm nếu muốn.

---

## Teleop WASD + ghi dataset thật (2026-09-04)

Để giải quyết vấn đề "dataset chỉ 1 episode", xây dựng công cụ thu demo
thật qua teleop bàn phím thay vì cần leader-arm/joystick riêng.

**Quyết định kiến trúc quan trọng**: `lerobot` upstream đã có sẵn class
`bi_openarm_follower` điều khiển OpenArm **trực tiếp qua CAN bus**
(socketcan, CAN FD 1Mbps/5Mbps) — nhưng KHÔNG dùng, vì nó bỏ qua toàn bộ
kiểm tra va chạm + giới hạn khớp của MoveIt đang chạy production. Thay
vào đó, script mới gọi `/api/move/pose` HTTP API đã có sẵn
(`moveit_api/robot_api_server.py`, port 5050 mặc định) — giữ nguyên
đường an toàn đã kiểm chứng, chỉ mượn **định dạng ghi dữ liệu**
(`LeRobotDataset.add_frame`/`save_episode`/`resume`) của lerobot để dataset
tương thích thẳng với fine-tune, không cần qua bước convert riêng của
`openarm_dataset`.

**2 thay đổi**:
1. `camera_bridge_node.py` — thêm cache frame mới nhất + endpoint
   `GET /frame/<slot>/latest` trả JPEG thô (dùng để script teleop lấy ảnh
   mà không cần tự làm ROS2 node).
2. `VLA/teleop_record_openarm.py` (mới, chạy trên laptop bằng
   `.venv_local`) — đọc phím raw-terminal (`tty`/`termios`/`select`,
   không cần `pynput`/X server, chạy tốt qua SSH thuần text). Bàn phím:
   WASD = x/y/z tay đang chọn, R/F = z+/z-, Tab = đổi tay, Space =
   đóng/mở gripper, N/M/X = bắt đầu/lưu/huỷ episode, `[`/`]` = giảm/tăng
   bước di chuyển. Mỗi bước di chuyển thành công = 1 frame ghi
   (camera front+left, joint state 16-dim, action = joint state sau khi
   di chuyển).

**Lưu ý độ mượt**: `/api/move/pose` dùng MoveGroup lập kế hoạch đầy đủ
(OMPL) mỗi lần gọi, không phải velocity-streaming — mỗi bước WASD là
"nhích rồi dừng", không mượt như teleop chuyên dụng (`moveit_servo`),
nhưng đủ để thu demo.

**Bug phát hiện khi test thật (2026-09-04)**: `get_ee_pose()`/`move_to_pose()`
trong `moveit_ee_controller.py` dùng `frame_id='world'` cho FK/IK, nhưng
kiểm tra `/tf_static` sống trên board thì **không có frame `world`** —
gốc cây TF thực tế là `openarm_base_link`. SRDF có khai báo virtual joint
`world -> openarm_base_link` (`openarm_bimanual.srdf:412`) nhưng KHÔNG có
node nào publish transform này thật (grep hết `bringup.launch.py`,
`robot_api.launch.py`, `sequence_executor.launch.py` — không file nào có
`static_transform_publisher`). Lỗ hổng có sẵn từ trước, chỉ lộ ra bây giờ
vì mọi thứ trước đó (QVIC sequence, waypoint_recorder) dùng joint-space
control (`JointConstraint`, không phụ thuộc frame) — teleop là lần đầu
dùng Cartesian EE pose control thật trên hardware này.

Gỡ tạm (khong sua launch file, chi chay 1 lenh o terminal rieng, khong
dung hardware/controller):
```bash
ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 --frame-id world --child-frame-id openarm_base_link
```
Fix vinh vien (con no): them static_transform_publisher nay vao
`bringup.launch.py` (hoac `sequence_executor.launch.py`) de khong phai
chay tay moi lan bringup.

**Bug thu 2, nghiem trong hon**: du da them static_transform_publisher va
xac nhan `world` co trong `/tf_static`, goi `/compute_fk` truc tiep bang
`ros2 service call` (khong qua code nao cua minh) van **treo vo thoi han**
(khong loi, khong response, timeout 8-20s+). Doi chieu voi log move_group:
khi TF loi ("Could not find a connection... two or more unconnected
trees"), move_group log ERROR nhung **khong bao gio goi response callback**
- day la exception khong duoc catch dung trong move_group's FK capability,
khien client cho vinh vien thay vi nhan loi ro rang. Kiem tra ky: khong
phai do script/quoting cua minh (an corrected test voi file YAML rieng,
van treo y het).

**Quyet dinh**: thay vi tiep tuc debug move_group/TF (khong biet ton bao
lau, thuoc ve ha tang MoveIt chu khong phai code du an), **chuyen teleop
tu Cartesian XYZ (`/api/move/pose`, can FK/IK/world) sang JOINT-SPACE
(`/api/move/joint`, chi can `JointConstraint`, KHONG can TF/world/FK)** -
dung dung con duong da chung minh on dinh trong production (QVIC sequence,
waypoint_recorder deu dung joint-space). Doi phim dieu khien: `1-7` chon
khop cua tay active, `W/S` tang/giam khop do (mac dinh 2 do/lan) thay vi
WASD/RF cho x/y/z.

`moveit_servo` (co san keyboard teleop chuyen dung cho Cartesian jogging,
khong qua compute_fk nen co the tranh duoc bug tren) **chua duoc cai** tren
board (khong co trong apt/ros2 pkg list) - de danh nang cap sau neu can
jogging muot hon, khong lam ngay vi can setup config/launch rieng.

**Xac nhan hoat dong tren phan cung that (2026-09-04)**: teleop joint-space
chay thanh cong - bam `1` (chon joint1 tay trai) roi `W` (tang 2 do) ->
tay trai joint1 di chuyen dung 2.0 do nhu lenh.

**Bug thu 3 phat hien khi test that**: sau ~30 lan bam W/S chi chinh
joint1/2/3, cac khop KHONG dung toi (joint4/5/6/7) troi vai do (vd
joint4=3.4deg, joint6=-2.7deg, joint7=5.61deg) du khong he bam phim nao
lien quan. Nguyen nhan: `/api/move/joint` (`move_single_joint()` trong
`moveit_ee_controller.py`) moi lan goi gui lai CA 7 khop, 6 khop "giu
nguyen" dung GIA TRI DO DUOC tai thoi diem goi (dung sai long ~2.9 do/0.05
rad) - qua nhieu lan goi, nhieu/backlash co khi tich luy thanh troi that.
Day la hanh vi co san cua API (dung cho UI slider tung khop, khong thiet
ke cho goi lien tuc hang chuc lan/phut nhu teleop).

**Fix trong teleop_record_openarm.py**: doi sang goi `/api/move/joints`
(ca nhom 1 lan, dung sai chat 0.001 rad cho MOI khop) voi vector 7 gia
tri TU THEO DOI trong script (khoi tao tu `/api/status` luc dau, chi cap
nhat khi thuc su di chuyen) thay vi doc lai gia tri do duoc moi lan - loai
bo hoan toan co che tich luy troi.

**CANH BAO CHO NGUOI DUNG**: neu da chay ban cu (goi `/api/move/joint`)
truoc khi fix, tay THAT co the da troi khoi vi tri ky vong vai do o
joint4/6/7 - nen kiem tra lai `ros2 run robot_control get_robot_state`
hoac ve home pose truoc khi tiep tuc thu demo, dac biet chu y joint1 tay
trai co gioi han co khi that ~-90 den +30 do (xem memory
project_iq9075_hardware_state).

**Chưa test xong trên phần cứng thật** — cần bạn tự chạy (theo quy tắc
không tự SSH vào IQ-9075). Cách chạy:
```bash
# Tren IQ-9075 (ban tu chay): dam bao robot_api_server.py va
# camera_bridge_node.py (ban moi, can rebuild) dang chay.
colcon build --packages-select camera_bridge && source install/setup.bash
ros2 run camera_bridge camera_bridge_node

# Tren laptop:
cd VLA && .venv_local/bin/python teleop_record_openarm.py \
    --api-host 192.168.1.226 --camera-host 192.168.1.226 \
    --task "pick up the object" --dataset-root datasets/openarm_teleop_v1
```
