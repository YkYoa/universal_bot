# Hướng dẫn đầy đủ: Deploy VLA model (pi-0.5 / GR00T) lên IQ-9075

Tài liệu này giải thích **từ kiến thức nền tới từng bước thực tế** đã làm
trong dự án — dành cho người chưa quen với toolchain Qualcomm/NPU, kể cả
khi bạn đã quen PyTorch/CUDA thông thường. Đọc theo thứ tự, mỗi phần đều
có ví dụ lệnh thật đã chạy trong repo này.

---

## Phần 0 — Vì sao việc này không đơn giản như "copy file model sang"

Khi bạn train một model bằng PyTorch trên GPU NVIDIA, model đó là **code
Python + trọng số (weights)**. Để chạy, máy cần: Python, PyTorch, CUDA
driver, GPU NVIDIA. Bạn có thể chạy `python inference.py` trực tiếp.

IQ-9075 **không có GPU NVIDIA**. Nó có 3 loại chip xử lý:

| Chip | Vai trò | Tốc độ AI |
|---|---|---|
| **Kryo CPU** (8 nhân) | Chạy code thường, logic, I/O | Chậm nhất cho AI |
| **Adreno GPU** | Đồ hoạ, có thể tăng tốc 1 phần AI qua OpenCL | Trung bình |
| **Hexagon NPU** | Chip chuyên dụng CHỈ để chạy mạng neural | Nhanh nhất — 100 TOPS |

Hexagon NPU nhanh gấp hàng chục lần CPU/GPU cho việc suy luận (inference)
model AI, nhưng đổi lại nó **không chạy code Python được**. Nó chỉ chạy
được một "graph" đã biên dịch sẵn thành mã máy riêng cho chip này — giống
như CPU x86 không chạy được file `.exe` biên dịch cho ARM.

Vậy nên toàn bộ quy trình là một **dây chuyền chuyển đổi**:

```
PyTorch model (.pth/.safetensors)
        │  (torch.onnx.export)
        ▼
ONNX (định dạng trung gian, giống "bytecode" cho AI model)
        │  (Qualcomm AI Hub biên dịch)
        ▼
QNN context binary (.bin — mã máy cho riêng Hexagon NPU)
        │  (nạp vào board qua qai_appbuilder)
        ▼
Chạy trên NPU thật
```

Mỗi mũi tên ở trên là **một bước riêng, có thể lỗi**, và mục tiêu tài liệu
này là giải thích rõ từng bước + lỗi thường gặp.

---

## Phần 1 — Tải model (Download)

### 1.1. Model ở đâu, tải bằng gì

Các model foundation cho robot (pi-0.5, GR00T) được host trên
**HuggingFace Hub** — giống GitHub nhưng cho model AI thay vì code. Công
cụ chính thức để tải là package `huggingface_hub`, dùng qua lệnh `hf`:

```bash
# Lệnh CŨ (đã deprecated, đừng dùng): huggingface-cli download ...
# Lệnh ĐÚNG hiện tại:
hf download nvidia/GR00T-N1.7-3B --local-dir /path/to/save/gr00t_n1.7_3b
```

`nvidia/GR00T-N1.7-3B` là "repo ID" — dạng `<tổ chức>/<tên model>`. Bạn
tìm repo ID này trên trang HuggingFace của model, ví dụ
`huggingface.co/nvidia/GR00T-N1.7-3B`.

### 1.2. Vì sao chọn model nào quan trọng

Trong dự án này chúng ta có 2 lựa chọn, mỗi cái có **license** khác nhau:

- `nvidia/GR00T-N1.5-3B` — chỉ dùng phi thương mại (non-commercial).
- `nvidia/GR00T-N1.7-3B` — license thoáng hơn (NVIDIA Open Model
  License), dữ liệu train mới hơn.

→ **Luôn đọc phần License trên trang HuggingFace của model trước khi
tải**, đặc biệt nếu sản phẩm cuối có mục đích thương mại.

### 1.3. Model nặng bao nhiêu, để ở đâu

Model 3 tỷ tham số (3B), lưu ở định dạng BF16 (16-bit thay vì 32-bit) →
khoảng 6GB. Nếu lưu FP32 (32-bit, chuẩn/mặc định của nhiều pipeline huấn
luyện) → khoảng 12GB.

**Quan trọng: kiểm tra dung lượng đĩa trước khi tải.** Trong dự án này,
ổ hệ thống (`/`) của board IQ-9075 chỉ còn 24GB trống — quá ít cho vài
model 6-14GB. Chúng ta dùng ổ phụ `/mnt/ssd` (88GB trống) cho board, và
`/data21tb` (4.6TB trống) cho server train. Lệnh kiểm tra:

```bash
df -h   # xem dung lượng trống của từng ổ đĩa
```

### 1.4. Vì sao cần cài thêm "package kiến trúc" ngoài file weight

File weight (`.safetensors`) chỉ chứa **con số** (trọng số đã học), không
chứa **code định nghĩa kiến trúc mạng** (bao nhiêu layer, layer nào nối
layer nào...). Bạn cần code kiến trúc gốc để load đúng weight vào đúng
chỗ. Với GR00T, đó là package `gr00t` từ repo GitHub chính chủ:

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
pip install -e .   # cài package "gr00t", định nghĩa class DiT, Gr00tN1d7ActionHead...
```

Với pi-0.5, code kiến trúc nằm trong package `lerobot`
(`lerobot.policies.pi05.modeling_pi05`).

---

## Phần 2 — Test / xác nhận model chạy đúng (trước khi đụng NPU)

**Nguyên tắc vàng: đúng trước, nhanh sau (correctness before speed).**
Đừng vội đưa model lên NPU nếu chưa chắc nó chạy đúng bằng PyTorch bình
thường trên CPU/GPU. Việc test:

```python
import torch
from gr00t.model.modules.dit import DiT
import json

with open("path/to/config.json") as f:
    cfg = json.load(f)

model = DiT(**cfg["diffusion_model_cfg"], cross_attention_dim=cfg["backbone_embedding_dim"])
model.eval()  # QUAN TRỌNG: tắt dropout/batchnorm training-mode

# Tạo input giả (đúng shape thật) để chạy thử
dummy_input = torch.randn(1, 40, 1536)   # (batch, action_horizon, inner_dim)
with torch.no_grad():                    # không cần tính gradient khi suy luận
    output = model(dummy_input, ...)
print(output.shape)   # kiểm tra shape đầu ra đúng kỳ vọng
```

Ở bước này bạn cũng nên **nạp trọng số THẬT** (không phải random-init) từ
file `.safetensors` để test đúng model đã huấn luyện:

```python
from safetensors import safe_open

with safe_open("model.safetensors", framework="pt") as f:
    tensor = f.get_tensor("action_head.model.transformer_blocks.0.attn1.to_k.weight")
```

Model lớn thường chia thành nhiều "shard" (file `model-00001-of-00002.safetensors`,
`model-00002-of-00002.safetensors`...) + 1 file `model.safetensors.index.json`
ghi rõ tensor nào nằm trong shard nào — đọc file index này trước để biết
cần mở shard nào.

---

## Phần 3 — Export sang ONNX

### 3.1. ONNX là gì

**ONNX** (Open Neural Network Exchange) là định dạng file mô tả một mạng
neural dưới dạng **đồ thị tính toán** (computational graph) — danh sách
các phép toán (MatMul, Add, Softmax...) và cách chúng nối với nhau, cộng
với các tensor trọng số. Nó độc lập với framework — export từ PyTorch,
có thể chạy bằng ONNX Runtime, TensorRT, hoặc (trường hợp của chúng ta)
biên dịch tiếp bằng QNN.

### 3.2. Lệnh export cơ bản

```python
torch.onnx.export(
    model,                          # model PyTorch (đã .eval())
    (dummy_input1, dummy_input2),   # tuple các input mẫu — QUYẾT ĐỊNH shape cố định
    "model.onnx",                   # đường dẫn file ra
    input_names=["x", "y"],         # đặt tên input (để dễ debug/gọi sau)
    output_names=["out"],
    opset_version=17,               # phiên bản "từ điển toán tử" ONNX — 17 là ổn định, tương thích rộng
    dynamo=False,                   # dùng exporter cũ (ổn định hơn cho QNN converter tại thời điểm này)
)
```

### 3.3. Ba cạm bẫy thật đã gặp trong dự án này

**Cạm bẫy 1 — Shape phải CỐ ĐỊNH.** Khác với PyTorch (chạy được input
shape bất kỳ), NPU cần biết trước chính xác kích thước mọi tensor. Nếu
bạn định dùng batch=1, độ dài chuỗi=64 — dummy input PHẢI đúng batch=1,
độ dài=64. Không dùng `dynamic_axes`.

**Cạm bẫy 2 — Input kiểu int64 làm QNN từ chối biên dịch.** Hexagon NPU
không hỗ trợ số nguyên 64-bit gốc. Nếu model có input như `timestep`
hay `position_ids` kiểu `torch.long` (= int64), bước biên dịch QNN sẽ báo
lỗi:
```
Must use --truncate_64bit_io when input tensors have type int64.
```
→ Cách sửa: thêm flag `--truncate_64bit_io` khi biên dịch (Phần 4).

**Cạm bẫy 3 — Model lớn hơn 2GB bị tách thành nhiều file, và điều này
làm bước upload lên AI Hub chậm khủng khiếp.** Định dạng protobuf mà ONNX
dùng có giới hạn cứng 2GB/file. Nếu model bạn export lớn hơn 2GB
(fp32, vài trăm triệu tham số trở lên), PyTorch **tự động** tách trọng số
ra thành file phụ riêng (`model.onnx` nhỏ chỉ chứa graph + `model.onnx.data`
chứa số, hoặc tệ hơn — hàng trăm file nhỏ, mỗi file 1 tensor). Chúng tôi đã
đo thực tế: upload dạng nhiều-file này chỉ đạt **~20KB/s** (chậm gấp
~1000 lần bình thường, dù mạng thật đo được 20-28MB/s) — mất hàng chục
giờ cho vài trăm MB.

→ **Cách né hoàn toàn:** giữ mỗi lần export **dưới 2GB** để nó tự động ở
dạng 1-file-duy-nhất (không external data). Với model lớn hơn nhiều (như
GR00T DiT 32 layer đầy đủ ~4.3GB), chỉ export **một lát cắt đại diện**
(vài layer, vì tất cả layer dùng chung loại toán tử lặp lại) thay vì toàn
bộ — điều này vẫn cho kết quả operator-coverage chính xác 100% cho toàn
bộ model, chỉ khác số lần lặp.

---

## Phần 4 — Biên dịch qua Qualcomm AI Hub (bước "dịch ONNX → mã máy NPU")

### 4.1. Qualcomm AI Hub là gì

**Qualcomm AI Hub** (`aihub.qualcomm.com`) là dịch vụ cloud của Qualcomm:
bạn nộp model ONNX lên, chọn **thiết bị đích** (ví dụ chính xác
`Dragonwing IQ-9075 EVK`), họ biên dịch bằng bộ công cụ QNN của họ, và —
quan trọng — **cho chạy thử luôn trên phần cứng thật trong lab của họ**
để đo latency, mà bạn không cần sở hữu board đó.

### 4.2. Cài đặt & xác thực

```bash
pip install qai-hub
```

Cần API token (lấy từ tài khoản aihub.qualcomm.com), lưu tại
`~/.qai_hub/client.ini`. Token này **là bí mật** — không copy qua kênh
không an toàn, không paste vào chat/log.

### 4.3. Ba bước: compile → profile → (tùy) inference

```python
import qai_hub as hub

DEVICE = hub.Device("Dragonwing IQ-9075 EVK")  # tên thiết bị CHÍNH XÁC, lấy từ hub.get_devices()

# Bước 1: biên dịch ONNX -> QNN context binary
compile_job = hub.submit_compile_job(
    model="model.onnx",
    device=DEVICE,
    options="--target_runtime qnn_context_binary --truncate_64bit_io",
)
target_model = compile_job.get_target_model()   # đây là model đã biên dịch, sẵn sàng chạy NPU

# Bước 2: đo hiệu năng THẬT trên phần cứng IQ-9075 trong lab Qualcomm
profile_job = hub.submit_profile_job(model=target_model, device=DEVICE)
profile = profile_job.download_profile()
print(profile["execution_summary"]["estimated_inference_time"])  # latency (microsecond)

# Đếm bao nhiêu layer chạy trên NPU vs rơi về CPU (chỉ số quan trọng nhất!)
detail = profile["execution_detail"]
from collections import Counter
print(Counter(l["compute_unit"] for l in detail))
# {'NPU': 185} => TỐT, 100% chạy NPU
# {'NPU': 150, 'CPU': 35} => một số op không được NPU hỗ trợ, rơi về CPU (chậm hơn)

# Bước 3 (tuỳ chọn): chạy inference thật với input cụ thể, lấy output để so sánh độ chính xác
infer_job = hub.submit_inference_job(model=target_model, device=DEVICE, inputs={"x": [my_numpy_array]})
output = infer_job.download_output_data()
```

`options="--target_runtime qnn_context_binary"` nghĩa là: biên dịch ra
định dạng "QNN context binary" — file `.bin` nạp thẳng vào Hexagon NPU
runtime. Có các target khác (ví dụ TFLite) nhưng với IQ-9075/Hexagon,
đây là lựa chọn đúng để đạt tốc độ NPU tối đa.

### 4.4. Đọc kết quả profile như thế nào

Con số quan trọng nhất **không phải** latency, mà là **tỷ lệ layer chạy
trên NPU**. Nếu 100% layer ở "NPU" — graph của bạn hoàn toàn tương thích.
Nếu có layer rơi về "CPU" — nghĩa là có toán tử (op) mà Hexagon compiler
không hỗ trợ, model sẽ chạy CHẬM HƠN NHIỀU vì phải chuyển dữ liệu qua lại
giữa NPU và CPU liên tục. Khi gặp trường hợp này, cách xử lý thường là:
đổi cách viết layer đó (dùng op khác tương đương), hoặc quantize (đôi khi
op chỉ được NPU hỗ trợ ở dạng INT8/FP16, không hỗ trợ FP32).

---

## Phần 5 — Quantize (nén số để chạy nhanh hơn)

### 5.1. Quantize là gì, tại sao cần

Model huấn luyện thường ở FP32 (số thực 32-bit) hoặc BF16 (16-bit).
**Quantize** là chuyển trọng số + phép tính sang số nguyên ít bit hơn,
thường là **INT8** (8-bit). Hexagon NPU đạt tốc độ tối đa (gần 100 TOPS)
CHỈ KHI chạy INT8 — FP32 chạy được (đã chứng minh ở dry-run) nhưng chậm
hơn nhiều so với tiềm năng thật của chip.

Đánh đổi: INT8 có ít "độ phân giải" số hơn FP32 → model có thể **mất
chút độ chính xác**. Việc quantize làm đúng cách sẽ giữ độ chính xác gần
như FP32; làm ẩu có thể làm model output sai lệch đáng kể.

### 5.2. Hai kiểu quantize

- **PTQ (Post-Training Quantization)** — quantize SAU khi model đã huấn
  luyện xong, không cần train lại, chỉ cần một ít dữ liệu mẫu để "hiệu
  chỉnh" (calibration). Nhanh, đơn giản, thường dùng đầu tiên. Đây là
  loại chúng ta dùng qua AI Hub.
- **QAT (Quantization-Aware Training)** — huấn luyện lại model với
  "nhận biết" trước việc sẽ bị quantize, cho độ chính xác tốt hơn PTQ
  nhưng tốn công (cần lại GPU, dữ liệu train, thời gian). Chỉ cần khi
  PTQ không đạt độ chính xác chấp nhận được.

### 5.3. Calibration data là gì

Để quantize, thuật toán cần biết **khoảng giá trị thực tế** mà mỗi tensor
trong model sẽ nhận (ví dụ: activation của layer này thường nằm trong
khoảng -3.0 đến 4.5) — từ đó chọn cách ánh xạ FP32→INT8 tối ưu. Dữ liệu
dùng để "quan sát" khoảng giá trị này gọi là **calibration data** — vài
chục đến vài trăm mẫu input THẬT (không phải random) là đủ, càng giống
dữ liệu thật lúc suy luận sản xuất càng tốt.

```python
import qai_hub as hub

quantize_job = hub.submit_quantize_job(
    model="model.onnx",
    calibration_data={"hidden_states": [sample1, sample2, ...]},  # vài chục sample thật
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT8,
)
quantized_model = quantize_job.get_target_model()

# Biên dịch model ĐÃ quantize (giống bước 4.3, nhưng model giờ nhẹ hơn ~4 lần)
compile_job = hub.submit_compile_job(model=quantized_model, device=DEVICE, ...)
```

### 5.4. Kiểm tra độ chính xác sau quantize — bước KHÔNG ĐƯỢC BỎ QUA

Sau quantize, **luôn so sánh output model đã quantize với output model
FP32 gốc trên cùng 1 input**, trước khi tin tưởng đưa vào robot thật:

```python
import numpy as np

fp32_output = ...      # đã lưu sẵn từ lúc test PyTorch (Phần 2)
quantized_output = ...  # lấy từ submit_inference_job trên model đã quantize

cos_sim = np.dot(fp32_output.flatten(), quantized_output.flatten()) / (
    np.linalg.norm(fp32_output) * np.linalg.norm(quantized_output)
)
max_diff = np.max(np.abs(fp32_output - quantized_output))
print(f"cosine similarity: {cos_sim:.6f}")   # càng gần 1.0 càng tốt (>0.99 thường chấp nhận được)
print(f"max abs diff: {max_diff:.6f}")
```

Nếu cosine similarity thấp (model output gần như "đoán bừa" so với bản
gốc) → PTQ không đủ tốt cho model này, cần thử QAT hoặc calibration data
tốt hơn — **đừng đưa thẳng vào robot thật khi chưa qua bước kiểm tra
này.**

### 5.5. Ví dụ thật: thu calibration data từ camera OpenArm (không phải synthetic)

Lần quantize đầu tiên của pi0.5 (Phần 5.3) chỉ dùng **1 sample tổng hợp**
(hidden_states từ `torch.randn`) làm calibration data → cosine similarity
chỉ đạt **0.9167** (thấp hơn GR00T 0.9556 cùng phương pháp 1-sample). Lý
do: 1 sample random không đại diện đúng khoảng giá trị hidden_states thật
mà model gặp khi chạy ảnh camera thật. Cách khắc phục — thu **calibration
data thật** ngay trong lúc chạy vòng lặp camera→model→action (Phần 6.2):

**Bước 1 — Gắn PyTorch forward hook vào layer cần calibrate**, bật/tắt
qua flag global để không tốn bộ nhớ khi không cần:

```python
CAPTURE_CALIB = {"enabled": False, "samples": []}

def _capture_hook(module, args, kwargs):
    if CAPTURE_CALIB["enabled"]:
        hs = args[0] if args else kwargs.get("hidden_states")
        if hs is not None:
            # BFloat16 khong duoc numpy ho tro truc tiep -> ep float32 truoc
            CAPTURE_CALIB["samples"].append(hs.detach().to(torch.float32).cpu().numpy().copy())

target_layer.register_forward_pre_hook(_capture_hook, with_kwargs=True)
```

Cắm 3 endpoint HTTP (`/calib/start`, `/calib/stop`, `/calib/save`) để bật
capture, chạy vài chục request thật (camera thật, không phải ảnh mẫu),
rồi lưu `np.concatenate(samples, axis=0)` ra `.npy`. Với pi0.5,
`num_inference_steps=10` nghĩa là **mỗi request tạo 10 sample** (1/bước
denoise) — chỉ cần ~20 request thật (~3-4 phút) đã ra 200+ sample.

**Cạm bẫy 4 — Sample thật KHÔNG khớp shape với ONNX đã export trước đó.**
ONNX ở Phần 3 được export với 1 input mẫu cố định (ví dụ
`torch.randn(1, 64, hidden_size)` — seq_len=64 chọn tùy ý lúc test). Khi
capture hidden_states thật qua hook, seq_len thực tế phụ thuộc vào cấu
hình model + độ dài prompt/token thật (ví dụ pi0.5 ra seq_len=51, không
phải 64) — **lệch shape với graph ONNX đã biên dịch, không dùng trực tiếp
làm calibration data được** vì QNN yêu cầu shape cố định tuyệt đối
(Cạm bẫy 1, Phần 3.3).

→ **Cách sửa: export lại ONNX dùng chính 1 sample thật (không phải
random) làm input trace**, để seq_len (và mọi shape khác) khớp chính xác
với toàn bộ tập calibration thật đã thu:

```python
calib_all = np.load("real_calib_hidden_states.npy")  # (220, seq_len_that, hidden)
SEQ_LEN = calib_all.shape[1]                          # LẤY TỪ DATA THẬT, không hardcode
hidden_states = torch.from_numpy(calib_all[0:1])      # sample #0 làm input export + ref FP32
# cos/sin chỉ phụ thuộc position_ids (không đổi theo nội dung ảnh) -> tính 1 lần, dùng chung
```

Sau đó `calibration_data` cho `submit_quantize_job` truyền **toàn bộ**
tập thật thay vì 1 sample:

```python
calib = {
    "hidden_states": [calib_all[i:i+1] for i in range(N)],  # N sample thật
    "cos": [cos] * N,   # lặp lại vì cos/sin giống nhau mọi sample
    "sin": [sin] * N,
}
```

**Kết quả thật đo trên board IQ-9075** (pi0.5 Gemma action-expert, 3 layer
đầu): cosine similarity tăng từ **0.9167** (calib=1 sample tổng hợp) lên
**0.9753** (calib=220 sample thật) — max relative diff giảm từ 42.39%
xuống 26.04%. Latency INT8 tăng nhẹ không đáng kể (1565us → 1930us), vẫn
giữ 224/224 layer chạy 100% NPU. Chi tiết: `VLA/command.md`.

### 5.6. Bảng benchmark tổng hợp (số liệu THẬT đo trên IQ-9075)

Tất cả số liệu dưới đây đo trực tiếp trên Hexagon NPU của board IQ-9075
qua Qualcomm AI Hub — không phải ước tính. "Layer thật" nghĩa là lát cắt
đại diện (vài layer đầu, không phải toàn bộ model) — xem Phần 3.3 vì sao
làm vậy (tránh bug external-data >2GB) và vì sao vẫn hợp lệ (mọi layer
lặp lại dùng chung 1 loại toán tử).

**Operator coverage (dry-run, trọng số random-init, FP32) — 2026-08-28:**

| Model | Lát cắt | Layer đo | % chạy NPU | Latency FP32 |
|---|---|---|---|---|
| GR00T N1.7 DiT | 4/32 layer | 4 layer | **100%** (185/185) | 6.18ms |
| pi0/pi0.5 Gemma action-expert | 3/18 layer | 3 layer | **100%** (187/187) | 4.99ms |

Cả 2 model đều 100% operator coverage — không op nào bị QNN từ chối/rơi
về CPU. Ước tính tuyến tính cho full stack (chưa tính VLM backbone, chỉ
riêng action-head × số bước denoise): GR00T ~197ms (32 layer × 4 bước,
`num_inference_timesteps=4`), pi0.5 ~300ms (18 layer × 10 bước,
`num_inference_steps=10`) — GR00T có lợi thế tốc độ thô nhờ ít bước
denoise hơn.

**Quantize INT8 với TRỌNG SỐ THẬT (checkpoint đã tải, không random) — 2026-08-28 đến 2026-09-04:**

| Model | Calibration data | Latency INT8 | Speedup vs FP32 | % NPU | Cosine similarity | Max relative diff |
|---|---|---|---|---|---|---|
| GR00T N1.7 DiT (4 layer) | 1 sample tổng hợp | 2.99ms | ~2.0x (vs 6.18ms) | 100% (197/197) | 0.9556 | 39.56% |
| pi0.5 Gemma (3 layer) | 1 sample tổng hợp | 1.565ms | ~3.2x (vs 4.99ms) | 100% (224/224) | 0.9167 | 42.39% |
| pi0.5 Gemma (3 layer) | **220 sample thật** (camera OpenArm) | 1.930ms | ~2.6x (vs 4.99ms) | 100% (224/224) | **0.9753** | **26.04%** |

**Kết luận rút ra từ bảng trên** (quan trọng khi đọc số liệu này):
1. **INT8 luôn nhanh hơn FP32 rõ rệt** (2-3.2x) trên cả 2 model, và **luôn
   giữ 100% layer trên NPU** — operator coverage chưa bao giờ là vấn đề.
2. **Calibration data mới là yếu tố quyết định độ chính xác**, không phải
   kiến trúc model: cùng pi0.5, chỉ đổi từ 1 sample tổng hợp sang 220
   sample thật đã đưa cosine similarity từ dưới ngưỡng chấp nhận (0.9167)
   lên gần ngưỡng tốt (0.9753), trong khi latency chỉ tăng ~23%.
3. GR00T DiT **chưa được thử lại với calibration data thật** — kết quả
   0.9556 hiện tại vẫn dùng 1 sample tổng hợp, nên **không so sánh công
   bằng** với pi0.5's 0.9753. Muốn biết model nào "quantize tốt hơn" thật
   sự, cần lặp lại quy trình Phần 5.5 cho GR00T trước.
4. Ngưỡng ">0.99 thường chấp nhận được" (Phần 5.4) **chưa đạt được ở cả
   2 model** dù đã cải thiện nhiều — cân nhắc thêm QAT hoặc calibration
   data lớn hơn/đa dạng hơn trước khi tin dùng model quantize cho robot
   thật.

**Latency CPU tham khảo (không phải NPU, chỉ để đối chiếu)**: pi0.5 full
model (3.3B tham số, FP32, chưa fine-tune OpenArm) chạy inference đầy đủ
(10 bước denoise) trên CPU x86 của server mất **5.77 giây/lần** — quá
chậm cho control loop thật, chỉ dùng để xác nhận pipeline đúng trước khi
tối ưu NPU.

Chi tiết đầy đủ từng lần đo, script, và ngày tháng: `VLA/command.md`.

---

## Phần 6 — Deploy thật lên board IQ-9075

### 6.1. Nạp context binary và chạy suy luận trên board

Trên chính board IQ-9075 (không phải server train), dùng package
`qai_appbuilder` (Qualcomm cung cấp) để nạp file `.bin` đã biên dịch và
gọi suy luận:

```python
from qai_appbuilder import QNNConfig, QNNContext, Runtime

QNNConfig.Config(
    qnn_lib_path="/path/to/qairt/lib/aarch64-oe-linux-gcc11.2",
    runtime=Runtime.HTP,   # "Htp" = Hexagon Tensor Processor, viết hoa chữ H đầu — chú ý case-sensitive
)

ctx = QNNContext("model_name", "/path/to/model_qcs9075.bin")
output = ctx.Inference([input_numpy_array])
```

Lưu ý: tên runtime là `"Htp"` (viết hoa H), API tự động ghép thành tên
file thư viện `libQnnHtp.so` — truyền sai case sẽ báo lỗi tìm không thấy
file dù file có tồn tại.

### 6.2. Ghép vào pipeline robot thật

Model chỉ trả về **con số hành động** (action vector) — không tự biết gì
về ROS2, camera, hay tay robot. Cần code "keo dán" (glue code):

```
Camera (ROS2 topic) ──► tiền xử lý ảnh (resize, chuẩn hoá) ──┐
                                                              ├──► model.Inference() ──► action vector
Joint state (ROS2 topic) ──► chuẩn hoá state ────────────────┘                              │
                                                                                              ▼
                                                          hậu xử lý (unnormalize, map sang
                                                          đúng thứ tự joint của tay OpenArm)
                                                                                              │
                                                                                              ▼
                                                          publish lên ROS2 topic điều khiển tay
```

Đây chính là vai trò của `vla_bridge.py` / `vla0_api.md` trong repo —
**luôn kiểm tra kỹ action space đầu ra của model khớp với action space
thật của robot** (ví dụ: model train trên robot khác có thể xuất
end-effector-delta 7 chiều, trong khi OpenArm cần joint-position 14
chiều — không khớp sẽ gây chuyển động sai/nguy hiểm nếu áp trực tiếp).

### 6.3. Checklist an toàn trước khi chạy trên tay thật

1. Chạy thử trong RViz/MoveIt fake execution trước.
2. Kiểm tra output nằm trong `joint_limits.yaml` trước khi actuate.
3. Chạy tay thật lần đầu ở tốc độ thấp, qua chế độ gravity-compensation,
   có người trực kill switch.
4. So khớp: output model unquantized (FP32) vs quantized phải gần giống
   nhau trên cùng input (Phần 5.4) trước khi tin dùng bản quantize.

---

## Phần 7 — Bảng thuật ngữ nhanh

| Thuật ngữ | Nghĩa ngắn gọn |
|---|---|
| **ONNX** | Định dạng file mô tả đồ thị tính toán của model, trung gian giữa PyTorch và các runtime khác |
| **Opset** | "Phiên bản từ điển toán tử" của ONNX — opset cao hơn hỗ trợ toán tử mới hơn nhưng có thể kém tương thích ngược |
| **QNN** | Qualcomm AI Engine Direct — bộ SDK biên dịch/chạy model trên chip Qualcomm (CPU/GPU/Hexagon NPU) |
| **Hexagon / HTP** | Tên chip NPU của Qualcomm; HTP = Hexagon Tensor Processor, tên runtime khi gọi QNN |
| **Context binary** | File `.bin` — model đã biên dịch sẵn thành mã máy cho đúng 1 loại chip Hexagon cụ thể |
| **AI Hub** | Dịch vụ cloud của Qualcomm để biên dịch + đo hiệu năng model trên phần cứng thật từ xa |
| **Quantize (PTQ/QAT)** | Nén trọng số/phép tính từ FP32 xuống ít bit hơn (thường INT8) để chạy nhanh hơn trên NPU |
| **Calibration data** | Vài chục-trăm mẫu input thật, dùng để quantize biết khoảng giá trị cần ánh xạ |
| **External data (ONNX)** | Khi model >2GB, trọng số bị tách ra file phụ riêng — cần tránh vì gây bug upload chậm với AI Hub |
| **Compute unit / operator coverage** | Layer nào của model chạy trên NPU vs bị rơi về CPU khi biên dịch — chỉ số quan trọng nhất để đánh giá "model này hợp NPU không" |
| **Flow matching / denoising steps** | Cách GR00T và pi0/pi0.5 sinh hành động: bắt đầu từ nhiễu ngẫu nhiên, qua N bước "khử nhiễu" dần thành hành động thật (giống cách ảnh AI được sinh ra) |
| **Action chunk** | Một lần suy luận model không sinh ra 1 hành động, mà cả một chuỗi hành động tương lai (ví dụ 40-50 bước), robot thực thi dần trong lúc chờ lần suy luận tiếp theo |

---

## Phần 8 — Tóm tắt quy trình đầy đủ (checklist)

```
[ ] 1. Tải checkpoint từ HuggingFace (hf download ...), kiểm tra dung lượng đĩa trước
[ ] 2. Cài package kiến trúc gốc (pip install -e . từ repo GitHub chính chủ)
[ ] 3. Load model + trọng số THẬT trong PyTorch, chạy thử trên CPU/GPU, xác nhận output hợp lý
[ ] 4. Export sang ONNX với shape CỐ ĐỊNH — kiểm tra không có input int64 chưa xử lý,
       kiểm tra file <2GB (hoặc export từng lát cắt nếu model lớn hơn)
[ ] 5. Nộp AI Hub: submit_compile_job(device=<đúng tên thiết bị đích>)
[ ] 6. submit_profile_job() — đọc tỷ lệ NPU/CPU coverage, PHẢI gần 100% NPU mới đáng đầu tư tiếp
[ ] 7. submit_quantize_job() với calibration data THẬT (không random) — thu qua
       forward hook trong lúc chạy camera thật (Phần 5.5); nhớ export lại ONNX
       dùng đúng shape của sample thật (seq_len...), không phải shape giả lập ban đầu
[ ] 8. So sánh output quantize vs FP32 (cosine similarity) — PHẢI đạt ngưỡng chấp nhận được
[ ] 9. Tải file .bin cuối cùng về board IQ-9075 thật
[ ] 10. Viết code Inference qua qai_appbuilder, nối với camera/joint state qua ROS2
[ ] 11. Test an toàn: RViz trước → gravity-comp tốc độ thấp có người trực → mới chạy bình thường
```
