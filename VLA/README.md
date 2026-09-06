# VLA — pi-0.5 / GR00T lên IQ-9075

- **[DEPLOY_GUIDE.md](DEPLOY_GUIDE.md)** — đọc trước: hướng dẫn đầy đủ từ
  kiến thức nền tới từng bước download → test → export → compile →
  quantize → deploy.
- **[command.md](command.md)** — nhật ký quyết định + kết quả dry-run thật
  (GR00T N1.7 vs pi0/pi0.5), số liệu latency/accuracy đo được, các lỗi kỹ
  thuật đã gặp và cách sửa.

## Cấu trúc script (đã đồng bộ về từ 2 máy remote)

Việc thật chạy trên `huyhoang-4090` (192.168.1.122, có GPU CUDA, chứa
checkpoint) và board `IQ-9075` (192.168.1.226, có Hexagon NPU thật) — các
file dưới đây là **bản sao đồng bộ về máy này để lưu trong repo/git**,
không tự chạy được ở đây (cần đúng conda env + checkpoint đã tải trên máy
gốc).

```
export/gr00t/    — script export DiT (action head) của GR00T N1.7, chạy trên huyhoang-4090
  01_export_dit_onnx.py        export full 32-layer (KHÔNG dùng nữa — quá chậm upload, xem command.md)
  01b_export_dit_small.py      export rút gọn 4/32 layer, random-init weights (dry-run đầu tiên)
  02_consolidate_onnx.py       gộp external-data thành 1 file (KHÔNG dùng nữa — chính là nguồn gốc bug chậm)
  03_aihub_compile.py          nộp AI Hub compile — bản full layer (KHÔNG dùng, quá chậm)
  03b_aihub_compile_small.py   nộp AI Hub compile — bản rút gọn external-data (vẫn lỗi, xem command.md)
  05_export_real_weights.py    ⭐ export 4 layer với TRỌNG SỐ THẬT từ checkpoint — dùng cho quantize

export/pi05/     — script export Gemma decoder layer (action expert) của pi0/pi0.5
  01_export_gemma_layer.py     ⭐ export 3/18 layer gemma_300m, random-init (dry-run)

board_scripts/   — chạy trên chính board IQ-9075 (source env.sh trước)
  env.sh                       biến môi trường QAIRT/QNN + đường dẫn — LUÔN source trước khi chạy script khác
  01_tiny_export.py            model tí hon tự tạo, dùng để test toàn bộ pipeline lần đầu
  02_aihub_compile.py          compile+profile model tí hon qua AI Hub
  03_device_run.py             nạp context binary .bin và chạy suy luận thật trên NPU (qai_appbuilder)
  04_gr00t_dit_compile.py      ⭐ compile+profile GR00T DiT (random-init) — kết quả: 185/185 layer NPU, 6.18ms
  05_pi05_gemma_compile.py     ⭐ compile+profile pi0.5 Gemma layer (random-init) — kết quả: 187/187 layer NPU, 4.99ms
  06_gr00t_quantize_real.py    ⭐ quantize INT8 (trọng số thật) + so sánh cosine-similarity với FP32
```

Script đánh dấu ⭐ là bản **đang dùng/kết quả cuối**; các bản khác giữ lại
làm lịch sử debug (xem `command.md` để hiểu vì sao mỗi lần thử thất bại).

## Máy & đường dẫn gốc (nơi script thật sự chạy được)

| Máy | Vai trò | Đường dẫn workspace |
|---|---|---|
| `huyhoang-4090` (192.168.1.122) | GPU CUDA, checkpoint, export ONNX | `/home/huyhoang/VLA/` |
| `ubuntu@192.168.1.226` (IQ-9075) | Hexagon NPU thật, chạy AI Hub job + inference | `/mnt/ssd/vla/` |
