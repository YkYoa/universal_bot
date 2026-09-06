#!/usr/bin/env python3
"""
Smoke test 2/3 — compile ONNX -> QNN context binary qua Qualcomm AI Hub,
target đúng thiết bị 'Dragonwing IQ-9075 EVK' (qcs9075, hexagon v73).

Đây là khác biệt lớn nhất so với CUDA/TensorRT:
  TensorRT  : build engine NGAY trên máy có GPU, runtime linh hoạt.
  QNN       : converter/quantizer chỉ có bản x86 -> compile ở AI Hub (cloud)
              hoặc máy x86, board chỉ NẠP và CHẠY context binary đã compile.
"""
import qai_hub as hub

ONNX = "/mnt/ssd/vla/export/tiny/tiny.onnx"
DEVICE = hub.Device("Dragonwing IQ-9075 EVK")

# fp16 trên HTP được hỗ trợ (htp-supports-fp16:true) -> bản chạy đầu tiên
# KHÔNG cần quantize INT8. Correctness trước, tốc độ sau.
compile_job = hub.submit_compile_job(
    model=ONNX,
    device=DEVICE,
    name="vla-smoke-tiny-block",
    options="--target_runtime qnn_context_binary",
)
print("compile job :", compile_job.url)
target = compile_job.get_target_model()
print("compile ok  :", compile_job.success if hasattr(compile_job, "success") else "n/a")

profile_job = hub.submit_profile_job(
    model=target, device=DEVICE, name="vla-smoke-tiny-block-profile"
)
print("profile job :", profile_job.url)

prof = profile_job.download_profile()
ex = prof["execution_summary"]
print("---- on-device latency ----")
print("  inference (us):", ex.get("estimated_inference_time"))
print("  peak mem (b)  :", ex.get("estimated_inference_peak_memory"))

# Số quan trọng nhất: bao nhiêu layer thực sự chạy trên NPU vs rơi về CPU
detail = prof.get("execution_detail", [])
units = {}
for L in detail:
    units[L.get("compute_unit", "?")] = units.get(L.get("compute_unit", "?"), 0) + 1
print("  layers per compute unit:", units)

out = "/mnt/ssd/vla/export/tiny/tiny_qcs9075.bin"
target.download(out)
print("context binary ->", out)
