#!/usr/bin/env python3
"""Dry-run 2/2: nop DiT that (GR00T N1.7, 1.09B tham so) len AI Hub, compile
+ profile cho dung thiet bi Dragonwing IQ-9075 EVK. Muc tieu: lay so lieu
that ve operator-coverage (bao nhieu layer roi ve CPU vs chay tren NPU) va
latency - day la du lieu quyet dinh co nen dau tu sau vao GR00T tren IQ-9075
hay khong, thay vi doan."""
import qai_hub as hub

ONNX = "/home/huyhoang/VLA/Gr00t-N1,7/export/dit_onnx_packed/dit.onnx"
DEVICE = hub.Device("Dragonwing IQ-9075 EVK")

compile_job = hub.submit_compile_job(
    model=ONNX,
    device=DEVICE,
    name="gr00t-n17-dit-dryrun",
    options="--target_runtime qnn_context_binary",
)
print("compile job :", compile_job.url)
target = compile_job.get_target_model()

profile_job = hub.submit_profile_job(
    model=target, device=DEVICE, name="gr00t-n17-dit-dryrun-profile"
)
print("profile job :", profile_job.url)

prof = profile_job.download_profile()
ex = prof["execution_summary"]
print("---- on-device latency (DiT 1 forward = 1 buoc denoise) ----")
print("  inference (us):", ex.get("estimated_inference_time"))
print("  peak mem (MB) :", ex.get("estimated_inference_peak_memory", 0) / 1e6)

detail = prof.get("execution_detail", [])
units = {}
for L in detail:
    units[L.get("compute_unit", "?")] = units.get(L.get("compute_unit", "?"), 0) + 1
print("  layers per compute unit:", units)
print(f"  total layers: {len(detail)}")

out = "/home/huyhoang/VLA/Gr00t-N1,7/export/dit_qcs9075.bin"
target.download(out)
print("context binary ->", out)
