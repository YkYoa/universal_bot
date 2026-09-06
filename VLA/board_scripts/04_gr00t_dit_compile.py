#!/usr/bin/env python3
"""Dry-run: nop GR00T N1.7 DiT (rut gon 4 layer, 144M tham so) len AI Hub,
compile+profile cho dung Dragonwing IQ-9075 EVK. Chay tu chinh board IQ-9075
(da xac nhan route toi AI Hub nhanh, khac voi server huyhoang-4090 bi cham
bat thuong ~20KB/s tren duong nay)."""
import qai_hub as hub

ONNX = "/mnt/ssd/vla/export/gr00t_dit/dit_small_single.onnx"
DEVICE = hub.Device("Dragonwing IQ-9075 EVK")

compile_job = hub.submit_compile_job(
    model=ONNX, device=DEVICE, name="gr00t-n17-dit4layer-dryrun",
    options="--target_runtime qnn_context_binary --truncate_64bit_io",
)
print("compile job :", compile_job.url)
target = compile_job.get_target_model()

profile_job = hub.submit_profile_job(
    model=target, device=DEVICE, name="gr00t-n17-dit4layer-dryrun-profile"
)
print("profile job :", profile_job.url)

prof = profile_job.download_profile()
ex = prof["execution_summary"]
print("---- on-device latency (4-layer DiT, 1 forward) ----")
print("  inference (us):", ex.get("estimated_inference_time"))
print("  peak mem (MB) :", ex.get("estimated_inference_peak_memory", 0) / 1e6)

detail = prof.get("execution_detail", [])
units = {}
for L in detail:
    units[L.get("compute_unit", "?")] = units.get(L.get("compute_unit", "?"), 0) + 1
print("  layers per compute unit:", units)
print(f"  total layers: {len(detail)}")

out = "/mnt/ssd/vla/export/gr00t_dit/dit_small_qcs9075.bin"
target.download(out)
print("context binary ->", out)
