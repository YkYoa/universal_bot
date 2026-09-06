#!/usr/bin/env python3
"""Dry-run: nop ban DiT rut gon (4 layer, 144M tham so, 577MB) len AI Hub,
compile+profile cho Dragonwing IQ-9075 EVK. 4 layer bao phu du ca 2 loai
block (self-attn + cross-attn xen ke), nen ket qua operator-coverage tong
quat hoa duoc cho ca 32 layer that."""
import qai_hub as hub

ONNX = "/home/huyhoang/VLA/Gr00t-N1,7/export/dit_onnx_small_packed/dit_small.onnx"
DEVICE = hub.Device("Dragonwing IQ-9075 EVK")

compile_job = hub.submit_compile_job(
    model=ONNX, device=DEVICE, name="gr00t-n17-dit4layer-dryrun",
    options="--target_runtime qnn_context_binary",
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

out = "/home/huyhoang/VLA/Gr00t-N1,7/export/dit_small_qcs9075.bin"
target.download(out)
print("context binary ->", out)
