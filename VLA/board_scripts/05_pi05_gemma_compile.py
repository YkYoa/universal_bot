#!/usr/bin/env python3
"""Dry-run pi0/pi0.5: nop 3-layer Gemma decoder (action expert, gemma_300m
config, block chay lap lai 10 lan/chunk) len AI Hub, compile+profile cho
Dragonwing IQ-9075 EVK."""
import qai_hub as hub

ONNX = "/mnt/ssd/vla/export/gr00t_dit/gemma3layer.onnx"
DEVICE = hub.Device("Dragonwing IQ-9075 EVK")

compile_job = hub.submit_compile_job(
    model=ONNX, device=DEVICE, name="pi05-gemma3layer-dryrun",
    options="--target_runtime qnn_context_binary --truncate_64bit_io",
)
print("compile job :", compile_job.url)
target = compile_job.get_target_model()

profile_job = hub.submit_profile_job(
    model=target, device=DEVICE, name="pi05-gemma3layer-dryrun-profile"
)
print("profile job :", profile_job.url)

prof = profile_job.download_profile()
ex = prof["execution_summary"]
print("---- on-device latency (3-layer Gemma, 1 forward) ----")
print("  inference (us):", ex.get("estimated_inference_time"))
print("  peak mem (MB) :", ex.get("estimated_inference_peak_memory", 0) / 1e6)

detail = prof.get("execution_detail", [])
units = {}
for L in detail:
    units[L.get("compute_unit", "?")] = units.get(L.get("compute_unit", "?"), 0) + 1
print("  layers per compute unit:", units)
print(f"  total layers: {len(detail)}")
