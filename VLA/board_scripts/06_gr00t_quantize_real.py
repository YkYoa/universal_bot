#!/usr/bin/env python3
"""
Quantize INT8 DiT (4 layer, TRONG SO THAT tu checkpoint GR00T-N1.7-3B) qua
Qualcomm AI Hub, roi so sanh output voi ban FP32 goc (cosine similarity +
max abs diff) - day la "numerical gate" bat buoc truoc khi tin dung ban
quantize cho robot that.
"""
import numpy as np
import qai_hub as hub

D = "/mnt/ssd/vla/export/gr00t_dit"
ONNX = f"{D}/dit_real.onnx"
DEVICE = hub.Device("Dragonwing IQ-9075 EVK")

hidden_states = np.load(f"{D}/in_hidden_states.npy")
encoder_hidden_states = np.load(f"{D}/in_encoder_hidden_states.npy")
timestep = np.load(f"{D}/in_timestep.npy")
ref_output = np.load(f"{D}/ref_output.npy")

# Calibration data - dung nguyen input that da co (o day chi 1 sample vi
# muc tieu la kiem tra pipeline; production nen dung vai chuc-tram sample
# that da thu tu camera+state cua tay OpenArm)
calib = {
    "hidden_states": [hidden_states],
    "encoder_hidden_states": [encoder_hidden_states],
    "timestep": [timestep],
}

print("[1/4] submit_quantize_job...")
qjob = hub.submit_quantize_job(
    model=ONNX,
    calibration_data=calib,
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT8,
    name="gr00t-n17-dit4layer-int8",
)
print("quantize job:", qjob.url)
quantized_model = qjob.get_target_model()

print("[2/4] submit_compile_job (quantized -> qnn_context_binary)...")
cjob = hub.submit_compile_job(
    model=quantized_model, device=DEVICE, name="gr00t-n17-dit4layer-int8-compiled",
    options="--target_runtime qnn_context_binary --truncate_64bit_io",
)
print("compile job:", cjob.url)
compiled = cjob.get_target_model()

print("[3/4] submit_profile_job (do toc do that tren NPU)...")
pjob = hub.submit_profile_job(model=compiled, device=DEVICE, name="gr00t-n17-dit4layer-int8-profile")
print("profile job:", pjob.url)
prof = pjob.download_profile()
ex = prof["execution_summary"]
detail = prof.get("execution_detail", [])
units = {}
for L in detail:
    units[L.get("compute_unit", "?")] = units.get(L.get("compute_unit", "?"), 0) + 1
print(f"  INT8 latency (us): {ex.get('estimated_inference_time')}")
print(f"  layers per compute unit: {units}")

print("[4/4] submit_inference_job (chay that, lay output de so sanh FP32)...")
ijob = hub.submit_inference_job(
    model=compiled, device=DEVICE,
    inputs={"hidden_states": [hidden_states], "encoder_hidden_states": [encoder_hidden_states], "timestep": [timestep]},
    name="gr00t-n17-dit4layer-int8-inference",
)
out = ijob.download_output_data()
quantized_output = list(out.values())[0][0]

a, b = ref_output.flatten().astype(np.float64), quantized_output.flatten().astype(np.float64)
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
max_diff = np.max(np.abs(a - b))
rel_diff = max_diff / (np.max(np.abs(a)) + 1e-12)

print("\n==== KET QUA SO SANH FP32 vs INT8 (numerical gate) ====")
print(f"  cosine similarity : {cos_sim:.6f}   (>0.99 thuong chap nhan duoc)")
print(f"  max abs diff       : {max_diff:.6f}")
print(f"  max relative diff  : {rel_diff*100:.2f}%")
print(f"  FP32 range         : [{a.min():.4f}, {a.max():.4f}]")
print(f"  INT8 range         : [{b.min():.4f}, {b.max():.4f}]")
