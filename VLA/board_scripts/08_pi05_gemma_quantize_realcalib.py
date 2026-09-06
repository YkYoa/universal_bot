#!/usr/bin/env python3
"""
Quantize INT8 (trong so THAT) cho 3 layer dau cua Gemma action expert,
dung 220 SAMPLE HIDDEN_STATES THAT bat qua hook /calib/start khi chay
camera OpenArm that (thay vi 1 sample tong hop random nhu
07_pi05_gemma_quantize_real.py). cos/sin giu nguyen 1 ban vi chi phu
thuoc position_ids (khong doi theo noi dung anh) nen lap lai cho khop so
luong voi hidden_states.

Chay:
  source /mnt/ssd/vla/env.sh
  $VLA_PY /mnt/ssd/vla/export/pi05_gemma_realcalib51/08_pi05_gemma_quantize_realcalib.py
"""
import numpy as np
import qai_hub as hub

D = "/mnt/ssd/vla/export/pi05_gemma_realcalib51"
ONNX = f"{D}/gemma3layer_real_calib51.onnx"
DEVICE = hub.Device("Dragonwing IQ-9075 EVK")

hidden_states = np.load(f"{D}/in_hidden_states.npy")
cos = np.load(f"{D}/in_cos.npy")
sin = np.load(f"{D}/in_sin.npy")
ref_output = np.load(f"{D}/ref_output.npy")
calib_all = np.load(f"{D}/calib_hidden_states_all220.npy")  # (220, 51, 1024)
N = calib_all.shape[0]
print(f"[calib] dung {N} sample THAT tu camera OpenArm (thay vi 1 sample tong hop)")

calib = {
    "hidden_states": [calib_all[i:i + 1] for i in range(N)],
    "cos": [cos] * N,
    "sin": [sin] * N,
}

print("[1/4] submit_quantize_job...")
qjob = hub.submit_quantize_job(
    model=ONNX, calibration_data=calib,
    weights_dtype=hub.QuantizeDtype.INT8, activations_dtype=hub.QuantizeDtype.INT8,
    name="pi05-gemma3layer-realcalib220-int8",
)
print("quantize job:", qjob.url)
quantized_model = qjob.get_target_model()

print("[2/4] submit_compile_job...")
cjob = hub.submit_compile_job(
    model=quantized_model, device=DEVICE, name="pi05-gemma3layer-realcalib220-int8-compiled",
    options="--target_runtime qnn_context_binary --truncate_64bit_io",
)
print("compile job:", cjob.url)
compiled = cjob.get_target_model()

print("[3/4] submit_profile_job...")
pjob = hub.submit_profile_job(model=compiled, device=DEVICE, name="pi05-gemma3layer-realcalib220-int8-profile")
print("profile job:", pjob.url)
prof = pjob.download_profile()
ex = prof["execution_summary"]
detail = prof.get("execution_detail", [])
units = {}
for L in detail:
    units[L.get("compute_unit", "?")] = units.get(L.get("compute_unit", "?"), 0) + 1
print(f"  INT8 latency (us): {ex.get('estimated_inference_time')}")
print(f"  layers per compute unit: {units}")

print("[4/4] submit_inference_job...")
ijob = hub.submit_inference_job(
    model=compiled, device=DEVICE,
    inputs={"hidden_states": [hidden_states], "cos": [cos], "sin": [sin]},
    name="pi05-gemma3layer-realcalib220-int8-inference",
)
out = ijob.download_output_data()
quantized_output = list(out.values())[0][0]

a, b = ref_output.flatten().astype(np.float64), quantized_output.flatten().astype(np.float64)
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
max_diff = np.max(np.abs(a - b))
rel_diff = max_diff / (np.max(np.abs(a)) + 1e-12)

print("\n==== KET QUA SO SANH FP32 (trong so THAT pi0.5) vs INT8 (calib=220 sample THAT) ====")
print(f"  cosine similarity : {cos_sim:.6f}  (so sanh voi 0.916709 cua calib=1 sample tong hop)")
print(f"  max abs diff       : {max_diff:.6f}")
print(f"  max relative diff  : {rel_diff*100:.2f}%")
print(f"  FP32 range         : [{a.min():.4f}, {a.max():.4f}]")
print(f"  INT8 range         : [{b.min():.4f}, {b.max():.4f}]")
