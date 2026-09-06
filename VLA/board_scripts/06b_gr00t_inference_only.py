#!/usr/bin/env python3
"""Chi chay lai buoc inference (job compile INT8 da xong tu truoc, id j5w744nmg)
- tranh phai quantize+compile lai tu dau, chi sua loi dtype timestep."""
import numpy as np
import qai_hub as hub

D = "/mnt/ssd/vla/export/gr00t_dit"
DEVICE = hub.Device("Dragonwing IQ-9075 EVK")

COMPILE_JOB_ID = "j5w744nmg"  # da SUCCESS tu lan chay truoc
compiled = hub.get_job(COMPILE_JOB_ID).get_target_model()

hidden_states = np.load(f"{D}/in_hidden_states.npy")
encoder_hidden_states = np.load(f"{D}/in_encoder_hidden_states.npy")
timestep = np.load(f"{D}/in_timestep.npy").astype(np.int32)  # fix: compile voi --truncate_64bit_io -> can int32
ref_output = np.load(f"{D}/ref_output.npy")

print("submit_inference_job...")
ijob = hub.submit_inference_job(
    model=compiled, device=DEVICE,
    inputs={"hidden_states": [hidden_states], "encoder_hidden_states": [encoder_hidden_states], "timestep": [timestep]},
    name="gr00t-n17-dit4layer-int8-inference-v2",
)
print("inference job:", ijob.url)
out = ijob.download_output_data()
quantized_output = list(out.values())[0][0]

a, b = ref_output.flatten().astype(np.float64), quantized_output.flatten().astype(np.float64)
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
max_diff = np.max(np.abs(a - b))
rel_diff = max_diff / (np.max(np.abs(a)) + 1e-12)

print("\n==== KET QUA SO SANH FP32 (trong so that) vs INT8 (numerical gate) ====")
print(f"  cosine similarity : {cos_sim:.6f}")
print(f"  max abs diff       : {max_diff:.6f}")
print(f"  max relative diff  : {rel_diff*100:.2f}%")
print(f"  FP32 range         : [{a.min():.4f}, {a.max():.4f}]")
print(f"  INT8 range         : [{b.min():.4f}, {b.max():.4f}]")
