#!/usr/bin/env python3
"""
Load TRONG SO THAT (khong random-init) cho 4 layer dau cua DiT tu checkpoint
GR00T-N1.7-3B da tai, export ONNX, luu ca input+output tham chieu (fp32) de
so sanh voi output sau quantize INT8.
"""
import json
import os

import torch
from safetensors import safe_open

from gr00t.model.modules.dit import DiT

CKPT = "/home/huyhoang/VLA/Gr00t-N1,7/weights/gr00t_n1.7_3b"
OUT = "/home/huyhoang/VLA/Gr00t-N1,7/export/dit_real_weights"
os.makedirs(OUT, exist_ok=True)
NUM_LAYERS = 4

with open(f"{CKPT}/config.json") as f:
    cfg = json.load(f)

dit_cfg = dict(cfg["diffusion_model_cfg"])
dit_cfg["num_layers"] = NUM_LAYERS
cross_attention_dim = cfg["backbone_embedding_dim"]
action_horizon = cfg["action_horizon"]

model = DiT(**dit_cfg, cross_attention_dim=cross_attention_dim).eval()

# ---- nap trong so that tu safetensors shard, chi cho 4 layer dau + shared parts ----
with open(f"{CKPT}/model.safetensors.index.json") as f:
    index = json.load(f)
weight_map = index["weight_map"]

PREFIX = "action_head.model."
wanted_prefixes = [f"{PREFIX}transformer_blocks.{i}." for i in range(NUM_LAYERS)]
wanted_prefixes += [f"{PREFIX}timestep_encoder.", f"{PREFIX}proj_out_1", f"{PREFIX}proj_out_2"]

needed_keys = [k for k in weight_map if k.startswith(PREFIX) and any(k.startswith(p) for p in wanted_prefixes)]
print(f"[info] loading {len(needed_keys)} real tensors for {NUM_LAYERS} layers")

shard_files = sorted({weight_map[k] for k in needed_keys})
state_dict = {}
for shard in shard_files:
    with safe_open(f"{CKPT}/{shard}", framework="pt") as f:
        for k in needed_keys:
            if weight_map[k] == shard:
                local_key = k[len(PREFIX):]  # bo prefix "action_head.model."
                state_dict[local_key] = f.get_tensor(k).to(torch.float32)

missing, unexpected = model.load_state_dict(state_dict, strict=False)
# strict=False vi transformer_blocks index 4..31 khong ton tai trong model rut gon (dung, khong phai loi)
real_missing = [m for m in missing if not any(f"transformer_blocks.{i}." in m for i in range(NUM_LAYERS, 32))]
print(f"[info] missing (that su thieu, khong tinh layer 4-31): {real_missing}")
print(f"[info] unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
assert len(real_missing) == 0, "Thieu weight that su can thiet - kiem tra lai key mapping"

torch.manual_seed(0)
inner_dim = dit_cfg["num_attention_heads"] * dit_cfg["attention_head_dim"]
VL_TOKENS = 64
hidden_states = torch.randn(1, action_horizon, inner_dim)
encoder_hidden_states = torch.randn(1, VL_TOKENS, cross_attention_dim)
timestep = torch.tensor([500], dtype=torch.long)

with torch.no_grad():
    ref = model(hidden_states, encoder_hidden_states, timestep)

onnx_path = f"{OUT}/dit_real.onnx"
torch.onnx.export(
    model, (hidden_states, encoder_hidden_states, timestep), onnx_path,
    input_names=["hidden_states", "encoder_hidden_states", "timestep"],
    output_names=["action_out"], opset_version=17, dynamo=False,
)

torch.save(
    {"hidden_states": hidden_states, "encoder_hidden_states": encoder_hidden_states,
     "timestep": timestep, "ref": ref},
    f"{OUT}/ref_real.pt",
)
import numpy as np
np.save(f"{OUT}/in_hidden_states.npy", hidden_states.numpy())
np.save(f"{OUT}/in_encoder_hidden_states.npy", encoder_hidden_states.numpy())
np.save(f"{OUT}/in_timestep.npy", timestep.numpy())
np.save(f"{OUT}/ref_output.npy", ref.numpy())

print(f"[ok] onnx (real weights) -> {onnx_path} ({os.path.getsize(onnx_path)/1e6:.1f} MB)")
print(f"[ok] ref out shape: {tuple(ref.shape)}  mean={ref.mean().item():.6f}  std={ref.std().item():.6f}")
