#!/usr/bin/env python3
"""
Ban re-export cua 02_export_gemma_layer_real_local.py, sua SEQ_LEN tu 64
(gia lap) -> 51 de KHOP voi seq_len that cua hidden_states bat duoc qua
hook /calib/start trong serve_pi05_http.py khi chay camera that (xac nhan
qua real_calib_hidden_states.npy shape (220, 51, 1024)). QNN can shape co
dinh luc compile nen ONNX phai trace dung seq_len nay thi 220 sample that
moi dung duoc lam calibration data.

Dung sample #0 trong real_calib_hidden_states.npy lam input export + tinh
ref_output (FP32 baseline) - thay vi torch.randn nhu ban cu - de tracing
va reference deu bam theo phan phoi hidden_states THAT.
"""
import os

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer, GemmaConfig

CKPT = "/home/hans/universal_bot/VLA/weights_local/pi0_base"
CALIB_NPY = "/home/hans/universal_bot/VLA/PI_0,5/export/real_calib_hidden_states.npy"
OUT = "/home/hans/universal_bot/VLA/PI_0,5/export/gemma_onnx_real_calib51"
os.makedirs(OUT, exist_ok=True)
NUM_LAYERS = 3

cfg = GemmaConfig(
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=8,
    num_key_value_heads=1,
    head_dim=256,
    hidden_act="gelu_pytorch_tanh",
    max_position_embeddings=256,
    attn_implementation="eager",
)

layers = torch.nn.ModuleList([GemmaDecoderLayer(cfg, layer_idx=i) for i in range(NUM_LAYERS)]).eval()

PREFIX = "paligemma_with_expert.gemma_expert.model.layers."
needed_keys = []
with safe_open(f"{CKPT}/model.safetensors", framework="pt") as f:
    all_keys = list(f.keys())
    for i in range(NUM_LAYERS):
        layer_prefix = f"{PREFIX}{i}."
        needed_keys.extend([k for k in all_keys if k.startswith(layer_prefix)])
    print(f"[info] loading {len(needed_keys)} real tensors for {NUM_LAYERS} layers")

    state_dict = {}
    for k in needed_keys:
        local_key = k[len(PREFIX):]
        state_dict[local_key] = f.get_tensor(k).to(torch.float32)

missing, unexpected = layers.load_state_dict(state_dict, strict=False)
print(f"[info] missing: {missing}")
assert len(missing) == 0, "Thieu weight that su can thiet - kiem tra lai key mapping"


class GemmaStack(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, hidden_states, cos, sin):
        position_embeddings = (cos, sin)
        for layer in self.layers:
            out = layer(hidden_states, position_embeddings=position_embeddings)
            hidden_states = out[0] if isinstance(out, tuple) else out
        return hidden_states


model = GemmaStack(layers).eval()

calib_all = np.load(CALIB_NPY)  # (220, 51, 1024) - hidden_states THAT bat qua hook
SEQ_LEN = calib_all.shape[1]
assert SEQ_LEN == 51, f"seq_len bat ngo: {SEQ_LEN} (ky vong 51 tu log truoc)"
hidden_states = torch.from_numpy(calib_all[0:1])  # sample #0 lam input trace + ref

position_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
head_dim = cfg.head_dim
inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
pos = position_ids[0].float()
freqs = torch.outer(pos, inv_freq)
emb = torch.cat([freqs, freqs], dim=-1)
cos, sin = emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)

with torch.no_grad():
    ref = model(hidden_states, cos, sin)

onnx_path = f"{OUT}/gemma3layer_real_calib51.onnx"
torch.onnx.export(
    model, (hidden_states, cos, sin), onnx_path,
    input_names=["hidden_states", "cos", "sin"], output_names=["out"],
    opset_version=17, dynamo=False,
)

n_params = sum(p.numel() for p in model.parameters())
onnx_size = os.path.getsize(onnx_path) / 1e6
print(f"[ok] Gemma {NUM_LAYERS}-layer (TRONG SO THAT, seq_len={SEQ_LEN}) params: {n_params/1e6:.1f}M")
print(f"[ok] onnx -> {onnx_path} ({onnx_size:.1f} MB)")
print(f"[ok] ref out shape: {tuple(ref.shape)}  mean={ref.mean().item():.6f}  std={ref.std().item():.6f}")

np.save(f"{OUT}/in_hidden_states.npy", hidden_states.numpy())
np.save(f"{OUT}/in_cos.npy", cos.numpy())
np.save(f"{OUT}/in_sin.npy", sin.numpy())
np.save(f"{OUT}/ref_output.npy", ref.numpy())
# ca 220 sample that, dung lam calibration_data day du (khong chi 1 sample nhu ban cu)
np.save(f"{OUT}/calib_hidden_states_all220.npy", calib_all)
print(f"[ok] calib set day du: {calib_all.shape} -> {OUT}/calib_hidden_states_all220.npy")
