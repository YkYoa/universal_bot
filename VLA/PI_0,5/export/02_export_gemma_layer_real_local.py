#!/usr/bin/env python3
"""
Load TRONG SO THAT (khong random-init) cho 3 layer dau cua Gemma action
expert (gemma_300m) tu checkpoint pi0_base da tai, export ONNX. Dung
phuong phap giong 05_export_real_weights.py da lam voi GR00T DiT.

Checkpoint pi0_base la 1 file duy nhat (khong shard), key khong co prefix
"model." (xac nhan qua safe_open truc tiep). Key layer gemma_expert:
  paligemma_with_expert.gemma_expert.model.layers.{i}.{self_attn,mlp,...}
"""
import json
import os

import torch
from safetensors import safe_open
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer, GemmaConfig

CKPT = "/home/hans/universal_bot/VLA/weights_local/pi0_base"
OUT = "/home/hans/universal_bot/VLA/export/gemma_onnx_real_local"
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
        layer_keys = [k for k in all_keys if k.startswith(layer_prefix)]
        needed_keys.extend(layer_keys)
    print(f"[info] loading {len(needed_keys)} real tensors for {NUM_LAYERS} layers")

    state_dict = {}
    for k in needed_keys:
        # paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight
        #   -> 0.self_attn.q_proj.weight  (khop voi ModuleList.state_dict() key)
        local_key = k[len(PREFIX):]
        state_dict[local_key] = f.get_tensor(k).to(torch.float32)

missing, unexpected = layers.load_state_dict(state_dict, strict=False)
print(f"[info] missing: {missing}")
print(f"[info] unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
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

torch.manual_seed(0)
SEQ_LEN = 64
hidden_states = torch.randn(1, SEQ_LEN, cfg.hidden_size)
position_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)

head_dim = cfg.head_dim
inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
pos = position_ids[0].float()
freqs = torch.outer(pos, inv_freq)
emb = torch.cat([freqs, freqs], dim=-1)
cos, sin = emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)

with torch.no_grad():
    ref = model(hidden_states, cos, sin)

onnx_path = f"{OUT}/gemma3layer_real.onnx"
torch.onnx.export(
    model, (hidden_states, cos, sin), onnx_path,
    input_names=["hidden_states", "cos", "sin"], output_names=["out"],
    opset_version=17, dynamo=False,
)

n_params = sum(p.numel() for p in model.parameters())
onnx_size = os.path.getsize(onnx_path) / 1e6
print(f"[ok] Gemma {NUM_LAYERS}-layer (TRONG SO THAT) params: {n_params/1e6:.1f}M")
print(f"[ok] onnx -> {onnx_path} ({onnx_size:.1f} MB)")
print(f"[ok] ref out shape: {tuple(ref.shape)}  mean={ref.mean().item():.6f}  std={ref.std().item():.6f}")

import numpy as np
np.save(f"{OUT}/in_hidden_states.npy", hidden_states.numpy())
np.save(f"{OUT}/in_cos.npy", cos.numpy())
np.save(f"{OUT}/in_sin.npy", sin.numpy())
np.save(f"{OUT}/ref_output.npy", ref.numpy())
