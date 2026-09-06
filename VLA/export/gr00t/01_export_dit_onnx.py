#!/usr/bin/env python3
"""
Dry-run 1/2: export GR00T N1.7's DiT action head (kien truc rui ro nhat theo
plan - flow-matching/diffusion, khac han pattern autoregressive-decode da
duoc Genie chung minh chay tot tren IQ-9075) sang ONNX voi shape co dinh.

Day la mot khoi DiT THAT, dung dung hyperparameter tu config.json cua
checkpoint GR00T-N1.7-3B da tai ve - khong phai model gia lap nhu smoke
test truoc. Muc tieu: lay so lieu operator-coverage that qua AI Hub, chua
can quan tam do chinh xac (dung random-init weight, khong load checkpoint
that o buoc nay).
"""
import json
import os

import torch

from gr00t.model.modules.dit import DiT

CKPT = "/home/huyhoang/VLA/Gr00t-N1,7/weights/gr00t_n1.7_3b"
OUT = "/home/huyhoang/VLA/Gr00t-N1,7/export/dit_onnx"
os.makedirs(OUT, exist_ok=True)

with open(f"{CKPT}/config.json") as f:
    cfg = json.load(f)

dit_cfg = dict(cfg["diffusion_model_cfg"])
cross_attention_dim = cfg["backbone_embedding_dim"]  # 2048
action_horizon = cfg["action_horizon"]  # 40

print("diffusion_model_cfg:", dit_cfg)
print("cross_attention_dim:", cross_attention_dim)
print("action_horizon:", action_horizon)

torch.manual_seed(0)
model = DiT(**dit_cfg, cross_attention_dim=cross_attention_dim).eval()

inner_dim = dit_cfg["num_attention_heads"] * dit_cfg["attention_head_dim"]
VL_TOKENS = 64  # so token vision+language dien hinh sau projection (uoc luong hop ly)

hidden_states = torch.randn(1, action_horizon, inner_dim)
encoder_hidden_states = torch.randn(1, VL_TOKENS, cross_attention_dim)
timestep = torch.tensor([500], dtype=torch.long)  # 1 buoc denoise (trong so 4 buoc)

with torch.no_grad():
    ref = model(hidden_states, encoder_hidden_states, timestep)

onnx_path = f"{OUT}/dit.onnx"
torch.onnx.export(
    model,
    (hidden_states, encoder_hidden_states, timestep),
    onnx_path,
    input_names=["hidden_states", "encoder_hidden_states", "timestep"],
    output_names=["action_out"],
    opset_version=17,
    dynamo=False,
)

torch.save(
    {"hidden_states": hidden_states, "encoder_hidden_states": encoder_hidden_states,
     "timestep": timestep, "ref": ref},
    f"{OUT}/ref.pt",
)

n_params = sum(p.numel() for p in model.parameters())
print(f"[ok] DiT params      : {n_params/1e6:.1f}M")
print(f"[ok] onnx             -> {onnx_path}")
print(f"[ok] ref out shape    : {tuple(ref.shape)}")
print(f"[ok] onnx file size   : {os.path.getsize(onnx_path)/1e6:.1f} MB")
