#!/usr/bin/env python3
"""
Ban rut gon cua dry-run: chi export num_layers=4 (thay vi 32) tu chinh DiT
that cua GR00T N1.7. Voi interleave_self_attention=True, block le/chan xen
ke self-attn va cross-attn - 4 layer la du de bao phu CA HAI loai block
(toan bo op-type xuat hien trong graph), chi khac ve SO LUONG block lap lai.

Ly do rut gon: ban day du 32 layer nang ~4.4GB, upload len AI Hub voi bang
thong server hien tai (~24 KB/s, bat thuong cham) se mat ~50 gio - khong
thuc te cho muc tieu "do luong operator-coverage nhanh, re" cua dry run.
4 layer chi nang vai tram MB, van cho CHINH XAC cung mot tap operator.
"""
import json
import os

import torch

from gr00t.model.modules.dit import DiT

CKPT = "/home/huyhoang/VLA/Gr00t-N1,7/weights/gr00t_n1.7_3b"
OUT = "/home/huyhoang/VLA/Gr00t-N1,7/export/dit_onnx_small"
os.makedirs(OUT, exist_ok=True)

with open(f"{CKPT}/config.json") as f:
    cfg = json.load(f)

dit_cfg = dict(cfg["diffusion_model_cfg"])
dit_cfg["num_layers"] = 4  # << rut gon, van xen ke self/cross-attn (idx%2)
cross_attention_dim = cfg["backbone_embedding_dim"]
action_horizon = cfg["action_horizon"]

print("diffusion_model_cfg (rut gon):", dit_cfg)

torch.manual_seed(0)
model = DiT(**dit_cfg, cross_attention_dim=cross_attention_dim).eval()

inner_dim = dit_cfg["num_attention_heads"] * dit_cfg["attention_head_dim"]
VL_TOKENS = 64

hidden_states = torch.randn(1, action_horizon, inner_dim)
encoder_hidden_states = torch.randn(1, VL_TOKENS, cross_attention_dim)
timestep = torch.tensor([500], dtype=torch.long)

with torch.no_grad():
    ref = model(hidden_states, encoder_hidden_states, timestep)

onnx_path = f"{OUT}/dit_small.onnx"
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
print(f"[ok] DiT params (4 layer): {n_params/1e6:.1f}M")
print(f"[ok] onnx -> {onnx_path}")
print(f"[ok] ref out shape: {tuple(ref.shape)}")

# gop external data thanh 1 file duy nhat luon, cho gon
import onnx
from onnx.external_data_helper import convert_model_to_external_data

m = onnx.load(onnx_path, load_external_data=True)
convert_model_to_external_data(m, all_tensors_to_one_file=True, location="dit_small.onnx.data", size_threshold=1024)
packed_dir = f"{OUT}_packed"
os.makedirs(packed_dir, exist_ok=True)
onnx.save_model(m, f"{packed_dir}/dit_small.onnx")
for f in os.listdir(packed_dir):
    p = os.path.join(packed_dir, f)
    print(f"  packed {f}: {os.path.getsize(p)/1e6:.1f} MB")
