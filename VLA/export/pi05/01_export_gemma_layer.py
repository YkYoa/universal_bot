#!/usr/bin/env python3
"""
Dry-run pi0/pi0.5: export mot vai GemmaDecoderLayer that (chuan HF
transformers), dung dung hyperparameter cua "gemma_300m" - action expert
that chay MOI buoc denoise (10 lan/action chunk) trong pi0_base checkpoint
da tai. Day la block lap lai nhieu nhat, tuong tu cach lam voi GR00T DiT.

Kien truc pi0/pi0.5 that (da xac nhan qua code, KHONG phai FAST-tokenizer
autoregressive nhu suy doan ban dau) - dung flow-matching giong GR00T:
PiGemmaModel(GemmaModel) lam "action expert" nhan noised-action + state,
joint-attention voi VLM prefix (gemma_2b) da cache.

gemma_300m config (tu modeling_pi05.py get_gemma_config):
  width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256
"""
import os
import torch

from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer, GemmaConfig

OUT = "/home/huyhoang/VLA/PI_0,5/export/gemma_onnx"
os.makedirs(OUT, exist_ok=True)

# gemma_300m - dung nguyen hyperparameter that
cfg = GemmaConfig(
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=3,          # rut gon: 3 layer du de bao phu toan bo op pattern lap lai (thay vi 18)
    num_attention_heads=8,
    num_key_value_heads=1,        # multi-query attention (GQA voi 1 KV head)
    head_dim=256,
    hidden_act="gelu_pytorch_tanh",
    max_position_embeddings=256,  # du cho action_horizon=50 + prefix
    attn_implementation="eager",  # an toan cho ONNX export, tranh fused-kernel path
)

torch.manual_seed(0)
layers = torch.nn.ModuleList([GemmaDecoderLayer(cfg, layer_idx=i) for i in range(cfg.num_hidden_layers)]).eval()

SEQ_LEN = 64  # uoc luong: action tokens (50) + vai token dieu kien
hidden_states = torch.randn(1, SEQ_LEN, cfg.hidden_size)
position_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)

# RoPE can position_embeddings (cos, sin) - tu tinh truoc nhu HF lam trong GemmaModel.forward
head_dim = cfg.head_dim
inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
pos = position_ids[0].float()
freqs = torch.outer(pos, inv_freq)
emb = torch.cat([freqs, freqs], dim=-1)
cos, sin = emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


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

with torch.no_grad():
    ref = model(hidden_states, cos, sin)

onnx_path = f"{OUT}/gemma3layer.onnx"
torch.onnx.export(
    model,
    (hidden_states, cos, sin),
    onnx_path,
    input_names=["hidden_states", "cos", "sin"],
    output_names=["out"],
    opset_version=17,
    dynamo=False,
)

n_params = sum(p.numel() for p in model.parameters())
onnx_size = os.path.getsize(onnx_path) / 1e6
print(f"[ok] Gemma 3-layer params: {n_params/1e6:.1f}M")
print(f"[ok] onnx -> {onnx_path} ({onnx_size:.1f} MB)")
print(f"[ok] ref out shape: {tuple(ref.shape)}")
