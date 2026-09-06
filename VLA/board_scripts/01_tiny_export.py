#!/usr/bin/env python3
"""
Smoke test 1/3 — PyTorch -> ONNX (fixed shape).

Mục đích: đi trọn pipeline bằng 1 model tí hon TRƯỚC khi đụng pi-0.5/GR00T,
để tách bạch lỗi "chưa quen toolchain" khỏi lỗi "op của VLA không convert được".

Điểm khác CUDA/TensorRT cần thấy tận mắt ở đây:
  - shape PHẢI cố định (batch=1, không dynamic_axes)
  - opset giữ thấp/ổn định để QNN converter nhận đủ op
"""
import os, torch, torch.nn as nn

OUT = "/mnt/ssd/vla/export/tiny"
os.makedirs(OUT, exist_ok=True)


class TinyBlock(nn.Module):
    """Giả lập 1 block transformer: matmul + softmax attention + MLP + LayerNorm.
    Đây đúng là các op mà VLA backbone sẽ dùng, nên nó test được op coverage thật."""

    def __init__(self, d=256, heads=4):
        super().__init__()
        self.h, self.d = heads, d
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):                       # x: (1, T, d)
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.h, D // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, D // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, D // self.h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (D // self.h) ** -0.5
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        return x + self.mlp(self.n2(x))


def main():
    torch.manual_seed(0)
    model = TinyBlock().eval()

    # SHAPE CỐ ĐỊNH — đây là ràng buộc cốt lõi của QNN so với TensorRT
    dummy = torch.randn(1, 32, 256)

    onnx_path = f"{OUT}/tiny.onnx"
    torch.onnx.export(
        model, (dummy,), onnx_path,
        input_names=["x"], output_names=["y"],
        opset_version=17,
        dynamo=False,          # exporter cũ ổn định hơn cho QNN converter
        # KHÔNG dynamic_axes -> graph tĩnh hoàn toàn
    )
    with torch.no_grad():
        ref = model(dummy)
    torch.save({"input": dummy, "ref": ref}, f"{OUT}/tiny_ref.pt")
    dummy.numpy().astype("float32").tofile(f"{OUT}/input.raw")
    with open(f"{OUT}/input_list.txt", "w") as f:
        f.write(f"x:={OUT}/input.raw\n")

    print(f"[ok] onnx      -> {onnx_path}")
    print(f"[ok] ref out   -> shape {tuple(ref.shape)}  mean {ref.mean():.6f}")
    print(f"[ok] raw input -> {OUT}/input.raw")


if __name__ == "__main__":
    main()
