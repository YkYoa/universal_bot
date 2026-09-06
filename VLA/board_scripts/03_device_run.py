#!/usr/bin/env python3
"""
Smoke test 3/3 — nap context binary (.bin) da compile qua qai_appbuilder,
chay THAT tren Hexagon NPU cua board nay, so ket qua voi ref PyTorch.

Day la buoc "correctness gate" (a) trong plan: cosine similarity / max-abs-diff
giua output NPU va output fp32 CPU tren CUNG input, TRUOC khi dung toi robot that.
"""
import numpy as np, torch, sys
from qai_appbuilder import QNNContext, QNNConfig

BIN = "/mnt/ssd/vla/export/tiny/tiny_qcs9075.bin"
REF = "/mnt/ssd/vla/export/tiny/tiny_ref.pt"

QNNConfig.Config(
    "/home/ubuntu/qcom/qairt/2.45.0.260326/lib/aarch64-oe-linux-gcc11.2",
    "Htp",  # backend = Hexagon Tensor Processor (NPU)
)

ctx = QNNContext("tiny_smoke", BIN)

ref = torch.load(REF)
x = ref["input"].numpy().astype(np.float32)
y_ref = ref["ref"].numpy()

y_npu = np.array(ctx.Inference([x.flatten()])).reshape(y_ref.shape)

cos = float(
    (y_npu.flatten() @ y_ref.flatten())
    / (np.linalg.norm(y_npu) * np.linalg.norm(y_ref) + 1e-9)
)
max_abs_diff = float(np.max(np.abs(y_npu - y_ref)))

print(f"cosine_similarity = {cos:.6f}")
print(f"max_abs_diff       = {max_abs_diff:.6f}")
print("PASS" if cos > 0.99 else "FAIL — kiem tra lai quantize/compile")
