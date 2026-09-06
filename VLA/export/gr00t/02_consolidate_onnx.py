#!/usr/bin/env python3
"""Gop toan bo external-data tensor cua dit.onnx thanh 1 file duy nhat
(dit.onnx + dit.onnx.data) - dung convention chuan de AI Hub / cac tool
khac doc du lieu de dang, thay vi hang tram file tensor roi."""
import onnx
from onnx.external_data_helper import convert_model_to_external_data

SRC = "/home/huyhoang/VLA/Gr00t-N1,7/export/dit_onnx/dit.onnx"
DST = "/home/huyhoang/VLA/Gr00t-N1,7/export/dit_onnx_packed/dit.onnx"

import os
os.makedirs(os.path.dirname(DST), exist_ok=True)

model = onnx.load(SRC, load_external_data=True)
convert_model_to_external_data(
    model, all_tensors_to_one_file=True, location="dit.onnx.data", size_threshold=1024
)
onnx.save_model(model, DST)

print("packed onnx ->", DST)
print("packed data ->", DST + ".data")
for f in os.listdir(os.path.dirname(DST)):
    p = os.path.join(os.path.dirname(DST), f)
    print(f"  {f}: {os.path.getsize(p)/1e6:.1f} MB")
