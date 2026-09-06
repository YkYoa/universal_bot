#!/usr/bin/env python3
"""HTTP inference server cho pi0_base (checkpoint goc, chua fine-tune tren
OpenArm) - CHI de kiem tra vong lap camera-that -> model -> action chay
duoc dau-cuoi lan dau tien (Stage 1 trong DEPLOY_GUIDE.md, "dung truoc,
nhanh sau"). Chay CPU vi GPU dang bi chiem boi job khac tren server dung
chung (xem VLA/command.md).

pi0_base khai bao 3 camera (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb)
+ state 32-D kieu Franka - CHUA khop voi OpenArm that (14 DoF). Camera thu 3
va state duoc zero-pad giong pattern cu trong vla0_api.md. Action tra ve o
day KHONG duoc dua thang vao tay that - checkpoint chua fine-tune tren
OpenArm nen action space/y nghia khong khop.

Endpoint:
  GET  /health
  POST /predict  {"front_b64": "<jpeg base64>", "left_b64": "<jpeg base64>", "task": "..."}
"""
import base64
import io
import time

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image as PILImage

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import PreTrainedPolicy, make_pre_post_processors

MODEL_PATH = "/home/huyhoang/VLA/PI_0,5/weights/pi0_base"
IMG_SIZE = 224  # config.json: image_resolution [224, 224]

print("Loading pi0_base config...")
config = PreTrainedConfig.from_pretrained(MODEL_PATH)
config.device = "cpu"

print("Loading pi0_base policy tren CPU (co the mat vai phut, model 3.3B tham so fp32)...")
t0 = time.time()
policy = PreTrainedPolicy.from_pretrained(MODEL_PATH, config=config)
policy.eval()
print(f"Loaded (from_pretrained mac dinh) trong {time.time() - t0:.1f}s")

# ── Fix: checkpoint pi0_base luu key KHONG co prefix "model." (vd
# "action_in_proj.bias"), nhung state_dict cua policy object lai can key
# CO prefix "model." (vd "model.action_in_proj.bias") - PreTrainedPolicy.
# from_pretrained mac dinh khong remap dung, dan toi ~777/777 key bi
# "missing" va model chay voi trong so RANDOM INIT (khong phai da train).
# Tu nap lai state_dict, tu remap prefix cho dung.
from safetensors.torch import load_file

raw_sd = load_file(f"{MODEL_PATH}/model.safetensors")
remapped_sd = {f"model.{k}": v for k, v in raw_sd.items()}
missing, unexpected = policy.load_state_dict(remapped_sd, strict=False)
print(f"[remap] missing={len(missing)} unexpected={len(unexpected)} (ky vong ca 2 gan 0)")
if len(missing) > 5:
    print(f"[remap] CANH BAO: van con {len(missing)} key thieu, vd: {missing[:5]}")
policy.eval()

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=MODEL_PATH,
    preprocessor_overrides={"device_processor": {"device": "cpu"}},
)
policy.reset()

STATE_DIM = int(policy.config.input_features["observation.state"].shape[0])
print(f"state dim ky vong: {STATE_DIM}")

# ── Hook thu thap calibration data THAT: bat input hidden_states thuc te
# di vao layer 0 cua gemma_expert (Gemma action-expert) trong luc chay
# inference that voi anh camera that. Vi num_inference_steps=10, MOI lan
# goi /predict tao ra 10 sample (1/buoc denoise) - vai lan goi la du calib
# set da dang. Bat tat qua bien global CAPTURE_CALIB.
CAPTURE_CALIB = {"enabled": False, "samples": []}


def _capture_hook(module, args, kwargs):
    if CAPTURE_CALIB["enabled"]:
        hs = args[0] if args else kwargs.get("hidden_states")
        if hs is not None:
            # BFloat16 khong duoc numpy ho tro truc tiep -> ep float32 truoc
            CAPTURE_CALIB["samples"].append(hs.detach().to(torch.float32).cpu().numpy().copy())


_target_layer = policy.model.paligemma_with_expert.gemma_expert.model.layers[0]
_target_layer.register_forward_pre_hook(_capture_hook, with_kwargs=True)
print(f"[calib] da gan hook vao layer: {type(_target_layer).__name__} (layer 0 cua gemma_expert)")

app = Flask(__name__)


def decode_image(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    img = PILImage.open(io.BytesIO(raw)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.uint8)  # HWC, uint8


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", "device": "cpu", "state_dim": STATE_DIM,
        "calib_capture_enabled": CAPTURE_CALIB["enabled"],
        "calib_samples_collected": len(CAPTURE_CALIB["samples"]),
    })


@app.route("/calib/start", methods=["POST"])
def calib_start():
    CAPTURE_CALIB["enabled"] = True
    return jsonify({"status": "capturing_on", "samples_so_far": len(CAPTURE_CALIB["samples"])})


@app.route("/calib/stop", methods=["POST"])
def calib_stop():
    CAPTURE_CALIB["enabled"] = False
    return jsonify({"status": "capturing_off", "samples_so_far": len(CAPTURE_CALIB["samples"])})


@app.route("/calib/save", methods=["POST"])
def calib_save():
    if not CAPTURE_CALIB["samples"]:
        return jsonify({"error": "chua co sample nao, goi /calib/start roi chay vai lan /predict truoc"}), 400
    arr = np.concatenate(CAPTURE_CALIB["samples"], axis=0)  # (N, seq_len, hidden) - N = tong so buoc denoise da bat
    out_path = "/home/huyhoang/VLA/PI_0,5/export/real_calib_hidden_states.npy"
    np.save(out_path, arr)
    return jsonify({"status": "saved", "path": out_path, "shape": list(arr.shape)})


@app.route("/predict", methods=["POST"])
def predict():
    # QUAN TRONG: select_action() dung "action queue" noi bo (n_action_steps=50
    # trong config.json) - chi chay model THAT 1 lan moi 50 lan goi, con lai
    # tra ve action da tinh san TU ANH TRUOC DO. Vi policy song xuyen suot
    # server (khong tao lai moi request), HTTP request nay coi la MOT
    # request doc lap (stateless, giong vla0_api.md) nen PHAI reset queue
    # truoc, ep chay full inference tren dung anh vua nhan.
    policy.reset()

    data = request.get_json(force=True)
    front = decode_image(data["front_b64"])
    left = decode_image(data.get("left_b64")) if data.get("left_b64") else np.zeros_like(front)
    right = np.zeros_like(front)  # camera thu 3 chua co that -> zero-pad
    task = data.get("task", "pick up the object")

    obs = {
        "observation.images.base_0_rgb": torch.from_numpy(front).permute(2, 0, 1).float() / 255.0,
        "observation.images.left_wrist_0_rgb": torch.from_numpy(left).permute(2, 0, 1).float() / 255.0,
        "observation.images.right_wrist_0_rgb": torch.from_numpy(right).permute(2, 0, 1).float() / 255.0,
        "observation.state": torch.zeros(STATE_DIM, dtype=torch.float32),  # chua co joint state that
        "task": [task],
    }
    for k in obs:
        if k != "task" and obs[k].dim() == 3:
            obs[k] = obs[k].unsqueeze(0)  # them batch dim
        elif k != "task":
            obs[k] = obs[k].unsqueeze(0)

    t0 = time.time()
    processed = preprocessor(obs)
    with torch.inference_mode():
        action = policy.select_action(processed)
    action = postprocessor(action)
    elapsed = time.time() - t0

    action_np = action.squeeze(0).cpu().numpy().tolist() if action.dim() > 1 else action.cpu().numpy().tolist()
    return jsonify({"action": action_np, "inference_time_sec": round(elapsed, 3)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9091, debug=False)
