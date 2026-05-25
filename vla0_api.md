# VLA Deployment API

HTTP API for robot control. Send camera images + task instruction → get action sequence.

- Base URL: `http://<host>:10000`
- Interactive Swagger UI: `GET /docs`

> ⚠️ **Warning:** The default checkpoint at `runs/vla0/model_last.pth` is trained on the **LIBERO dataset** (Franka 7-DoF end-effector control), not SO-ARM100 joints. Output is a **7-dim normalized end-effector delta action**, not 6-dim absolute joint targets in degrees. The deploy code performs no EE→joint conversion. Do not feed these values directly to a SO-ARM100. Verify the loaded checkpoint matches your robot before actuating.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict_base64` | Full 8-step action sequence in one response |
| POST | `/predict_base64_stream` | Stream actions step-by-step (NDJSON) |

---

## `GET /health`

**Request:** no body.

**Response** (`application/json`):

```json
{"status": "ok"}
```

---

## `POST /predict_base64`

### Request

Content-Type: `application/json`

```json
{
  "base64_rgb": ["<image1>", "<image2>"],
  "state": [-8.177, -98.898, 99.637, 50.923, -1.651, 6.600],
  "instr": "Push the apple to the block"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `base64_rgb` | `list[str]` | yes | 1 or 2 encoded camera frames in order `[front, left]`. If only 1 sent, server pads a zero frame for the missing slot. |
| `state` | `list[float]` | yes | 6 current joint positions (degrees). Accepted by the API but **not consumed** by the Qwen-VL text-action path; included for compatibility with future state-conditioned checkpoints. |
| `instr` | `str \| null` | no | Natural-language task instruction |

> **Note:** Model trained with 2 cameras. Sending only the front camera works (server zero-pads the left slot), but predictions degrade compared to providing both. Sending more than 2 → HTTP 400.

**Image encoding** — each `base64_rgb` entry is built from an RGB `uint8` array shape `(H, W, 3)`, values `0–255`, recommended `224 x 224 x 3`:

```python
import base64, io, numpy as np

def encode(img: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, img.astype(np.uint8))
    return base64.b64encode(buf.getvalue()).decode()
```

**`state` joint order** (request only — 6 dims, degrees):

| Index | Joint | Unit |
|---|---|---|
| 0 | shoulder_pan | degrees |
| 1 | shoulder_lift | degrees |
| 2 | elbow_flex | degrees |
| 3 | wrist_flex | degrees |
| 4 | wrist_roll | degrees |
| 5 | gripper | 0 (open) → higher (closed) |

**Response action vector** (7 dims, LIBERO end-effector delta action space):

| Index | Meaning | Range (from `dataset_stats.pkl`) | Notes |
|---|---|---|---|
| 0 | EE delta x | ±0.9375 | Normalized position delta |
| 1 | EE delta y | ±0.9375 | Normalized position delta |
| 2 | EE delta z | ±0.9375 | Normalized position delta |
| 3 | EE delta rotation (axis-angle x) | [-0.258, 0.356] | radians |
| 4 | EE delta rotation (axis-angle y) | ±0.375 | radians |
| 5 | EE delta rotation (axis-angle z) | ±0.368 | radians |
| 6 | Gripper command | [-1, 1] | -1 = open, +1 = close |

### Response

Content-Type: `application/json`

2D array, shape `(8, 7)` — 8 future timesteps × 7-dim EE delta action:

```json
[
  [ 0.12, -0.10,  0.66,  0.00,  0.09, -0.08, -1.00],
  [ 0.13, -0.11,  0.67,  0.00,  0.09, -0.08, -1.00],
  [ 0.14, -0.11,  0.67,  0.00,  0.09, -0.08,  1.00],
  ...
]
```

Values = normalized EE deltas + gripper command (NOT joint degrees). Must be unnormalized + mapped to your robot's control space before actuation.

#### Debug — inspect raw response

Print full response to verify shape and value ranges before any indexing/post-processing:

```python
import json, requests
resp = requests.post(f"{URL}/predict_base64", json=payload).json()
print(json.dumps(resp, indent=2))
# Or with numpy:
import numpy as np
arr = np.array(resp)
print("shape:", arr.shape)
print("min per dim:", arr.min(axis=0))
print("max per dim:", arr.max(axis=0))
print("full array:\n", arr)
```

---

## `POST /predict_base64_stream`

Same prediction; emits actions one at a time. Begin actuating timestep 0 before later steps finish generating.

### Request

Same as `/predict_base64`.

### Response

Content-Type: `application/x-ndjson` (chunked stream, one JSON object per line).

8 action lines + 1 timing line:

```
{"index": 0, "value": [[ 0.12, -0.10,  0.66,  0.00,  0.09, -0.08, -1.00]]}
{"index": 1, "value": [[ 0.13, -0.11,  0.67,  0.00,  0.09, -0.08, -1.00]]}
...
{"index": 7, "value": [[ 0.18, -0.13,  0.69,  0.00,  0.10, -0.08,  1.00]]}
{"time_taken": 3.42}
```

**Action line:**

| Field | Type | Description |
|---|---|---|
| `index` | int | Timestep, `0..7` |
| `value` | `list[list[float]]` | Shape `(1, 7)`. Inner list = 7-dim LIBERO EE delta action for this step (see action vector table above) |

**Final line:**

| Field | Type | Description |
|---|---|---|
| `time_taken` | float | Total inference time, seconds |

Parse:

```python
import json, numpy as np, requests

r = requests.post(url, json=payload, stream=True)
for line in r.iter_lines():
    data = json.loads(line)
    if "value" in data:
        action = np.array(data["value"][0])   # shape (7,)
        send_to_robot(action)
```

#### Debug — print every streamed line

Inspect full raw stream before indexing into `value`:

```python
import json, requests
r = requests.post(url, json=payload, stream=True)
for line in r.iter_lines():
    if not line:
        continue
    print(line.decode())   # raw NDJSON line
    data = json.loads(line)
    print("parsed:", data)
```

---

## Minimal Client

```python
import base64, io, numpy as np, requests

URL = "http://localhost:10000"

def encode(img):
    buf = io.BytesIO()
    np.save(buf, img.astype(np.uint8))
    return base64.b64encode(buf.getvalue()).decode()

front = np.zeros((224, 224, 3), dtype=np.uint8)   # replace with camera frame
left  = np.zeros((224, 224, 3), dtype=np.uint8)

payload = {
    "base64_rgb": [encode(front), encode(left)],   # or [encode(front)] for single-cam
    "state": [-8.177, -98.898, 99.637, 50.923, -1.651, 6.600],
    "instr": "Push the apple to the block",
}

resp = requests.post(f"{URL}/predict_base64", json=payload).json()
print(resp)                      # full raw response — verify before indexing
actions = np.array(resp)
print("shape:", actions.shape)   # expect (8, 7) for the libero checkpoint
# shape (8, 7) → 7-dim LIBERO EE delta actions for next 8 timesteps
```
