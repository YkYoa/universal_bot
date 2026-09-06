#!/usr/bin/env bash
# ── Môi trường QAIRT/QNN cho VLA trên IQ-9075 (QCS9075, Hexagon V73) ──
# Dùng: source /mnt/ssd/vla/env.sh

export QAIRT_HOME="/home/ubuntu/qcom/qairt/2.45.0.260326"
export QNN_SDK_ROOT="$QAIRT_HOME"

# QUAN TRỌNG: chỉ dùng lib aarch64. KHÔNG source envsetup.sh của SDK trên board
# (nó đẩy x86_64-linux-clang lên đầu LD_LIBRARY_PATH -> libQnnSystem.so 0 byte -> "file too short")
export PATH="$QAIRT_HOME/bin/aarch64-oe-linux-gcc11.2:$PATH"
export LD_LIBRARY_PATH="$QAIRT_HOME/lib/aarch64-oe-linux-gcc11.2:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$QAIRT_HOME/lib/hexagon-v73/unsigned;/usr/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp;/dsp/cdsp"

# Hexagon arch của QCS9075
export QNN_HEXAGON_ARCH="v73"
export QNN_TARGET_SOC="qualcomm-qcs9075"

# Không để HF cache ăn hết 24G của rootfs -> đẩy sang NVMe (88G free)
export HF_HOME="/mnt/ssd/vla/models/.hf"
export HF_HUB_CACHE="/mnt/ssd/vla/models/.hf/hub"
export TORCH_HOME="/mnt/ssd/vla/models/.torch"

export VLA_ROOT="/mnt/ssd/vla"
export VLA_PY="/mnt/ssd/miniconda3/envs/qai/bin/python"

echo "[vla] QAIRT=$QAIRT_HOME  hexagon=$QNN_HEXAGON_ARCH  root=$VLA_ROOT"
