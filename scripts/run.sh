#!/bin/bash
# ================================
# Transformer MT Project - Run Script
# 自动训练机器翻译模型
# ================================

# 1. 自动选择设备（如果有GPU就用CUDA）
DEVICE="cpu"
if python - <<'EOF'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
EOF
then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi

echo "[Info] Using device: $DEVICE"

# 2. 创建虚拟环境
echo "[Info] Creating environment..."
conda create -n transformer_mt python=3.10 -y
# conda activate transformer_mt

# 3. 安装依赖（你也可以只用 pip install -r requirements.txt）
echo "[Info] Installing dependencies..."
pip install -r requirements.txt

# 4. 运行训练脚本
echo "[Info] Starting training..."
python train_mt_alldata.py \
    --seed 42 \
    --epochs 30 \
    --batch_size 16 \
    --max_len 64 \
    --device "$DEVICE"

echo "[Info] Training completed."
