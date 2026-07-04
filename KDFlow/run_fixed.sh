#!/bin/bash
set -e
set -x

# ==================================================================
# Fixed version based on official example pattern
# Key insight: Must start Ray manually BEFORE training, not use local mode
# ==================================================================

echo "=== Step 1: Stop any existing Ray instance ==="
ray stop || true
sleep 2

echo "=== Step 2: Start Ray cluster with 2 GPUs ==="
ray start --head --node-ip-address=127.0.0.1 --num-gpus=2

echo "=== Step 3: Verify Ray status ==="
ray status

echo "=== Step 4: Run training (without KDFLOW_RAY_ADDRESS=local) ==="
# Key fixes:
# 1. Remove KDFLOW_RAY_ADDRESS=local to use manual Ray cluster
# 2. CRITICAL: Reduce TEACHER_FORWARD_N_BATCHES to avoid overwhelming teacher
# 3. Use TEACHER_TP_SIZE=1 for simplicity on 2 GPUs
CUDA_LIB_PATH=/czsun/zhi/envs/kdflow/lib/python3.10/site-packages/nvidia/cu13/lib \
CUDA_VISIBLE_DEVICES=6,7 \
RUN_REFERENCE_SFT=false \
SFT_REFERENCE_MODEL=/czsun/zhi/xywang/anchored_learning/LlamaFactory/saves/Qwen3-4B_medcalc_train_1e-5 \
DATA_JSON=/czsun/zhi/xywang/anchored_learning/LlamaFactory/data/medcalc_train.json \
BASE_MODEL=/czsun/models/Qwen3-4B \
NUM_GPUS_PER_NODE=2 \
TEACHER_TP_SIZE=1 \
TEACHER_MEM_FRACTION_STATIC=0.5 \
TEACHER_FORWARD_N_BATCHES=5 \
TRAIN_BATCH_SIZE=16 \
MICRO_TRAIN_BATCH_SIZE=1 \
ATTN_IMPLEMENTATION=sdpa \
PACKING_SAMPLES=False \
RING_ATTN_SIZE=1 \
bash examples/anchored_learning/train_anchored_learning.sh

echo "=== Step 5: Training completed, stopping Ray ==="
ray stop
