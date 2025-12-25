#!/bin/bash

# 单节点多GPU训练脚本

echo "================================================"
echo "Data Parallel Training - Single Node"
echo "================================================"

# 检测可用GPU数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

# 设置环境变量
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO  # 调试信息，生产环境可设为WARN

echo ""
echo "1. Running Basic DDP Example..."
echo "---------------------------------------"
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    dp_basic.py

echo ""
echo "2. Running Advanced DDP Example..."
echo "---------------------------------------"
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    dp_ddp.py

echo ""
echo "3. Running Matrix Multiplication Benchmark..."
echo "---------------------------------------"
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29502 \
    dp_matrix_multiply.py --mode benchmark

echo ""
echo "================================================"
echo "All Examples Completed!"
echo "================================================"