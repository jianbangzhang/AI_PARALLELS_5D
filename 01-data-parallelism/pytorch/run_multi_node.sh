#!/bin/bash

# 多节点分布式训练启动脚本（适用于 DDP 或 FSDP）
# 用法: 在节点0上运行: ./run_multi_node.sh <节点数> <每节点GPU数> <脚本.py>
# 示例: ./run_multi_node.sh 2 4 dp_ddp.py

if [ $# -lt 3 ]; then
    echo "用法: $0 <节点数> <每节点GPU数> <训练脚本.py> [其他参数]"
    exit 1
fi

NUM_NODES=$1
GPUS_PER_NODE=$2
SCRIPT=$3
shift 3
EXTRA_ARGS="$@"

WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

# 主节点地址（假设已通过环境变量或参数设置，这里用第一个参数或默认）
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}  # 多节点时需替换为节点0的IP
MASTER_PORT=29500

echo "启动多节点训练: ${NUM_NODES} 节点, 每节点 ${GPUS_PER_NODE} GPU, 总 world_size=${WORLD_SIZE}"
echo "主节点地址: ${MASTER_ADDR}:${MASTER_PORT}"
echo "训练脚本: ${SCRIPT}"

torchrun \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --rdzv_id=456 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    ${SCRIPT} ${EXTRA_ARGS}