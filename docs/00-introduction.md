# 分布式计算总体介绍

## 📋 目录

- [为什么需要分布式计算](#为什么需要分布式计算)
- [基础概念](#基础概念)
- [通信原语](#通信原语)
- [硬件架构](#硬件架构)
- [软件栈](#软件栈)
- [性能指标](#性能指标)
- [常见挑战](#常见挑战)

---

## 为什么需要分布式计算

### 深度学习模型规模的爆炸式增长

```
年份    代表模型              参数量        训练数据量
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2018    BERT-Large          340M         16GB
2019    GPT-2               1.5B         40GB
2020    GPT-3               175B         570GB
2021    Megatron-Turing     530B         >1TB
2022    PaLM                540B         >2TB
2023    GPT-4               ~1.8T*       未知
2024    Gemini Ultra        未公开       未知

* 推测值，基于MoE架构
```

### 单设备的三大限制

#### 1. 内存限制

**问题**：大模型无法装入单个GPU

```python
# GPT-3 (175B参数) 内存需求
参数存储:     175B × 4 bytes (FP32)    = 700 GB
梯度存储:     175B × 4 bytes           = 700 GB
优化器状态:   175B × 8 bytes (Adam)   = 1,400 GB
激活值:       ~100 GB (取决于batch size)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总计:                                   ≈ 3 TB

A100 80GB显存: 仅能存储 2.6% 的模型！
```

#### 2. 计算限制

**问题**：训练时间过长

```
GPT-3 训练 (单A100估算):
- 浮点运算: 3.14 × 10²³ FLOPs
- A100性能: 312 TFLOPS (FP16)
- 理论时间: 3.14×10²³ / (312×10¹²) = 1,006,410 秒
- 实际时间: ≈ 355 天 (考虑效率40%)

使用1024个A100:
- 训练时间: ≈ 8.3 小时 ✓
```

#### 3. 数据限制

**问题**：训练数据无法一次性加载

```
现代数据集规模:
- Common Crawl:     ~250 TB (网页文本)
- The Pile:         ~800 GB (多样化文本)
- LAION-5B:         ~240 TB (图像-文本对)
- C4:               ~750 GB (清洗后网页)

单机内存: 512 GB << 数据集大小
需要: 分布式数据加载 + 流式处理
```

---

## 基础概念

### 1. 进程与设备

```
分布式系统组织层次:

集群 (Cluster)
    ├── 节点1 (Node 1)
    │   ├── GPU 0  ← Rank 0
    │   ├── GPU 1  ← Rank 1
    │   ├── GPU 2  ← Rank 2
    │   └── GPU 3  ← Rank 3
    ├── 节点2 (Node 2)
    │   ├── GPU 0  ← Rank 4
    │   └── ...
    └── 节点N (Node N)

关键概念:
- Rank: 进程的全局唯一标识 (0 到 world_size-1)
- Local Rank: 进程在节点内的标识 (0 到 GPUs_per_node-1)
- World Size: 总进程数
```

#### 代码示例

```python
import torch.distributed as dist

# 初始化分布式环境
dist.init_process_group(backend='nccl')

# 获取基本信息
rank = dist.get_rank()              # 当前进程的全局rank
world_size = dist.get_world_size()  # 总进程数
local_rank = int(os.environ['LOCAL_RANK'])  # 节点内rank

print(f"我是 Rank {rank}/{world_size}, Local Rank {local_rank}")
```

### 2. 进程组 (Process Group)

```python
# 创建不同的进程组
import torch.distributed as dist

# 全局组
world_group = dist.group.WORLD

# 数据并行组 (例如: 0,1,2,3 一组, 4,5,6,7 一组)
dp_group = dist.new_group([0, 1, 2, 3])

# 模型并行组 (例如: 0,4 一组, 1,5 一组, ...)
mp_group = dist.new_group([0, 4])

# 在特定组内通信
dist.all_reduce(tensor, group=dp_group)
```

### 3. 并行维度

```
5D并行空间:

Dimension 1: Data Parallelism (DP)
    ├─ 数据切分: [Batch 0-31] [Batch 32-63] [Batch 64-95] ...
    └─ 模型复制: Model₁  Model₂  Model₃ ...

Dimension 2: Pipeline Parallelism (PP)
    ├─ 层切分: [Layer 1-8] → [Layer 9-16] → [Layer 17-24]
    └─ 流水线: Device₀ → Device₁ → Device₂

Dimension 3: Tensor Parallelism (TP)
    ├─ 张量切分: Weight = [W₀ | W₁ | W₂ | W₃]
    └─ 并行计算: Y₀ = X @ W₀,  Y₁ = X @ W₁, ...

Dimension 4: Sequence Parallelism (SP)
    ├─ 序列切分: Seq = [Token 0-511 | Token 512-1023 | ...]
    └─ 分布处理: 每个设备处理部分序列

Dimension 5: Expert Parallelism (EP)
    ├─ 专家切分: [Expert₀, Expert₁] [Expert₂, Expert₃] ...
    └─ 路由分配: Token → Expert (稀疏激活)
```

---

## 通信原语

### 1. 点对点通信 (Point-to-Point)

#### Send / Recv

```python
import torch.distributed as dist

if rank == 0:
    # 发送方
    tensor = torch.tensor([1, 2, 3], device='cuda')
    dist.send(tensor, dst=1)
    
elif rank == 1:
    # 接收方
    tensor = torch.zeros(3, device='cuda')
    dist.recv(tensor, src=0)
    print(f"接收到: {tensor}")
```

**应用场景**: 流水线并行 (Pipeline Parallelism)

```
GPU 0 --send(activation)--> GPU 1 --send(activation)--> GPU 2
GPU 0 <--recv(gradient)--- GPU 1 <--recv(gradient)--- GPU 2
```

### 2. 集合通信 (Collective Communication)

#### Broadcast

```python
# 从rank 0广播数据到所有进程
tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
dist.broadcast(tensor, src=0)

# 所有进程现在都有相同的tensor值
```

**可视化**:
```
初始状态:
Rank 0: [1, 2, 3]
Rank 1: [0, 0, 0]
Rank 2: [0, 0, 0]
Rank 3: [0, 0, 0]

Broadcast后:
Rank 0: [1, 2, 3]
Rank 1: [1, 2, 3]  ← 从Rank 0复制
Rank 2: [1, 2, 3]  ← 从Rank 0复制
Rank 3: [1, 2, 3]  ← 从Rank 0复制
```

#### AllReduce

```python
# 聚合所有进程的tensor并广播结果
tensor = torch.tensor([rank], device='cuda', dtype=torch.float32)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# rank 0: tensor = [0+1+2+3] = [6]
# rank 1: tensor = [0+1+2+3] = [6]
# rank 2: tensor = [0+1+2+3] = [6]
# rank 3: tensor = [0+1+2+3] = [6]
```

**可视化**:
```
初始状态:
Rank 0: [1]     Rank 1: [2]     Rank 2: [3]     Rank 3: [4]
           ↘         ↓         ↓         ↙
                 [1+2+3+4=10]
           ↗         ↑         ↑         ↖
Rank 0: [10]    Rank 1: [10]    Rank 2: [10]    Rank 3: [10]
```

**应用场景**: 数据并行中的梯度同步

```python
# 每个GPU计算自己的梯度
loss.backward()

# 同步梯度 (平均)
for param in model.parameters():
    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    param.grad /= world_size  # 取平均
```

#### AllGather

```python
# 收集所有进程的tensor
tensor = torch.tensor([rank], device='cuda')
tensor_list = [torch.zeros(1, device='cuda') for _ in range(world_size)]
dist.all_gather(tensor_list, tensor)

# 所有进程都有: [0, 1, 2, 3]
```

**可视化**:
```
初始状态:
Rank 0: [0]
Rank 1: [1]
Rank 2: [2]
Rank 3: [3]

AllGather后:
Rank 0: [0, 1, 2, 3]
Rank 1: [0, 1, 2, 3]
Rank 2: [0, 1, 2, 3]
Rank 3: [0, 1, 2, 3]
```

#### ReduceScatter

```python
# 聚合后分散到各进程
input_list = [torch.tensor([i], device='cuda') for i in range(world_size)]
output = torch.zeros(1, device='cuda')
dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)

# rank 0: output = [0+0+0+0] = [0]
# rank 1: output = [1+1+1+1] = [4]
# rank 2: output = [2+2+2+2] = [8]
# rank 3: output = [3+3+3+3] = [12]
```

#### AllToAll

```python
# 所有进程互相交换数据
input_tensor = torch.arange(world_size, device='cuda') * (rank + 1)
output_tensor = torch.zeros(world_size, device='cuda')
dist.all_to_all_single(output_tensor, input_tensor)
```

**可视化** (4个进程):
```
初始状态:
Rank 0: [0, 1, 2, 3]
Rank 1: [0, 2, 4, 6]
Rank 2: [0, 3, 6, 9]
Rank 3: [0, 4, 8, 12]

AllToAll后:
Rank 0: [0, 0, 0, 0]  ← 收集所有进程的第0个元素
Rank 1: [1, 2, 3, 4]  ← 收集所有进程的第1个元素
Rank 2: [2, 4, 6, 8]  ← 收集所有进程的第2个元素
Rank 3: [3, 6, 9, 12] ← 收集所有进程的第3个元素
```

**应用场景**: 专家并行中的Token路由

### 3. 通信原语对比

| 原语 | 通信模式 | 数据量变化 | 应用场景 | 时间复杂度 |
|-----|---------|-----------|---------|-----------|
| **Send/Recv** | 点对点 | 不变 | 流水线并行 | O(n) |
| **Broadcast** | 一对多 | 复制 | 参数同步 | O(n log p) |
| **Reduce** | 多对一 | 聚合 | 梯度收集 | O(n log p) |
| **AllReduce** | 全对全 | 聚合+复制 | 梯度同步 | O(n log p) |
| **Gather** | 多对一 | 拼接 | 结果收集 | O(n) |
| **AllGather** | 全对全 | 拼接 | 张量重组 | O(n p) |
| **Scatter** | 一对多 | 分割 | 数据分发 | O(n) |
| **ReduceScatter** | 全对全 | 聚合+分割 | 张量并行 | O(n log p) |
| **AllToAll** | 全对全 | 重排 | 专家路由 | O(n p) |

其中 n = 数据大小, p = 进程数

---

## 硬件架构

### 1. GPU架构

#### NVIDIA GPU对比

| GPU | 架构 | FP32 TFLOPS | FP16 TFLOPS | 显存 | 带宽 | 发布年份 |
|-----|------|-------------|-------------|------|------|---------|
| V100 | Volta | 15.7 | 125 | 32GB | 900 GB/s | 2017 |
| A100 | Ampere | 19.5 | 312 | 80GB | 2 TB/s | 2020 |
| H100 | Hopper | 67 | 989 | 80GB | 3.35 TB/s | 2022 |
| H200 | Hopper | 67 | 989 | 141GB | 4.8 TB/s | 2023 |

#### 单GPU内存层次

```
寄存器 (Registers)
    ↓ ~1 cycle
共享内存 (Shared Memory) - 每个SM: 64-128 KB
    ↓ ~30 cycles  
L1缓存 (L1 Cache) - 每个SM: 128 KB
    ↓
L2缓存 (L2 Cache) - 全局: 40-60 MB
    ↓ ~200 cycles
HBM显存 (Global Memory) - 40-80 GB
    ↓ ~400 cycles
主机内存 (Host Memory) - 256-512 GB
    ↓ PCIe/NVLink
```

### 2. 节点内互连

#### NVLink

```
NVLink拓扑 (8× A100):

GPU 0 ═══╗       ╔═══ GPU 1
         ║       ║
GPU 2 ═══╬═══════╬═══ GPU 3
         ║       ║
GPU 4 ═══╬═══════╬═══ GPU 5
         ║       ║
GPU 6 ═══╝       ╚═══ GPU 7

特性:
- 带宽: 600 GB/s (每个GPU)
- 延迟: ~1-2 μs
- 全连接: 任意两个GPU直接通信
```

#### NVSwitch

```
NVSwitch架构 (DGX A100):

    ┌─────────────────┐
    │   NVSwitch 0    │
    └─────────────────┘
      ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
      │ │ │ │ │ │ │ │
    GPU0 1 2 3 4 5 6 7
      │ │ │ │ │ │ │ │
      ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
    ┌─────────────────┐
    │   NVSwitch 1    │
    └─────────────────┘

特性:
- 带宽: 600 GB/s × 8 = 4.8 TB/s (总)
- 无阻塞交换
- 支持多GPU通信
```

### 3. 节点间互连

#### InfiniBand

| 代次 | 速率 | 带宽 | 延迟 | 年份 |
|-----|------|------|------|------|
| QDR | 40 Gbps | 5 GB/s | ~1 μs | 2010 |
| FDR | 56 Gbps | 7 GB/s | ~1 μs | 2011 |
| EDR | 100 Gbps | 12.5 GB/s | ~0.5 μs | 2014 |
| HDR | 200 Gbps | 25 GB/s | ~0.5 μs | 2018 |
| NDR | 400 Gbps | 50 GB/s | ~0.5 μs | 2022 |

#### 以太网

| 类型 | 速率 | 带宽 | 延迟 | 应用 |
|-----|------|------|------|------|
| 1GbE | 1 Gbps | 125 MB/s | ~100 μs | 小规模 |
| 10GbE | 10 Gbps | 1.25 GB/s | ~10 μs | 中等规模 |
| 25GbE | 25 Gbps | 3.125 GB/s | ~5 μs | 中等规模 |
| 100GbE | 100 Gbps | 12.5 GB/s | ~2 μs | 大规模 |
| 400GbE | 400 Gbps | 50 GB/s | ~2 μs | 超大规模 |

### 4. 存储系统

```
分布式存储层次:

本地NVMe SSD
    ├── 容量: 1-4 TB
    ├── 带宽: 3-7 GB/s
    └── 延迟: ~100 μs

网络文件系统 (NFS/Lustre)
    ├── 容量: 1-100 PB
    ├── 带宽: 10-100 GB/s
    └── 延迟: ~1-10 ms

对象存储 (S3/Ceph)
    ├── 容量: 无限
    ├── 带宽: 1-10 GB/s
    └── 延迟: ~10-100 ms
```

---

## 软件栈

### 1. 通信库

#### NCCL (NVIDIA Collective Communications Library)

```python
# NCCL后端 (GPU通信)
dist.init_process_group(
    backend='nccl',
    init_method='env://',
    world_size=world_size,
    rank=rank
)

特性:
✓ 针对NVIDIA GPU优化
✓ 支持NVLink/NVSwitch
✓ 多种集合通信原语
✓ 自动选择最优算法
```

#### Gloo

```python
# Gloo后端 (CPU通信)
dist.init_process_group(
    backend='gloo',
    init_method='tcp://10.0.0.1:23456',
    world_size=world_size,
    rank=rank
)

特性:
✓ CPU-CPU通信
✓ 跨平台 (Linux/Mac/Windows)
✓ 适合小规模
```

#### MPI

```python
# MPI后端
dist.init_process_group(
    backend='mpi',
    world_size=world_size,
    rank=rank
)

特性:
✓ 成熟稳定
✓ 支持异构集群
✓ 丰富的通信原语
✓ 需要MPI实现 (OpenMPI/MPICH)
```

### 2. 深度学习框架

```
框架栈:

应用层:
    ├── Transformers (Hugging Face)
    ├── Detectron2 (Facebook)
    └── MMDetection (OpenMMLab)

分布式训练框架:
    ├── DeepSpeed (Microsoft)
    ├── Megatron-LM (NVIDIA)
    ├── FairScale (Meta)
    └── Colossal-AI (HPC-AI Tech)

深度学习框架:
    ├── PyTorch
    ├── TensorFlow
    └── JAX

通信库:
    ├── NCCL
    ├── Gloo
    └── MPI

硬件驱动:
    ├── CUDA
    ├── ROCm (AMD)
    └── oneAPI (Intel)
```

---

## 性能指标

### 1. 计算性能

#### FLOPS (浮点运算每秒)

```
单位:
- FLOPS:  10⁰
- KFLOPS: 10³
- MFLOPS: 10⁶
- GFLOPS: 10⁹
- TFLOPS: 10¹²
- PFLOPS: 10¹⁵
- EFLOPS: 10¹⁸

A100 80GB 理论峰值:
- FP64: 9.7 TFLOPS
- FP32: 19.5 TFLOPS
- TF32: 156 TFLOPS
- FP16: 312 TFLOPS
- INT8: 624 TOPS
```

#### MFU (Model FLOPs Utilization)

```python
def calculate_mfu(observed_flops, peak_flops):
    """
    MFU = 实际FLOPS / 理论峰值FLOPS
    
    好的MFU:
    - 训练: 40-60%
    - 推理: 60-80%
    """
    return (observed_flops / peak_flops) * 100

# 示例
observed = 125  # TFLOPS
peak = 312      # TFLOPS (A100 FP16)
mfu = calculate_mfu(observed, peak)  # 40.06%
```

### 2. 通信性能

#### 带宽

```
带宽 = 数据量 / 传输时间

示例:
- 数据量: 1 GB
- 时间: 0.01 秒
- 带宽: 100 GB/s
```

#### 延迟

```
延迟组成:
    启动延迟 (α) + 传输时间 (β × 数据量)

典型值:
- NVLink:  α ≈ 1-2 μs,   β ≈ 1/600 GB/s
- PCIe 4.0: α ≈ 10 μs,   β ≈ 1/32 GB/s
- InfiniBand HDR: α ≈ 1 μs, β ≈ 1/200 Gbps
- Ethernet 100G: α ≈ 10 μs, β ≈ 1/100 Gbps
```

### 3. 扩展性

#### 强扩展 (Strong Scaling)

```
固定问题规模，增加资源

效率 = (单GPU时间 × GPU数) / 多GPU时间

理想: 效率 = 100% (线性加速)
实际: 效率 = 70-90%
```

#### 弱扩展 (Weak Scaling)

```
固定每GPU负载，增加资源和问题规模

效率 = 单GPU时间 / 多GPU时间

理想: 效率 = 100%
实际: 效率 = 85-95%
```

---

## 常见挑战

### 1. 通信开销

```
问题:
- 通信时间占比过高 (>30%)
- 影响扩展性

解决方案:
✓ 计算-通信重叠
✓ 梯度压缩
✓ 通信融合
✓ 混合并行
```

### 2. 负载不均衡

```
问题:
- 不同GPU工作量不同
- 出现等待和空闲

解决方案:
✓ 动态负载均衡
✓ 更细粒度的分割
✓ 异步执行
```

### 3. 内存墙

```
问题:
- 模型太大无法装入显存
- 激活值占用大量内存

解决方案:
✓ 模型并行 (TP/PP)
✓ 梯度检查点
✓ CPU offload
✓ 混合精度训练
```

### 4. 同步开销

```
问题:
- 全局同步导致等待
- 流水线气泡

解决方案:
✓ 异步SGD
✓ 局部SGD
✓ 1F1B调度
✓ 微批次流水线
```

---

## 下一步

完成基础概念学习后，建议按以下顺序深入：

1. [数据并行](01-data-parallelism.md) - 最简单，从这里开始
2. [流水线并行](02-pipeline-parallelism.md) - 理解模型分割
3. [张量并行](03-tensor-parallelism.md) - 深入算子级并行
4. [序列并行](04-sequence-parallelism.md) - 处理长序列
5. [专家并行](05-expert-parallelism.md) - MoE架构
6. [混合并行](06-hybrid-parallelism.md) - 组合策略

---

## 参考资料

### 论文
- [Megatron-LM](https://arxiv.org/abs/1909.08053) - NVIDIA, 2019
- [ZeRO](https://arxiv.org/abs/1910.02054) - Microsoft, 2020
- [GPipe](https://arxiv.org/abs/1811.06965) - Google, 2019

### 文档
- [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [DeepSpeed Tutorials](https://www.deepspeed.ai/tutorials/)

### 书籍
- "Distributed Deep Learning" - Jeff Daily et al.
- "High Performance Python" - Micha Gorelick & Ian Ozsvald

---

<div align="center">
  <strong>准备好开始分布式训练之旅了吗？🚀</strong>
</div>