# 数据并行 (Data Parallelism)

## 📋 目录

- [核心原理](#核心原理)
- [实现方式](#实现方式)
- [DDP详解](#ddp详解)
- [FSDP详解](#fsdp详解)
- [性能优化](#性能优化)
- [实战案例](#实战案例)
- [常见问题](#常见问题)

---

## 核心原理

### 什么是数据并行？

数据并行是**最简单、最常用**的并行策略：

```
基本思想:
1. 每个GPU持有完整模型的副本
2. 将训练数据分成多个batch
3. 每个GPU处理不同的数据batch
4. 通过AllReduce同步梯度
5. 所有GPU使用相同的梯度更新模型
```

### 工作流程

```python
"""
数据并行训练流程 (4个GPU示例)
"""

# 步骤1: 初始化 - 每个GPU复制完整模型
GPU 0: Model (W₀ = W_init)
GPU 1: Model (W₁ = W_init)  # 与GPU 0相同
GPU 2: Model (W₂ = W_init)  # 与GPU 0相同
GPU 3: Model (W₃ = W_init)  # 与GPU 0相同

# 步骤2: 数据分发 - 不同GPU处理不同数据
GPU 0: Data batch [0-31]
GPU 1: Data batch [32-63]
GPU 2: Data batch [64-95]
GPU 3: Data batch [96-127]

# 步骤3: 前向传播 - 各自独立计算
GPU 0: loss₀ = forward(Data[0-31])
GPU 1: loss₁ = forward(Data[32-63])
GPU 2: loss₂ = forward(Data[64-95])
GPU 3: loss₃ = forward(Data[96-127])

# 步骤4: 反向传播 - 各自独立计算梯度
GPU 0: grad₀ = backward(loss₀)
GPU 1: grad₁ = backward(loss₁)
GPU 2: grad₂ = backward(loss₂)
GPU 3: grad₃ = backward(loss₃)

# 步骤5: 梯度同步 - AllReduce求平均
AllReduce(grad₀, grad₁, grad₂, grad₃)
→ grad_avg = (grad₀ + grad₁ + grad₂ + grad₃) / 4

# 步骤6: 参数更新 - 所有GPU使用相同梯度
GPU 0: W₀ = W₀ - lr × grad_avg
GPU 1: W₁ = W₁ - lr × grad_avg
GPU 2: W₂ = W₂ - lr × grad_avg
GPU 3: W₃ = W₃ - lr × grad_avg

# 结果: 所有GPU的模型参数保持同步
```

### 关键特点

| 特性 | 说明 |
|-----|------|
| ✅ **简单直观** | 最容易理解和实现的并行方式 |
| ✅ **高效率** | 通信开销相对较小 |
| ✅ **线性加速** | 理想情况下可达到近线性加速 |
| ❌ **内存限制** | 每个GPU必须能装下完整模型 |
| ❌ **通信瓶颈** | 大模型梯度同步开销大 |

---

## 实现方式

### 1. DataParallel (DP) - 单机多卡

```python
import torch
import torch.nn as nn

# 简单包装即可
model = nn.Linear(1000, 1000)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.cuda()

# 训练
data = torch.randn(128, 1000).cuda()
output = model(data)
```

**特点**:
- ✅ 最简单，一行代码启用
- ❌ 单进程多线程，GIL限制
- ❌ GPU 0负载重（参数服务器）
- ❌ 不支持多节点
- ⚠️ **已不推荐使用**，建议用DDP

### 2. DistributedDataParallel (DDP) - 推荐

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group(backend='nccl')
rank = dist.get_rank()

# 创建模型并移到对应GPU
model = MyModel().cuda(rank)
model = DDP(model, device_ids=[rank])

# 训练
for data, target in dataloader:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

**特点**:
- ✅ 多进程，无GIL限制
- ✅ 支持多节点训练
- ✅ 通信高效（Ring AllReduce）
- ✅ 负载均衡
- ✅ 社区标准，生产级

### 3. Fully Sharded Data Parallel (FSDP) - 最新

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 创建模型
model = MyLargeModel()

# FSDP包装
model = FSDP(
    model,
    auto_wrap_policy=my_auto_wrap_policy,
    mixed_precision=mixed_precision_policy,
)

# 训练（与DDP相同）
for data, target in dataloader:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

**特点**:
- ✅ ZeRO-3风格分片
- ✅ 大幅减少内存占用
- ✅ 支持超大模型
- ✅ PyTorch原生支持
- ⚠️ 通信开销略高于DDP

---

## DDP详解

### 架构设计

```
DDP架构:

进程0 (GPU 0)          进程1 (GPU 1)          进程2 (GPU 2)
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Model副本   │      │  Model副本   │      │  Model副本   │
│   + Grad     │      │   + Grad     │      │   + Grad     │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                     │
       └─────────────────────┴─────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  AllReduce通信   │
                    │   (NCCL/Gloo)   │
                    └─────────────────┘
```

### 完整代码示例

```python
"""
完整的DDP训练脚本
运行: torchrun --nproc_per_node=4 train_ddp.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def setup(rank, world_size):
    """初始化分布式环境"""
    # 环境变量由torchrun自动设置
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',  # GPU用nccl，CPU用gloo
        rank=rank,
        world_size=world_size
    )
    
    # 设置当前进程使用的GPU
    torch.cuda.set_device(rank)


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


class ToyModel(nn.Module):
    """示例模型"""
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def train_ddp(rank, world_size):
    """DDP训练函数"""
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)
    
    # 创建模型并移到GPU
    model = ToyModel().to(rank)
    
    # DDP包装
    ddp_model = DDP(model, device_ids=[rank])
    
    # 优化器和损失函数
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # 数据集和DataLoader
    # 重要：使用DistributedSampler确保每个进程看到不同数据
    dataset = torch.randn(1000, 10)  # 示例数据
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler
    )
    
    # 训练循环
    for epoch in range(10):
        # 设置epoch以shuffle数据
        sampler.set_epoch(epoch)
        
        for batch_idx, data in enumerate(dataloader):
            data = data.to(rank)
            targets = torch.randn(data.size(0), 5).to(rank)
            
            # 前向传播
            outputs = ddp_model(data)
            loss = loss_fn(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # DDP自动处理梯度同步
            optimizer.step()
            
            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    
    # 使用torch.multiprocessing启动多进程
    import torch.multiprocessing as mp
    mp.spawn(
        train_ddp,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
```

### DDP通信机制

#### Gradient Bucketing

```python
"""
DDP将梯度分组成buckets进行通信
"""

# 默认bucket大小: 25 MB
model = DDP(
    model,
    device_ids=[rank],
    bucket_cap_mb=25  # 可调整
)

# 通信流程:
# 1. 反向传播开始
# 2. 当一个bucket的所有梯度ready时
# 3. 立即启动AllReduce (不等所有梯度)
# 4. 实现计算和通信重叠
```

**可视化**:
```
时间线:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Backward Pass:    ████████████████████████
                  ↓   ↓   ↓   ↓   ↓   ↓
Bucket 0 Ready:   ■
AllReduce 0:      ═══════╗
Bucket 1 Ready:       ■   ║
AllReduce 1:          ═══════╗
Bucket 2 Ready:           ■   ║
AllReduce 2:              ═══════╗
                                  ║
计算和通信重叠:     ████▓▓▓▓▓▓▓▓▓▓║
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### Gradient Accumulation

```python
"""
梯度累积 - 模拟更大的batch size
"""

accumulation_steps = 4  # 累积4个step
optimizer.zero_grad()

for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target)
    
    # 归一化loss
    loss = loss / accumulation_steps
    loss.backward()
    
    # 每accumulation_steps个step更新一次
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 优点:
# 1. 减少通信频率 (4倍)
# 2. 模拟更大batch size
# 3. 节省显存
```

### 混合精度训练

```python
"""
使用AMP (Automatic Mixed Precision)
"""

from torch.cuda.amp import autocast, GradScaler

model = DDP(model, device_ids=[rank])
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    # 自动混合精度
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # 缩放loss并反向传播
    scaler.scale(loss).backward()
    
    # 梯度裁剪（可选）
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 更新参数
    scaler.step(optimizer)
    scaler.update()

# 好处:
# - 2倍加速
# - 减少50%显存
# - 几乎无精度损失
```

---

## FSDP详解

### ZeRO原理

FSDP实现了ZeRO (Zero Redundancy Optimizer) Stage 3:

```
传统DDP内存占用:
┌─────────────────────────────────────┐
│ GPU 0: 参数 + 梯度 + 优化器状态     │ 100%
│ GPU 1: 参数 + 梯度 + 优化器状态     │ 100%
│ GPU 2: 参数 + 梯度 + 优化器状态     │ 100%
│ GPU 3: 参数 + 梯度 + 优化器状态     │ 100%
└─────────────────────────────────────┘

ZeRO-3 (FSDP)内存占用:
┌─────────────────────────────────────┐
│ GPU 0: 参数₀ + 梯度₀ + 优化器₀      │ 25%
│ GPU 1: 参数₁ + 梯度₁ + 优化器₁      │ 25%
│ GPU 2: 参数₂ + 梯度₂ + 优化器₂      │ 25%
│ GPU 3: 参数₃ + 梯度₃ + 优化器₃      │ 25%
└─────────────────────────────────────┘

内存节省: 4× (对于4个GPU)
```

### FSDP工作流程

```python
"""
FSDP前向传播
"""

# 步骤1: All-Gather参数
for layer in model:
    # 收集所有GPU的参数分片
    full_params = all_gather(layer.params_shard)
    
    # 步骤2: 前向计算
    output = layer.forward(input, full_params)
    
    # 步骤3: 释放完整参数（保留分片）
    del full_params

"""
FSDP反向传播
"""

# 步骤1: All-Gather参数
for layer in reversed(model):
    full_params = all_gather(layer.params_shard)
    
    # 步骤2: 反向计算
    grad = layer.backward(grad_output, full_params)
    
    # 步骤3: Reduce-Scatter梯度
    layer.grad_shard = reduce_scatter(grad)
    
    # 步骤4: 释放完整参数和梯度
    del full_params, grad
```

### FSDP完整示例

```python
"""
FSDP训练示例
"""

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)


def setup_fsdp():
    """配置FSDP策略"""
    
    # 混合精度策略
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,      # 参数用FP16
        reduce_dtype=torch.float16,     # 梯度reduce用FP16
        buffer_dtype=torch.float32,     # buffer用FP32
    )
    
    # Sharding策略
    sharding_strategy = ShardingStrategy.FULL_SHARD  # ZeRO-3
    # 其他选项:
    # - SHARD_GRAD_OP: ZeRO-2 (分片梯度和优化器)
    # - NO_SHARD: 类似DDP
    # - HYBRID_SHARD: 节点内全分片，节点间复制
    
    return mixed_precision_policy, sharding_strategy


class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        # Attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feedforward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


def build_model():
    """构建大模型"""
    dim = 2048
    num_layers = 48
    num_heads = 32
    
    # 使用enable_wrap自动包装
    mixed_precision, sharding_strategy = setup_fsdp()
    
    # 自动包装策略: 根据参数量决定
    auto_wrap_policy = size_based_auto_wrap_policy(
        min_num_params=1e8  # 100M参数以上的模块独立分片
    )
    
    with enable_wrap(
        wrapper_cls=FSDP,
        mixed_precision=mixed_precision,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
    ):
        model = nn.Sequential(
            nn.Embedding(50000, dim),
            *[wrap(TransformerBlock(dim, num_heads)) for _ in range(num_layers)],
            nn.Linear(dim, 50000)
        )
    
    # 整个模型最外层再包装一次
    model = FSDP(
        model,
        mixed_precision=mixed_precision,
        sharding_strategy=sharding_strategy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # 预取优化
    )
    
    return model


def train_fsdp():
    """训练"""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    
    # 创建模型
    model = build_model().cuda(rank)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 训练
    for epoch in range(10):
        for data, target in dataloader:
            data, target = data.cuda(rank), target.cuda(rank)
            
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if rank == 0:
                print(f"Loss: {loss.item():.4f}")
    
    dist.destroy_process_group()
```

### FSDP vs DDP 对比

| 特性 | DDP | FSDP |
|-----|-----|------|
| 内存效率 | 低 (每GPU完整模型) | 高 (分片) |
| 通信量 | 中等 | 较高 |
| 实现复杂度 | 简单 | 中等 |
| 最大模型 | ~10B (8×A100) | ~100B+ |
| 训练速度 | 快 | 略慢10-20% |
| 适用场景 | 小中型模型 | 超大模型 |

---

## 性能优化

### 1. 通信优化

#### 使用混合精度

```python
# FP16可以减少2倍通信量
scaler = GradScaler()

with autocast():
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 梯度压缩

```python
# PowerSGD梯度压缩
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook

model = DDP(model, device_ids=[rank])

# 注册通信hook
state = powerSGD_hook.PowerSGDState(
    process_group=None,
    matrix_approximation_rank=4,  # 压缩秩
    start_powerSGD_iter=10,  # 从第10次迭代开始压缩
)
model.register_comm_hook(state, powerSGD_hook.powerSGD_hook)

# 可以减少5-10倍通信量
```

### 2. 计算优化

#### 使用编译器优化

```python
# PyTorch 2.0+ 支持torch.compile
model = torch.compile(model, mode="reduce-overhead")
model = DDP(model, device_ids=[rank])

# 可以加速10-30%
```

#### CUDA Graphs

```python
# 对于固定输入shape的模型
use_cuda_graph = True
if use_cuda_graph:
    # Warmup
    for _ in range(10):
        output = model(sample_input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # Capture
    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad()
    with torch.cuda.graph(g):
        static_output = model(static_input)
        static_loss = criterion(static_output, static_target)
        static_loss.backward()
    
    # Replay
    for data, target in dataloader:
        static_input.copy_(data)
        static_target.copy_(target)
        g.replay()
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 数据加载优化

```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,          # 多进程加载
    pin_memory=True,        # 锁页内存
    prefetch_factor=2,      # 预取
    persistent_workers=True # 保持worker进程
)
```

---

## 实战案例

### 案例1: GPT-2训练

```python
"""
使用DDP训练GPT-2 (1.5B参数)
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def train_gpt2():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 加载模型
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    model = model.cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # 数据
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    # ... 数据加载代码 ...
    
    # 训练
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            input_ids = batch['input_ids'].cuda(rank)
            labels = input_ids.clone()
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    # 保存模型 (只在rank 0保存)
    if rank == 0:
        model.module.save_pretrained('./gpt2-finetuned')
```

### 案例2: 超参数搜索

```python
"""
并行超参数搜索
"""

def hyperparameter_search():
    # 每个GPU测试不同的超参数
    rank = dist.get_rank()
    
    # 超参数网格
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4]
    lr = learning_rates[rank]
    
    # 训练
    model = create_model()
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    val_loss = train(model, optimizer)
    
    # 收集所有结果
    all_results = [None] * dist.get_world_size()
    dist.all_gather_object(all_results, {'lr': lr, 'val_loss': val_loss})
    
    if rank == 0:
        best = min(all_results, key=lambda x: x['val_loss'])
        print(f"Best LR: {best['lr']}, Loss: {best['val_loss']}")
```

---

## 常见问题

### Q1: 为什么DDP比DP快？

**A**: 多进程 vs 单进程多线程
- DDP: 每个GPU独立的Python进程，无GIL
- DP: 单进程，受Python GIL限制

### Q2: Batch Size如何设置？

**A**: 全局Batch Size = local_batch_size × world_size

```python
# 示例: 4 GPUs, 全局BS=128
local_batch_size = 128 // 4  # = 32
global_batch_size = local_batch_size * 4  # = 128
```

### Q3: 如何保存和加载模型？

```python
# 保存 (只在rank 0)
if rank == 0:
    torch.save(model.module.state_dict(), 'model.pt')

# 加载 (所有rank)
dist.barrier()  # 等待保存完成
model.module.load_state_dict(torch.load('model.pt'))
```

### Q4: OOM怎么办？

**解决方案**:
1. 减小local batch size
2. 使用梯度累积
3. 启用梯度检查点
4. 使用FSDP
5. 使用混合精度

```python
# 梯度检查点示例
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def forward(self, x):
        # 对大层使用checkpoint
        x = checkpoint(self.big_layer, x)
        return x
```

### Q5: 多节点如何运行？

```bash
# 节点0 (master)
torchrun \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    --nproc_per_node=8 \
    train.py

# 节点1
torchrun \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    --nproc_per_node=8 \
    train.py
```

---

## 总结

### 数据并行选择指南

```
决策树:

模型能放入单GPU? 
├─ Yes → 使用DDP (简单高效)
└─ No → 继续判断
    
    模型 < 50B?
    ├─ Yes → 使用FSDP (ZeRO-3)
    └─ No → 考虑模型并行 (TP/PP)
```

### 最佳实践

✅ 首选DDP - 适合90%的场景
✅ 大模型用FSDP - 参数>10B时
✅ 启用混合精度 - 免费2倍加速
✅ 合理设置batch size - 保证GPU利用率>80%
✅ 使用梯度累积 - 模拟更大batch
✅ 监控通信开销 - 应<20%训练时间