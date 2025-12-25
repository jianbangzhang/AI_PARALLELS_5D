# 流水线并行 (Pipeline Parallelism)

## 📋 目录

- [核心原理](#核心原理)
- [调度策略](#调度策略)
- [GPipe详解](#gpipe详解)
- [PipeDream详解](#pipedream详解)
- [1F1B调度](#1f1b调度)
- [实现细节](#实现细节)
- [性能优化](#性能优化)
- [实战案例](#实战案例)

---

## 核心原理

### 什么是流水线并行？

```
基本思想:
将深度神经网络按层切分成多个阶段(Stage)
每个阶段在不同的设备上执行
数据像流水线一样在设备间传递
```

### 为什么需要流水线并行？

**问题**: 模型太深，单个GPU装不下

```python
# 例如: GPT-3 (96层 Transformer)
Model:
    Embedding Layer
    ├─ Transformer Block 1-24   → GPU 0
    ├─ Transformer Block 25-48  → GPU 1  
    ├─ Transformer Block 49-72  → GPU 2
    └─ Transformer Block 73-96  → GPU 3
    Output Layer

单个GPU内存: 不够！
解决方案: 流水线并行
```

### 朴素流水线的问题

```
时间线 (4个阶段，1个batch):

Stage 0:  [F0]─────────[B0]─────────
Stage 1:  ────[F0]─────────[B0]─────
Stage 2:  ────────[F0]─────────[B0]─
Stage 3:  ────────────[F0]─────────[B0]

F = Forward, B = Backward
问题: 大量空闲时间(气泡) ≈ 75%
```

---

## 调度策略

### 1. GPipe - 微批次流水线

**核心思想**: 将batch切分成多个micro-batch，填充流水线

```
时间线 (4个阶段，8个micro-batch):

       Micro-batch:  0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7
       ──────────────────────────────────────────────────────────────
Stage 0:             F0 F1 F2 F3 F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7
Stage 1:                F0 F1 F2 F3 F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7
Stage 2:                   F0 F1 F2 F3 F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7
Stage 3:                      F0 F1 F2 F3 F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7

特点:
✓ 同步训练
✓ 气泡率降低: (K-1)/(K-1+M) = 3/11 ≈ 27%
✓ 简单实现
✗ 内存峰值高 (需存储所有micro-batch激活)
```

**气泡率计算**:
```
K = 阶段数
M = micro-batch数量
气泡率 = (K-1) / (K-1 + M)

示例:
- 4阶段, 8 micro-batch: 气泡率 = 3/11 ≈ 27%
- 4阶段, 16 micro-batch: 气泡率 = 3/19 ≈ 16%
- 8阶段, 16 micro-batch: 气泡率 = 7/23 ≈ 30%
```

### 2. PipeDream - 异步流水线

**核心思想**: 不同micro-batch使用不同版本的权重

```
时间线:

Stage 0:  F0─F1─F2─F3─B0─B1─B2─B3─U0─F4─F5─...
Stage 1:  ───F0─F1─F2─F3─B0─B1─B2─B3─U0─F4─...
Stage 2:  ──────F0─F1─F2─F3─B0─B1─B2─B3─U0─...
Stage 3:  ─────────F0─F1─F2─F3─B0─B1─B2─B3─U0

U = Update

特点:
✓ 气泡率更低
✓ 内存占用低
✗ 不同版本权重(权重过时问题)
✗ 实现复杂
```

### 3. 1F1B (One Forward One Backward)

**核心思想**: 交替执行前向和反向，平衡内存

```
时间线 (4阶段, 8 micro-batch):

       微批次:      0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7
       ────────────────────────────────────────────────────────────
Stage 0:           F0 F1 F2 F3 F4 B0 F5 B1 F6 B2 F7 B3    B4    B5 B6 B7
Stage 1:              F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4    B5 B6 B7
Stage 2:                 F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 B6 B7
Stage 3:                    F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 B7

阶段:
1. Warmup:  填充流水线 (纯前向)
2. 1F1B:    交替前向/反向
3. Cooldown: 排空流水线 (纯反向)

特点:
✓ 内存占用低 (只保存K个micro-batch激活)
✓ 气泡率与GPipe相同
✓ 实现相对简单
→ 工业界主流选择!
```

---

## GPipe详解

### 架构设计

```python
"""
GPipe核心组件
"""

class GPipe:
    def __init__(self, model, balance, chunks):
        """
        Args:
            model: 完整模型
            balance: 每个阶段的层数 [24, 24, 24, 24]
            chunks: micro-batch数量
        """
        self.stages = self.partition_model(model, balance)
        self.chunks = chunks
    
    def partition_model(self, model, balance):
        """将模型分割成多个阶段"""
        stages = []
        start = 0
        for num_layers in balance:
            end = start + num_layers
            stage = nn.Sequential(*list(model.children())[start:end])
            stages.append(stage)
            start = end
        return stages
```

### 完整实现

```python
"""
GPipe完整实现
"""

import torch
import torch.nn as nn
from torch.distributed.pipeline.sync import Pipe


class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.ff(x))
        return x


def create_gpipe_model(num_layers=96, dim=2048, heads=32, devices=4):
    """创建GPipe模型"""
    
    # 创建完整模型
    layers = []
    layers.append(nn.Embedding(50000, dim))
    for _ in range(num_layers):
        layers.append(TransformerBlock(dim, heads))
    layers.append(nn.Linear(dim, 50000))
    
    model = nn.Sequential(*layers)
    
    # 计算每个设备的层数
    layers_per_device = (num_layers + 2) // devices
    balance = [layers_per_device] * devices
    
    # 调整最后一个设备
    balance[-1] = (num_layers + 2) - sum(balance[:-1])
    
    # 创建Pipe模型
    model = Pipe(
        model,
        balance=balance,
        chunks=8,  # 8个micro-batch
        checkpoint='always'  # 激活检查点
    )
    
    return model


def train_gpipe():
    """GPipe训练"""
    
    # 创建模型
    model = create_gpipe_model(
        num_layers=96,
        dim=2048, 
        heads=32,
        devices=4
    )
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练循环
    for epoch in range(10):
        for batch in dataloader:
            # 输入数据
            input_ids = batch['input_ids']  # [batch_size, seq_len]
            
            # 前向传播
            # GPipe自动处理micro-batch切分
            output = model(input_ids).local_value()
            
            # 计算损失
            loss = F.cross_entropy(
                output.view(-1, output.size(-1)),
                input_ids.view(-1)
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# 内存优化: 激活检查点
"""
不使用检查点:
- 需要存储所有micro-batch的激活
- 内存 = M × activation_size

使用检查点:
- 重新计算激活而不存储
- 内存 = √(M) × activation_size
- 时间增加 ~33%
"""
```

### GPipe关键特性

#### 1. 激活检查点

```python
# 自动激活检查点
pipe_model = Pipe(
    model,
    balance=balance,
    chunks=8,
    checkpoint='always'  # 'always', 'except_last', 'never'
)

"""
检查点策略:
- always: 所有层都checkpoint (最省内存)
- except_last: 最后一个stage不checkpoint
- never: 不使用checkpoint (最快)
"""
```

#### 2. Micro-batch调度

```python
def forward(self, input):
    """GPipe前向传播伪代码"""
    
    # 切分成micro-batches
    micro_batches = torch.chunk(input, self.chunks, dim=0)
    
    # Warmup阶段
    outputs = []
    for i in range(self.num_stages):
        for j in range(i + 1):
            micro_batch = micro_batches[j]
            # 前向传播到stage i
            output = self.stages[i](micro_batch)
            if i == self.num_stages - 1:
                outputs.append(output)
    
    # 稳定阶段
    for i in range(self.num_stages, len(micro_batches)):
        micro_batch = micro_batches[i]
        # 前向
        output = self.forward_stage(micro_batch)
        outputs.append(output)
        
        # 反向 (与前向重叠)
        self.backward_stage(i - self.num_stages)
    
    # Cooldown阶段
    for i in range(len(micro_batches) - self.num_stages):
        self.backward_stage(len(micro_batches) - self.num_stages + i)
    
    return torch.cat(outputs, dim=0)
```

---

## PipeDream详解

### 权重版本管理

```python
"""
PipeDream的核心挑战: 不同micro-batch使用不同权重版本
"""

class PipeDream:
    def __init__(self, stages, num_versions):
        self.stages = stages
        # 每个阶段维护多个权重版本
        self.weight_versions = [
            [copy.deepcopy(stage.state_dict()) 
             for _ in range(num_versions)]
            for stage in stages
        ]
    
    def forward(self, micro_batch, version_id):
        """使用特定版本的权重"""
        # 加载对应版本的权重
        self.load_version(version_id)
        
        # 前向传播
        output = self.stage(micro_batch)
        
        return output, version_id
    
    def backward(self, grad, version_id):
        """使用对应版本的权重进行反向"""
        # 加载对应版本
        self.load_version(version_id)
        
        # 反向传播
        grad_input = self.stage.backward(grad)
        
        return grad_input
```

### 权重更新策略

```python
"""
PipeDream权重更新
"""

# 策略1: Weight Stashing
def weight_stashing():
    """
    为每个in-flight micro-batch保存一份权重
    
    内存开销: K × weight_size
    K = pipeline depth
    """
    versions = {}
    
    for micro_batch_id in range(num_micro_batches):
        # 保存当前权重版本
        version_id = micro_batch_id % pipeline_depth
        versions[version_id] = copy.deepcopy(model.state_dict())
        
        # 使用对应版本进行前向/反向
        output = forward(micro_batch, version=versions[version_id])
        loss.backward()
        
        # 更新权重
        optimizer.step()


# 策略2: Vertical Sync
def vertical_sync():
    """
    定期同步所有stage的权重
    
    减少版本不一致
    """
    sync_interval = 4  # 每4个micro-batch同步一次
    
    if step % sync_interval == 0:
        # 广播stage 0的权重到所有stage
        for stage_id in range(1, num_stages):
            sync_weights(stage_0, stage_id)
```

---

## 1F1B调度

### 详细时间线

```python
"""
1F1B调度详解 (4 stages, 8 micro-batches)
"""

def one_f_one_b_schedule():
    num_stages = 4
    num_microbatches = 8
    
    # 阶段1: Warmup (填充流水线)
    # 每个stage依次开始，执行 (num_stages - stage_id - 1) 次前向
    for stage_id in range(num_stages):
        warmup_iters = num_stages - stage_id - 1
        for i in range(warmup_iters):
            forward(stage_id, microbatch_id=i)
    
    # 阶段2: 1F1B (稳定阶段)
    # 交替执行1次前向和1次反向
    num_1f1b_iters = num_microbatches - (num_stages - 1)
    for i in range(num_1f1b_iters):
        for stage_id in range(num_stages):
            # 前向
            forward(stage_id, microbatch_id=warmup_iters + i)
            # 反向
            backward(stage_id, microbatch_id=i)
    
    # 阶段3: Cooldown (排空流水线)
    # 只执行反向传播
    for stage_id in range(num_stages):
        cooldown_iters = num_stages - stage_id - 1
        for i in range(cooldown_iters):
            backward(stage_id, microbatch_id=num_1f1b_iters + i)
```

### 完整实现

```python
"""
1F1B完整实现
"""

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc


class PipelineStage(nn.Module):
    """流水线阶段"""
    
    def __init__(self, stage_id, model, num_stages):
        super().__init__()
        self.stage_id = stage_id
        self.model = model
        self.num_stages = num_stages
        
        # 存储激活值 (用于反向传播)
        self.activations = {}
    
    def forward_stage(self, micro_batch_id, input_tensor):
        """阶段前向传播"""
        
        # 前向计算
        with torch.enable_grad():
            output = self.model(input_tensor)
            
            # 保存激活值
            if self.stage_id < self.num_stages - 1:
                self.activations[micro_batch_id] = output.detach()
                output.requires_grad = True
        
        # 发送到下一阶段
        if self.stage_id < self.num_stages - 1:
            next_stage = self.stage_id + 1
            rpc.rpc_async(
                f'worker{next_stage}',
                forward_stage,
                args=(micro_batch_id, output)
            )
        
        return output
    
    def backward_stage(self, micro_batch_id, grad_output):
        """阶段反向传播"""
        
        # 获取保存的激活值
        activation = self.activations.pop(micro_batch_id)
        
        # 反向计算
        activation.backward(grad_output)
        grad_input = activation.grad
        
        # 发送到上一阶段
        if self.stage_id > 0:
            prev_stage = self.stage_id - 1
            rpc.rpc_async(
                f'worker{prev_stage}',
                backward_stage,
                args=(micro_batch_id, grad_input)
            )
        
        return grad_input


def train_1f1b(stage_id, model, num_stages, num_microbatches):
    """1F1B训练"""
    
    stage = PipelineStage(stage_id, model, num_stages)
    optimizer = torch.optim.Adam(stage.parameters())
    
    # Warmup阶段
    warmup_iters = num_stages - stage_id - 1
    for i in range(warmup_iters):
        if stage_id == 0:
            # 第一个stage从dataloader获取数据
            input_tensor = next(dataloader)
        else:
            # 其他stage接收上一阶段的输出
            input_tensor = receive_activation(i)
        
        stage.forward_stage(i, input_tensor)
    
    # 1F1B阶段
    num_1f1b_iters = num_microbatches - (num_stages - 1)
    for i in range(num_1f1b_iters):
        # 1 Forward
        micro_batch_id = warmup_iters + i
        if stage_id == 0:
            input_tensor = next(dataloader)
        else:
            input_tensor = receive_activation(micro_batch_id)
        
        stage.forward_stage(micro_batch_id, input_tensor)
        
        # 1 Backward
        if stage_id == num_stages - 1:
            # 最后一个stage计算损失
            grad_output = compute_loss_gradient(i)
        else:
            grad_output = receive_gradient(i)
        
        stage.backward_stage(i, grad_output)
        
        # 更新参数
        optimizer.step()
        optimizer.zero_grad()
    
    # Cooldown阶段
    cooldown_iters = num_stages - stage_id - 1
    for i in range(cooldown_iters):
        micro_batch_id = num_1f1b_iters + i
        
        if stage_id == num_stages - 1:
            grad_output = compute_loss_gradient(micro_batch_id)
        else:
            grad_output = receive_gradient(micro_batch_id)
        
        stage.backward_stage(micro_batch_id, grad_output)
        optimizer.step()
        optimizer.zero_grad()
```

### 内存分析

```python
"""
1F1B vs GPipe 内存对比
"""

# GPipe内存占用
gpipe_memory = num_microbatches * activation_size_per_microbatch

# 1F1B内存占用
one_f_one_b_memory = num_stages * activation_size_per_microbatch

# 示例: 4 stages, 16 microbatches
# GPipe:  16 × activation_size
# 1F1B:   4 × activation_size
# 节省:   75% 内存!
```

---

## 实现细节

### 1. 层分割策略

```python
"""
如何决定每个stage包含多少层?
"""

def balance_by_parameters(model, num_stages):
    """按参数量均衡分割"""
    total_params = sum(p.numel() for p in model.parameters())
    params_per_stage = total_params / num_stages
    
    balance = []
    current_params = 0
    current_layers = 0
    
    for layer in model:
        layer_params = sum(p.numel() for p in layer.parameters())
        current_params += layer_params
        current_layers += 1
        
        if current_params >= params_per_stage:
            balance.append(current_layers)
            current_params = 0
            current_layers = 0
    
    return balance


def balance_by_computation(model, num_stages, sample_input):
    """按计算量均衡分割"""
    from torch.profiler import profile, ProfilerActivity
    
    layer_times = []
    
    # Profile每一层
    for layer in model:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _ = layer(sample_input)
        
        layer_time = sum([e.cuda_time for e in prof.key_averages()])
        layer_times.append(layer_time)
    
    # 贪心分割
    total_time = sum(layer_times)
    target_time = total_time / num_stages
    
    balance = []
    current_time = 0
    current_layers = 0
    
    for time in layer_times:
        current_time += time
        current_layers += 1
        
        if current_time >= target_time:
            balance.append(current_layers)
            current_time = 0
            current_layers = 0
    
    return balance
```

### 2. 通信优化

```python
"""
点对点通信优化
"""

import torch.distributed as dist


class OptimizedPipelineStage:
    def __init__(self, stage_id, prev_rank, next_rank):
        self.stage_id = stage_id
        self.prev_rank = prev_rank
        self.next_rank = next_rank
        
        # 预分配通信缓冲区
        self.send_buffer = None
        self.recv_buffer = None
    
    def send_activation(self, tensor):
        """异步发送激活值"""
        if self.send_buffer is None:
            self.send_buffer = tensor.clone()
        else:
            self.send_buffer.copy_(tensor)
        
        # 异步发送
        handle = dist.isend(self.send_buffer, dst=self.next_rank)
        return handle
    
    def recv_activation(self, shape):
        """异步接收激活值"""
        if self.recv_buffer is None:
            self.recv_buffer = torch.zeros(shape, device='cuda')
        
        # 异步接收
        handle = dist.irecv(self.recv_buffer, src=self.prev_rank)
        return handle, self.recv_buffer
    
    def forward_with_overlap(self, input_tensor):
        """计算和通信重叠"""
        
        # 开始接收下一个micro-batch (如果有)
        if self.has_next_microbatch():
            recv_handle, next_input = self.recv_activation(input_tensor.shape)
        
        # 前向计算当前micro-batch
        output = self.model(input_tensor)
        
        # 异步发送输出
        send_handle = self.send_activation(output)
        
        # 等待通信完成
        if self.has_next_microbatch():
            recv_handle.wait()
        send_handle.wait()
        
        return output
```

---

## 性能优化

### 1. Micro-batch数量选择

```python
"""
如何选择最优的micro-batch数量?
"""

def optimal_microbatches(num_stages, target_bubble_rate=0.1):
    """
    气泡率 = (num_stages - 1) / (num_stages - 1 + num_microbatches)
    
    求解 num_microbatches:
    num_microbatches = (num_stages - 1) * (1/target_bubble_rate - 1)
    """
    num_microbatches = int((num_stages - 1) * (1/target_bubble_rate - 1))
    return num_microbatches


# 示例
for num_stages in [2, 4, 8, 16]:
    optimal_m = optimal_microbatches(num_stages, target_bubble_rate=0.1)
    print(f"Stages: {num_stages}, Optimal Micro-batches: {optimal_m}")

"""
输出:
Stages: 2, Optimal Micro-batches: 9
Stages: 4, Optimal Micro-batches: 27
Stages: 8, Optimal Micro-batches: 63
Stages: 16, Optimal Micro-batches: 135
"""
```

### 2. 激活检查点

```python
"""
选择性激活检查点
"""

def selective_checkpointing(model, checkpoint_ratio=0.5):
    """
    只对部分层使用检查点
    
    策略: checkpoint计算量大的层
    """
    from torch.utils.checkpoint import checkpoint
    
    layer_costs = profile_layer_costs(model)
    threshold = sorted(layer_costs)[int(len(layer_costs) * checkpoint_ratio)]
    
    class SelectiveCheckpointModel(nn.Module):
        def forward(self, x):
            for i, layer in enumerate(self.layers):
                if layer_costs[i] > threshold:
                    # 使用checkpoint
                    x = checkpoint(layer, x)
                else:
                    # 正常前向
                    x = layer(x)
            return x
    
    return SelectiveCheckpointModel()
```

### 3. 虚拟流水线

```python
"""
虚拟流水线 (Interleaved Pipeline)
进一步减少气泡
"""

def interleaved_schedule(num_stages, num_virtual_stages=2):
    """
    每个物理stage包含多个虚拟stage
    
    例如: 4个物理GPU, 每个包含2个虚拟stage
    总共8个虚拟stage
    
    GPU 0: [Stage 0, Stage 4]
    GPU 1: [Stage 1, Stage 5]
    GPU 2: [Stage 2, Stage 6]
    GPU 3: [Stage 3, Stage 7]
    
    执行顺序:
    GPU 0: F0→F4→F0→F4→B0→B4→B0→B4
    """
    
    virtual_stages_per_device = num_virtual_stages
    total_virtual_stages = num_stages * virtual_stages_per_device
    
    # 交错调度
    schedule = []
    for device_id in range(num_stages):
        device_schedule = []
        for v in range(virtual_stages_per_device):
            stage_id = device_id + v * num_stages
            device_schedule.append(stage_id)
        schedule.append(device_schedule)
    
    return schedule
```

---

## 实战案例

### 案例1: GPT-3训练

```python
"""
使用流水线并行训练GPT-3 (175B参数)
"""

def train_gpt3_pipeline():
    # 配置
    num_layers = 96
    hidden_size = 12288
    num_heads = 96
    num_stages = 8  # 8个GPU
    num_microbatches = 64
    
    # 创建模型
    model = GPT3Model(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads
    )
    
    # 分割模型
    layers_per_stage = num_layers // num_stages  # 12 layers per stage
    balance = [layers_per_stage] * num_stages
    
    # 创建流水线
    pipe_model = Pipe(
        model,
        balance=balance,
        chunks=num_microbatches,
        checkpoint='always'
    )
    
   # 训练
    optimizer = torch.optim.AdamW(pipe_model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for batch in dataloader:
            output = pipe_model(batch['input_ids']).local_value()
            loss = F.cross_entropy(output.view(-1, vocab_size), 
                                   batch['labels'].view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

### 案例2: BERT预训练

```python
"""
BERT流水线并行预训练
"""

from transformers import BertConfig, BertForPreTraining

def train_bert_pipeline():
    # BERT-Large配置
    config = BertConfig(
        vocab_size=30522,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16
    )
    
    model = BertForPreTraining(config)
    
    # 4-stage流水线
    pipe_model = Pipe(
        model,
        balance=[6, 6, 6, 6],  # 每stage 6层
        chunks=16
    )
    
    # MLM + NSP训练
    for batch in dataloader:
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        next_sentence_label = batch['next_sentence_label']
        
        outputs = pipe_model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        ).local_value()
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

---

## 常见问题

### Q1: 流水线并行 vs 数据并行?

| 特性 | 数据并行 | 流水线并行 |
|-----|---------|----------|
| **内存** | 每GPU完整模型 | 每GPU部分模型 |
| **通信** | AllReduce梯度 | P2P激活/梯度 |
| **效率** | 高 (>90%) | 中 (70-85%) |
| **适用** | 小中型模型 | 超深模型 |

### Q2: 如何选择阶段数?

```python
"""
经验法则:
- 阶段数 = GPU数量 (简单)
- 考虑通信开销: 阶段数过多 → 通信频繁
- 考虑气泡率: 阶段数过少 → 利用率低

推荐:
- 2-8个阶段: 适合大多数场景
- 8-16个阶段: 超大模型
"""
```

### Q3: 流水线并行能与数据并行结合吗?

**可以！这就是2D并行**:

```python
"""
2D并行: 流水线 + 数据
"""

# 16 GPUs配置
pipeline_parallel_size = 4  # 4个流水线阶段
data_parallel_size = 4      # 4路数据并行

# GPU分配:
# Stage 0: GPU 0,1,2,3    (数据并行组)
# Stage 1: GPU 4,5,6,7
# Stage 2: GPU 8,9,10,11
# Stage 3: GPU 12,13,14,15

# 每个stage内部做数据并行
# 不同stage间做流水线并行
```

### Q4: 气泡时间能完全消除吗?

**不能，但可以最小化**:

```
理论下限 (无限micro-batch):
气泡率 = 0

实际下限:
- GPipe: ~10-20%
- 1F1B: ~10-20%
- 虚拟流水线: ~5-10%
```

---

## 总结

### 流水线并行选择指南

```
决策树:

模型是否很深 (>24层)?
├─ No → 考虑其他并行方式
└─ Yes → 继续判断

    内存是否受限?
    ├─ Yes → 使用1F1B (省内存)
    └─ No → 使用GPipe (简单)

    是否追求极致性能?
    ├─ Yes → 使用虚拟流水线
    └─ No → 标准1F1B足够
```

### 最佳实践

1. ✅ **首选1F1B** - 内存和性能平衡最好
2. ✅ **Micro-batch数 > 4×阶段数** - 控制气泡率<20%
3. ✅ **激活检查点** - 大模型必须启用
4. ✅ **合理分割** - 按计算量而非层数
5. ✅ **结合数据并行** - 2D并行效果更好

---

## 下一步

学完流水线并行后，继续学习:
- [张量并行](03-tensor-parallelism.md) - 处理超宽层
- [混合并行](06-hybrid-parallelism.md) - 组合PP+TP+DP

---

<div align="center">
  <strong>流水线并行让超深模型训练成为可能！🚀</strong>
</div></parameter>