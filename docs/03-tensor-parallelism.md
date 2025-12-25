张量并行 (Tensor Parallelism)
📋 目录

* 核心原理

* Megatron-LM方法

* 列并行与行并行

* Transformer并行化

* 通信优化

* 实现细节

* 性能分析

* 实战案例

核心原理
什么是张量并行？

```
基本思想:
将单个层的权重矩阵分割到多个GPU上
每个GPU计算部分结果
通过集合通信合并最终输出
```

为什么需要张量并行？
问题: 单个层太大，一个GPU装不下

```python
# 例如: 超大线性层
layer = nn.Linear(12288, 49152)  # 12B × 49K

参数量: 12,288 × 49,152 = 603M 参数
内存: 603M × 4 bytes (FP32) = 2.4 GB (仅权重!)

加上激活值、梯度、优化器状态:
总内存 ≈ 2.4 GB × 16 = 38.4 GB

单个A100 80GB: 可以装下
但100B+模型的层: 装不下！
```

张量并行 vs 数据并行

```
数据并行:
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ GPU 0       │  │ GPU 1       │  │ GPU 2       │
│ 完整模型    │  │ 完整模型    │  │ 完整模型    │
│ Data[0-31]  │  │ Data[32-63] │  │ Data[64-95] │
└─────────────┘  └─────────────┘  └─────────────┘

张量并行:
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ GPU 0       │  │ GPU 1       │  │ GPU 2       │
│ 权重[0:1/3] │  │ 权重[1/3:2/3]│ │ 权重[2/3:1] │
│ 完整数据    │  │ 完整数据    │  │ 完整数据    │
└─────────────┘  └─────────────┘  └─────────────┘
```

Megatron-LM方法
核心思想
Megatron-LM: NVIDIA提出的Transformer张量并行方法

```
关键洞察:
1. Transformer中的矩阵乘法可以按列或按行切分
2. 多头注意力天然适合并行 (每个head独立)
3. 精心设计切分方式，最小化通信
```

两种基本切分方式
1. 列并行 (Column Parallel)

```python
"""
列并行: 按输出维度切分

Y = XW, 其中 W: [k, n]
切分 W = [W₀, W₁, W₂, W₃]

每个GPU计算部分输出:
GPU 0: Y₀ = X @ W₀  → [batch, n/4]
GPU 1: Y₁ = X @ W₁  → [batch, n/4]
GPU 2: Y₂ = X @ W₂  → [batch, n/4]
GPU 3: Y₃ = X @ W₃  → [batch, n/4]

最终输出: Y = [Y₀, Y₁, Y₂, Y₃]  (拼接)
通信: 无需通信! (仅需拼接)
"""
```

可视化:

```
输入 X [batch×k]:
┌──────────────────┐
│                  │
│    X (完整)      │
│                  │
└──────────────────┘

权重 W [k×n]:
┌────┬────┬────┬────┐
│ W₀ │ W₁ │ W₂ │ W₃ │  ← 按列切分
│    │    │    │    │
│    │    │    │    │
└────┴────┴────┴────┘
GPU0  GPU1 GPU2  GPU3

输出 Y [batch×n]:
┌────┬────┬────┬────┐
│ Y₀ │ Y₁ │ Y₂ │ Y₃ │
└────┴────┴────┴────┘
```

2. 行并行 (Row Parallel)

```python
"""
行并行: 按输入维度切分

Y = XW, 其中 W: [k, n]
切分 W 按行:
  ┌ W₀ ┐
W=│ W₁ │
  │ W₂ │
  └ W₃ ┘

需要先切分输入 X:
GPU 0: Y₀ = X₀ @ W₀  → [batch, n]
GPU 1: Y₁ = X₁ @ W₁  → [batch, n]
GPU 2: Y₂ = X₂ @ W₂  → [batch, n]
GPU 3: Y₃ = X₃ @ W₃  → [batch, n]

最终输出: Y = Y₀ + Y₁ + Y₂ + Y₃  (AllReduce求和)
通信: 需要AllReduce!
"""
```

可视化:

```
输入 X [batch×k] (需先切分):
┌────┬────┬────┬────┐
│ X₀ │ X₁ │ X₂ │ X₃ │  ← 按特征维切分
└────┴────┴────┴────┘
GPU0  GPU1 GPU2  GPU3

权重 W [k×n]:
┌──────────────────┐
│       W₀         │  GPU 0
├──────────────────┤
│       W₁         │  GPU 1
├──────────────────┤
│       W₂         │  GPU 2
├──────────────────┤
│       W₃         │  GPU 3
└──────────────────┘

部分结果:
GPU 0: Y₀ [batch×n]
GPU 1: Y₁ [batch×n]
GPU 2: Y₂ [batch×n]
GPU 3: Y₃ [batch×n]
        ↓
    AllReduce
        ↓
Y = Y₀+Y₁+Y₂+Y₃ [batch×n]
```

列并行与行并行
代码实现

```python
"""
列并行和行并行的PyTorch实现
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class ColumnParallelLinear(nn.Module):
    """列并行线性层"""
    
    def __init__(
        self, 
        input_size: int,
        output_size: int,
        tensor_parallel_group,
        bias: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.tensor_parallel_group = tensor_parallel_group
        
        # 获取并行组信息
        world_size = dist.get_world_size(tensor_parallel_group)
        rank = dist.get_rank(tensor_parallel_group)
        
        # 计算每个GPU的输出维度
        assert output_size % world_size == 0
        self.output_size_per_partition = output_size // world_size
        
        # 创建权重 (只存储自己的分片)
        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size
            )
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.output_size_per_partition)
            )
        else:
            self.register_parameter('bias', None)
        
        # 初始化
        self._initialize_weights()
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_: [batch, seq_len, input_size]
        
        Returns:
            output: [batch, seq_len, output_size_per_partition]
        """
        # 列并行: Y = X @ W^T
        # 每个GPU计算部分输出
        output = torch.matmul(input_, self.weight.t())
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def _initialize_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class RowParallelLinear(nn.Module):
    """行并行线性层"""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        tensor_parallel_group,
        bias: bool = True,
        input_is_parallel: bool = False
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.tensor_parallel_group = tensor_parallel_group
        self.input_is_parallel = input_is_parallel
        
        # 获取并行组信息
        world_size = dist.get_world_size(tensor_parallel_group)
        rank = dist.get_rank(tensor_parallel_group)
        
        # 计算每个GPU的输入维度
        assert input_size % world_size == 0
        self.input_size_per_partition = input_size // world_size
        
        # 创建权重 (只存储自己的分片)
        self.weight = nn.Parameter(
            torch.empty(
                output_size,
                self.input_size_per_partition
            )
        )
        
        # Bias只在rank 0创建 (避免重复)
        if bias and rank == 0:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter('bias', None)
        
        self._initialize_weights()
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_: [batch, seq_len, input_size] 或
                   [batch, seq_len, input_size_per_partition] (如果已切分)
        
        Returns:
            output: [batch, seq_len, output_size]
        """
        # 如果输入还没切分，先切分
        if not self.input_is_parallel:
            input_ = self._split_along_last_dim(input_)
        
        # 行并行: Y_i = X_i @ W_i^T
        # 每个GPU计算部分结果
        output_parallel = torch.matmul(input_, self.weight.t())
        
        # AllReduce求和
        output = self._reduce_from_tensor_parallel_region(
            output_parallel
        )
        
        # 添加bias (只在rank 0)
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def _split_along_last_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """按最后一维切分tensor"""
        world_size = dist.get_world_size(self.tensor_parallel_group)
        rank = dist.get_rank(self.tensor_parallel_group)
        
        last_dim = tensor.size(-1)
        assert last_dim % world_size == 0
        
        chunk_size = last_dim // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size
        
        return tensor[..., start_idx:end_idx].contiguous()
    
    def _reduce_from_tensor_parallel_region(
        self, 
        input_: torch.Tensor
    ) -> torch.Tensor:
        """从张量并行区域reduce"""
        # AllReduce求和
        dist.all_reduce(
            input_,
            op=dist.ReduceOp.SUM,
            group=self.tensor_parallel_group
        )
        return input_
    
    def _initialize_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
```

通信分析

```python
"""
列并行 vs 行并行 通信对比
"""

# 列并行
"""
前向: 无通信 (仅拼接)
反向: AllReduce梯度 (∇X)

通信量: input_size × hidden_size × sizeof(dtype)
"""

# 行并行
"""
前向: AllReduce输出 (Y)
反向: 无通信 (梯度已切分)

通信量: batch_size × seq_len × output_size × sizeof(dtype)
"""

# 结论: 
# - 列并行: 反向通信
# - 行并行: 前向通信
# - 巧妙组合可以最小化通信!
```

Transformer并行化
MLP层并行化

```python
"""
Transformer MLP层的张量并行

标准MLP:
    h1 = GeLU(X @ W1 + b1)  # [batch, seq, 4*hidden]
    h2 = h1 @ W2 + b2        # [batch, seq, hidden]

并行化策略:
    W1: 列并行 (输出4*hidden切分)
    W2: 行并行 (输入4*hidden切分)
"""


class ParallelMLP(nn.Module):
    """并行MLP"""
    
    def __init__(self, hidden_size, ffn_hidden_size, tp_group):
        super().__init__()
        
        # W1: 列并行
        # hidden_size → ffn_hidden_size (切分输出)
        self.dense_h_to_4h = ColumnParallelLinear(
            hidden_size,
            ffn_hidden_size,
            tensor_parallel_group=tp_group,
            bias=True
        )
        
        self.activation = nn.GELU()
        
        # W2: 行并行
        # ffn_hidden_size → hidden_size (切分输入)
        self.dense_4h_to_h = RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            tensor_parallel_group=tp_group,
            bias=True,
            input_is_parallel=True  # 输入已切分
        )
    
    def forward(self, hidden_states):
        """
        前向传播
        
        通信分析:
        1. dense_h_to_4h (列并行): 
           - 前向: 无通信
           - 反向: AllReduce ∇hidden_states
        
        2. activation: 独立计算，无通信
        
        3. dense_4h_to_h (行并行):
           - 前向: AllReduce output
           - 反向: 无通信
        
        总通信: 2次AllReduce (前向1次 + 反向1次)
        """
        # [batch, seq, hidden] → [batch, seq, ffn_hidden/tp_size]
        intermediate = self.dense_h_to_4h(hidden_states)
        
        # 激活函数 (无通信)
        intermediate = self.activation(intermediate)
        
        # [batch, seq, ffn_hidden/tp_size] → [batch, seq, hidden]
        # 内部AllReduce
        output = self.dense_4h_to_h(intermediate)
        
        return output
```

注意力层并行化

```python
"""
Multi-Head Attention的张量并行

标准Attention:
    Q = X @ W_Q  # [batch, seq, num_heads * head_dim]
    K = X @ W_K
    V = X @ W_V
    
    # 分头
    Q = Q.view(batch, seq, num_heads, head_dim)
    K = K.view(batch, seq, num_heads, head_dim)
    V = V.view(batch, seq, num_heads, head_dim)
    
    # 计算attention
    scores = Q @ K.transpose(-2, -1) / sqrt(head_dim)
    attn = softmax(scores)
    output = attn @ V
    
    # 合并头
    output = output.view(batch, seq, num_heads * head_dim)
    output = output @ W_O

并行化策略:
    W_Q, W_K, W_V: 列并行 (按head切分)
    W_O: 行并行 (输入已按head切分)
"""


class ParallelAttention(nn.Module):
    """并行注意力层"""
    
    def __init__(
        self, 
        hidden_size, 
        num_attention_heads,
        tp_group
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.tp_group = tp_group
        
        # 张量并行组信息
        tp_world_size = dist.get_world_size(tp_group)
        
        # 确保head数可以整除
        assert num_attention_heads % tp_world_size == 0
        self.num_attention_heads_per_partition = (
            num_attention_heads // tp_world_size
        )
        
        self.head_dim = hidden_size // num_attention_heads
        self.hidden_size_per_partition = (
            self.num_attention_heads_per_partition * self.head_dim
        )
        
        # QKV投影: 列并行
        # 输出维度 = 3 * hidden_size (Q, K, V拼接)
        # 切分后每个GPU: 3 * hidden_size_per_partition
        self.query_key_value = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,
            tensor_parallel_group=tp_group,
            bias=True
        )
        
        # 输出投影: 行并行
        self.dense = RowParallelLinear(
            hidden_size,
            hidden_size,
            tensor_parallel_group=tp_group,
            bias=True,
            input_is_parallel=True
        )
    
    def forward(self, hidden_states, attention_mask=None):
        """
        前向传播
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len]
        
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # QKV投影 (列并行)
        # [batch, seq, hidden] → [batch, seq, 3*hidden_per_partition]
        qkv = self.query_key_value(hidden_states)
        
        # 切分成Q, K, V
        qkv = qkv.view(
            batch_size,
            seq_len,
            self.num_attention_heads_per_partition,
            3 * self.head_dim
        )
        
        # [batch, seq, num_heads_per_partition, 3*head_dim]
        # → 3 × [batch, num_heads_per_partition, seq, head_dim]
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # 计算attention scores
        # [batch, num_heads_per_partition, seq, seq]
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        
        # 应用mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 应用attention
        # [batch, num_heads_per_partition, seq, head_dim]
        context = torch.matmul(attn_weights, v)
        
        # 转置回来并合并heads
        # [batch, seq, num_heads_per_partition, head_dim]
        context = context.permute(0, 2, 1, 3).contiguous()
        
        # [batch, seq, hidden_per_partition]
        context = context.view(
            batch_size,
            seq_len,
            self.hidden_size_per_partition
        )
        
        # 输出投影 (行并行)
        # [batch, seq, hidden_per_partition] → [batch, seq, hidden]
        output = self.dense(context)
        
        return output
```

完整Transformer Block

```python
"""
完整的张量并行Transformer Block
"""


class ParallelTransformerBlock(nn.Module):
    """并行Transformer块"""
    
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        ffn_hidden_size,
        tp_group
    ):
        super().__init__()
        
        # LayerNorm (无需并行)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        
        # 注意力层 (张量并行)
        self.attention = ParallelAttention(
            hidden_size,
            num_attention_heads,
            tp_group
        )
        
        # LayerNorm
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        
        # MLP层 (张量并行)
        self.mlp = ParallelMLP(
            hidden_size,
            ffn_hidden_size,
            tp_group
        )
    
    def forward(self, hidden_states, attention_mask=None):
        """
        前向传播
        
        通信分析 (每个block):
        1. Attention:
           - QKV投影 (列并行): 反向AllReduce
           - 输出投影 (行并行): 前向AllReduce
        
        2. MLP:
           - 第一层 (列并行): 反向AllReduce
           - 第二层 (行并行): 前向AllReduce
        
        总计: 4次AllReduce (前向2次 + 反向2次)
        """
        # 注意力 + 残差
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = residual + attention_output
        
        # MLP + 残差
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states
```

通信优化
1. 通信与计算重叠

```python
"""
使用异步通信重叠计算
"""

class OptimizedRowParallelLinear(nn.Module):
    """优化的行并行线性层"""
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # 局部计算
        output_parallel = torch.matmul(input_, self.weight.t())
        
        # 异步AllReduce
        handle = dist.all_reduce(
            output_parallel,
            op=dist.ReduceOp.SUM,
            group=self.tensor_parallel_group,
            async_op=True  # 异步
        )
        
        # 可以在这里做其他计算...
        # 例如: dropout, bias等
        
        # 等待通信完成
        handle.wait()
        
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
        
        return output_parallel
```

2. 通信融合

```python
"""
将多个小通信合并成一个大通信
"""

def fused_all_reduce(tensors, group):
    """融合多个tensor的AllReduce"""
    
    # 将所有tensor拼接成一个大tensor
    sizes = [t.numel() for t in tensors]
    flat_tensors = [t.flatten() for t in tensors]
    fused_tensor = torch.cat(flat_tensors)
    
    # 一次AllReduce
    dist.all_reduce(fused_tensor, op=dist.ReduceOp.SUM, group=group)
    
    # 切分回原来的tensor
    offset = 0
    for i, size in enumerate(sizes):
        tensors[i].copy_(
            fused_tensor[offset:offset + size].view_as(tensors[i])
        )
        offset += size
```

3. 减少通信频率

```python
"""
梯度累积 + 减少AllReduce频率
"""

class CommunicationOptimizedModel(nn.Module):
    def __init__(self, model, accumulation_steps, tp_group):
        super().__init__()
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.tp_group = tp_group
        self.tp_world_size = dist.get_world_size(tp_group)
        self.step_count = 0
    
    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        
        # 每accumulation_steps才执行一次AllReduce
        self.step_count += 1
        if self.step_count % self.accumulation_steps == 0:
            self._all_reduce_gradients()
        
        return output
    
    def _all_reduce_gradients(self):
        """AllReduce所有梯度"""
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(
                    param.grad,
                    op=dist.ReduceOp.SUM,
                    group=self.tp_group
                )
                param.grad.div_(self.tp_world_size)
```

实现细节
1. 初始化同步

```python
"""
确保所有GPU的权重初始化相同
"""

def synchronized_init(tensor, group):
    """同步初始化"""
    rank = dist.get_rank(group)
    
    if rank == 0:
        # 只在rank 0初始化
        nn.init.xavier_uniform_(tensor)
    
    # 广播到所有进程
    dist.broadcast(tensor, src=0, group=group)
```

2. 随机数种子管理

```python
"""
张量并行需要相同的dropout mask等随机操作
"""

def set_parallel_seed(base_seed, tp_rank):
    """为张量并行设置相同种子"""
    torch.manual_seed(base_seed + tp_rank * 1000)  # 简单方式确保相同

class ParallelDropout(nn.Dropout):
    def forward(self, input_):
        if not self.training:
            return input_
        
        # 所有rank使用相同种子
        seed = torch.initial_seed()  # 或使用全局step
        torch.manual_seed(seed)
        return super().forward(input_)
```

3. 检查点保存/加载

```python
"""
张量并行模型的检查点处理
"""

def save_tensor_parallel_checkpoint(model, path, tp_rank, tp_size):
    """每个rank保存自己的分片"""
    state_dict = model.state_dict()
    checkpoint = {
        'model_state_dict': state_dict,
        'tp_rank': tp_rank,
        'tp_size': tp_size,
    }
    torch.save(checkpoint, f"{path}/rank_{tp_rank:02d}.pt")

def load_tensor_parallel_checkpoint(model, path, tp_rank):
    """加载对应rank的分片"""
    ckpt_path = f"{path}/rank_{tp_rank:02d}.pt"
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
```

性能分析
通信开销分析

```python
"""
理论通信时间估算
"""

def estimate_communication_time(
    hidden_size: int,
    seq_len: int,
    batch_size: int,
    tp_size: int,
    num_layers: int = 1,
    bandwidth_gbps: float = 600  # NVLink典型带宽 ~600 GB/s (双向)
):
    """
    估算一个Transformer Block的通信时间（单位：秒）
    
    假设FP16精度，2 bytes/element
    使用Ring AllReduce算法（NCCL默认）
    """
    bytes_per_element = 2
    
    # 每个Block约4次AllReduce（Attention + MLP各2次）
    num_allreduce = 4 * num_layers
    
    # 每次AllReduce传输的数据量（激活）
    data_per_allreduce = batch_size * seq_len * hidden_size * bytes_per_element  # bytes
    
    # Ring AllReduce通信量 = 2 * (tp_size - 1) / tp_size * data
    ring_factor = 2 * (tp_size - 1) / tp_size
    total_bytes = num_allreduce * data_per_allreduce * ring_factor
    
    # 带宽转换为bytes/s
    bandwidth_bytes_per_sec = bandwidth_gbps * 1e9 / 8
    
    comm_time_sec = total_bytes / bandwidth_bytes_per_sec
    
    return comm_time_sec

# 示例：GPT-3 175B规模，tp=8, A100 NVLink
print(estimate_communication_time(
    hidden_size=12288,
    seq_len=2048,
    batch_size=1,
    tp_size=8,
    num_layers=96
))  # ≈ 0.15-0.3秒（取决于具体配置）
```

实际性能对比（典型A100 8卡，FP16）

| 配置               | 模型规模 | TP Size | MFU (Model FLOPs Utilization) | 通信占比 |
|--------------------|----------|---------|-------------------------------|----------|
| 数据并行           | 13B     | 1       | ~55%                          | 低       |
| 张量并行 (Megatron)| 175B    | 8       | ~48-52%                       | 15-25%   |
| 3D并行 (TP+PP+DP)  | 175B    | 8       | ~58%                          | <10%     |

关键结论：
- TP=8时，通信开销通常占总时间的15-30%
- 通过异步通信 + 通信融合，可将通信占比降至<10%
- 序列越长、batch越小，通信占比越高（激活通信主导）

实战案例
1. Megatron-LM (NVIDIA官方)

- 最早实现张量并行的框架
- 支持GPT-3 175B在1024 A100上训练
- 代码：https://github.com/NVIDIA/Megatron-LM
- 关键特性：完整的列/行并行实现、通信重叠、模型检查点分片

2. DeepSpeed (Microsoft)

- 集成张量并行 + ZeRO + Pipeline并行
- 支持Llama-70B在单节点8卡高效训练
- 更易用API，自动处理通信优化

3. HuggingFace + Accelerate + Megatron集成

- 社区主流方案：transformers + megatron-lm插件
- 示例训练70B模型：
```bash
deepspeed --num_gpus=8 train.py \
    --deepspeed ds_config_zero3.json \
    --tensor-parallel-size 8
```

4. 实际部署经验总结

- TP大小推荐：8为最佳甜点（NVLink全互联）
- 超过8卡建议结合Pipeline并行（3D并行）
- 长序列（>4096）时注意激活内存爆炸，可结合序列并行
- 推理阶段：张量并行可显著降低单卡内存需求（如70B模型只需4×A100）

张量并行是训练超大规模Transformer模型的核心技术之一，通过巧妙的矩阵切分和通信优化，实现了模型参数在多GPU间的有效分布，是从13B到万亿参数模型演进的关键使能技术。