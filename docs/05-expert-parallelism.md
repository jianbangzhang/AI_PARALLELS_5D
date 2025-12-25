### 专家并行 (Expert Parallelism)

📋 目录

* 核心原理

* MoE架构回顾

* 专家并行实现

* 与其他并行的组合

* 通信分析

* 性能分析

* 实战案例

#### 核心原理
什么是专家并行？

```
基本思想:
将MoE层中的多个专家（Experts）分布到不同GPU
每个GPU持有完整的一个或多个专家
路由器（Router）将token派发到对应专家GPU
```

MoE架构回顾

```
标准MoE层:
Router → Top-K专家选择
Dispatch → 专家计算
Combine → 加权合并
```

#### 专家并行实现
All-to-All通信

```python
"""
专家并行核心: All-to-All dispatch/combine
"""
# 前向:
tokens → Router → top-k专家ID
All-to-All: 将tokens发送到对应专家GPU
每个GPU: 只计算本地专家
All-to-All: 合并专家输出

# 反向: 类似反向All-to-All
```

#### 与其他并行的组合
- **EP + TP**：每个专家内部再用TP切分（大专家）
- **EP + DP**：非MoE层用数据并行复制
- **必须开启SP**：当EP+TP时，需要SP避免激活爆炸

#### 通信分析
- **主要通信**：2次All-to-All（dispatch + combine）
- **通信量**：batch × seq × hidden × top-k × bytes
- **负载均衡**：容量因子（capacity factor）控制，避免热点专家

#### 性能分析

| 模型          | 参数量   | 活跃参数 | EP Size | 训练速度 vs Dense |
|---------------|----------|----------|---------|-------------------|
| Mixtral 8x7B  | 47B      | 13B      | 8       | 2.5x              |
| DeepSeek-MoE  | 145B     | 16B      | 16      | 3-4x              |

结论：
- 计算FLOPs仅增加~top-k倍
- 参数容量指数级增长
- 通信占比20-40%，需高速互联（如NVLink）

#### 实战案例
- **Mixtral 8x7B**：8专家，EP+TP训练
- **DeepSpeed-MoE**：支持EP+TP+DP
- **Megatron-MoE**：NVIDIA官方，支持万亿级MoE
- **经验**：top-k=2，容量因子1.2-2.0最佳

