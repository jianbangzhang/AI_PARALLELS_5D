### 混合并行 (Hybrid Parallelism)

📋 目录

* 核心原理

* 3D并行 (DP + TP + PP)

* 4D并行 (加入SP/EP)

* 配置策略

* 性能分析

* 实战案例

#### 核心原理
为什么需要混合并行？
单一并行无法同时解决：
- 参数内存（TP/EP）
- 激活内存（SP/PP）
- 批大小/吞吐（DP）

#### 3D并行 (DP + TP + PP)

```
总GPU = DP × TP × PP

示例: 512 GPU
DP=64, TP=4, PP=2 → 训练万亿参数模型
```

可视化:

```
节点间: Data Parallel (AllReduce梯度)
节点内: Tensor Parallel (层内切分)
跨节点: Pipeline Parallel (层间流水线)
```

#### 4D并行 (加入SP/EP)

```
常见组合:
- DP + TP + PP + SP (长上下文Dense模型)
- DP + TP + PP + EP (MoE模型)
- DP + TP + EP + SP (大MoE，必须SP)
```

#### 配置策略
- **总GPU数 = DP × TP × PP × (EP或1)**
- **推荐甜点**：
  - TP=4-8（NVLink全互联）
  - PP=1-4（避免bubble）
  - EP=专家数（MoE专用）
  - SP=TP大小（激活优化）
- **搜索空间**：使用自动搜索工具（如Alpa）

#### 性能分析
混合并行是训练超大规模模型的终极方案，通过多维度并行组合，实现参数、激活、计算的最优分配，是从百亿到万亿参数模型训练的核心技术。
| 并行策略       | 模型规模   | GPU数 | MFU    | 内存效率 |
|----------------|------------|-------|--------|----------|
| 纯DP           | 70B        | 128   | 45%    | 低       |
| 3D (DP+TP+PP)  | 175B       | 1024  | 55%    | 高       |
| 4D + SP        | 70B@128K   | 256   | 60%    | 极高     |
| 4D + EP        | 1T (活跃100B)| 2048| 58%    | 参数极高 |

#### 实战案例
- **GPT-3 175B**：Megatron 3D并行，1024 A100
- **Llama3 405B**：TP+PP+DP+SP，长上下文优化
- **Mixtral/DeepSeek**：EP主导的4D并行
- **框架支持**：
  - Megatron-LM：完整3D/4D
  - DeepSpeed：ZeRO + Pipeline + MoE
  - NeMo：FSDP + TP + SP + EP

混合并行是训练超大规模模型的终极方案，通过多维度并行组合，实现参数、激活、计算的最优分配，是从百亿到万亿参数模型训练的核心技术。