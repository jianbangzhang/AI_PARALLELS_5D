# 5D并行学习资源与参考资料

这是一份详尽的学习资源清单，涵盖理论、论文、官方文档、教程和开源项目。

---

## 目录
1. [基础知识](#1-基础知识)
2. [数据并行 (DP)](#2-数据并行-dp)
3. [流水线并行 (PP)](#3-流水线并行-pp)
4. [张量并行 (TP)](#4-张量并行-tp)
5. [序列并行 (SP)](#5-序列并行-sp)
6. [专家并行 (EP)](#6-专家并行-ep)
7. [混合并行](#7-混合并行)
8. [工程实践](#8-工程实践)
9. [开源框架](#9-开源框架)
10. [视频教程](#10-视频教程)

---

## 1. 基础知识

### 1.1 分布式计算基础

#### 📚 书籍
- **《Distributed Computing: Principles, Algorithms, and Systems》** by Ajay D. Kshemkalyani
  - 分布式系统经典教材
  - 涵盖基本原理和算法

- **《Parallel Programming in C with MPI and OpenMP》** by Michael J. Quinn
  - MPI和OpenMP编程指南
  - 适合C++实现参考

#### 📄 论文
- **"Data Parallelism"** - Ian Foster (1995)
  - 数据并行的开创性论文
  - 📎 [PDF](https://www.mcs.anl.gov/~itf/dbpp/)

- **"Efficient Large-Scale Language Model Training"** - Shoeybi et al. (2019)
  - 大规模模型训练综述
  - 📎 [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)

#### 🌐 在线资源
- **PyTorch Distributed Overview**
  - https://pytorch.org/tutorials/beginner/dist_overview.html
  - PyTorch官方分布式教程

- **NCCL Documentation**
  - https://docs.nvidia.com/deeplearning/nccl/
  - GPU间通信库文档

- **MPI Tutorial**
  - https://mpitutorial.com/
  - 详细的MPI编程教程

### 1.2 通信原语

#### 📺 视频
- **"Understanding Collective Communication"** - NVIDIA
  - https://www.youtube.com/watch?v=KJGlMRPe-bw
  - AllReduce、AllGather等原语讲解

#### 📖 文档
- **NCCL Collective Operations**
  - https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
  - 详细的集合通信操作说明

---

## 2. 数据并行 (DP)

### 2.1 核心论文

1. **"Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"**
   - Goyal et al., Facebook AI Research (2017)
   - 📎 [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
   - 💡 关键内容：大批量训练、学习率缩放、warm-up策略

2. **"PyTorch Distributed: Experiences on Accelerating Data Parallel Training"**
   - Li et al. (2020)
   - 📎 [arXiv:2006.15704](https://arxiv.org/abs/2006.15704)
   - 💡 关键内容：DDP实现细节、优化技巧

3. **"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"**
   - Rajbhandari et al., Microsoft (2020)
   - 📎 [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
   - 💡 关键内容：参数分片、梯度分片、优化器状态分片

### 2.2 官方文档

#### PyTorch
- **DistributedDataParallel (DDP)**
  - https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
  - DDP API完整文档

- **DDP Tutorial**
  - https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
  - 完整的DDP使用教程，包含代码示例

- **FSDP Tutorial**
  - https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
  - Fully Sharded Data Parallel教程

#### TensorFlow
- **Distributed Training Guide**
  - https://www.tensorflow.org/guide/distributed_training
  - TensorFlow分布式训练指南

### 2.3 开源实现

- **Horovod**
  - https://github.com/horovod/horovod
  - Uber开发的分布式训练框架
  - 支持TensorFlow、PyTorch、MXNet

- **Hivemind**
  - https://github.com/learning-at-home/hivemind
  - 去中心化深度学习

### 2.4 博客文章

- **"Introduction to Distributed Data Parallel"** - PyTorch Blog
  - https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
  
- **"Efficient Training on Multiple GPUs"** - Hugging Face
  - https://huggingface.co/docs/transformers/perf_train_gpu_many

---

## 3. 流水线并行 (PP)

### 3.1 核心论文

1. **"GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism"**
   - Huang et al., Google (2019)
   - 📎 [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)
   - 💡 关键内容：微批次流水线、同步训练、内存优化

2. **"PipeDream: Generalized Pipeline Parallelism for DNN Training"**
   - Narayanan et al., Microsoft (2019)
   - 📎 [SOSP 2019](https://cs.stanford.edu/~matei/papers/2019/sosp_pipedream.pdf)
   - 💡 关键内容：异步流水线、权重版本管理、1F1B调度

3. **"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"**
   - Shoeybi et al., NVIDIA (2020)
   - 📎 [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
   - 💡 关键内容：结合TP和PP

4. **"Chimera: Efficiently Training Large-Scale Neural Networks"**
   - Li et al. (2021)
   - 📎 [SOSP 2021](https://dl.acm.org/doi/10.1145/3477132.3483547)
   - 💡 关键内容：动态流水线调度

### 3.2 官方文档

#### PyTorch
- **torch.distributed.pipeline**
  - https://pytorch.org/docs/stable/pipeline.html
  - PyTorch官方流水线并行API

- **Pipeline Parallelism Tutorial**
  - https://pytorch.org/tutorials/intermediate/dist_pipeline_parallel_tutorial.html

#### DeepSpeed
- **Pipeline Parallelism**
  - https://www.deepspeed.ai/tutorials/pipeline/
  - DeepSpeed的流水线并行教程

### 3.3 开源实现

- **FairScale Pipeline**
  - https://github.com/facebookresearch/fairscale
  - Facebook的模型并行库

- **PipeDream**
  - https://github.com/msr-fiddle/pipedream
  - PipeDream官方实现

### 3.4 技术博客

- **"How to Train Really Large Models on Many GPUs?"** - Lilian Weng
  - https://lilianweng.github.io/posts/2021-09-25-train-large/
  - 大模型训练全面综述

---

## 4. 张量并行 (TP)

### 4.1 核心论文

1. **"Megatron-LM: Training Multi-Billion Parameter Language Models"**
   - Shoeybi et al., NVIDIA (2019)
   - 📎 [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
   - 💡 关键内容：列并行、行并行、通信优化

2. **"Tensor Parallelism in Large-Scale Transformers"**
   - 📎 Megatron-LM GitHub: https://github.com/NVIDIA/Megatron-LM
   - 💡 关键内容：Transformer层的张量切分策略

3. **"Colossal-AI: A Unified Deep Learning System"**
   - Li et al. (2021)
   - 📎 [arXiv:2110.14883](https://arxiv.org/abs/2110.14883)
   - 💡 关键内容：多维张量并行

### 4.2 官方文档

#### Megatron-LM
- **Megatron-LM Documentation**
  - https://github.com/NVIDIA/Megatron-LM
  - NVIDIA官方实现和文档

- **Tensor and Pipeline Parallelism**
  - https://github.com/NVIDIA/Megatron-LM/blob/main/docs/PARALLELISM.md

#### DeepSpeed
- **Model Parallelism**
  - https://www.deepspeed.ai/training/#model-parallelism
  - DeepSpeed的模型并行指南

### 4.3 开源实现

- **Megatron-LM**
  - https://github.com/NVIDIA/Megatron-LM
  - NVIDIA官方实现，最权威

- **Colossal-AI**
  - https://github.com/hpcaitech/ColossalAI
  - 支持多种并行策略

- **Alpa**
  - https://github.com/alpa-projects/alpa
  - 自动并行优化

### 4.4 教程与博客

- **"Tensor Parallelism in PyTorch"** - Lei Mao's Blog
  - https://leimao.github.io/blog/PyTorch-Distributed-Training/

- **"Understanding Tensor Parallelism"** - Hugging Face
  - https://huggingface.co/docs/transformers/v4.15.0/parallelism

---

## 5. 序列并行 (SP)

### 5.1 核心论文

1. **"Reducing Activation Recomputation in Large Transformer Models"**
   - Korthikanti et al., NVIDIA (2022)
   - 📎 [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)
   - 💡 关键内容：序列维度分割、激活内存优化

2. **"Sequence Parallelism: Long Sequence Training from System Perspective"**
   - Li et al. (2021)
   - 📎 [arXiv:2105.13120](https://arxiv.org/abs/2105.13120)
   - 💡 关键内容：Ring Attention、块状序列处理

3. **"DeepSpeed Ulysses: System Optimizations for Enabling Training"**
   - Jacobs et al., Microsoft (2023)
   - 📎 [DeepSpeed Blog](https://www.microsoft.com/en-us/research/blog/deepspeed-ulysses/)
   - 💡 关键内容：All-to-All通信优化

### 5.2 官方文档

#### DeepSpeed
- **Sequence Parallelism**
  - https://www.deepspeed.ai/tutorials/ds-sequence/
  - DeepSpeed序列并行教程

- **DeepSpeed Ulysses**
  - https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-ulysses

#### Megatron-LM
- **Sequence Parallel in Megatron**
  - https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/

### 5.3 开源实现

- **Ring Attention**
  - https://github.com/lhao499/ring-attention
  - 超长序列注意力实现

- **FlashAttention**
  - https://github.com/Dao-AILab/flash-attention
  - 高效注意力实现，可与SP结合

### 5.4 技术文章

- **"Long Sequence Training from System Perspective"**
  - https://www.microsoft.com/en-us/research/blog/deepspeed-ulysses-system-optimizations-for-enabling-training-of-extreme-long-sequence-transformer-models/

---

## 6. 专家并行 (EP)

### 6.1 核心论文

1. **"Switch Transformers: Scaling to Trillion Parameter Models"**
   - Fedus et al., Google (2021)
   - 📎 [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
   - 💡 关键内容：简化的MoE路由、专家容量

2. **"GShard: Scaling Giant Models with Conditional Computation"**
   - Lepikhin et al., Google (2020)
   - 📎 [arXiv:2006.16668](https://arxiv.org/abs/2006.16668)
   - 💡 关键内容：分片MoE、负载均衡

3. **"BASE Layers: Simplifying Training of Large Models"**
   - Lewis et al., Meta (2021)
   - 📎 [arXiv:2103.16716](https://arxiv.org/abs/2103.16716)
   - 💡 关键内容：专家训练稳定性

4. **"ST-MoE: Designing Stable and Transferable MoE Models"**
   - Zoph et al., Google (2022)
   - 📎 [arXiv:2202.08906](https://arxiv.org/abs/2202.08906)
   - 💡 关键内容：路由器设计、专家初始化

### 6.2 官方文档

#### DeepSpeed
- **MoE Training**
  - https://www.deepspeed.ai/tutorials/mixture-of-experts/
  - DeepSpeed MoE完整教程

- **DeepSpeed-MoE API**
  - https://deepspeed.readthedocs.io/en/latest/moe.html

#### FairSeq
- **MoE Implementation**
  - https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm
  - Meta的MoE实现

### 6.3 开源实现

- **DeepSpeed-MoE**
  - https://github.com/microsoft/DeepSpeedExamples/tree/master/MoE
  - Microsoft官方MoE示例

- **Tutel**
  - https://github.com/microsoft/tutel
  - 高性能MoE库

- **Switch Transformers**
  - https://github.com/google-research/t5x/tree/main/t5x/examples/scalable_t5
  - Google官方实现

- **Mixtral**
  - https://github.com/mistralai/mistral-src
  - Mistral AI的开源MoE模型

### 6.4 技术博客

- **"Mixture of Experts Explained"** - Hugging Face
  - https://huggingface.co/blog/moe

- **"Scaling to MoE Models"** - Microsoft Research
  - https://www.microsoft.com/en-us/research/blog/deepspeed-advancing-moe-inference-and-training-to-power-next-generation-ai-scale/

---

## 7. 混合并行

### 7.1 核心论文

1. **"Efficient Large-Scale Language Model Training on GPU Clusters"**
   - Narayanan et al., NVIDIA (2021)
   - 📎 [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)
   - 💡 关键内容：3D并行 (DP+PP+TP)

2. **"Colossal-AI: A Unified Deep Learning System"**
   - Li et al., HPC-AI Tech (2021)
   - 📎 [arXiv:2110.14883](https://arxiv.org/abs/2110.14883)
   - 💡 关键内容：多维并行自动化

3. **"Alpa: Automating Inter- and Intra-Operator Parallelism"**
   - Zheng et al., UC Berkeley (2022)
   - 📎 [OSDI 2022](https://arxiv.org/abs/2201.12023)
   - 💡 关键内容：自动并行策略搜索

### 7.2 官方文档

#### DeepSpeed
- **3D Parallelism**
  - https://www.deepspeed.ai/tutorials/megatron/
  - DeepSpeed + Megatron集成

#### Megatron-LM
- **Multi-Dimensional Parallelism**
  - https://github.com/NVIDIA/Megatron-LM/blob/main/examples/pretrain_gpt_distributed.sh
  - 完整的多维并行启动脚本

### 7.3 开源框架

- **Megatron-DeepSpeed**
  - https://github.com/microsoft/Megatron-DeepSpeed
  - 结合两大框架的优势

- **Colossal-AI**
  - https://github.com/hpcaitech/ColossalAI
  - 支持各种并行组合

---

## 8. 工程实践

### 8.1 性能优化

#### 论文
- **"ZeRO-Infinity: Breaking GPU Memory Wall"**
  - Rajbhandari et al. (2021)
  - 📎 [arXiv:2104.07857](https://arxiv.org/abs/2104.07857)

- **"Activation Checkpointing"**
  - Chen et al. (2016)
  - 📎 [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)

#### 文档
- **PyTorch Performance Tuning**
  - https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

- **NVIDIA NCCL Best Practices**
  - https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/best_practices.html

### 8.2 调试与监控

- **TensorBoard Profiling**
  - https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras

- **PyTorch Profiler**
  - https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html

- **NVIDIA Nsight Systems**
  - https://developer.nvidia.com/nsight-systems

### 8.3 配置示例

- **DeepSpeed Configuration JSON**
  - https://www.deepspeed.ai/docs/config-json/

- **Megatron Launch Scripts**
  - https://github.com/NVIDIA/Megatron-LM/tree/main/examples

---

## 9. 开源框架

### 9.1 主流框架对比

| 框架 | DP | PP | TP | SP | EP | 易用性 | 性能 |
|------|----|----|----|----|----|----|------|
| **DeepSpeed** | ✅ | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Megatron-LM** | ✅ | ✅ | ✅ | ✅ | ❌ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Colossal-AI** | ✅ | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **FairScale** | ✅ | ✅ | ❌ | ❌ | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Alpa** | ✅ | ✅ | ✅ | ❌ | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 9.2 框架详细介绍

#### DeepSpeed
- **GitHub**: https://github.com/microsoft/DeepSpeed
- **文档**: https://www.deepspeed.ai/
- **特点**: Microsoft开发，功能最全，社区活跃
- **推荐场景**: 工业级大模型训练

#### Megatron-LM
- **GitHub**: https://github.com/NVIDIA/Megatron-LM
- **特点**: NVIDIA开发，性能最优
- **推荐场景**: GPU集群上的超大模型

#### Colossal-AI
- **GitHub**: https://github.com/hpcaitech/ColossalAI
- **文档**: https://colossalai.org/
- **特点**: 易用性好，自动化程度高
- **推荐场景**: 快速原型开发

---

## 10. 视频教程

### 10.1 入门课程

- **"Distributed Deep Learning"** - Stanford CS336
  - https://stanford-cs336.github.io/spring2024/
  - 斯坦福大学课程

- **"Large Language Models"** - UW CSE 599
  - https://courses.cs.washington.edu/courses/cse599g1/
  - 华盛顿大学课程

### 10.2 技术讲座

- **"Training GPT-3 Scale Models"** - NVIDIA GTC
  - YouTube: NVIDIA Developer Channel
  - Megatron-LM技术详解

- **"DeepSpeed: Extreme-scale Model Training"** - Microsoft
  - https://www.youtube.com/watch?v=wbG0jGU5qvY

### 10.3 会议演讲

- **MLSys Conference**
  - https://mlsys.org/
  - 系统与机器学习会议录像

- **NVIDIA GTC Sessions**
  - https://www.nvidia.com/gtc/
  - GPU技术大会

---

## 11. 实战项目

### 11.1 模型训练示例

- **Train GPT-2 with DeepSpeed**
  - https://github.com/microsoft/DeepSpeedExamples/tree/master/training/gpt2

- **Train BERT with Megatron**
  - https://github.com/NVIDIA/Megatron-LM/tree/main/examples

### 11.2 Benchmark项目

- **MLPerf Training**
  - https://mlcommons.org/en/training-normal-21/
  - 业界标准性能测试

---

## 12. 学习路线建议

### 阶段1: 基础 (1-2周)
1. 学习MPI基础
2. 理解集合通信原语
3. 掌握PyTorch分布式基础

### 阶段2: 数据并行 (1周)
1. 实现简单的DDP程序
2. 理解梯度同步机制
3. 学习FSDP/ZeRO

### 阶段3: 模型并行 (2-3周)
1. 实现流水线并行
2. 实现张量并行
3. 理解通信-计算重叠

### 阶段4: 高级并行 (2-3周)
1. 学习序列并行
2. 学习MoE和专家并行
3. 实现混合并行策略

### 阶段5: 工程实践 (持续)
1. 性能调优
2. 大规模集群部署
3. 故障容错处理

---

## 13. 推荐阅读顺序

### 必读论文 (按顺序)
1. Megatron-LM (理解TP/PP基础)
2. GPipe (理解流水线)
3. ZeRO (理解内存优化)
4. Switch Transformers (理解MoE)
5. Alpa (理解自动并行)

### 必看文档
1. PyTorch DDP Tutorial
2. DeepSpeed Getting Started
3. Megatron-LM Examples
4. NCCL User Guide

### 动手实践项目
1. 复现本仓库所有示例
2. 训练一个小型GPT模型
3. 实现自定义并行策略
4. 性能对比实验

---

## 14. 社区资源

### 论坛与讨论
- **PyTorch Discuss**
  - https://discuss.pytorch.org/c/distributed/

- **DeepSpeed GitHub Issues**
  - https://github.com/microsoft/DeepSpeed/issues

### 微信公众号
- HPC-AI科技
- NVIDIA英伟达
- Microsoft Research
