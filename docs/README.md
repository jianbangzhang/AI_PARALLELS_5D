# 文档索引

欢迎来到 MatrixDistributedComputing-5DParallel 的文档中心！

## 📖 文档导航

### 入门指南

1. **[00-introduction.md](00-introduction.md)** - 分布式计算总体介绍
   - 为什么需要分布式计算
   - 基础概念和术语
   - 通信原语介绍
   - 硬件和网络要求

2. **[setup-guide.md](setup-guide.md)** - 环境配置指南
   - 系统要求
   - 依赖安装
   - 环境变量配置
   - 验证安装

### 5D并行详解

3. **[01-data-parallelism.md](01-data-parallelism.md)** - 数据并行 (DP)
   - 核心原理
   - DDP vs FSDP
   - 实现细节
   - 性能优化

4. **[02-pipeline-parallelism.md](02-pipeline-parallelism.md)** - 流水线并行 (PP)
   - 流水线架构
   - 调度策略 (GPipe, PipeDream, 1F1B)
   - 气泡问题
   - 实战案例

5. **[03-tensor-parallelism.md](03-tensor-parallelism.md)** - 张量并行 (TP)
   - 列并行与行并行
   - Megatron-LM方法
   - 通信优化
   - 与其他并行的结合

6. **[04-sequence-parallelism.md](04-sequence-parallelism.md)** - 序列并行 (SP)
   - 序列切分策略
   - 激活内存优化
   - Ring Attention
   - 超长序列训练

7. **[05-expert-parallelism.md](05-expert-parallelism.md)** - 专家并行 (EP)
   - MoE架构
   - 路由机制
   - 负载均衡
   - Switch Transformer实现

8. **[06-hybrid-parallelism.md](06-hybrid-parallelism.md)** - 混合并行策略
   - 3D/4D/5D并行
   - 并行度选择
   - 配置优化
   - 实际部署案例

### 性能与测试

9. **[benchmarks.md](benchmarks.md)** - 性能测试对比
   - 测试方法
   - 性能指标
   - 扩展性分析
   - 不同硬件配置的结果

## 📚 推荐阅读顺序

### 初学者路线
```
00-introduction.md 
    ↓
setup-guide.md
    ↓
01-data-parallelism.md
    ↓
实践：运行 01-data-parallelism/pytorch/dp_basic.py
```

### 进阶学习路线
```
02-pipeline-parallelism.md → 03-tensor-parallelism.md
    ↓                              ↓
实践：PP示例                  实践：TP示例
    ↓                              ↓
04-sequence-parallelism.md → 05-expert-parallelism.md
    ↓                              ↓
06-hybrid-parallelism.md (整合所有知识)
    ↓
benchmarks.md (性能分析)
```

## 🔗 外部资源

### 官方文档
- [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)

### 重要论文
- [Megatron-LM (2019)](https://arxiv.org/abs/1909.08053)
- [ZeRO (2020)](https://arxiv.org/abs/1910.02054)
- [GPipe (2019)](https://arxiv.org/abs/1811.06965)
- [Switch Transformers (2021)](https://arxiv.org/abs/2101.03961)

## 💡 使用建议

### 文档结构说明
每个文档都包含以下部分：
1. **概述**: 快速了解核心概念
2. **原理**: 深入技术细节
3. **实现**: 具体代码示例
4. **优化**: 性能调优技巧
5. **常见问题**: FAQ和故障排查

### 代码示例说明
- 📝 **理论说明**: 文档中的图表和伪代码
- 💻 **可运行代码**: 对应文件夹中的完整实现
- 🧪 **测试用例**: tests/ 目录中的单元测试

## 🛠️ 文档贡献

发现文档问题或想要改进？
1. 在 [Issues](../../issues) 中报告问题
2. 提交 Pull Request 改进文档
3. 分享你的实践经验

## 📊 文档更新日志

- **2024-12**: 初始版本发布
  - 完成5D并行所有文档
  - 添加代码示例
  - 性能测试结果

---

<div align="center">
  <strong>Happy Learning! 🚀</strong>
</div>