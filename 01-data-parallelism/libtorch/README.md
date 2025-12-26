# 数据并行 - LibTorch C++ 实现

使用LibTorch C++ API实现的数据并行训练示例。

## 📋 目录

- [环境要求](#环境要求)
- [安装LibTorch](#安装libtorch)
- [编译](#编译)
- [运行示例](#运行示例)
- [示例说明](#示例说明)

---

## 环境要求

### 必需组件

- **C++ 编译器**: GCC 9.0+ 或 Clang 10.0+
- **CMake**: 3.18+
- **LibTorch**: 2.0+
- **CUDA**: 11.8+ (GPU训练必需)
- **MPI**: OpenMPI 4.0+ 或 MPICH 3.3+

### 系统要求

- **操作系统**: Linux (Ubuntu 20.04+ 推荐)
- **GPU**: NVIDIA GPU with CUDA support
- **内存**: 16GB+ RAM

---

## 安装LibTorch

### 方法1: 下载预编译版本 (推荐)

```bash
# 进入安装目录
cd /opt

# 下载LibTorch (cxx11 ABI版本)
# CUDA 11.8
sudo wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip

# 解压
sudo unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip

# 设置环境变量
export LIBTORCH_PATH=/opt/libtorch
export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH

# 添加到 ~/.bashrc
echo 'export LIBTORCH_PATH=/opt/libtorch' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 方法2: 从源码编译

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
mkdir build && cd build
python3 ../tools/build_libtorch.py
```

---

## 安装MPI

### Ubuntu/Debian

```bash
# OpenMPI
sudo apt-get update
sudo apt-get install -y openmpi-bin openmpi-common libopenmpi-dev

# 或 MPICH
sudo apt-get install -y mpich libmpich-dev

# 验证安装
mpirun --version
```

### 从源码编译OpenMPI (可选)

```bash
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
tar -xzf openmpi-4.1.6.tar.gz
cd openmpi-4.1.6

./configure --prefix=/opt/openmpi --with-cuda=/usr/local/cuda
make -j$(nproc)
sudo make install

# 设置环境变量
export PATH=/opt/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/opt/openmpi/lib:$LD_LIBRARY_PATH
```

---

## 编译

### 使用编译脚本 (推荐)

```bash
cd libtorch
chmod +x scripts/build.sh
./scripts/build.sh
```

### 手动编译

```bash
# 创建build目录
mkdir build && cd build

# 配置CMake
cmake .. \
    -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH \
    -DCMAKE_BUILD_TYPE=Release

# 编译
make -j$(nproc)

# 结果: build/dp_libtorch
```

### 验证编译

```bash
cd build
./dp_libtorch --help
```

---

## 运行示例

### 1. 基础训练示例

```bash
cd build

# 单节点4个GPU
mpirun -np 4 ./dp_libtorch basic train

# 指定GPU
mpirun -np 4 \
    --bind-to none \
    --map-by slot \
    -x NCCL_DEBUG=INFO \
    ./dp_libtorch basic train
```

### 2. 性能测试

```bash
# 测试不同batch size的性能
mpirun -np 4 ./dp_libtorch basic benchmark
```

### 3. 矩阵乘法示例

```bash
# 单次矩阵乘法
mpirun -np 4 ./dp_libtorch matrix single

# 性能测试
mpirun -np 4 ./dp_libtorch matrix benchmark

# 验证正确性
mpirun -np 4 ./dp_libtorch matrix verify
```

### 4. 多节点训练

```bash
# 创建hostfile
cat > hostfile << EOF
node1 slots=4
node2 slots=4
EOF

# 运行
mpirun -np 8 \
    --hostfile hostfile \
    -x LIBTORCH_PATH \
    -x LD_LIBRARY_PATH \
    ./dp_libtorch basic train
```

---

## 示例说明

### basic - 基础DDP训练

**功能**: 演示基本的数据并行训练流程

**模式**:
- `train`: 完整训练循环
- `benchmark`: 性能测试

**特性**:
- 简单的3层全连接网络
- 自动梯度同步
- 性能统计

**代码**: `src/dp_basic.cpp`

### matrix - 分布式矩阵乘法

**功能**: 演示数据并行的矩阵计算

**模式**:
- `single`: 单次计算
- `benchmark`: 不同大小的性能测试
- `verify`: 验证计算正确性

**策略**:
- 矩阵A按行切分
- 矩阵B在所有GPU复制
- 独立计算后无需通信

**代码**: `src/dp_matrix_multiply.cpp`

---

## 项目结构

```
libtorch/
├── CMakeLists.txt          # CMake配置
├── README.md               # 本文件
├── include/
│   └── dp_utils.hpp        # 工具类和辅助函数
├── src/
│   ├── main.cpp            # 主程序入口
│   ├── dp_basic.cpp        # 基础训练实现
│   └── dp_matrix_multiply.cpp  # 矩阵乘法实现
└── scripts/
    └── build.sh            # 编译脚本
```

---

## 常见问题

### Q1: 找不到LibTorch

**错误**: `Could not find package Torch`

**解决**:
```bash
export LIBTORCH_PATH=/path/to/libtorch
export CMAKE_PREFIX_PATH=$LIBTORCH_PATH
```

### Q2: 链接错误

**错误**: `undefined reference to torch::xxx`

**解决**:
```bash
# 确保链接了正确的LibTorch库
export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH
ldconfig
```

### Q3: CUDA版本不匹配

**错误**: `CUDA version mismatch`

**解决**:
```bash
# 检查CUDA版本
nvcc --version

# 下载对应版本的LibTorch
# CUDA 11.8 → libtorch cu118
# CUDA 12.1 → libtorch cu121
```

### Q4: MPI初始化失败

**错误**: `MPI_Init failed`

**解决**:
```bash
# 检查MPI安装
which mpirun
mpirun --version

# 测试MPI
mpirun -np 2 hostname
```

### Q5: NCCL错误

**错误**: `NCCL error: unhandled system error`

**解决**:
```bash
# 设置调试信息
export NCCL_DEBUG=INFO

# 禁用InfiniBand (测试用)
export NCCL_IB_DISABLE=1

# 指定网络接口
export NCCL_SOCKET_IFNAME=eth0
```

---

## 性能优化

### 1. 编译优化

```cmake
# 在CMakeLists.txt中添加
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
```

### 2. NCCL优化

```bash
# 设置NCCL参数
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
```

### 3. 数据加载优化

```cpp
// 使用pin memory
auto tensor = torch::randn({batch_size, input_size})
    .to(device, /*non_blocking=*/true);
```

---

## 调试技巧

### 1. 详细日志

```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
mpirun -np 4 ./dp_libtorch basic train
```

### 2. GDB调试

```bash
mpirun -np 4 xterm -e gdb -ex run --args ./dp_libtorch basic train
```

### 3. 性能分析

```bash
# NVIDIA Nsight Systems
nsys profile -o profile.qdrep mpirun -np 4 ./dp_libtorch basic train

# CUDA-MEMCHECK
cuda-memcheck mpirun -np 4 ./dp_libtorch basic train
```

---

## 参考资料

- [LibTorch文档](https://pytorch.org/cppdocs/)
- [LibTorch C++ API](https://pytorch.org/cppdocs/api/library_root.html)
- [MPI教程](https://mpitutorial.com/)
- [NCCL文档](https://docs.nvidia.com/deeplearning/nccl/)

---

## 许可证

MIT License

---

<div align="center">
  <strong>Happy Coding with LibTorch! 🚀</strong>
</div>