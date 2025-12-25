# 环境配置指南

本指南将帮助你配置完整的分布式训练环境，包括PyTorch、LibTorch和MPI。

## 📋 目录

- [系统要求](#系统要求)
- [Python环境配置](#python环境配置)
- [LibTorch配置](#libtorch配置)
- [MPI配置](#mpi配置)
- [GPU和CUDA配置](#gpu和cuda配置)
- [网络配置](#网络配置)
- [验证安装](#验证安装)
- [常见问题](#常见问题)

---

## 系统要求

### 最低配置
- **操作系统**: Linux (Ubuntu 20.04+, CentOS 7+, RHEL 8+)
- **CPU**: 8核以上
- **内存**: 32GB RAM
- **GPU**: NVIDIA GPU (Compute Capability ≥ 7.0)
- **存储**: 100GB 可用空间

### 推荐配置
- **操作系统**: Ubuntu 22.04 LTS
- **CPU**: 32核以上 (AMD EPYC / Intel Xeon)
- **内存**: 256GB RAM
- **GPU**: 4× NVIDIA A100 80GB (或 H100)
- **网络**: InfiniBand HDR (200Gbps) 或 NVLink
- **存储**: NVMe SSD 1TB+

---

## Python环境配置

### 1. 安装Conda

```bash
# 下载Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# 初始化
source $HOME/miniconda3/bin/activate
conda init bash
source ~/.bashrc
```

### 2. 创建虚拟环境

```bash
# 创建环境
conda create -n dist-parallel python=3.10 -y
conda activate dist-parallel

# 安装基础工具
conda install -y cmake ninja git
```

### 3. 安装PyTorch

#### CUDA 11.8
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 12.1
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

#### CPU Only (测试用)
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

### 4. 安装分布式训练框架

```bash
# DeepSpeed
pip install deepspeed>=0.10.0

# FairScale
pip install fairscale>=0.4.13

# Accelerate
pip install accelerate>=0.21.0

# 其他依赖
pip install -r requirements.txt
```

### 5. 验证PyTorch安装

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"
python -c "import torch.distributed as dist; print('Distributed模块正常')"
```

---

## LibTorch配置

### 1. 下载LibTorch

```bash
# 选择版本
PYTORCH_VERSION="2.1.0"
CUDA_VERSION="cu118"  # 或 cu121, cpu

# 下载 (cxx11 ABI版本)
cd /opt
sudo wget https://download.pytorch.org/libtorch/${CUDA_VERSION}/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2B${CUDA_VERSION}.zip

# 解压
sudo unzip libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}+${CUDA_VERSION}.zip
sudo mv libtorch /opt/libtorch

# 清理
sudo rm libtorch-*.zip
```

### 2. 设置环境变量

```bash
# 添加到 ~/.bashrc
cat >> ~/.bashrc << 'EOF'
# LibTorch
export LIBTORCH_PATH=/opt/libtorch
export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$LIBTORCH_PATH:$CMAKE_PREFIX_PATH
EOF

# 使环境变量生效
source ~/.bashrc
```

### 3. 验证LibTorch安装

创建测试文件 `test_libtorch.cpp`:

```cpp
#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "LibTorch安装成功！" << std::endl;
    std::cout << "Tensor:\n" << tensor << std::endl;
    
    if (torch::cuda::is_available()) {
        std::cout << "CUDA可用，GPU数量: " 
                  << torch::cuda::device_count() << std::endl;
    } else {
        std::cout << "CUDA不可用" << std::endl;
    }
    
    return 0;
}
```

编译并运行:

```bash
g++ test_libtorch.cpp -o test_libtorch \
    -I$LIBTORCH_PATH/include \
    -I$LIBTORCH_PATH/include/torch/csrc/api/include \
    -L$LIBTORCH_PATH/lib \
    -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,$LIBTORCH_PATH/lib

./test_libtorch
```

---

## MPI配置

### 1. 安装MPICH

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y mpich libmpich-dev

# 验证
mpichversion
which mpirun
```

### 2. 安装OpenMPI (可选)

```bash
# Ubuntu/Debian
sudo apt-get install -y openmpi-bin openmpi-common libopenmpi-dev

# 或者从源码编译 (推荐)
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
tar -xzf openmpi-4.1.6.tar.gz
cd openmpi-4.1.6

./configure --prefix=/opt/openmpi \
    --enable-mpi-cxx \
    --with-cuda=/usr/local/cuda

make -j$(nproc)
sudo make install

# 设置环境变量
export PATH=/opt/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/opt/openmpi/lib:$LD_LIBRARY_PATH
```

### 3. 安装mpi4py (Python绑定)

```bash
conda activate dist-parallel
pip install mpi4py
```

### 4. 验证MPI安装

```bash
# 检查版本
mpirun --version

# 测试MPI
cat > mpi_test.c << 'EOF'
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    printf("Hello from rank %d of %d\n", world_rank, world_size);
    
    MPI_Finalize();
    return 0;
}
EOF

mpicc mpi_test.c -o mpi_test
mpirun -np 4 ./mpi_test
```

---

## GPU和CUDA配置

### 1. 检查GPU

```bash
# 检查GPU信息
nvidia-smi

# 检查CUDA版本
nvcc --version
cat /usr/local/cuda/version.txt
```

### 2. 安装CUDA (如果未安装)

```bash
# Ubuntu 22.04 - CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# 设置环境变量
cat >> ~/.bashrc << 'EOF'
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF

source ~/.bashrc
```

### 3. 安装cuDNN

```bash
# 下载cuDNN (需要NVIDIA账号)
# 从 https://developer.nvidia.com/cudnn 下载

# 解压并安装
tar -xvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### 4. 安装NCCL

```bash
# Ubuntu
sudo apt-get install -y libnccl2 libnccl-dev

# 或从源码编译
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j src.build
sudo make install
```

### 5. 验证CUDA和NCCL

```bash
# 测试CUDA
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.nccl.version())"

# NCCL测试
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 4
```

---

## 网络配置

### 1. 单节点配置

```bash
# 检查网络接口
ip addr show

# 设置防火墙 (如果需要)
sudo ufw allow 29500/tcp  # PyTorch默认端口
sudo ufw allow 12345/tcp  # 自定义端口
```

### 2. 多节点配置

#### 创建hostfile

```bash
# 创建 ~/hostfile
cat > ~/hostfile << 'EOF'
node1 slots=4
node2 slots=4
node3 slots=4
node4 slots=4
EOF
```

#### SSH无密码登录

```bash
# 生成密钥
ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa

# 复制到所有节点
for node in node1 node2 node3 node4; do
    ssh-copy-id $node
done

# 测试
for node in node1 node2 node3 node4; do
    ssh $node "hostname"
done
```

### 3. InfiniBand配置 (可选)

```bash
# 安装驱动
sudo apt-get install -y infiniband-diags ibutils

# 检查IB状态
ibstat
ibstatus

# 测试带宽
ib_write_bw -d mlx5_0 -a
```

---

## 验证安装

### 完整验证脚本

创建 `verify_setup.sh`:

```bash
#!/bin/bash

echo "=== 验证安装 ==="
echo

# 1. Python和包
echo "1. 检查Python环境..."
python --version
pip list | grep -E "torch|deepspeed|fairscale|mpi4py"
echo

# 2. PyTorch
echo "2. 检查PyTorch..."
python << EOF
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
print(f"NCCL版本: {torch.cuda.nccl.version()}")
EOF
echo

# 3. MPI
echo "3. 检查MPI..."
mpirun --version
which mpirun
echo

# 4. CUDA
echo "4. 检查CUDA..."
nvcc --version
nvidia-smi --query-gpu=name,memory.total --format=csv
echo

# 5. 网络
echo "5. 检查网络..."
ip addr show | grep inet
echo

# 6. 环境变量
echo "6. 检查环境变量..."
echo "CUDA_HOME: $CUDA_HOME"
echo "LIBTORCH_PATH: $LIBTORCH_PATH"
echo

echo "=== 验证完成 ==="
```

运行验证:

```bash
chmod +x verify_setup.sh
./verify_setup.sh
```

### 快速测试脚本

```bash
# 克隆仓库
git clone https://github.com/your-username/MatrixDistributedComputing-5DParallel.git
cd MatrixDistributedComputing-5DParallel

# 测试数据并行
cd 01-data-parallelism/pytorch
torchrun --nproc_per_node=2 dp_basic.py

# 测试MPI
cd ../../01-data-parallelism/cpp
make
mpirun -np 2 ./dp_mpi
```

---

## 常见问题

### Q1: NCCL初始化失败

**问题**: `NCCL error: unhandled system error`

**解决**:
```bash
# 检查NCCL版本兼容性
python -c "import torch; print(torch.cuda.nccl.version())"

# 设置环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 禁用InfiniBand (测试用)
```

### Q2: LibTorch链接错误

**问题**: `cannot find -ltorch`

**解决**:
```bash
# 确认路径
echo $LIBTORCH_PATH
ls -l $LIBTORCH_PATH/lib/libtorch.so

# 重新设置LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH
ldconfig
```

### Q3: MPI进程通信超时

**问题**: `MPI_Init timeout`

**解决**:
```bash
# 检查SSH连接
ssh localhost hostname

# 检查防火墙
sudo ufw status
sudo ufw allow from 192.168.0.0/16

# 使用localhost环回
mpirun -np 4 --host localhost:4 ./your_program
```

### Q4: GPU内存不足

**问题**: `CUDA out of memory`

**解决**:
```python
# 启用梯度检查点
model.gradient_checkpointing_enable()

# 使用混合精度
from torch.cuda.amp import autocast
with autocast():
    output = model(input)

# 减小batch size
batch_size = batch_size // 2
```

### Q5: 多节点通信失败

**问题**: `Connection refused`

**解决**:
```bash
# 设置正确的master地址
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=29500

# 检查端口是否开放
nc -zv $MASTER_ADDR $MASTER_PORT

# 确保所有节点都能访问
ping -c 3 $MASTER_ADDR
```

---

## Docker部署 (可选)

### 使用Docker快速启动

```bash
# 构建镜像
docker build -t dist-parallel:latest -f docker/Dockerfile.pytorch .

# 运行容器
docker run --gpus all --ipc=host --ulimit memlock=-1 \
    -v $(pwd):/workspace \
    -it dist-parallel:latest bash

# 测试
torchrun --nproc_per_node=2 01-data-parallelism/pytorch/dp_basic.py
```

---

## 性能调优建议

### 系统级优化

```bash
# 1. 设置CPU亲和性
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 2. 禁用CPU降频
sudo cpupower frequency-set -g performance

# 3. 增大共享内存
sudo sysctl -w kernel.shmmax=68719476736
sudo sysctl -w kernel.shmall=16777216

# 4. 优化网络缓冲区
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
```

### PyTorch优化

```python
# 启用cudnn benchmark
torch.backends.cudnn.benchmark = True

# 启用TF32 (Ampere架构)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 设置NCCL优化
os.environ['NCCL_IB_GID_INDEX'] = '3'
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
```

---

## 下一步

环境配置完成后，你可以：

1. 📖 阅读 [00-introduction.md](00-introduction.md) 了解基础概念
2. 💻 运行 [01-data-parallelism](../01-data-parallelism) 中的示例
3. 📊 查看 [benchmarks.md](benchmarks.md) 了解性能指标
4. 🚀 开始训练你的第一个分布式模型！

---

<div align="center">
  <strong>Happy Coding! 🎉</strong>
</div>