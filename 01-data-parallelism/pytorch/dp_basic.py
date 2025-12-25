"""
基础数据并行实现 - PyTorch DDP

运行方式:
    单节点:
    多节点: 参见 run_multi_node.sh
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import time


class SimpleModel(nn.Module):
    """简单的三层神经网络"""
    def __init__(self, input_size=1024, hidden_size=2048, output_size=512):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class SyntheticDataset(torch.utils.data.Dataset):
    """合成数据集用于测试"""
    def __init__(self, size=10000, input_dim=1024, output_dim=512):
        self.size = size
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 生成随机数据
        data = torch.randn(self.input_dim)
        target = torch.randn(self.output_dim)
        return data, target


def setup(rank, world_size):
    """初始化分布式环境"""
    # 环境变量通常由torchrun自动设置
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',  # GPU用nccl, CPU用gloo
        rank=rank,
        world_size=world_size
    )
    
    # 设置当前进程使用的GPU
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print(f"✓ Initialized DDP with {world_size} processes")


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def train_one_epoch(model, dataloader, optimizer, criterion, epoch, rank, world_size):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(dataloader):
        # 将数据移到GPU
        data = data.cuda(rank)
        target = target.cuda(rank)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播 (DDP自动同步梯度)
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 打印进度 (只在rank 0打印)
        if rank == 0 and batch_idx % 10 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f}")
    
    elapsed = time.time() - start_time
    avg_loss = total_loss / num_batches
    
    # 同步所有进程的loss (可选)
    loss_tensor = torch.tensor([avg_loss], device=f'cuda:{rank}')
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    avg_loss = loss_tensor.item()
    
    if rank == 0:
        throughput = len(dataloader.dataset) / elapsed
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.2f} samples/sec")
        print(f"  Per-GPU Throughput: {throughput/world_size:.2f} samples/sec")


def main():
    """主训练函数"""
    # 获取rank和world_size
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # 初始化分布式环境
    setup(rank, world_size)
    
    if rank == 0:
        print("\n" + "="*60)
        print("Data Parallel Training - Basic DDP")
        print("="*60)
        print(f"World Size: {world_size}")
        print(f"Local Rank: {local_rank}")
    
    # 创建模型并移到GPU
    model = SimpleModel(
        input_size=1024,
        hidden_size=2048,
        output_size=512
    ).cuda(local_rank)
    
    # DDP包装
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank
    )
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model Parameters: {total_params:,}")
    
    # 创建数据集和dataloader
    dataset = SyntheticDataset(size=10000, input_dim=1024, output_dim=512)
    
    # 使用DistributedSampler确保每个进程看到不同的数据
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    if rank == 0:
        print(f"\nBatch Size (per GPU): 32")
        print(f"Global Batch Size: {32 * world_size}")
        print(f"Total Batches per Epoch: {len(dataloader)}")
        print("\nStarting Training...\n")
    
    # 训练循环
    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        # 设置sampler的epoch (用于shuffle)
        sampler.set_epoch(epoch)
        
        # 训练一个epoch
        train_one_epoch(model, dataloader, optimizer, criterion, 
                       epoch, local_rank, world_size)
    
    if rank == 0:
        print("\n" + "="*60)
        print("Training Completed!")
        print("="*60)
    
    # 保存模型 (只在rank 0保存)
    if rank == 0:
        torch.save(model.module.state_dict(), 'ddp_model.pt')
        print("\n✓ Model saved to ddp_model.pt")
    
    # 清理
    cleanup()


if __name__ == "__main__":
    main()