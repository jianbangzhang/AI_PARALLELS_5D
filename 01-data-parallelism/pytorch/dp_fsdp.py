import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import os
import argparse

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

class ToyDataset(Dataset):
    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return torch.randn(10), torch.tensor(1 if idx % 2 == 0 else 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # 初始化进程组（推荐使用 torchrun 自动设置环境变量）
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(args.local_rank)

    # 模型
    model = ToyModel().cuda()

    # FSDP 包装（自动分片策略）
    wrap_policy = size_based_auto_wrap_policy(min_num_params=1000)
    model = FSDP(model, auto_wrap_policy=wrap_policy, device_id=args.local_rank, use_orig_params=True)

    # 数据
    dataset = ToyDataset()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)

    # 优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().cuda()

    model.train()
    for epoch in range(5):
        sampler.set_epoch(epoch)
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch + 1}/5, Average Loss: {total_loss / len(dataloader):.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()