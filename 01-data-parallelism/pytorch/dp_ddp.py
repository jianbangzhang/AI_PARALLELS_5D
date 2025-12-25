"""
高级DDP实现 - 包含混合精度训练和梯度累积

运行方式:
    torchrun --nproc_per_node=4 dp_ddp.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import time
from tqdm import tqdm


class ConvNet(nn.Module):
    """卷积神经网络示例"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ImageDataset(torch.utils.data.Dataset):
    """模拟图像数据集"""
    def __init__(self, size=50000):
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 模拟32x32的RGB图像
        image = torch.randn(3, 32, 32)
        label = torch.randint(0, 10, (1,)).item()
        return image, label


def setup_distributed():
    """设置分布式环境"""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


class Trainer:
    """DDP训练器"""
    
    def __init__(
        self,
        model,
        rank,
        world_size,
        local_rank,
        use_amp=True,
        gradient_accumulation_steps=1
    ):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # 将模型移到GPU并包装为DDP
        self.model = model.cuda(local_rank)
        self.model = DDP(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # 性能优化
        )
        
        # 混合精度训练
        self.scaler = GradScaler() if use_amp else None
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 统计信息
        self.global_step = 0
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 只在rank 0显示进度条
        if self.rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = dataloader
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.cuda(self.local_rank)
            labels = labels.cuda(self.local_rank)
            
            # 混合精度前向传播
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # 缩放损失用于梯度累积
                loss = loss / self.gradient_accumulation_steps
                
                # 反向传播
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=1.0
                    )
                    
                    # 更新参数
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=1.0
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # 更新进度条
            if self.rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
                })
        
        # 学习率调度
        self.scheduler.step()
        
        # 计算平均loss
        avg_loss = total_loss / num_batches
        
        # 同步所有进程的loss
        loss_tensor = torch.tensor([avg_loss], device=f'cuda:{self.local_rank}')
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, dataloader):
        """验证"""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        for images, labels in dataloader:
            images = images.cuda(self.local_rank)
            labels = labels.cuda(self.local_rank)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
            else:
                outputs = self.model(images)
            
            _, predicted = outputs.max(1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        # 同步所有进程的统计信息
        stats = torch.tensor(
            [total_correct, total_samples],
            device=f'cuda:{self.local_rank}'
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        
        accuracy = stats[0].item() / stats[1].item()
        return accuracy
    
    def save_checkpoint(self, path, epoch):
        """保存检查点"""
        if self.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'global_step': self.global_step,
            }
            if self.scaler:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            torch.save(checkpoint, path)
            print(f"\n✓ Checkpoint saved to {path}")


def main():
    """主函数"""
    # 设置分布式
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print("\n" + "="*60)
        print("Advanced DDP Training with Mixed Precision")
        print("="*60)
        print(f"World Size: {world_size}")
        print(f"Using Mixed Precision: True")
        print(f"Gradient Accumulation Steps: 4")
    
    # 创建模型
    model = ConvNet(num_classes=10)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model Parameters: {total_params:,}")
    
    # 创建数据集
    train_dataset = ImageDataset(size=50000)
    val_dataset = ImageDataset(size=10000)
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    if rank == 0:
        print(f"Batch Size (per GPU): 64")
        print(f"Global Batch Size: {64 * world_size * 4}")  # 包含梯度累积
        print("\nStarting Training...\n")
    
    # 创建训练器
    trainer = Trainer(
        model,
        rank,
        world_size,
        local_rank,
        use_amp=True,
        gradient_accumulation_steps=4
    )
    
    # 训练循环
    num_epochs = 10
    best_accuracy = 0.0
    
    for epoch in range(1, num_epochs + 1):
        train_sampler.set_epoch(epoch)
        
        # 训练
        avg_loss = trainer.train_epoch(train_loader, epoch)
        
        # 验证
        accuracy = trainer.validate(val_loader)
        
        if rank == 0:
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  Val Accuracy: {accuracy*100:.2f}%")
            
            # 保存最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                trainer.save_checkpoint('best_model.pt', epoch)
    
    if rank == 0:
        print("\n" + "="*60)
        print(f"Training Completed! Best Accuracy: {best_accuracy*100:.2f}%")
        print("="*60)
    
    # 清理
    dist.destroy_process_group()


if __name__ == "__main__":
    main()