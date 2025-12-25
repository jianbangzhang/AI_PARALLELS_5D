import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

def worker(rank, world_size):
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    if rank == 0:
        print(f"Rank {rank}: 初始梯度 = {torch.tensor([1.0, 2.0])}")

    # 每个进程的模拟梯度
    local_grad = torch.tensor([float(rank + 1)] * 2)

    # 手动 all_reduce 求和
    dist.all_reduce(local_grad, op=dist.ReduceOp.SUM)

    # 平均梯度（模拟 DDP 中的梯度同步）
    local_grad /= world_size

    print(f"Rank {rank}: 同步后平均梯度 = {local_grad}")

    # 更完整的模型梯度同步示例
    model = nn.Linear(3, 1)
    model.weight.data.fill_(rank + 1.0)
    model.weight.grad = torch.ones_like(model.weight.data) * (rank + 1)

    dist.all_reduce(model.weight.grad, op=dist.ReduceOp.SUM)
    model.weight.grad /= world_size

    print(f"Rank {rank}: 模型权重梯度平均后 = {model.weight.grad.item():.2f}")

    dist.barrier()
    dist.destroy_process_group()

def main():
    world_size = 4
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()