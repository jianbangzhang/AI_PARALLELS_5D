"""
数据并行矩阵乘法示例

演示如何使用数据并行进行大规模矩阵计算
"""

import os
import torch
import torch.distributed as dist
import time
import argparse


def setup_distributed():
    """设置分布式环境"""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def distributed_matrix_multiply(
    M, K, N,
    rank, world_size, local_rank,
    dtype=torch.float32
):
    """
    分布式矩阵乘法: C = A @ B
    
    数据并行策略:
    - 矩阵B在所有GPU上复制
    - 矩阵A按行切分到不同GPU
    - 每个GPU计算自己的部分: C_local = A_local @ B
    
    Args:
        M: 矩阵A的行数
        K: 矩阵A的列数 / 矩阵B的行数
        N: 矩阵B的列数
        rank: 当前进程rank
        world_size: 总进程数
        local_rank: 本地rank
        dtype: 数据类型
    """
    # 计算每个GPU的行数
    local_M = M // world_size
    if rank < M % world_size:
        local_M += 1
    
    # 在每个GPU上创建局部矩阵A
    # 使用不同的随机种子确保数据不同
    torch.manual_seed(rank + 42)
    A_local = torch.randn(local_M, K, dtype=dtype, device=f'cuda:{local_rank}')
    
    # 矩阵B在所有GPU上复制(相同)
    torch.manual_seed(42)  # 相同的种子
    B = torch.randn(K, N, dtype=dtype, device=f'cuda:{local_rank}')
    
    if rank == 0:
        print(f"\nMatrix Dimensions:")
        print(f"  A: [{M} × {K}] (split across {world_size} GPUs)")
        print(f"  B: [{K} × {N}] (replicated)")
        print(f"  C: [{M} × {N}] (result)")
        print(f"\nPer-GPU:")
        print(f"  A_local: [{local_M} × {K}]")
        print(f"  Memory (A_local): {A_local.numel() * A_local.element_size() / 1e9:.2f} GB")
        print(f"  Memory (B): {B.numel() * B.element_size() / 1e9:.2f} GB")
    
    # 同步所有进程
    dist.barrier()
    
    # 计时开始
    torch.cuda.synchronize()
    start_time = time.time()
    
    # 执行矩阵乘法
    C_local = torch.matmul(A_local, B)
    
    # 同步等待计算完成
    torch.cuda.synchronize()
    dist.barrier()
    
    elapsed_time = time.time() - start_time
    
    # 计算FLOPs
    flops = 2 * local_M * K * N  # 每个GPU的FLOPs
    total_flops = 2 * M * K * N  # 总FLOPs
    tflops = total_flops / elapsed_time / 1e12
    
    if rank == 0:
        print(f"\nPerformance:")
        print(f"  Time: {elapsed_time:.4f} seconds")
        print(f"  TFLOPS: {tflops:.2f}")
        print(f"  Per-GPU TFLOPS: {tflops/world_size:.2f}")
        
        # 计算理论带宽
        data_volume = (M * K + K * N + M * N) * dtype.itemsize
        bandwidth = data_volume / elapsed_time / 1e9
        print(f"  Effective Bandwidth: {bandwidth:.2f} GB/s")
    
    return C_local, elapsed_time, tflops


def gather_results(C_local, M, N, rank, world_size, local_rank):
    """
    收集所有GPU的结果到rank 0
    
    可选操作，用于验证正确性
    """
    if rank == 0:
        # 创建接收缓冲区
        C_list = []
        for i in range(world_size):
            local_M = M // world_size
            if i < M % world_size:
                local_M += 1
            C_list.append(torch.zeros(
                local_M, N,
                dtype=C_local.dtype,
                device=f'cuda:{local_rank}'
            ))
        
        # Gather所有结果
        C_list[0].copy_(C_local)
        for i in range(1, world_size):
            dist.recv(C_list[i], src=i)
        
        # 拼接成完整矩阵
        C_full = torch.cat(C_list, dim=0)
        return C_full
    else:
        # 发送结果到rank 0
        dist.send(C_local, dst=0)
        return None


def benchmark_different_sizes(rank, world_size, local_rank):
    """测试不同矩阵大小的性能"""
    
    if rank == 0:
        print("\n" + "="*80)
        print("Benchmarking Different Matrix Sizes")
        print("="*80)
    
    # 测试不同的矩阵大小
    test_sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]
    
    results = []
    
    for M, K, N in test_sizes:
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Testing Size: [{M} × {K}] @ [{K} × {N}]")
            print(f"{'='*80}")
        
        try:
            C_local, elapsed, tflops = distributed_matrix_multiply(
                M, K, N,
                rank, world_size, local_rank,
                dtype=torch.float16  # 使用FP16提高性能
            )
            
            results.append({
                'size': (M, K, N),
                'time': elapsed,
                'tflops': tflops
            })
            
            # 清理内存
            del C_local
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if rank == 0:
                print(f"  Error: {e}")
                print(f"  Size too large for available memory")
            break
    
    if rank == 0:
        print(f"\n{'='*80}")
        print("Summary")
        print(f"{'='*80}")
        print(f"{'Size':<25} {'Time (s)':<12} {'TFLOPS':<12}")
        print(f"{'-'*80}")
        for result in results:
            M, K, N = result['size']
            size_str = f"{M}×{K}×{N}"
            print(f"{size_str:<25} {result['time']:<12.4f} {result['tflops']:<12.2f}")


def verify_correctness(rank, world_size, local_rank):
    """验证计算正确性"""
    
    if rank == 0:
        print("\n" + "="*80)
        print("Verifying Correctness")
        print("="*80)
    
    # 使用小矩阵验证
    M, K, N = 128, 128, 128
    
    # 分布式计算
    C_local, _, _ = distributed_matrix_multiply(
        M, K, N,
        rank, world_size, local_rank,
        dtype=torch.float32
    )
    
    # 收集结果
    C_distributed = gather_results(C_local, M, N, rank, world_size, local_rank)
    
    if rank == 0:
        # 在rank 0上计算正确结果
        torch.manual_seed(42)
        A_full = torch.randn(M, K, dtype=torch.float32, device=f'cuda:{local_rank}')
        # 填充A的不同部分
        for r in range(1, world_size):
            local_M = M // world_size
            start_idx = r * local_M
            end_idx = start_idx + local_M
            torch.manual_seed(r + 42)
            A_full[start_idx:end_idx] = torch.randn(
                local_M, K,
                dtype=torch.float32,
                device=f'cuda:{local_rank}'
            )
        
        torch.manual_seed(42)
        B = torch.randn(K, N, dtype=torch.float32, device=f'cuda:{local_rank}')
        
        C_correct = torch.matmul(A_full, B)
        
        # 比较结果
        max_error = torch.max(torch.abs(C_distributed - C_correct)).item()
        relative_error = max_error / torch.max(torch.abs(C_correct)).item()
        
        print(f"\nCorrectness Check:")
        print(f"  Max Absolute Error: {max_error:.6e}")
        print(f"  Relative Error: {relative_error:.6e}")
        
        if relative_error < 1e-4:
            print(f"  ✓ Results are correct!")
        else:
            print(f"  ✗ Results may be incorrect!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Data Parallel Matrix Multiplication')
    parser.add_argument('--mode', type=str, default='benchmark',
                       choices=['single', 'benchmark', 'verify'],
                       help='运行模式')
    parser.add_argument('--M', type=int, default=4096, help='矩阵A的行数')
    parser.add_argument('--K', type=int, default=4096, help='矩阵A的列数')
    parser.add_argument('--N', type=int, default=4096, help='矩阵B的列数')
    
    args = parser.parse_args()
    
    # 设置分布式
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print("\n" + "="*80)
        print("Data Parallel Matrix Multiplication")
        print("="*80)
        print(f"World Size: {world_size}")
        print(f"Device: {torch.cuda.get_device_name(local_rank)}")
    
    try:
        if args.mode == 'single':
            # 单次计算
            distributed_matrix_multiply(
                args.M, args.K, args.N,
                rank, world_size, local_rank,
                dtype=torch.float16
            )
        elif args.mode == 'benchmark':
            # 性能测试
            benchmark_different_sizes(rank, world_size, local_rank)
        elif args.mode == 'verify':
            # 验证正确性
            verify_correctness(rank, world_size, local_rank)
    
    except Exception as e:
        if rank == 0:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
    
    finally:
        # 清理
        dist.destroy_process_group()


if __name__ == "__main__":
    main()