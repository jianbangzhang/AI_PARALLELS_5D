"""
矩阵生成工具

提供各种类型的测试矩阵生成功能，用于分布式计算测试。
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
import random


class MatrixGenerator:
    """矩阵生成器类"""
    
    def __init__(self, seed: Optional[int] = None, device: str = 'cuda'):
        """
        初始化生成器
        
        Args:
            seed: 随机种子
            device: 设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if seed is not None:
            self.set_seed(seed)
    
    @staticmethod
    def set_seed(seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def random_matrix(
        self, 
        shape: Tuple[int, ...], 
        dtype: torch.dtype = torch.float32,
        low: float = -1.0,
        high: float = 1.0
    ) -> torch.Tensor:
        """
        生成随机矩阵
        
        Args:
            shape: 矩阵形状
            dtype: 数据类型
            low: 最小值
            high: 最大值
            
        Returns:
            随机矩阵
        """
        matrix = (high - low) * torch.rand(shape, dtype=dtype, device=self.device) + low
        return matrix
    
    def normal_matrix(
        self,
        shape: Tuple[int, ...],
        mean: float = 0.0,
        std: float = 1.0,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        生成正态分布矩阵
        
        Args:
            shape: 矩阵形状
            mean: 均值
            std: 标准差
            dtype: 数据类型
            
        Returns:
            正态分布矩阵
        """
        matrix = torch.randn(shape, dtype=dtype, device=self.device) * std + mean
        return matrix
    
    def sparse_matrix(
        self,
        shape: Tuple[int, ...],
        sparsity: float = 0.9,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        生成稀疏矩阵
        
        Args:
            shape: 矩阵形状
            sparsity: 稀疏度 (0-1)
            dtype: 数据类型
            
        Returns:
            稀疏矩阵
        """
        matrix = torch.randn(shape, dtype=dtype, device=self.device)
        mask = torch.rand(shape, device=self.device) > sparsity
        matrix = matrix * mask
        return matrix
    
    def identity_matrix(
        self,
        size: int,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        生成单位矩阵
        
        Args:
            size: 矩阵大小
            dtype: 数据类型
            
        Returns:
            单位矩阵
        """
        return torch.eye(size, dtype=dtype, device=self.device)
    
    def diagonal_matrix(
        self,
        diagonal: List[float],
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        生成对角矩阵
        
        Args:
            diagonal: 对角线元素
            dtype: 数据类型
            
        Returns:
            对角矩阵
        """
        diag_tensor = torch.tensor(diagonal, dtype=dtype, device=self.device)
        return torch.diag(diag_tensor)
    
    def orthogonal_matrix(
        self,
        size: int,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        生成正交矩阵
        
        Args:
            size: 矩阵大小
            dtype: 数据类型
            
        Returns:
            正交矩阵
        """
        # 使用QR分解生成正交矩阵
        random_matrix = torch.randn(size, size, dtype=dtype, device=self.device)
        q, r = torch.linalg.qr(random_matrix)
        return q
    
    def symmetric_matrix(
        self,
        size: int,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        生成对称矩阵
        
        Args:
            size: 矩阵大小
            dtype: 数据类型
            
        Returns:
            对称矩阵
        """
        matrix = torch.randn(size, size, dtype=dtype, device=self.device)
        return (matrix + matrix.T) / 2
    
    def positive_definite_matrix(
        self,
        size: int,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        生成正定矩阵
        
        Args:
            size: 矩阵大小
            dtype: 数据类型
            
        Returns:
            正定矩阵
        """
        # A^T * A 总是正定的
        random_matrix = torch.randn(size, size, dtype=dtype, device=self.device)
        return torch.mm(random_matrix.T, random_matrix)
    
    def block_diagonal_matrix(
        self,
        block_sizes: List[int],
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        生成块对角矩阵
        
        Args:
            block_sizes: 每个块的大小
            dtype: 数据类型
            
        Returns:
            块对角矩阵
        """
        total_size = sum(block_sizes)
        matrix = torch.zeros(total_size, total_size, dtype=dtype, device=self.device)
        
        offset = 0
        for size in block_sizes:
            block = torch.randn(size, size, dtype=dtype, device=self.device)
            matrix[offset:offset+size, offset:offset+size] = block
            offset += size
        
        return matrix
    
    def toeplitz_matrix(
        self,
        c: List[float],
        r: Optional[List[float]] = None,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        生成Toeplitz矩阵
        
        Args:
            c: 第一列
            r: 第一行 (如果为None，则使用c)
            dtype: 数据类型
            
        Returns:
            Toeplitz矩阵
        """
        if r is None:
            r = c
        
        n = len(c)
        m = len(r)
        
        matrix = torch.zeros(n, m, dtype=dtype, device=self.device)
        
        for i in range(n):
            for j in range(m):
                if i >= j:
                    matrix[i, j] = c[i - j]
                else:
                    matrix[i, j] = r[j - i]
        
        return matrix
    
    def hankel_matrix(
        self,
        c: List[float],
        r: Optional[List[float]] = None,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        生成Hankel矩阵
        
        Args:
            c: 第一列
            r: 最后一行
            dtype: 数据类型
            
        Returns:
            Hankel矩阵
        """
        if r is None:
            r = [0] * len(c)
        
        n = len(c)
        m = len(r)
        
        matrix = torch.zeros(n, m, dtype=dtype, device=self.device)
        
        for i in range(n):
            for j in range(m):
                if i + j < n:
                    matrix[i, j] = c[i + j]
                else:
                    matrix[i, j] = r[i + j - n + 1]
        
        return matrix


class DistributedMatrixGenerator(MatrixGenerator):
    """分布式矩阵生成器"""
    
    def __init__(
        self, 
        rank: int, 
        world_size: int,
        seed: Optional[int] = None,
        device: str = 'cuda'
    ):
        """
        初始化分布式生成器
        
        Args:
            rank: 当前进程rank
            world_size: 总进程数
            seed: 随机种子
            device: 设备
        """
        super().__init__(seed, device)
        self.rank = rank
        self.world_size = world_size
        
        # 为每个rank设置不同的种子
        if seed is not None:
            self.set_seed(seed + rank)
    
    def partition_matrix(
        self,
        matrix: torch.Tensor,
        dim: int = 0
    ) -> torch.Tensor:
        """
        分割矩阵给当前rank
        
        Args:
            matrix: 完整矩阵
            dim: 分割维度
            
        Returns:
            当前rank的矩阵分片
        """
        size = matrix.shape[dim]
        chunk_size = size // self.world_size
        start_idx = self.rank * chunk_size
        end_idx = start_idx + chunk_size if self.rank < self.world_size - 1 else size
        
        if dim == 0:
            return matrix[start_idx:end_idx]
        elif dim == 1:
            return matrix[:, start_idx:end_idx]
        else:
            raise ValueError(f"Unsupported dim: {dim}")
    
    def local_random_matrix(
        self,
        global_shape: Tuple[int, ...],
        partition_dim: int = 0,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        生成局部随机矩阵
        
        Args:
            global_shape: 全局形状
            partition_dim: 分割维度
            dtype: 数据类型
            
        Returns:
            局部矩阵
        """
        local_shape = list(global_shape)
        local_shape[partition_dim] = global_shape[partition_dim] // self.world_size
        
        return self.random_matrix(tuple(local_shape), dtype)


def generate_test_data(
    batch_size: int,
    input_size: int,
    output_size: int,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成测试数据
    
    Args:
        batch_size: 批次大小
        input_size: 输入维度
        output_size: 输出维度
        device: 设备
        
    Returns:
        (输入数据, 目标数据)
    """
    gen = MatrixGenerator(device=device)
    inputs = gen.random_matrix((batch_size, input_size))
    targets = gen.random_matrix((batch_size, output_size))
    return inputs, targets


if __name__ == "__main__":
    # 测试代码
    print("=== 测试矩阵生成器 ===\n")
    
    gen = MatrixGenerator(seed=42, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试各种矩阵
    print("1. 随机矩阵:")
    rand_mat = gen.random_matrix((3, 3))
    print(rand_mat)
    print()
    
    print("2. 正态分布矩阵:")
    normal_mat = gen.normal_matrix((3, 3))
    print(normal_mat)
    print()
    
    print("3. 稀疏矩阵 (90% 稀疏):")
    sparse_mat = gen.sparse_matrix((5, 5), sparsity=0.9)
    print(sparse_mat)
    print()
    
    print("4. 单位矩阵:")
    identity_mat = gen.identity_matrix(4)
    print(identity_mat)
    print()
    
    print("5. 对称矩阵:")
    sym_mat = gen.symmetric_matrix(4)
    print(sym_mat)
    print("是否对称:", torch.allclose(sym_mat, sym_mat.T))
    print()
    
    print("6. 正定矩阵:")
    pd_mat = gen.positive_definite_matrix(4)
    eigenvalues = torch.linalg.eigvalsh(pd_mat)
    print("特征值:", eigenvalues)
    print("是否正定:", torch.all(eigenvalues > 0))
    print()
    
    print("=== 测试完成 ===")