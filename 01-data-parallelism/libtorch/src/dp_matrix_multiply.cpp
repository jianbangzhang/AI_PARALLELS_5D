/**
 * 数据并行矩阵乘法 - LibTorch
 * 
 * 演示分布式矩阵计算
 */

#include "dp_utils.hpp"
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace dp_utils;

/**
 * 分布式矩阵乘法类
 */
class DistributedMatrixMultiply {
public:
    DistributedMatrixMultiply(
        int rank,
        int world_size,
        torch::Device device,
        c10::intrusive_ptr<c10d::ProcessGroup> process_group
    ) : rank_(rank), 
        world_size_(world_size),
        device_(device),
        process_group_(process_group),
        logger_(rank) {}
    
    /**
     * 执行分布式矩阵乘法 C = A @ B
     * 
     * 策略:
     * - 矩阵A按行切分到不同GPU
     * - 矩阵B在所有GPU上复制
     * - 每个GPU独立计算自己的部分
     */
    torch::Tensor multiply(
        int64_t M, int64_t K, int64_t N,
        torch::ScalarType dtype = torch::kFloat32
    ) {
        // 计算每个GPU的行数
        int64_t local_M = M / world_size_;
        if (rank_ < M % world_size_) {
            local_M++;
        }
        
        logger_.info("\nMatrix Multiplication: C = A @ B");
        logger_.info("  A: [", M, " × ", K, "] (distributed)");
        logger_.info("  B: [", K, " × ", N, "] (replicated)");
        logger_.info("  C: [", M, " × ", N, "] (result)");
        logger_.info("  Local A: [", local_M, " × ", K, "]");
        
        // 创建局部矩阵A (每个GPU不同数据)
        torch::manual_seed(rank_ + 42);
        auto A_local = torch::randn({local_M, K}, 
                                   torch::TensorOptions()
                                   .dtype(dtype)
                                   .device(device_));
        
        // 创建矩阵B (所有GPU相同数据)
        torch::manual_seed(42);
        auto B = torch::randn({K, N},
                             torch::TensorOptions()
                             .dtype(dtype)
                             .device(device_));
        
        // 计算内存占用
        double mem_A = A_local.numel() * A_local.element_size() / 1e9;
        double mem_B = B.numel() * B.element_size() / 1e9;
        
        logger_.info("  Memory (A_local): ", std::fixed, std::setprecision(2), mem_A, " GB");
        logger_.info("  Memory (B): ", std::fixed, std::setprecision(2), mem_B, " GB");
        
        // 同步
        if (process_group_) {
            torch::cuda::synchronize();
            // MPI barrier
        }
        
        // 计时
        auto start = std::chrono::high_resolution_clock::now();
        
        // 矩阵乘法
        auto C_local = torch::matmul(A_local, B);
        
        // 同步
        torch::cuda::synchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        // 计算性能
        int64_t flops = 2LL * local_M * K * N;  // 每个GPU的FLOPs
        int64_t total_flops = 2LL * M * K * N;  // 总FLOPs
        double tflops = total_flops / elapsed.count() / 1e12;
        
        logger_.info("\nPerformance:");
        logger_.info("  Time: ", std::fixed, std::setprecision(4), elapsed.count(), " seconds");
        logger_.info("  TFLOPS: ", std::fixed, std::setprecision(2), tflops);
        logger_.info("  Per-GPU TFLOPS: ", std::fixed, std::setprecision(2), tflops / world_size_);
        
        return C_local;
    }
    
    /**
     * 收集结果到rank 0
     */
    torch::Tensor gather_results(const torch::Tensor& C_local, int64_t M, int64_t N) {
        if (rank_ == 0) {
            // 创建完整结果矩阵
            auto C_full = torch::zeros({M, N}, C_local.options());
            
            // 复制rank 0的结果
            int64_t local_M = M / world_size_;
            C_full.slice(0, 0, local_M).copy_(C_local);
            
            // 接收其他rank的结果
            for (int src_rank = 1; src_rank < world_size_; ++src_rank) {
                int64_t src_local_M = M / world_size_;
                if (src_rank < M % world_size_) {
                    src_local_M++;
                }
                
                int64_t start_idx = (M / world_size_) * src_rank;
                auto recv_tensor = C_full.slice(0, start_idx, start_idx + src_local_M);
                
                // 使用process group接收
                std::vector<torch::Tensor> tensors = {recv_tensor};
                // 注: 实际需要使用p2p通信API
            }
            
            return C_full;
        } else {
            // 发送结果到rank 0
            // 注: 实际需要使用p2p通信API
            return torch::Tensor();
        }
    }
    
    /**
     * 验证正确性
     */
    void verify_correctness() {
        logger_.separator();
        logger_.info("Verifying Correctness");
        logger_.separator();
        
        int64_t M = 128, K = 128, N = 128;
        
        // 分布式计算
        auto C_local = multiply(M, K, N, torch::kFloat32);
        
        if (rank_ == 0) {
            // 在rank 0上计算正确结果
            torch::manual_seed(42);
            
            // 构建完整矩阵A
            auto A_full = torch::zeros({M, K}, 
                                      torch::TensorOptions()
                                      .dtype(torch::kFloat32)
                                      .device(device_));
            
            for (int r = 0; r < world_size_; ++r) {
                int64_t local_M = M / world_size_;
                if (r < M % world_size_) {
                    local_M++;
                }
                
                int64_t start_idx = (M / world_size_) * r;
                
                torch::manual_seed(r + 42);
                A_full.slice(0, start_idx, start_idx + local_M).copy_(
                    torch::randn({local_M, K},
                                torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(device_))
                );
            }
            
            // 矩阵B
            torch::manual_seed(42);
            auto B = torch::randn({K, N},
                                 torch::TensorOptions()
                                 .dtype(torch::kFloat32)
                                 .device(device_));
            
            // 计算正确结果
            auto C_correct = torch::matmul(A_full, B);
            
            // 比较 rank 0 的结果
            int64_t local_M = M / world_size_;
            auto C_correct_local = C_correct.slice(0, 0, local_M);
            
            auto max_error = torch::max(torch::abs(C_local - C_correct_local)).item<double>();
            auto max_val = torch::max(torch::abs(C_correct_local)).item<double>();
            double relative_error = max_error / max_val;
            
            logger_.info("Max Absolute Error: ", std::scientific, max_error);
            logger_.info("Relative Error: ", std::scientific, relative_error);
            
            if (relative_error < 1e-4) {
                logger_.info("✓ Results are correct!");
            } else {
                logger_.error("✗ Results may be incorrect!");
            }
        }
        
        logger_.separator();
    }
    
    /**
     * 性能基准测试
     */
    void benchmark() {
        logger_.separator();
        logger_.info("Performance Benchmark - Different Matrix Sizes");
        logger_.separator();
        
        std::vector<std::tuple<int64_t, int64_t, int64_t>> test_sizes = {
            {1024, 1024, 1024},
            {2048, 2048, 2048},
            {4096, 4096, 4096},
            {8192, 8192, 8192},
        };
        
        logger_.info(std::left, std::setw(25), "Size", 
                    std::setw(15), "Time (s)",
                    std::setw(15), "TFLOPS");
        logger_.info(std::string(55, '-'));
        
        for (const auto& [M, K, N] : test_sizes) {
            try {
                // 使用FP16提高性能
                auto C_local = multiply(M, K, N, torch::kFloat16);
                
                // 清理内存
                C_local.reset();
                
            } catch (const std::exception& e) {
                logger_.error("Size ", M, "×", K, "×", N, " failed: ", e.what());
                break;
            }
        }
        
        logger_.separator();
    }
    
private:
    int rank_;
    int world_size_;
    torch::Device device_;
    c10::intrusive_ptr<c10d::ProcessGroup> process_group_;
    Logger logger_;
};

/**
 * 导出函数
 */
extern "C" {
    void run_dp_matrix_multiply(int rank, int world_size, const char* mode) {
        torch::Device device(torch::kCUDA, rank);
        auto process_group = init_process_group(rank, world_size);
        
        DistributedMatrixMultiply dmm(rank, world_size, device, process_group);
        
        std::string mode_str(mode);
        
        try {
            if (mode_str == "single") {
                // 单次计算
                dmm.multiply(4096, 4096, 4096, torch::kFloat16);
            } else if (mode_str == "benchmark") {
                // 性能测试
                dmm.benchmark();
            } else if (mode_str == "verify") {
                // 验证正确性
                dmm.verify_correctness();
            } else {
                if (rank == 0) {
                    std::cerr << "Unknown mode: " << mode_str << std::endl;
                }
            }
        } catch (const std::exception& e) {
            if (rank == 0) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
        }
    }
}