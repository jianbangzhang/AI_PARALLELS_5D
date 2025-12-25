/**
 * 数据并行 - MPI实现
 * 
 * 编译: make
 * 运行: mpirun -np 4 ./dp_mpi
 */

#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include "../include/matrix_ops.hpp"

class DataParallelTrainer {
private:
    int rank_;
    int world_size_;
    int M_, K_, N_;  // 矩阵维度
    
    std::vector<double> A_local_;  // 局部矩阵A
    std::vector<double> B_;        // 完整矩阵B
    std::vector<double> C_local_;  // 局部结果
    
public:
    DataParallelTrainer(int rank, int world_size, int M, int K, int N)
        : rank_(rank), world_size_(world_size), M_(M), K_(K), N_(N) {
        
        // 计算每个进程的行数
        int local_M = M / world_size;
        if (rank < M % world_size) {
            local_M++;
        }
        
        // 分配内存
        A_local_.resize(local_M * K);
        B_.resize(K * N);
        C_local_.resize(local_M * N);
        
        // 初始化数据
        init_matrices(local_M);
    }
    
    void init_matrices(int local_M) {
        /**
         * 初始化矩阵
         * A_local: 每个进程不同的数据
         * B: 所有进程相同的数据
         */
        std::mt19937 gen(rank_ + 42);  // 不同的种子
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        
        // 初始化A_local
        for (auto& val : A_local_) {
            val = dis(gen);
        }
        
        // 初始化B (所有进程使用相同种子)
        gen.seed(42);
        for (auto& val : B_) {
            val = dis(gen);
        }
        
        if (rank_ == 0) {
            std::cout << "\nMatrix Initialization:" << std::endl;
            std::cout << "  A: [" << M_ << " × " << K_ << "] (distributed)" << std::endl;
            std::cout << "  B: [" << K_ << " × " << N_ << "] (replicated)" << std::endl;
            std::cout << "  Local A: [" << local_M << " × " << K_ << "]" << std::endl;
        }
    }
    
    double train_step() {
        /**
         * 训练一个step (矩阵乘法 + 梯度同步)
         */
        int local_M = A_local_.size() / K_;
        
        // 计时开始
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
        
        // 1. 前向传播: C_local = A_local @ B
        matrix_multiply(
            A_local_.data(),
            B_.data(),
            C_local_.data(),
            local_M, K_, N_
        );
        
        // 2. 计算损失 (简化: MSE with target=0)
        double local_loss = 0.0;
        for (const auto& val : C_local_) {
            local_loss += val * val;
        }
        local_loss /= C_local_.size();
        
        // 3. 反向传播 (简化: 计算梯度)
        std::vector<double> grad_A_local(A_local_.size());
        std::vector<double> grad_B(B_.size(), 0.0);
        
        // grad_A = 2 * C @ B^T
        // grad_B = 2 * A^T @ C
        
        // 简化实现: 假设梯度已计算
        for (size_t i = 0; i < grad_A_local.size(); ++i) {
            grad_A_local[i] = 2.0 * A_local_[i];
        }
        
        // 4. 梯度同步: AllReduce grad_B
        std::vector<double> grad_B_local(B_.size());
        for (size_t i = 0; i < grad_B_local.size(); ++i) {
            grad_B_local[i] = 2.0 * B_[i] / world_size_;
        }
        
        MPI_Allreduce(
            grad_B_local.data(),
            grad_B.data(),
            grad_B.size(),
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD
        );
        
        // 5. 参数更新
        double learning_rate = 0.01;
        for (size_t i = 0; i < A_local_.size(); ++i) {
            A_local_[i] -= learning_rate * grad_A_local[i];
        }
        for (size_t i = 0; i < B_.size(); ++i) {
            B_[i] -= learning_rate * grad_B[i];
        }
        
        // 同步所有进程
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        
        // 计算时间
        std::chrono::duration<double> elapsed = end - start;
        
        // 同步loss
        double global_loss;
        MPI_Allreduce(&local_loss, &global_loss, 1, MPI_DOUBLE, 
                     MPI_SUM, MPI_COMM_WORLD);
        global_loss /= world_size_;
        
        return global_loss;
    }
    
    void benchmark() {
        /**
         * 性能测试
         */
        if (rank_ == 0) {
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "Starting Training Benchmark" << std::endl;
            std::cout << std::string(60, '=') << std::endl;
        }
        
        int num_iterations = 10;
        std::vector<double> times;
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            auto start = std::chrono::high_resolution_clock::now();
            double loss = train_step();
            auto end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double> elapsed = end - start;
            times.push_back(elapsed.count());
            
            if (rank_ == 0) {
                std::cout << "Iteration " << std::setw(2) << iter + 1 
                         << "/" << num_iterations
                         << " | Loss: " << std::fixed << std::setprecision(6) << loss
                         << " | Time: " << std::setprecision(4) << elapsed.count() << "s"
                         << std::endl;
            }
        }
        
        // 计算统计信息
        if (rank_ == 0) {
            double avg_time = 0.0;
            for (double t : times) avg_time += t;
            avg_time /= times.size();
            
            // 计算TFLOPS
            long long flops = 2LL * M_ * K_ * N_;  // 矩阵乘法FLOPs
            double tflops = flops / avg_time / 1e12;
            
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "Performance Summary" << std::endl;
            std::cout << std::string(60, '=') << std::endl;
            std::cout << "Average Time: " << avg_time << " seconds" << std::endl;
            std::cout << "TFLOPS: " << tflops << std::endl;
            std::cout << "Per-Process TFLOPS: " << tflops / world_size_ << std::endl;
            std::cout << std::string(60, '=') << std::endl;
        }
    }
};

void run_correctness_test(int rank, int world_size) {
    /**
     * 验证计算正确性
     */
    if (rank == 0) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Correctness Test" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }
    
    int M = 128, K = 128, N = 128;
    int local_M = M / world_size;
    
    // 创建测试数据
    std::vector<double> A_local(local_M * K);
    std::vector<double> B(K * N);
    std::vector<double> C_local(local_M * N);
    
    // 初始化为简单值
    for (int i = 0; i < local_M * K; ++i) {
        A_local[i] = rank + 1.0;
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = 2.0;
    }
    
    // 计算
    matrix_multiply(A_local.data(), B.data(), C_local.data(), local_M, K, N);
    
    // 验证结果
    double expected = (rank + 1.0) * 2.0 * K;  // 每个元素的期望值
    bool correct = true;
    for (double val : C_local) {
        if (std::abs(val - expected) > 1e-6) {
            correct = false;
            break;
        }
    }
    
    // 收集结果
    int all_correct;
    int local_correct = correct ? 1 : 0;
    MPI_Allreduce(&local_correct, &all_correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    
    if (rank == 0) {
        if (all_correct) {
            std::cout << "✓ Correctness test PASSED" << std::endl;
        } else {
            std::cout << "✗ Correctness test FAILED" << std::endl;
        }
        std::cout << std::string(60, '=') << std::endl;
    }
}

int main(int argc, char** argv) {
    // 初始化MPI
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (rank == 0) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Data Parallel Training with MPI" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Number of Processes: " << world_size << std::endl;
    }
    
    try {
        // 运行正确性测试
        run_correctness_test(rank, world_size);
        
        // 创建训练器并运行benchmark
        int M = 2048, K = 2048, N = 2048;
        DataParallelTrainer trainer(rank, world_size, M, K, N);
        trainer.benchmark();
        
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    
    // 清理
    MPI_Finalize();
    return 0;
}