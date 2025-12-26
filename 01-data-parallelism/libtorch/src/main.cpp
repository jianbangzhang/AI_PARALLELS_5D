/**
 * 数据并行主程序 - LibTorch
 * 
 * 使用方法:
 *   编译: cd build && cmake .. && make
 *   运行: mpirun -np 4 ./dp_libtorch [模式] [功能]
 *   
 * 模式:
 *   basic       - 基础训练示例
 *   matrix      - 矩阵乘法示例
 *   
 * 功能:
 *   train       - 训练模型
 *   benchmark   - 性能测试
 *   verify      - 验证正确性
 */

#include <mpi.h>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <cudnn.h>
#include <iostream>
#include <string>
#include <cstdlib>

// 外部函数声明
extern "C" {
    void run_dp_basic(int rank, int world_size, const char* mode);
    void run_dp_matrix_multiply(int rank, int world_size, const char* mode);
}

void print_usage(const char* program_name) {
    std::cout << "\nUsage: " << program_name << " [example] [mode]\n";
    std::cout << "\nExamples:\n";
    std::cout << "  basic    - Basic DDP training\n";
    std::cout << "  matrix   - Matrix multiplication\n";
    std::cout << "\nModes:\n";
    std::cout << "  train      - Training (for basic)\n";
    std::cout << "  benchmark  - Performance benchmark\n";
    std::cout << "  verify     - Correctness verification\n";
    std::cout << "  single     - Single run (for matrix)\n";
    std::cout << "\nExamples:\n";
    std::cout << "  mpirun -np 4 " << program_name << " basic train\n";
    std::cout << "  mpirun -np 4 " << program_name << " basic benchmark\n";
    std::cout << "  mpirun -np 4 " << program_name << " matrix benchmark\n";
    std::cout << "  mpirun -np 4 " << program_name << " matrix verify\n";
    std::cout << std::endl;
}

void print_system_info(int rank, int world_size) {
    if (rank == 0) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Data Parallel Training with LibTorch" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // LibTorch版本
        std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;
        
        // CUDA信息
        if (torch::cuda::is_available()) {
            std::cout << "CUDA available: Yes" << std::endl;
            std::cout << "cuDNN version: " << CUDNN_VERSION << std::endl;
            std::cout << "Number of GPUs: " << torch::cuda::device_count() << std::endl;
            
            // 打印每个GPU信息
            for (int i = 0; i < torch::cuda::device_count(); ++i) {
                cudaDeviceProp props;
                cudaGetDeviceProperties(&props, i);                
                std::cout << "  GPU " << i << ": " << props.name 
                         << " (" << props.totalGlobalMem / 1e9 << " GB)"
                         << std::endl;
            }
        } else {
            std::cout << "CUDA available: No (CPU mode)" << std::endl;
        }
        
        // MPI信息
        std::cout << "\nMPI Configuration:" << std::endl;
        std::cout << "  World size: " << world_size << std::endl;
        std::cout << "  Number of processes: " << world_size << std::endl;
        
        // 环境变量
        const char* master_addr = std::getenv("MASTER_ADDR");
        const char* master_port = std::getenv("MASTER_PORT");
        if (master_addr) {
            std::cout << "  Master address: " << master_addr << std::endl;
        }
        if (master_port) {
            std::cout << "  Master port: " << master_port << std::endl;
        }
        
        std::cout << std::string(60, '=') << std::endl;
    }
}

int main(int argc, char** argv) {
    // 初始化MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "Warning: MPI does not support MPI_THREAD_MULTIPLE" << std::endl;
    }
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // 检查CUDA是否可用
    if (!torch::cuda::is_available()) {
        if (rank == 0) {
            std::cerr << "Error: CUDA is not available!" << std::endl;
            std::cerr << "This program requires CUDA-enabled GPUs." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // 检查GPU数量
    if (world_size > torch::cuda::device_count()) {
        if (rank == 0) {
            std::cerr << "Error: Not enough GPUs!" << std::endl;
            std::cerr << "Requested " << world_size << " processes but only "
                     << torch::cuda::device_count() << " GPUs available." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // 设置当前进程使用的GPU
    c10::cuda::set_device(rank);
    
    // 打印系统信息
    print_system_info(rank, world_size);
    
    // 解析命令行参数
    std::string example = "basic";
    std::string mode = "train";
    
    if (argc > 1) {
        example = argv[1];
    }
    if (argc > 2) {
        mode = argv[2];
    }
    
    if (rank == 0) {
        std::cout << "\nRunning: " << example << " / " << mode << "\n" << std::endl;
    }
    
    // 运行指定的示例
    try {
        if (example == "basic") {
            run_dp_basic(rank, world_size, mode.c_str());
        } else if (example == "matrix") {
            run_dp_matrix_multiply(rank, world_size, mode.c_str());
        } else if (example == "help" || example == "-h" || example == "--help") {
            if (rank == 0) {
                print_usage(argv[0]);
            }
        } else {
            if (rank == 0) {
                std::cerr << "Unknown example: " << example << std::endl;
                print_usage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "\nError occurred: " << e.what() << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Program completed successfully!" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }
    
    // 清理MPI
    MPI_Finalize();
    
    return 0;
}