/**
 * 基础数据并行实现 - LibTorch
 * 
 * 演示使用LibTorch进行分布式数据并行训练
 */

#include "dp_utils.hpp"
#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <iostream>
#include <memory>

using namespace dp_utils;

/**
 * 简单的神经网络模型
 */
struct SimpleNetImpl : torch::nn::Module {
    SimpleNetImpl(int64_t input_size, int64_t hidden_size, int64_t output_size) {
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_size, output_size));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }
    
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};
TORCH_MODULE(SimpleNet);

/**
 * 数据并行训练器
 */
class DataParallelTrainer : public DistributedTrainer {
public:
    DataParallelTrainer(
        int rank,
        int world_size,
        torch::Device device,
        c10::intrusive_ptr<c10d::ProcessGroup> process_group
    ) : DistributedTrainer(rank, world_size, device),
        process_group_(process_group),
        logger_(rank) {
        
        // 创建模型
        model_ = SimpleNet(1024, 2048, 512);
        model_->to(device_);
        
        // 同步初始权重
        for (auto& param : model_->parameters()) {
            broadcast(param, process_group_, 0);
        }
        
        // 创建优化器
        optimizer_ = std::make_unique<torch::optim::Adam>(
            model_->parameters(),
            torch::optim::AdamOptions(0.001)
        );
        
        logger_.info("Model initialized with ", 
                    count_parameters(*model_), " parameters");
    }
    
    void train_epoch(int epoch) override {
        model_->train();
        
        const int num_batches = 100;
        const int batch_size = 32;
        
        double total_loss = 0.0;
        PerformanceMetrics metrics;
        
        logger_.separator();
        logger_.info("Epoch ", epoch, " - Training");
        logger_.separator();
        
        ProgressBar pbar(num_batches, rank_);
        
        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            metrics.start();
            
            // 生成随机数据
            auto data = torch::randn({batch_size, 1024}, device_);
            auto target = torch::randn({batch_size, 512}, device_);
            
            // 前向传播
            optimizer_->zero_grad();
            auto output = model_->forward(data);
            auto loss = torch::mse_loss(output, target);
            
            // 反向传播
            loss.backward();
            
            // 同步梯度
            sync_gradients(*model_, process_group_, world_size_);
            
            // 更新参数
            optimizer_->step();
            
            metrics.stop(batch_size);
            total_loss += loss.item<double>();
            
            pbar.update();
        }
        
        // 计算平均loss
        double avg_loss = total_loss / num_batches;
        
        // 同步所有进程的loss
        auto loss_tensor = torch::tensor({avg_loss}, device_);
        c10d::AllreduceOptions opts;
        opts.reduceOp = c10d::ReduceOp::AVG;
        std::vector<torch::Tensor> tensors = {loss_tensor};
        auto work = process_group_->allreduce(tensors, opts);
        work->wait();
        avg_loss = loss_tensor.item<double>();
        
        logger_.info("\nEpoch ", epoch, " Summary:");
        logger_.info("  Avg Loss: ", avg_loss);
        logger_.info("  Throughput: ", metrics.get_throughput(), " samples/sec");
        logger_.info("  Per-GPU Throughput: ", 
                    metrics.get_throughput() / world_size_, " samples/sec");
    }
    
    double validate() override {
        model_->eval();
        torch::NoGradGuard no_grad;
        
        const int num_batches = 20;
        const int batch_size = 32;
        double total_loss = 0.0;
        
        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            auto data = torch::randn({batch_size, 1024}, device_);
            auto target = torch::randn({batch_size, 512}, device_);
            
            auto output = model_->forward(data);
            auto loss = torch::mse_loss(output, target);
            
            total_loss += loss.item<double>();
        }
        
        double avg_loss = total_loss / num_batches;
        
        // 同步所有进程的loss
        auto loss_tensor = torch::tensor({avg_loss}, device_);
        c10d::AllreduceOptions opts;
        opts.reduceOp = c10d::ReduceOp::AVG;
        std::vector<torch::Tensor> tensors = {loss_tensor};
        auto work = process_group_->allreduce(tensors, opts);
        work->wait();
        
        return loss_tensor.item<double>();
    }
    
    void save_checkpoint(const std::string& path) override {
        if (rank_ == 0) {
            torch::save(model_, path);
            logger_.info("Model saved to ", path);
        }
    }
    
private:
    SimpleNet model_{nullptr};
    std::unique_ptr<torch::optim::Adam> optimizer_;
    c10::intrusive_ptr<c10d::ProcessGroup> process_group_;
    Logger logger_;
};

/**
 * 运行基础训练示例
 */
void run_basic_training(int rank, int world_size) {
    // 设置设备
    torch::Device device(torch::kCUDA, rank);
    
    // 初始化进程组
    auto process_group = init_process_group(rank, world_size);
    
    Logger logger(rank);
    logger.separator();
    logger.info("Data Parallel Training - Basic Example");
    logger.separator();
    logger.info("World Size: ", world_size);
    logger.info("Rank: ", rank);
    logger.info("Device: ", device);
    
    // 创建训练器
    DataParallelTrainer trainer(rank, world_size, device, process_group);
    
    // 训练循环
    const int num_epochs = 5;
    
    logger.info("\nStarting training for ", num_epochs, " epochs...\n");
    
    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        trainer.train_epoch(epoch);
        
        // 验证
        double val_loss = trainer.validate();
        logger.info("Validation Loss: ", val_loss, "\n");
    }
    
    // 保存模型
    trainer.save_checkpoint("model_basic.pt");
    
    logger.separator();
    logger.info("Training completed!");
    logger.separator();
}

/**
 * 性能测试
 */
void run_performance_benchmark(int rank, int world_size) {
    torch::Device device(torch::kCUDA, rank);
    auto process_group = init_process_group(rank, world_size);
    
    Logger logger(rank);
    logger.separator();
    logger.info("Performance Benchmark");
    logger.separator();
    
    // 不同的batch size测试
    std::vector<int> batch_sizes = {16, 32, 64, 128, 256};
    
    for (int batch_size : batch_sizes) {
        // 创建模型
        SimpleNet model(1024, 2048, 512);
        model->to(device);
        
        // 同步初始权重
        for (auto& param : model->parameters()) {
            broadcast(param, process_group, 0);
        }
        
        torch::optim::Adam optimizer(model->parameters());
        
        // 预热
        for (int i = 0; i < 10; ++i) {
            auto data = torch::randn({batch_size, 1024}, device);
            auto target = torch::randn({batch_size, 512}, device);
            
            optimizer.zero_grad();
            auto output = model->forward(data);
            auto loss = torch::mse_loss(output, target);
            loss.backward();
            
            sync_gradients(*model, process_group, world_size);
            optimizer.step();
        }
        
        // 测试
        const int num_iterations = 100;
        PerformanceMetrics metrics;
        
        torch::cuda::synchronize();
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; ++i) {
            auto data = torch::randn({batch_size, 1024}, device);
            auto target = torch::randn({batch_size, 512}, device);
            
            optimizer.zero_grad();
            auto output = model->forward(data);
            auto loss = torch::mse_loss(output, target);
            loss.backward();
            
            sync_gradients(*model, process_group, world_size);
            optimizer.step();
        }
        
        torch::cuda::synchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        double throughput = (batch_size * num_iterations) / elapsed.count();
        
        logger.info("Batch Size: ", batch_size);
        logger.info("  Time: ", elapsed.count(), " seconds");
        logger.info("  Throughput: ", throughput, " samples/sec");
        logger.info("  Global Throughput: ", throughput * world_size, " samples/sec\n");
    }
    
    logger.separator();
}

/**
 * 导出函数供main.cpp调用
 */
extern "C" {
    void run_dp_basic(int rank, int world_size, const char* mode) {
        std::string mode_str(mode);
        
        try {
            if (mode_str == "train") {
                run_basic_training(rank, world_size);
            } else if (mode_str == "benchmark") {
                run_performance_benchmark(rank, world_size);
            } else {
                if (rank == 0) {
                    std::cerr << "Unknown mode: " << mode_str << std::endl;
                    std::cerr << "Available modes: train, benchmark" << std::endl;
                }
            }
        } catch (const std::exception& e) {
            if (rank == 0) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
        }
    }
}