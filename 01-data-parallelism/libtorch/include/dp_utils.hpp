/**
 * 数据并行工具类 - LibTorch实现
 * 
 * 提供分布式训练的通用工具函数
 */

#ifndef DP_UTILS_HPP
#define DP_UTILS_HPP

#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <iostream>
#include <memory>
#include <chrono>
#include <vector>

namespace dp_utils {

/**
 * 分布式训练器基类
 */
class DistributedTrainer {
public:
    DistributedTrainer(int rank, int world_size, torch::Device device)
        : rank_(rank), world_size_(world_size), device_(device) {}
    
    virtual ~DistributedTrainer() = default;
    
    // 训练一个epoch
    virtual void train_epoch(int epoch) = 0;
    
    // 验证
    virtual double validate() = 0;
    
    // 保存模型
    virtual void save_checkpoint(const std::string& path) = 0;
    
protected:
    int rank_;
    int world_size_;
    torch::Device device_;
};

/**
 * 性能统计类
 */
class PerformanceMetrics {
public:
    PerformanceMetrics() : total_time_(0.0), num_samples_(0) {}
    
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    void stop(int batch_size) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time_;
        total_time_ += elapsed.count();
        num_samples_ += batch_size;
    }
    
    double get_throughput() const {
        return num_samples_ / total_time_;
    }
    
    double get_avg_time() const {
        return total_time_ / num_samples_;
    }
    
    void reset() {
        total_time_ = 0.0;
        num_samples_ = 0;
    }
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    double total_time_;
    int num_samples_;
};

/**
 * 初始化分布式进程组
 */
inline c10::intrusive_ptr<c10d::ProcessGroup> init_process_group(
    int rank,
    int world_size,
    const std::string& master_addr = "127.0.0.1",
    int master_port = 29500
) {
    // 创建文件存储
    std::string store_path = "/tmp/torch_distributed_" + std::to_string(master_port);
    auto store = c10::make_intrusive<c10d::FileStore>(store_path, world_size);
    
    // NCCL配置
    c10d::ProcessGroupNCCL::Options options;
    options.timeout = std::chrono::milliseconds(30000);
    
    // 创建进程组 - 注意类型转换
    c10::intrusive_ptr<c10d::ProcessGroup> process_group = 
        c10::make_intrusive<c10d::ProcessGroupNCCL>(
            store, rank, world_size, options
        );
    
    return process_group;
}

/**
 * AllReduce操作
 */
inline void all_reduce(
    torch::Tensor& tensor,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    c10d::ReduceOp op = c10d::ReduceOp::SUM
) {
    std::vector<torch::Tensor> tensors = {tensor};
    c10d::AllreduceOptions opts;
    opts.reduceOp = op;
    auto work = process_group->allreduce(tensors, opts);
    work->wait();
}

/**
 * Broadcast操作
 */
inline void broadcast(
    torch::Tensor& tensor,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int src_rank = 0
) {
    std::vector<torch::Tensor> tensors = {tensor};
    c10d::BroadcastOptions opts;
    opts.rootRank = src_rank;
    opts.rootTensor = 0;
    auto work = process_group->broadcast(tensors, opts);
    work->wait();
}

/**
 * 同步所有梯度
 */
inline void sync_gradients(
    torch::nn::Module& model,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int world_size
) {
    for (auto& param : model.parameters()) {
        if (param.grad().defined()) {
            all_reduce(param.grad(), process_group, c10d::ReduceOp::SUM);
            param.grad().div_(world_size);
        }
    }
}

/**
 * 打印张量信息
 */
inline void print_tensor_info(const torch::Tensor& tensor, const std::string& name) {
    std::cout << name << " shape: " << tensor.sizes() 
              << ", device: " << tensor.device()
              << ", dtype: " << tensor.dtype() << std::endl;
}

/**
 * 计算模型参数量
 */
inline int64_t count_parameters(const torch::nn::Module& model) {
    int64_t total = 0;
    for (const auto& param : model.parameters()) {
        total += param.numel();
    }
    return total;
}

/**
 * 简单的数据集类
 */
class SyntheticDataset : public torch::data::Dataset<SyntheticDataset> {
public:
    SyntheticDataset(int64_t size, std::vector<int64_t> input_shape, int64_t num_classes)
        : size_(size), input_shape_(input_shape), num_classes_(num_classes) {}
    
    torch::data::Example<> get(size_t index) override {
        auto data = torch::randn(input_shape_);
        auto target = torch::randint(0, num_classes_, {});
        return {data, target};
    }
    
    torch::optional<size_t> size() const override {
        return size_;
    }
    
private:
    int64_t size_;
    std::vector<int64_t> input_shape_;
    int64_t num_classes_;
};

/**
 * 日志工具
 */
class Logger {
public:
    Logger(int rank, bool verbose = true) 
        : rank_(rank), verbose_(verbose) {}
    
    template<typename... Args>
    void log(Args... args) {
        if (rank_ == 0 && verbose_) {
            (std::cout << ... << args) << std::endl;
        }
    }
    
    template<typename... Args>
    void info(Args... args) {
        if (rank_ == 0) {
            std::cout << "[INFO] ";
            (std::cout << ... << args) << std::endl;
        }
    }
    
    template<typename... Args>
    void error(Args... args) {
        if (rank_ == 0) {
            std::cerr << "[ERROR] ";
            (std::cerr << ... << args) << std::endl;
        }
    }
    
    void separator(char c = '=', int width = 60) {
        if (rank_ == 0) {
            std::cout << std::string(width, c) << std::endl;
        }
    }
    
private:
    int rank_;
    bool verbose_;
};

/**
 * 简单的进度条
 */
class ProgressBar {
public:
    ProgressBar(int total, int rank = 0) 
        : total_(total), current_(0), rank_(rank) {}
    
    void update(int n = 1) {
        if (rank_ != 0) return;
        
        current_ += n;
        int progress = (current_ * 100) / total_;
        int bar_width = 50;
        int pos = (current_ * bar_width) / total_;
        
        std::cout << "\r[";
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << progress << "% (" << current_ << "/" << total_ << ")";
        std::cout.flush();
        
        if (current_ >= total_) {
            std::cout << std::endl;
        }
    }
    
private:
    int total_;
    int current_;
    int rank_;
};

} // namespace dp_utils

#endif // DP_UTILS_HPP