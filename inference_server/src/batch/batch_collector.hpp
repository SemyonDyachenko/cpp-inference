#pragma once

#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include "../inference/inference.hpp"

class BatchCollector {
public:
    BatchCollector(Inference& inference, int max_batch_size, int timeout_ms);
    ~BatchCollector();

    std::future<Inference::Result> add(std::vector<float> image);

private:
    void processBatch();

    Inference& inference_;
    int max_batch_size_;
    int timeout_ms_;

    std::mutex mutex_;
    std::condition_variable cv_;

    struct PendingRequest {
        std::vector<float> image;
        std::promise<Inference::Result> promise;
    };

    std::vector<PendingRequest> pending_;
    std::thread batch_thread_;
    bool running_ = true;
};
