#include "batch_collector.hpp"

#include <chrono>

BatchCollector::BatchCollector(Inference& inference, int max_batch_size, int timeout_ms)
    : inference_(inference),
      max_batch_size_(max_batch_size),
      timeout_ms_(timeout_ms) {
    batch_thread_ = std::thread(&BatchCollector::processBatch, this);
}

BatchCollector::~BatchCollector() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
    }
    cv_.notify_all();
    batch_thread_.join();
}

std::future<Inference::Result> BatchCollector::add(std::vector<float> image) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.push_back(PendingRequest{std::move(image), {}});
    auto future = pending_.back().promise.get_future();

    if (static_cast<int>(pending_.size()) >= max_batch_size_) {
        cv_.notify_all();
    }

    return future;
}

void BatchCollector::processBatch() {
    std::unique_lock<std::mutex> lock(mutex_);

    while (running_) {
        cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms_), [this] {
            return !running_ || static_cast<int>(pending_.size()) >= max_batch_size_;
        });

        if (pending_.empty()) {
            continue;
        }

        std::vector<PendingRequest> batch = std::move(pending_);
        pending_.clear();

        lock.unlock();

        std::vector<std::vector<float>> images;
        images.reserve(batch.size());
        for (auto& request : batch) {
            images.push_back(std::move(request.image));
        }

        std::vector<Inference::Result> results = inference_.predictBatch(images);

        for (size_t i = 0; i < batch.size(); ++i) {
            batch[i].promise.set_value(results[i]);
        }

        lock.lock();
    }
}
