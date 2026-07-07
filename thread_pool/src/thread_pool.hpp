#pragma once

#include <vector>
#include <thread>
#include <functional>
#include <queue>
#include <mutex>
#include <future>


class ThreadPool {
    public:
        explicit ThreadPool(size_t num_threads);

        ~ThreadPool();

        template <typename F>
        auto enqueue(F&& task) -> std::future<typename std::result_of<F()>::type> {
            using return_type = typename std::result_of<F()>::type;

            auto task_ptr = std::make_shared<std::packaged_task<return_type()>>(
                std::move(task)
            );

            std::future<return_type> fut = task_ptr->get_future();

            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                tasks.push([task_ptr] { (*task_ptr)(); } );
            }
            
            cv.notify_one();
            return std::move(fut);
        }
    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;

        std::mutex queue_mutex;
        std::condition_variable cv;
        bool stop = false;

        void workerLoop();
};