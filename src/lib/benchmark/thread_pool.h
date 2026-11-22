#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace SVD_Project::Benchmark {

class ThreadPool {
 public:
  explicit ThreadPool(size_t num_threads) {
    size_t count = std::max<size_t>(1, num_threads);
    workers_.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      workers_.emplace_back([this]() { worker_loop(); });
    }
  }

  ~ThreadPool() { shutdown(); }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void enqueue(std::function<void()> task) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      tasks_.push(std::move(task));
    }
    cv_.notify_one();
  }

  void wait_all() {
    std::unique_lock<std::mutex> lock(mu_);
    done_cv_.wait(lock, [this]() { return tasks_.empty() && active_ == 0; });
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (stop_) return;
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
      if (w.joinable()) w.join();
    }
    workers_.clear();
  }

  size_t size() const { return workers_.size(); }

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  size_t active_{0};
  bool stop_{false};

  void worker_loop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
        if (stop_ && tasks_.empty()) return;
        task = std::move(tasks_.front());
        tasks_.pop();
        ++active_;
      }
      task();
      {
        std::lock_guard<std::mutex> lock(mu_);
        --active_;
        if (tasks_.empty() && active_ == 0) {
          done_cv_.notify_all();
        }
      }
    }
  }
};

}  // namespace SVD_Project::Benchmark
