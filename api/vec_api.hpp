#pragma once

#include "api/vec_value.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

namespace fhegpu {

enum class VecExecMode { Sync, Async };

struct VecExecConfig {
    VecExecMode mode = VecExecMode::Sync;
    std::uint64_t delay_seed = 0;
    int max_delay_ms = 0;
};

class VecExecutor {
public:
    explicit VecExecutor(VecExecConfig config = {});
    ~VecExecutor();
    VecExecutor(const VecExecutor &) = delete;
    VecExecutor &operator=(const VecExecutor &) = delete;

    VecValue compute(const ComputeOp &op, const std::vector<VecValue> &inputs);
    void stop();

private:
    struct Worker {
        std::mutex mutex;
        std::condition_variable cv;
        std::deque<std::function<void()>> tasks;
        bool stopping = false;
        std::thread thread;
    };

    VecPayload compute_now(const ComputeOp &op, const std::vector<VecValue> &inputs);
    VecMetadata compute_metadata(const ComputeOp &op, const std::vector<VecValue> &inputs) const;
    Worker &worker_for(const Place &place);
    int next_delay_ms();

    VecExecConfig config_;
    std::mutex workers_mutex_;
    std::map<Place, std::unique_ptr<Worker>> workers_;
    std::mt19937_64 random_;
    std::mutex random_mutex_;
    bool stopped_ = false;
};

} // namespace fhegpu
