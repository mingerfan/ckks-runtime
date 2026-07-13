#pragma once

#include "api/vec_api.hpp"

#include <condition_variable>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>

namespace fhegpu {

class ClusterPanic : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct MockClusterConfig {
    int world_size = 1;
    std::uint64_t delay_seed = 0;
    int max_delay_ms = 0;
    std::set<TransferId> fail_publish;
    std::set<TransferId> fail_wait;
    std::set<TransferId> corrupt_output_count;
    std::set<TransferId> corrupt_output_type;
    std::optional<ComputeKind> fail_compute;
    std::set<TransferId> fail_communicate;
};

class MockCluster {
public:
    explicit MockCluster(MockClusterConfig config);
    void preflight(int rank, std::uint64_t fingerprint);
    void publish(TransferId id, std::size_t slot, VecValue value);
    VecValue receive(TransferId id, std::size_t slot);
    int delay_ms(TransferId id, std::size_t slot) const;
    bool corrupt_output_count(TransferId id) const;
    bool corrupt_output_type(TransferId id) const;
    [[noreturn]] void abort_all(const std::string &reason);
    bool aborted() const;

private:
    using Key = std::pair<TransferId, std::size_t>;
    MockClusterConfig config_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::map<int, std::uint64_t> fingerprints_;
    std::map<Key, VecValue> messages_;
    bool aborted_ = false;
    std::string abort_reason_;
};

struct MockStats {
    std::size_t compute_calls = 0;
    std::size_t communicate_calls = 0;
    std::size_t wait_calls = 0;
    std::size_t completed_handles = 0;
    std::vector<std::string> implementations;
};

class MockVecApi {
public:
    using Value = VecValue;
    struct CommHandle {
        TransferId id = 0;
        std::vector<std::size_t> local_slots;
        std::future<void> sender;
        bool has_sender = false;
        bool waited = false;
    };

    MockVecApi(int rank, std::shared_ptr<MockCluster> cluster,
               VecExecConfig exec_config = {}, std::optional<ComputeKind> fail_compute = std::nullopt,
               std::set<TransferId> fail_communicate = {});
    ~MockVecApi();

    std::string name() const { return "MockVecApi"; }
    Value compute(const ComputeOp &op, const std::vector<Value> &inputs);
    CommHandle communicate_async(const CommAction &action, const std::vector<Value> &local_inputs);
    std::vector<Value> wait(CommHandle &handle);
    void synchronize(Value &value);
    void preflight(std::uint64_t fingerprint);
    [[noreturn]] void abort_all(int exit_code, const std::string &reason);
    ValueKind kind(const Value &value) const { return value.kind(); }
    MockStats stats() const;

private:
    int rank_;
    std::shared_ptr<MockCluster> cluster_;
    VecExecutor executor_;
    std::optional<ComputeKind> fail_compute_;
    std::set<TransferId> fail_communicate_;
    mutable std::mutex stats_mutex_;
    MockStats stats_;
};

} // namespace fhegpu
