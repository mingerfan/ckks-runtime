#include "api/mock_api.hpp"

#include <chrono>
#include <sstream>
#include <thread>

namespace fhegpu {

MockCluster::MockCluster(MockClusterConfig config) : config_(std::move(config)) {
    if (config_.world_size <= 0 || config_.max_delay_ms < 0) throw std::runtime_error("invalid MockCluster configuration");
}

void MockCluster::preflight(int rank, std::string plan_source_sha256,
                            bool skip_artifact_digest_checks) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!plan_digests_.emplace(rank, std::move(plan_source_sha256)).second)
        throw std::runtime_error("rank called preflight twice");
    skip_digest_checks_.emplace(rank, skip_artifact_digest_checks);
    cv_.notify_all();
    cv_.wait(lock, [&] { return aborted_ || plan_digests_.size() == static_cast<std::size_t>(config_.world_size); });
    if (aborted_) throw ClusterPanic(abort_reason_);
    const bool skip = skip_digest_checks_.begin()->second;
    for (const auto &entry : skip_digest_checks_) {
        if (entry.second != skip) {
            aborted_ = true;
            abort_reason_ = "skip_artifact_digest_checks mismatch across mock ranks";
            cv_.notify_all();
            throw ClusterPanic(abort_reason_);
        }
    }
    if (!skip) {
        const auto &expected = plan_digests_.begin()->second;
        for (const auto &entry : plan_digests_) if (entry.second != expected) {
            aborted_ = true;
            abort_reason_ = "plan source SHA-256 mismatch across mock ranks";
            cv_.notify_all();
            throw ClusterPanic(abort_reason_);
        }
    }
}

int MockCluster::delay_ms(TransferId id, std::size_t slot) const {
    if (config_.max_delay_ms == 0) return 0;
    std::uint64_t x = config_.delay_seed ^ (id * 0x9e3779b97f4a7c15ULL) ^ (slot * 0xbf58476d1ce4e5b9ULL);
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 27; x *= 0x94d049bb133111ebULL; x ^= x >> 31;
    return static_cast<int>(x % static_cast<std::uint64_t>(config_.max_delay_ms + 1));
}

bool MockCluster::corrupt_output_count(TransferId id) const { return config_.corrupt_output_count.count(id) != 0; }
bool MockCluster::corrupt_output_type(TransferId id) const { return config_.corrupt_output_type.count(id) != 0; }
bool MockCluster::corrupt_output_metadata(TransferId id) const { return config_.corrupt_output_metadata.count(id) != 0; }
bool MockCluster::capability_available(RequiredCapability capability) const {
    return config_.missing_capabilities.count(capability) == 0;
}
bool MockCluster::key_available(const KeyRequirement &key) const {
    return config_.missing_keys.count(key) == 0;
}

void MockCluster::publish(TransferId id, std::size_t slot, VecValue value) {
    const int delay = delay_ms(id, slot);
    if (delay) std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    std::lock_guard<std::mutex> lock(mutex_);
    if (aborted_) throw ClusterPanic(abort_reason_);
    if (config_.fail_publish.count(id)) throw std::runtime_error("injected mock publish failure");
    if (!messages_.emplace(Key{id, slot}, value.deep_copy()).second) throw std::runtime_error("duplicate mock message");
    cv_.notify_all();
}

VecValue MockCluster::receive(TransferId id, std::size_t slot) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return aborted_ || messages_.count(Key{id, slot}) != 0; });
    if (aborted_) throw ClusterPanic(abort_reason_);
    if (config_.fail_wait.count(id)) throw std::runtime_error("injected mock wait failure");
    return messages_.at(Key{id, slot}).deep_copy();
}

[[noreturn]] void MockCluster::abort_all(const std::string &reason) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!aborted_) { aborted_ = true; abort_reason_ = reason; }
    }
    cv_.notify_all();
    throw ClusterPanic(abort_reason_);
}

bool MockCluster::aborted() const { std::lock_guard<std::mutex> lock(mutex_); return aborted_; }

MockVecApi::MockVecApi(int rank, std::shared_ptr<MockCluster> cluster, VecExecConfig exec_config,
                       std::optional<ComputeKind> fail_compute, std::set<TransferId> fail_communicate)
    : rank_(rank), cluster_(std::move(cluster)), executor_(exec_config),
      fail_compute_(fail_compute), fail_communicate_(std::move(fail_communicate)) {
    if (!cluster_) throw std::runtime_error("MockVecApi requires a cluster");
}

MockVecApi::~MockVecApi() = default;

MockVecApi::Value MockVecApi::encode_plaintext(const ValueDesc &output_desc,
                                               const std::vector<double> &slots) {
    return make_plain(slots, output_desc.context, poly_degree_, output_desc.level,
                      output_desc.scale_log2, output_desc.ntt);
}

MockVecApi::Value MockVecApi::compute(const ComputeOp &op, const std::vector<Value> &inputs) {
    { std::lock_guard<std::mutex> lock(stats_mutex_); ++stats_.compute_calls; }
    if (fail_compute_ && *fail_compute_ == op.kind) throw std::runtime_error("injected compute failure");
    return executor_.compute(op, inputs);
}

MockVecApi::CommHandle MockVecApi::communicate_async(const CommAction &action, const std::vector<Value> &local_inputs) {
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++stats_.communicate_calls;
        stats_.implementations.push_back(action.kind == CommKind::Replicate && action.hint == CommHint::Broadcast
                                             ? "point-to-point broadcast fallback" : "point-to-point");
    }
    if (fail_communicate_.count(action.id)) throw std::runtime_error("injected communicate_async failure");
    CommHandle handle;
    handle.id = action.id;
    for (std::size_t i = 0; i < action.destinations.size(); ++i)
        if (action.destinations[i].rank == rank_) handle.local_slots.push_back(i);

    if (action.sources.at(0).rank == rank_) {
        if (local_inputs.size() != 1) throw std::runtime_error("mock sender requires one local input");
        const Value source = local_inputs[0];
        const auto destinations = action.destinations;
        const TransferId id = action.id;
        auto cluster = cluster_;
        handle.has_sender = true;
        handle.sender = std::async(std::launch::async, [source, destinations, id, cluster] {
            try {
                for (std::size_t i = 0; i < destinations.size(); ++i) cluster->publish(id, i, source);
            } catch (const ClusterPanic &) {
                throw;
            } catch (const std::exception &error) {
                cluster->abort_all(std::string("mock sender failed: ") + error.what());
            }
        });
    } else if (!local_inputs.empty()) {
        throw std::runtime_error("non-source rank supplied communication input");
    }
    return handle;
}

std::vector<MockVecApi::Value> MockVecApi::wait(CommHandle &handle) {
    if (handle.waited) throw std::runtime_error("communication handle waited twice");
    handle.waited = true;
    { std::lock_guard<std::mutex> lock(stats_mutex_); ++stats_.wait_calls; }
    if (handle.has_sender) handle.sender.get();
    std::vector<Value> outputs;
    for (std::size_t slot : handle.local_slots) outputs.push_back(cluster_->receive(handle.id, slot));
    if (cluster_->corrupt_output_count(handle.id) && !outputs.empty()) outputs.pop_back();
    if (cluster_->corrupt_output_type(handle.id) && !outputs.empty()) {
        VecPayload payload = outputs.front().materialize();
        payload.kind = payload.kind == ValueKind::Ciphertext ? ValueKind::Plaintext : ValueKind::Ciphertext;
        payload.metadata.components = payload.kind == ValueKind::Plaintext ? 1 : 2;
        outputs.front() = VecValue::ready(std::move(payload));
    }
    if (cluster_->corrupt_output_metadata(handle.id) && !outputs.empty()) {
        VecPayload payload = outputs.front().materialize();
        ++payload.metadata.level;
        outputs.front() = VecValue::ready(std::move(payload));
    }
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++stats_.completed_handles;
    }
    return outputs;
}

void MockVecApi::synchronize(Value &value) { static_cast<void>(value.materialize()); }

void MockVecApi::preflight(std::string_view plan_source_sha256,
                           bool skip_artifact_digest_checks,
                           const TargetConfig &target,
                           const OperatorSpec &operator_spec,
                           const PlanRequirements &requirements) {
    poly_degree_ = operator_spec.poly_degree;
    if (target.capability_version != 1) throw std::runtime_error("MockVecApi does not support target capability_version");
    for (RequiredCapability capability : requirements.capabilities)
        if (!cluster_->capability_available(capability))
            throw std::runtime_error("Api lacks required capability: " + to_string(capability));
    for (const auto &key : requirements.keys)
        if (!cluster_->key_available(key)) {
            std::string message = "Api lacks required " + to_string(key.kind) + " key at " + to_string(key.place);
            if (key.rotation_step) message += " for rotation step " + std::to_string(*key.rotation_step);
            throw std::runtime_error(message);
        }
    cluster_->preflight(rank_, std::string(plan_source_sha256), skip_artifact_digest_checks);
}

void MockVecApi::validate_value(const Value &value, const ValueDesc &expected) const {
    const VecMetadata metadata = value.metadata();
    if (value.kind() != expected.kind || metadata.context != expected.context ||
        metadata.degree != poly_degree_ || metadata.level != expected.level ||
        metadata.scale_log2 != expected.scale_log2 || metadata.ntt != expected.ntt ||
        metadata.components != expected.components)
        throw std::runtime_error("Api value metadata does not match ValueDesc " + std::to_string(expected.id));
}

[[noreturn]] void MockVecApi::abort_all(int, const std::string &reason) { cluster_->abort_all(reason); }

MockStats MockVecApi::stats() const { std::lock_guard<std::mutex> lock(stats_mutex_); return stats_; }

} // namespace fhegpu
