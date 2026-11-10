#pragma once

#include "runtime/execution_config.hpp"
#include "runtime/plaintext_bundle.hpp"
#include "runtime/thread_trace.hpp"
#include "runtime/verifier.hpp"
#include "runtime/waterline.hpp"
#include <deque>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace fhegpu {

namespace detail {

template <class Api, class = void>
struct HasPostedOutputs : std::false_type {};

template <class Api>
struct HasPostedOutputs<
    Api, std::void_t<decltype(std::declval<Api &>().posted_outputs(
             std::declval<typename Api::CommHandle &>()))>> : std::true_type {};

template <class Api, class = void>
struct UsesBackgroundCommunicationIssuer : std::false_type {};

template <class Api>
struct UsesBackgroundCommunicationIssuer<
    Api, std::void_t<decltype(Api::background_communication_issuer)>>
    : std::bool_constant<Api::background_communication_issuer> {};

template <class Api, class = void>
struct HasComputeReuse : std::false_type {};

template <class Api>
struct HasComputeReuse<Api, std::void_t<
    decltype(std::declval<Api &>().supports_reuse(std::declval<const ComputeOp &>())),
    decltype(std::declval<Api &>().compute_reuse(std::declval<const ComputeOp &>(),
        std::declval<typename Api::Value>(),
        std::declval<const std::vector<typename Api::Value> &>()))>> : std::true_type {};

// Whether the Api can say that waiting on a posted handle may be put off: its
// outputs already carry the completion event consumers wait on, on the device,
// and nothing is left for the wait to reclaim.  An Api without the method
// keeps the wait where it has always been.
template <class Api, class = void>
struct CanDeferCommunicationWait : std::false_type {};

template <class Api>
struct CanDeferCommunicationWait<
    Api, std::void_t<decltype(std::declval<const Api &>().wait_can_be_deferred(
             std::declval<const typename Api::CommHandle &>()))>>
    : std::true_type {};

} // namespace detail

namespace detail {
template<class Api, class = void> struct HasWaterlineApi : std::false_type {};
template<class Api,class=void>struct HasWaterlineValidation:std::false_type{};
template<class Api>struct HasWaterlineValidation<Api,std::void_t<decltype(std::declval<Api &>().validate_waterline_resources(std::declval<const RuntimePlan &>()))>>:std::true_type{};
template<class Api,class=void>struct HasObservePlan:std::false_type{};
template<class Api>struct HasObservePlan<Api,std::void_t<decltype(std::declval<Api &>().observe_plan(std::declval<const RuntimePlan &>()))>>:std::true_type{};
template<class Api,class=void>struct HasWaterlineAllocatorCapacity:std::false_type{};
template<class Api>struct HasWaterlineAllocatorCapacity<Api,std::void_t<decltype(std::declval<Api &>().waterline_allocator_capacity())>>:std::true_type{};
template<class Api,class=void>struct HasWaterlineFixedResources:std::false_type{};
template<class Api>struct HasWaterlineFixedResources<Api,std::void_t<decltype(std::declval<Api &>().waterline_fixed_resources())>>:std::true_type{};
template<class Api> struct HasWaterlineApi<Api, std::void_t<
    decltype(std::declval<Api &>().prepare_resources(std::declval<const ValueDesc &>(), PrepareBackend::Cpu)),
    decltype(std::declval<Api &>().value_resources(std::declval<const ValueDesc &>())),
    decltype(std::declval<Api &>().instruction_workspace(std::declval<const Instruction &>())),
    decltype(std::declval<Api &>().prepare_plaintext_async(std::declval<const ValueDesc &>(),
        std::declval<const std::vector<double> &>(), PrepareBackend::Cpu)),
    decltype(std::declval<Api &>().is_ready(std::declval<typename Api::Value &>()))>> : std::true_type {};
}
enum class DiffMode { FinalOnly, AllValuesAfterRun };
enum class DeviceExecutionMode {
    Sequential,
    PerDeviceWorkers,
    PerDeviceReadyWorkers,
    PerDeviceDependencyWorkers
};

// The one place the file format's mode names are tied to this scheduler's
// enum.  Kept here rather than in execution_config.cpp so the config header
// does not have to know what the runtime's scheduler looks like today.
inline DeviceExecutionMode device_execution_mode_of(const ExecutionConfig &config) {
    switch (config.runtime.device_execution_mode) {
        case ConfigDeviceExecutionMode::Sequential:
            return DeviceExecutionMode::Sequential;
        case ConfigDeviceExecutionMode::PerDevice:
            return DeviceExecutionMode::PerDeviceWorkers;
        case ConfigDeviceExecutionMode::ReadyBackfill:
            return DeviceExecutionMode::PerDeviceReadyWorkers;
        case ConfigDeviceExecutionMode::DependencyWorkers:
            return DeviceExecutionMode::PerDeviceDependencyWorkers;
    }
    return DeviceExecutionMode::Sequential;
}

struct RuntimeResources {
    const LoadedOperatorSpec &operator_spec;
    std::optional<std::filesystem::path> plaintext_bundle_dir;
    bool skip_artifact_digest_checks = false;
    BundleReadOptions plaintext_bundle_options{};

    // Optional sidecar.  When present its binding is checked against the plan
    // before anything executes; when absent the runtime behaves exactly as it
    // did before this file existed, which keeps every existing caller working
    // and keeps "no configuration" a reachable baseline.
    const LoadedExecutionConfig *execution_config = nullptr;
};

template <class Value>
struct ArtifactValue { Place place; Value value; };

// Wall time inside Api::compute. Asynchronous APIs report submission time.
struct RuntimeTiming {
    std::size_t encode_calls = 0, bundle_read_calls = 0, fence_calls = 0;
    std::uint64_t encode_nanoseconds = 0, bundle_read_nanoseconds = 0;
    std::uint64_t bundle_read_bytes = 0, raw_peak_bytes = 0, fence_nanoseconds = 0;
    std::uint64_t bundle_resident_bytes = 0, bundle_resident_load_nanoseconds = 0;
    // JSON manifest: hashing this rank's blobs once when the bundle is opened.
    std::uint64_t bundle_verify_nanoseconds = 0;
    std::size_t compute_calls = 0;
    std::size_t boot_calls = 0;
    std::uint64_t compute_including_boot_nanoseconds = 0;
    std::uint64_t boot_nanoseconds = 0;
    // Steps of compiler-expanded Boots (V6). Counted apart from boot_calls,
    // which stays the number of whole Boot instructions.
    std::size_t boot_step_calls = 0;
    std::uint64_t boot_step_nanoseconds = 0;
    std::uint64_t setup_nanoseconds = 0;
    std::uint64_t initialization_nanoseconds = 0;
    std::uint64_t online_execution_nanoseconds = 0;
    std::size_t communication_post_calls = 0;
    std::size_t communication_wait_calls = 0;
    std::uint64_t communication_post_nanoseconds = 0;
    std::uint64_t communication_wait_nanoseconds = 0;
    std::size_t device_backfill_tasks = 0;
    std::size_t device_ready_wait_calls = 0;
    std::uint64_t device_ready_wait_nanoseconds = 0;

    std::uint64_t compute_excluding_boot_nanoseconds() const {
        return compute_including_boot_nanoseconds - boot_nanoseconds;
    }
};

template <class Value>
struct RunArtifact {
    std::map<ValueId, ArtifactValue<Value>> values;
    RuntimeTiming timing;
};

template <class Api>
class ValueStore {
public:
    using Value = typename Api::Value;
    struct Ready { Place place; Value value; };
    struct Pending { Place place; std::size_t group; std::size_t local_slot; };
    struct Released { Place place; };
    using Entry = std::variant<Ready, Pending, Released>;

    void define_ready(ValueId id, Place place, Value value) {
        if (!entries_.emplace(id, Ready{place, std::move(value)}).second)
            throw std::runtime_error("ValueId defined twice in ValueStore: " + std::to_string(id));
    }
    void define_pending(ValueId id, Place place, std::size_t group, std::size_t local_slot) {
        if (!entries_.emplace(id, Pending{place, group, local_slot}).second)
            throw std::runtime_error("ValueId defined twice in ValueStore: " + std::to_string(id));
    }
    Entry &lookup(ValueId id) {
        auto it = entries_.find(id);
        if (it == entries_.end()) throw std::runtime_error("ValueId is absent from local ValueStore: " + std::to_string(id));
        return it->second;
    }
    const std::unordered_map<ValueId, Entry> &entries() const { return entries_; }
private:
    std::unordered_map<ValueId, Entry> entries_;
};

template <class Api>
class SequentialRuntime {
public:
    using Value = typename Api::Value;
    using CommHandle = typename Api::CommHandle;

    SequentialRuntime(int rank, int world_size, int local_devices, Api &api,
                      DeviceExecutionMode device_execution_mode =
                          DeviceExecutionMode::Sequential,
                      std::size_t device_worker_count = 0)
        : rank_(rank), world_size_(world_size), local_devices_(local_devices), api_(api),
          device_execution_mode_(device_execution_mode),
          device_worker_count_(device_worker_count) {}

    RunArtifact<Value> run(const LoadedRuntimePlan &loaded_plan,
                           const RuntimeResources &resources,
                           const std::unordered_map<ValueId, Value> &local_inputs,
                           DiffMode diff_mode = DiffMode::FinalOnly) {
        plan_ = &loaded_plan.plan;
        plan_source_sha256_ = loaded_plan.source_sha256;
        current_ = nullptr;
        current_value_.reset();
        store_ = ValueStore<Api>{};
        groups_.clear();
        uses_.clear();
        reset_parallel_execution_state();
        bundle_slots_.clear();
        bundle_index_.reset();
        descriptors_.clear();
        for (const auto &value : plan_->values) descriptors_.emplace(value.id, &value);
        return_all_values_ = diff_mode == DiffMode::AllValuesAfterRun;
        timing_ = RuntimeTiming{};
        ThreadTrace::set_thread_name("rank-" + std::to_string(rank_) + "-main");
        open_trace();
        open_wait_trace();
        try {
            const auto setup_start = std::chrono::steady_clock::now();
            if (resources.skip_artifact_digest_checks) {
                std::cerr << "WARNING: rank " << rank_
                          << " is running with skip_artifact_digest_checks=true; artifact digest comparisons are disabled\n";
                std::cerr.flush();
            }
            const PlanRequirements requirements = PlanVerifier::verify(
                *plan_, resources.operator_spec, resources.skip_artifact_digest_checks);
            PlanVerifier::verify_runtime_target(*plan_, rank_, world_size_, local_devices_);
            // Before the bundle is loaded and before any input is bound: a
            // configuration generated for another plan must not get as far as
            // touching data.
            if (resources.execution_config != nullptr)
                verify_execution_config(*resources.execution_config, loaded_plan, rank_,
                                        world_size_, local_devices_);
            if (plan_->waterline && device_execution_mode_ != DeviceExecutionMode::PerDeviceWorkers)
                throw std::runtime_error("Waterline uses the per-device executor; select PerDeviceWorkers");
            if (plan_->waterline && !detail::HasWaterlineApi<Api>::value)
                throw std::runtime_error("Api does not support complete plaintext preparation/resource queries");
            check_memory_instruction_support(diff_mode);
            count_uses();
            load_bundle(resources);
            api_.preflight(loaded_plan.preflight_identity.empty() ? loaded_plan.source_sha256
                                                                : loaded_plan.preflight_identity,
                           resources.skip_artifact_digest_checks,
                           plan_->target, resources.operator_spec.spec, requirements);
            if constexpr(detail::HasWaterlineValidation<Api>::value)
                if(plan_->waterline)api_.validate_waterline_resources(*plan_);
            // Optional: an Api may inspect the whole plan (e.g. to defer
            // linear combinations) before any value is bound or encoded.
            if constexpr(detail::HasObservePlan<Api>::value)api_.observe_plan(*plan_);
            if constexpr(detail::HasWaterlineAllocatorCapacity<Api>::value)
                if(plan_->waterline)waterline_allocator_capacity_=api_.waterline_allocator_capacity();
            bind_inputs(local_inputs);

            const auto initialization_start = std::chrono::steady_clock::now();
            timing_.setup_nanoseconds = elapsed_nanoseconds(
                setup_start, initialization_start);
            if (!worker_initialization()) {
                execute_phase(plan_->initialization);
                finish_all_groups();
                drain_initialization_transfers();
            }

            const auto online_execution_start = std::chrono::steady_clock::now();
            timing_.initialization_nanoseconds = elapsed_nanoseconds(
                initialization_start, online_execution_start);
            trace_event("phase", 0, 0, std::nullopt, "online_start", 0);
            if (device_execution_mode_ ==
                DeviceExecutionMode::PerDeviceDependencyWorkers) {
                execute_dependency_parallel_phases();
            } else if (device_execution_mode_ ==
                       DeviceExecutionMode::PerDeviceReadyWorkers) {
                execute_ready_parallel_phases();
            } else if (device_execution_mode_ == DeviceExecutionMode::PerDeviceWorkers) {
                execute_device_parallel_phases();
            } else {
                execute_phase(plan_->execution);
                execute_phase(plan_->finalization);
                finish_all_groups();
            }
            synchronize_final_outputs();
            api_.drain();
            timing_.online_execution_nanoseconds = elapsed_nanoseconds(
                online_execution_start, std::chrono::steady_clock::now());
            if (worker_initialization()) timing_.online_execution_nanoseconds -= timing_.initialization_nanoseconds;
            trace_event("phase", 0, 0, std::nullopt, "online_end",
                        timing_.online_execution_nanoseconds);
            auto artifact = make_artifact(diff_mode);
            store_ = ValueStore<Api>{};
            groups_.clear();
            reset_parallel_execution_state();
            ThreadTrace::write_json(rank_);
            return artifact;
        } catch (const std::exception &error) {
            const std::string diagnostic = format_error(error.what());
            std::cerr << diagnostic << std::endl;
            std::cerr.flush();
            try {
                // Distributed backends may terminate the job here. Abort them
                // before attempting a drain that could wait for a failed peer.
                api_.abort_all(1, diagnostic);
            } catch (...) {
                // Local backends report abort by throwing. Finish their already
                // submitted work and drop runtime-owned values before returning
                // the original error to a caller that keeps this runtime alive.
                const auto failure = std::current_exception();
                try { api_.drain(); }
                catch (const std::exception &cleanup) {
                    std::cerr << "runtime failure cleanup also failed: " << cleanup.what() << std::endl;
                }
                catch (...) {
                    std::cerr << "runtime failure cleanup also failed with a non-standard exception" << std::endl;
                }
                store_ = ValueStore<Api>{};
                groups_.clear();
                reset_parallel_execution_state();
                std::rethrow_exception(failure);
            }
        }
    }

private:
    bool worker_initialization() const {
        return plan_->waterline.has_value() ||
            (device_execution_mode_ == DeviceExecutionMode::PerDeviceWorkers && local_devices_ == 0);
    }

    void check_memory_instruction_support(DiffMode mode) {
        const auto check_phase = [&](const std::vector<Instruction> &instructions) {
            for (const auto &instruction : instructions) {
                const bool release = std::holds_alternative<ReleaseOp>(instruction.body);
                const auto *op = std::get_if<ComputeOp>(&instruction.body);
                if (!release && (!op || !op->reuse_input)) continue;
                current_ = &instruction;
                if (mode == DiffMode::AllValuesAfterRun)
                    throw std::runtime_error("AllValuesAfterRun is incompatible with Release/reuse_input");
                if (op && op->reuse_input) {
                    if constexpr (detail::HasComputeReuse<Api>::value) {
                        if (!api_.supports_reuse(*op))
                            throw std::runtime_error("Api does not support reuse_input for " + to_string(op->kind));
                    } else {
                        throw std::runtime_error("Api does not support reuse_input");
                    }
                }
                current_ = nullptr;
            }
        };
        check_phase(plan_->initialization);
        check_phase(plan_->execution);
        check_phase(plan_->finalization);
    }

    static std::uint64_t elapsed_nanoseconds(
        std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point finish) {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start)
                .count());
    }

    void open_trace() {
        runtime_trace_.reset();
        const char *trace_path = std::getenv("POSEIDON_RUNTIME_TRACE");
        if (trace_path == nullptr || trace_path[0] == '\0' ||
            std::string_view(trace_path) == "0")
            return;
        std::string path = std::string_view(trace_path) == "1"
                               ? "/tmp/poseidon-runtime"
                               : std::string(trace_path);
        const std::string rank_text = std::to_string(rank_);
        std::size_t offset = 0;
        bool replaced = false;
        while ((offset = path.find("%r", offset)) != std::string::npos) {
            path.replace(offset, 2, rank_text);
            offset += rank_text.size();
            replaced = true;
        }
        if (!replaced) path += ".rank" + rank_text + ".csv";
        runtime_trace_ = std::make_unique<std::ofstream>(
            path, std::ios::out | std::ios::trunc);
        if (!*runtime_trace_)
            throw std::runtime_error("cannot open Poseidon Runtime trace file");
        // end_ns is the steady clock the online timing uses, read when the
        // event is written -- right after the operation returns -- so an
        // event started at end_ns - duration_ns.  thread tells the submitting
        // threads apart.  Both let the trace run give a timeline of its own,
        // without the launch overhead a profiler adds.
        *runtime_trace_
            << "rank,event,id,ordinal,consumer_ordinal,op,duration_ns,end_ns,thread\n";
        runtime_trace_->flush();
    }

    void trace_event(const char *event, std::uint64_t id,
                     std::uint64_t ordinal,
                     std::optional<std::uint64_t> consumer_ordinal,
                     std::string_view op,
                     std::uint64_t duration_nanoseconds) {
        if (!runtime_trace_) return;
        const auto end_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
        const std::size_t thread =
            std::hash<std::thread::id>{}(std::this_thread::get_id());
        ThreadTraceLockGuard lock(trace_mutex_, "runtime.trace");
        *runtime_trace_ << rank_ << ',' << event << ',' << id << ','
                        << ordinal << ',';
        if (consumer_ordinal) *runtime_trace_ << *consumer_ordinal;
        *runtime_trace_ << ',' << op << ',' << duration_nanoseconds << ','
                        << end_ns << ',' << thread << '\n';
        runtime_trace_->flush();
    }

    struct PendingGroup {
        CommAction action;
        std::vector<ValueId> local_output_ids;
        std::vector<std::size_t> local_output_slots;
        std::vector<bool> local_output_posted;
        CommHandle handle;
        std::uint64_t instruction_ordinal = 0;
        bool completed = false;
    };

    struct ParallelGroup;

    struct ValueUses {
        std::size_t remaining = 0;
        bool release_requested = false;
    };

    struct ParallelValue {
        Place place;
        std::mutex mutex;
        std::condition_variable condition;
        bool defined = false;
        ValueUses uses;
        std::optional<Value> value;
        std::shared_ptr<ParallelGroup> group;
        std::optional<Value> completion;
    };

    struct ParallelOutput {
        ValueId id = 0;
        std::size_t action_slot = 0;
        std::shared_ptr<ParallelValue> value;
    };

    struct ParallelGroup {
        CommAction action;
        std::uint64_t instruction_ordinal = 0;
        std::vector<ValueId> local_input_ids;
        std::vector<ParallelOutput> local_outputs;
        CommHandle handle;
        std::mutex mutex;
        bool cross_rank = false;
        bool posted = false;
        bool completed = false;
    };

    struct ParallelTask {
        std::function<void()> execute;
        const Instruction *instruction = nullptr;
        std::vector<ValueId> inputs, outputs;
        bool preparation = false;
        PrepareBackend backend = PrepareBackend::Cpu;
        bool final_output = false;
    };

    enum class ParallelTaskReadiness {
        Blocked,
        PendingCommunication,
        Ready
    };

    struct DependencyTaskQueue;

    struct DependencyTask {
        std::uint64_t ordinal = 0;
        std::vector<ValueId> input_ids;
        std::function<void()> execute;
        std::weak_ptr<DependencyTaskQueue> queue;
        std::size_t queue_position = 0;
        std::atomic<std::size_t> pending_inputs{0};
        std::string op_label;
        bool completed = false;
    };

    struct DependencyTaskCompare {
        bool operator()(const std::shared_ptr<DependencyTask> &left,
                        const std::shared_ptr<DependencyTask> &right) const {
            return left->ordinal > right->ordinal;
        }
    };

    struct DependencyTaskQueue {
        std::mutex mutex;
        std::condition_variable condition;
        std::priority_queue<
            std::shared_ptr<DependencyTask>,
            std::vector<std::shared_ptr<DependencyTask>>,
            DependencyTaskCompare> ready;
        std::vector<std::shared_ptr<DependencyTask>> tasks;
        std::size_t next_unfinished = 0;
        std::size_t completed = 0;
        bool compute_queue = false;
        std::size_t index = 0;
    };

    // Optional dependency-wait tracing. When POSEIDON_RUNTIME_WAIT_TRACE is
    // set, every worker queue wait is recorded with the instruction it is
    // blocked on and the producers of its missing inputs, and written as
    // JSON once parallel execution finishes.
    struct ValueProducer {
        std::string queue_kind;
        std::size_t queue_index = 0;
        std::uint64_t ordinal = 0;
        std::string op;
    };

    struct WaitTraceProducer {
        ValueId value = 0;
        std::string queue_kind;
        std::size_t queue_index = 0;
        std::uint64_t ordinal = 0;
        std::string op;
    };

    struct WaitTraceRecord {
        std::uint64_t start_ns = 0;
        std::uint64_t duration_ns = 0;
        std::string queue_kind;
        std::size_t queue_index = 0;
        std::optional<std::uint64_t> waited_ordinal;
        std::string waited_op;
        std::vector<ValueId> missing_inputs;
        std::vector<WaitTraceProducer> producers;
        std::string wake_reason;
    };

    void open_wait_trace() {
        wait_trace_enabled_ = false;
        wait_trace_records_.clear();
        const char *trace_path = std::getenv("POSEIDON_RUNTIME_WAIT_TRACE");
        if (trace_path == nullptr || trace_path[0] == '\0' ||
            std::string_view(trace_path) == "0")
            return;
        std::string path = std::string_view(trace_path) == "1"
                               ? "/tmp/poseidon-runtime-waits"
                               : std::string(trace_path);
        const std::string rank_text = std::to_string(rank_);
        std::size_t offset = 0;
        bool replaced = false;
        while ((offset = path.find("%r", offset)) != std::string::npos) {
            path.replace(offset, 2, rank_text);
            offset += rank_text.size();
            replaced = true;
        }
        if (!replaced) path += ".rank" + rank_text + ".json";
        wait_trace_path_ = std::move(path);
        wait_trace_enabled_ = true;
    }

    const char *device_execution_mode_name() const {
        switch (device_execution_mode_) {
            case DeviceExecutionMode::Sequential:
                return "Sequential";
            case DeviceExecutionMode::PerDeviceWorkers:
                return "PerDeviceWorkers";
            case DeviceExecutionMode::PerDeviceReadyWorkers:
                return "PerDeviceReadyWorkers";
            case DeviceExecutionMode::PerDeviceDependencyWorkers:
                return "PerDeviceDependencyWorkers";
        }
        return "Unknown";
    }

    static std::string json_escape(std::string_view text) {
        std::string escaped;
        escaped.reserve(text.size());
        for (const char character : text) {
            switch (character) {
                case '"': escaped += "\\\""; break;
                case '\\': escaped += "\\\\"; break;
                case '\n': escaped += "\\n"; break;
                case '\r': escaped += "\\r"; break;
                case '\t': escaped += "\\t"; break;
                default: escaped += character; break;
            }
        }
        return escaped;
    }

    // Snapshot what this queue is blocked on before sleeping: the earliest
    // unfinished task in program order and which of its inputs are still
    // un-materialized. Sampled while holding the queue mutex.
    std::optional<WaitTraceRecord> sample_wait_trace(
        const std::shared_ptr<DependencyTaskQueue> &queue) {
        WaitTraceRecord record;
        record.start_ns = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        record.queue_kind = queue->compute_queue ? "compute" : "communication";
        record.queue_index = queue->index;
        if (queue->next_unfinished >= queue->tasks.size()) return record;
        const auto &head = queue->tasks[queue->next_unfinished];
        record.waited_ordinal = head->ordinal;
        record.waited_op = head->op_label;
        for (ValueId id : head->input_ids) {
            const auto found = parallel_values_.find(id);
            if (found == parallel_values_.end()) continue;
            bool materialized = false;
            {
                std::lock_guard<std::mutex> lock(found->second->mutex);
                materialized = found->second->value.has_value();
            }
            if (materialized) continue;
            record.missing_inputs.push_back(id);
            const auto producer = value_producers_.find(id);
            if (producer == value_producers_.end()) continue;
            record.producers.push_back(WaitTraceProducer{
                id, producer->second.queue_kind, producer->second.queue_index,
                producer->second.ordinal, producer->second.op});
        }
        return record;
    }

    void record_wait_trace(WaitTraceRecord record) {
        std::lock_guard<std::mutex> lock(wait_trace_mutex_);
        wait_trace_records_.push_back(std::move(record));
    }

    static void write_json_array(std::ostream &out,
                                 const std::vector<ValueId> &ids) {
        out << '[';
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i != 0) out << ',';
            out << ids[i];
        }
        out << ']';
    }

    static void write_json_waited_for(std::ostream &out,
                                      const WaitTraceRecord &record) {
        out << "\"waited_for\": ";
        if (!record.waited_ordinal) {
            out << "null";
            return;
        }
        out << "{\"ordinal\": " << *record.waited_ordinal
            << ", \"op\": \"" << json_escape(record.waited_op) << "\"}";
    }

    static void write_json_producers(std::ostream &out,
                                     const WaitTraceRecord &record) {
        out << "\"producers\": [";
        for (std::size_t i = 0; i < record.producers.size(); ++i) {
            if (i != 0) out << ',';
            const auto &producer = record.producers[i];
            out << "{\"value\": " << producer.value
                << ", \"queue\": {\"kind\": \"" << producer.queue_kind
                << "\", \"index\": " << producer.queue_index
                << "}, \"ordinal\": " << producer.ordinal
                << ", \"op\": \"" << json_escape(producer.op) << "\"}";
        }
        out << ']';
    }

    void write_wait_trace() {
        if (!wait_trace_enabled_) return;
        std::sort(wait_trace_records_.begin(), wait_trace_records_.end(),
                  [](const WaitTraceRecord &left,
                     const WaitTraceRecord &right) {
                      return left.start_ns < right.start_ns;
                  });
        std::uint64_t total_ns = 0;
        std::uint64_t max_ns = 0;
        std::map<std::pair<std::string, std::size_t>,
                 std::pair<std::size_t, std::uint64_t>>
            per_queue;
        std::map<std::string, std::pair<std::size_t, std::uint64_t>>
            per_op;
        for (const auto &record : wait_trace_records_) {
            total_ns += record.duration_ns;
            max_ns = std::max(max_ns, record.duration_ns);
            auto &queue_stats =
                per_queue[{record.queue_kind, record.queue_index}];
            ++queue_stats.first;
            queue_stats.second += record.duration_ns;
            auto &op_stats = per_op[record.waited_op];
            ++op_stats.first;
            op_stats.second += record.duration_ns;
        }
        std::ofstream out(wait_trace_path_, std::ios::out | std::ios::trunc);
        if (!out)
            throw std::runtime_error(
                "cannot open Poseidon Runtime wait trace file: " +
                wait_trace_path_);
        out << "{\n  \"format_version\": 1,\n";
        out << "  \"mode\": \"" << device_execution_mode_name() << "\",\n";
        out << "  \"worker_count\": " << wait_trace_worker_count_ << ",\n";
        out << "  \"waits\": [\n";
        for (std::size_t i = 0; i < wait_trace_records_.size(); ++i) {
            const auto &record = wait_trace_records_[i];
            out << "    {\"start_ns\": " << record.start_ns
                << ", \"duration_ns\": " << record.duration_ns
                << ", \"queue\": {\"kind\": \"" << record.queue_kind
                << "\", \"index\": " << record.queue_index << "}, ";
            write_json_waited_for(out, record);
            out << ", \"missing_inputs\": ";
            write_json_array(out, record.missing_inputs);
            out << ", ";
            write_json_producers(out, record);
            out << ", \"wake_reason\": \"" << record.wake_reason << "\"}";
            if (i + 1 != wait_trace_records_.size()) out << ',';
            out << '\n';
        }
        out << "  ],\n  \"summary\": {\n";
        out << "    \"wait_count\": " << wait_trace_records_.size() << ",\n";
        out << "    \"total_ns\": " << total_ns << ",\n";
        out << "    \"max_ns\": " << max_ns << ",\n";
        out << "    \"mean_ns\": "
            << (wait_trace_records_.empty()
                    ? 0
                    : total_ns / wait_trace_records_.size())
            << ",\n";
        out << "    \"per_queue\": [\n";
        std::size_t queue_count = 0;
        for (const auto &entry : per_queue) {
            out << "      {\"kind\": \"" << entry.first.first
                << "\", \"index\": " << entry.first.second
                << ", \"count\": " << entry.second.first
                << ", \"total_ns\": " << entry.second.second << "}";
            if (++queue_count != per_queue.size()) out << ',';
            out << '\n';
        }
        out << "    ],\n    \"top_stalled_ops\": [\n";
        std::vector<std::pair<std::string,
                              std::pair<std::size_t, std::uint64_t>>>
            ops(per_op.begin(), per_op.end());
        std::sort(ops.begin(), ops.end(),
                  [](const auto &left, const auto &right) {
                      return left.second.second > right.second.second;
                  });
        const std::size_t op_count = std::min<std::size_t>(ops.size(), 20);
        for (std::size_t i = 0; i < op_count; ++i) {
            out << "      {\"op\": \"" << json_escape(ops[i].first)
                << "\", \"count\": " << ops[i].second.first
                << ", \"total_ns\": " << ops[i].second.second << "}";
            if (i + 1 != op_count) out << ',';
            out << '\n';
        }
        out << "    ]\n  }\n}\n";
    }

    const ValueDesc &desc(ValueId id) const {
        auto found = descriptors_.find(id);
        if (found != descriptors_.end()) return *found->second;
        throw std::runtime_error("missing value descriptor for " + std::to_string(id));
    }

    void count_uses() {
        for (const auto &value : plan_->values)
            if (value.place.rank == rank_) uses_.emplace(value.id, ValueUses{});
        const auto count_phase = [&](const auto &instructions, bool parallel) {
            for (const auto &instruction : instructions) {
                if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) {
                    if (op->place.rank == rank_)
                        for (ValueId id : op->inputs) ++uses_.at(id).remaining;
                } else if (const auto *action = std::get_if<CommAction>(&instruction.body)) {
                    std::size_t submissions = 1;
                    if (parallel) {
                        // Workers post one slice per local destination; the
                        // communication issuer posts all remote destinations together.
                        submissions = is_cross_rank_action(*action) ? 1 : 0;
                        submissions += std::count_if(
                            action->destinations.begin(), action->destinations.end(),
                            [this](const Place &place) { return place.rank == rank_; });
                    }
                    for (std::size_t i = 0; i < action->inputs.size(); ++i)
                        if (action->sources[i].rank == rank_)
                            uses_.at(action->inputs[i]).remaining += submissions;
                }
            }
        };
        count_phase(plan_->initialization, worker_initialization());
        // Every parallel executor slices a communication per local destination
        // and submits each slice on its own.
        const bool parallel = device_execution_mode_ != DeviceExecutionMode::Sequential;
        count_phase(plan_->execution, parallel);
        count_phase(plan_->finalization, parallel);
    }

    bool can_clear(const ValueUses &uses) const {
        return uses.release_requested && uses.remaining == 0;
    }

    void clear_sequential_value(ValueId id) {
        if (!can_clear(uses_.at(id))) return;
        auto &entry = store_.lookup(id);
        if (const auto *ready = std::get_if<typename ValueStore<Api>::Ready>(&entry)) {
            const auto place = ready->place;
            entry = typename ValueStore<Api>::Released{place};
        }
        // Pending data is discarded when the communication result arrives.
    }

    void submitted_sequential_uses(const std::vector<ValueId> &ids) {
        for (ValueId id : ids) {
            auto &uses = uses_.at(id);
            if (uses.remaining == 0)
                throw std::runtime_error("submission count underflow for ValueId " + std::to_string(id));
            --uses.remaining;
            clear_sequential_value(id);
        }
    }

    void execute_release(ValueId id, std::uint64_t ordinal, bool parallel) {
        if (desc(id).place.rank != rank_) return;
        if (parallel) {
            auto value = parallel_value(id);
            {
                ThreadTraceLockGuard lock(value->mutex, "runtime.parallel_value");
                value->uses.release_requested = true;
                if (can_clear(value->uses)) value->value.reset();
            }
            value->condition.notify_all();
            notify_parallel_progress();
        } else {
            uses_.at(id).release_requested = true;
            clear_sequential_value(id);
        }
        // Waterline's dispatcher reaps whole submitted batches. Repeating the
        // same query for every Release only adds contention with that collector.
        if (!parallel || !waterline_) api_.collect_completed();
        trace_event("release", id, ordinal, std::nullopt, "Release", 0);
    }

    void submitted_parallel_uses(const std::vector<ValueId> &ids) {
        for (ValueId id : ids) {
            auto value = parallel_value(id);
            ThreadTraceLockGuard lock(value->mutex, "runtime.parallel_value");
            if (value->uses.remaining == 0)
                throw std::runtime_error("parallel submission count underflow for ValueId " + std::to_string(id));
            --value->uses.remaining;
            if (can_clear(value->uses)) value->value.reset();
        }
    }

    std::vector<ValueDesc> communication_output_descs(
        const CommAction &action) const {
        std::vector<ValueDesc> result;
        result.reserve(action.outputs.size());
        for (ValueId id : action.outputs) result.push_back(desc(id));
        return result;
    }

    std::vector<std::optional<Value>> posted_outputs(
        const CommAction &action, CommHandle &handle) {
        if constexpr (detail::HasPostedOutputs<Api>::value) {
            auto outputs = api_.posted_outputs(handle);
            if (outputs.size() != action.outputs.size())
                throw std::runtime_error(
                    "Api posted_outputs returned wrong output count");
            for (std::size_t slot = 0; slot < outputs.size(); ++slot)
                if (action.destinations[slot].rank != rank_ && outputs[slot])
                    throw std::runtime_error("Api posted_outputs returned a remote output");
            return outputs;
        }
        return std::vector<std::optional<Value>>(action.outputs.size());
    }

    void load_bundle(const RuntimeResources &resources) {
        if (!plan_->plaintext_bundle) return;
        if (!resources.plaintext_bundle_dir)
            throw std::runtime_error("RuntimePlan requires a plaintext bundle directory");
        std::vector<std::string> local_contents;
        for (const auto *phase : {&plan_->initialization, &plan_->execution})
        for (const auto &instruction : *phase) {
            const EncodePayload *payload = nullptr;
            if (const auto *encode = std::get_if<EncodeOp>(&instruction.body)) {
                if (desc(encode->output).place.rank == rank_) payload = &encode->payload;
            } else if (const auto *prepare = std::get_if<PreparePlaintextOp>(&instruction.body)) {
                if (desc(prepare->output).place.rank == rank_) payload = &prepare->payload;
            }
            if (payload) if (const auto *bundle = std::get_if<BundleEncodePayload>(payload)) local_contents.push_back(bundle->content);
        }
        if (plan_->format_version >= 3) {
            bundle_index_ = PlaintextBundleLoader::open(*resources.plaintext_bundle_dir,
                *plan_->plaintext_bundle, local_contents,
                resources.operator_spec.spec.poly_degree / 2,
                resources.skip_artifact_digest_checks, resources.plaintext_bundle_options);
            timing_.bundle_resident_bytes = bundle_index_->resident_bytes();
            timing_.bundle_resident_load_nanoseconds = static_cast<std::uint64_t>(bundle_index_->resident_load_seconds() * 1e9);
            timing_.bundle_verify_nanoseconds = static_cast<std::uint64_t>(bundle_index_->verify_seconds() * 1e9);
            return;
        }
        auto bundle = PlaintextBundleLoader::load(*resources.plaintext_bundle_dir,
                                                   *plan_->plaintext_bundle,
                                                   local_contents,
                                                   resources.operator_spec.spec.poly_degree / 2,
                                                   resources.skip_artifact_digest_checks, resources.plaintext_bundle_options);
        bundle_slots_ = std::move(bundle.slots_by_content);
    }

    void bind_inputs(const std::unordered_map<ValueId, Value> &inputs) {
        std::size_t expected = 0;
        for (ValueId id : plan_->external_inputs) {
            const auto &expected_desc = desc(id);
            if (expected_desc.place.rank != rank_) continue;
            ++expected;
            auto it = inputs.find(id);
            if (it == inputs.end()) throw std::runtime_error("missing local external input " + std::to_string(id));
            api_.validate_value(it->second, expected_desc);
            store_.define_ready(id, expected_desc.place, it->second);
        }
        if (inputs.size() != expected) throw std::runtime_error("unexpected local external input binding");
    }

    void execute_phase(const std::vector<Instruction> &instructions) {
        for (const auto &instruction : instructions) {
            current_ = &instruction;
            if (const auto *encode = std::get_if<EncodeOp>(&instruction.body)) execute_encode(*encode);
            else if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) execute_compute(*op);
            else if (const auto *action = std::get_if<CommAction>(&instruction.body)) execute_communication(*action);
            else if (std::holds_alternative<FenceOp>(instruction.body)) execute_fence();
            else execute_release(std::get<ReleaseOp>(instruction.body).value,
                                 instruction.ordinal, false);
        }
        current_ = nullptr;
    }

    std::vector<double> read_encode_slots(const EncodePayload &payload, ValueId output, std::uint64_t ordinal) {
        std::lock_guard<std::mutex> lock(prepare_read_mutex_);
        std::vector<double> slots;
        if (const auto *inline_payload = std::get_if<InlineEncodePayload>(&payload))
            slots = inline_payload->values;
        else {
            const auto &content = std::get<BundleEncodePayload>(payload).content;
            const auto start = std::chrono::steady_clock::now();
            slots = bundle_index_ ? bundle_index_->read(content) : bundle_slots_.at(content);
            if (bundle_index_) {
                const auto elapsed = elapsed_nanoseconds(start, std::chrono::steady_clock::now());
                ++timing_.bundle_read_calls;
                timing_.bundle_read_nanoseconds += elapsed;
                timing_.bundle_read_bytes += slots.size() * sizeof(double);
                timing_.raw_peak_bytes = std::max(timing_.raw_peak_bytes,
                    static_cast<std::uint64_t>(slots.size() * sizeof(double) * 2));
                trace_event("bundle_read", output, ordinal, std::nullopt, "Read", elapsed);
            }
        }
        for (double &value : slots) if (value == 0.0) value = 0.0;
        return slots;
    }
    Value encode_value(const EncodeOp &op, std::uint64_t ordinal) {
        const auto &output_desc = desc(op.output);
        auto slots = read_encode_slots(op.payload, op.output, ordinal);
        const auto start = std::chrono::steady_clock::now();
        Value output = api_.encode_plaintext(output_desc, slots);
        api_.validate_value(output, output_desc);
        const auto elapsed = elapsed_nanoseconds(start, std::chrono::steady_clock::now());
        {std::lock_guard<std::mutex> lock(parallel_timing_mutex_);
         ++timing_.encode_calls;timing_.encode_nanoseconds+=elapsed;}
        trace_event("encode", op.output, ordinal, std::nullopt, "Encode", elapsed);
        return output;
    }

    void execute_encode(const EncodeOp &op) {
        const auto &output_desc = desc(op.output);
        if (output_desc.place.rank != rank_) return;
        store_.define_ready(op.output, output_desc.place, encode_value(op, current_->ordinal));
    }

    void execute_fence() {
        const auto start = std::chrono::steady_clock::now();
        finish_all_groups();
        api_.drain();
        record_fence(current_ ? current_->ordinal : 0, start);
    }

    void record_fence(std::uint64_t ordinal, std::chrono::steady_clock::time_point start) {
        const auto elapsed = elapsed_nanoseconds(start, std::chrono::steady_clock::now());
        ++timing_.fence_calls;
        timing_.fence_nanoseconds += elapsed;
        trace_event("fence", 0, ordinal, std::nullopt, "Fence", elapsed);
    }

    Value &ensure_ready(ValueId id, const Place &expected_place) {
        auto &entry = store_.lookup(id);
        if (std::holds_alternative<typename ValueStore<Api>::Released>(entry))
            throw std::runtime_error("ValueId was released: " + std::to_string(id));
        if (auto *ready = std::get_if<typename ValueStore<Api>::Ready>(&entry)) {
            if (ready->place != expected_place) throw std::runtime_error("Ready value Place mismatch for " + std::to_string(id));
            return ready->value;
        }
        const auto pending = std::get<typename ValueStore<Api>::Pending>(entry);
        if (pending.place != expected_place) throw std::runtime_error("Pending value Place mismatch for " + std::to_string(id));
        finish_group(pending.group);
        auto &ready = std::get<typename ValueStore<Api>::Ready>(store_.lookup(id));
        return ready.value;
    }

    void execute_compute(const ComputeOp &op) {
        if (op.place.rank != rank_) return;
        std::vector<Value> inputs;
        for (ValueId id : op.inputs) inputs.push_back(ensure_ready(id, op.place));
        const auto start = std::chrono::steady_clock::now();
        if (op.reuse_input) {
            auto &entry = store_.lookup(op.inputs.front());
            entry = typename ValueStore<Api>::Released{op.place};
        }
        Value output = compute_output(op, inputs);
        const auto finish = std::chrono::steady_clock::now();
        inputs.clear();
        submitted_sequential_uses(op.inputs);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            finish - start);
        const auto elapsed_nanoseconds =
            static_cast<std::uint64_t>(elapsed.count());
        trace_event("compute", op.output, current_->ordinal, std::nullopt,
                    to_string(op.kind), elapsed_nanoseconds);
        ++timing_.compute_calls;
        timing_.compute_including_boot_nanoseconds += elapsed_nanoseconds;
        if (op.kind == ComputeKind::Boot) {
            ++timing_.boot_calls;
            timing_.boot_nanoseconds += elapsed_nanoseconds;
        } else if (op.kind == ComputeKind::BootStep) {
            ++timing_.boot_step_calls;
            timing_.boot_step_nanoseconds += elapsed_nanoseconds;
        }
        const auto &output_desc = desc(op.output);
        api_.validate_value(output, output_desc);
        store_.define_ready(op.output, op.place, std::move(output));
    }

    Value compute_output(const ComputeOp &op, std::vector<Value> &inputs) {
        if (!op.reuse_input) return api_.compute(op, inputs);
        if constexpr (detail::HasComputeReuse<Api>::value) {
            Value reused = std::move(inputs.front());
            inputs.erase(inputs.begin());
            return api_.compute_reuse(op, std::move(reused), inputs);
        } else {
            throw std::runtime_error("Api does not support reuse_input");
        }
    }

    void execute_communication(const CommAction &action) {
        std::vector<Value> local_inputs;
        for (std::size_t i = 0; i < action.inputs.size(); ++i)
            if (action.sources[i].rank == rank_) local_inputs.push_back(ensure_ready(action.inputs[i], action.sources[i]));
        PendingGroup group;
        group.action = action;
        group.instruction_ordinal = current_->ordinal;
        const auto post_start = std::chrono::steady_clock::now();
        group.handle = api_.communicate_async(
            action, local_inputs, communication_output_descs(action));
        auto posted = posted_outputs(action, group.handle);
        const auto post_elapsed = elapsed_nanoseconds(
            post_start, std::chrono::steady_clock::now());
        ++timing_.communication_post_calls;
        timing_.communication_post_nanoseconds += post_elapsed;
        trace_event("comm_post", action.id, group.instruction_ordinal,
                    std::nullopt, to_string(action.kind), post_elapsed);
        const std::size_t group_id = groups_.size();
        for (std::size_t i = 0; i < action.outputs.size(); ++i) {
            if (action.destinations[i].rank != rank_) continue;
            const std::size_t local_slot = group.local_output_ids.size();
            group.local_output_ids.push_back(action.outputs[i]);
            group.local_output_slots.push_back(i);
            group.local_output_posted.push_back(posted[i].has_value());
            if (posted[i]) {
                const auto &expected_desc = desc(action.outputs[i]);
                api_.validate_value(*posted[i], expected_desc);
                store_.define_ready(action.outputs[i], action.destinations[i],
                                    std::move(*posted[i]));
            } else {
                store_.define_pending(action.outputs[i], action.destinations[i],
                                      group_id, local_slot);
            }
        }
        groups_.push_back(std::move(group));
        local_inputs.clear();
        std::vector<ValueId> local_ids;
        for (std::size_t i = 0; i < action.inputs.size(); ++i)
            if (action.sources[i].rank == rank_) local_ids.push_back(action.inputs[i]);
        submitted_sequential_uses(local_ids);
    }

    void finish_group(std::size_t group_id) {
        auto &group = groups_.at(group_id);
        if (group.completed) return;
        const auto wait_start = std::chrono::steady_clock::now();
        const std::uint64_t trace_wait_start =
            ThreadTrace::enabled() ? ThreadTrace::timestamp_ns() : 0;
        auto outputs = api_.wait(group.handle);
        if (ThreadTrace::enabled())
            ThreadTrace::record_duration(
                "runtime.communication_wait", &group, trace_wait_start,
                ThreadTrace::timestamp_ns());
        const auto wait_elapsed = elapsed_nanoseconds(
            wait_start, std::chrono::steady_clock::now());
        ++timing_.communication_wait_calls;
        timing_.communication_wait_nanoseconds += wait_elapsed;
        trace_event("comm_wait", group.action.id, group.instruction_ordinal,
                    current_ ? std::optional<std::uint64_t>(current_->ordinal)
                             : std::nullopt,
                    to_string(group.action.kind), wait_elapsed);
        if (outputs.size() != group.action.outputs.size()) throw std::runtime_error("Api wait returned wrong output count");
        for (std::size_t slot = 0; slot < outputs.size(); ++slot)
            if (group.action.destinations[slot].rank != rank_ && outputs[slot])
                throw std::runtime_error("Api wait returned a remote output");
        for (std::size_t i = 0; i < group.local_output_ids.size(); ++i) {
            const ValueId id = group.local_output_ids[i];
            auto &output = outputs[group.local_output_slots[i]];
            if (group.local_output_posted[i]) {
                if (output) throw std::runtime_error("Api delivered a communication output twice");
                continue;
            }
            if (!output) throw std::runtime_error("Api wait omitted a pending communication output");
            const auto &expected_desc = desc(id);
            api_.validate_value(*output, expected_desc);
            if (can_clear(uses_.at(id))) {
                output.reset();
                store_.lookup(id) = typename ValueStore<Api>::Released{expected_desc.place};
            } else {
                store_.lookup(id) = typename ValueStore<Api>::Ready{
                    expected_desc.place, std::move(*output)};
            }
        }
        group.completed = true;
    }

    std::shared_ptr<ParallelValue> parallel_value(ValueId id) const {
        const auto found = parallel_values_.find(id);
        if (found == parallel_values_.end())
            throw std::runtime_error(
                "ValueId is absent from parallel ValueStore: " +
                std::to_string(id));
        return found->second;
    }

    std::shared_ptr<ParallelValue> define_parallel_value(
        ValueId id, const Place &place) {
        auto value = std::make_shared<ParallelValue>();
        value->place = place;
        value->uses = uses_.at(id);
        if (!parallel_values_.emplace(id, value).second)
            throw std::runtime_error(
                "ValueId defined twice in parallel ValueStore: " +
                std::to_string(id));
        return value;
    }

    void prepare_parallel_values() {
        if (return_all_values_)
            throw std::runtime_error(
                "per-device worker execution does not support AllValuesAfterRun");
        for (const auto &item : store_.entries()) {
            const auto *ready =
                std::get_if<typename ValueStore<Api>::Ready>(&item.second);
            if (std::holds_alternative<typename ValueStore<Api>::Released>(item.second)) {
                auto value = define_parallel_value(item.first, desc(item.first).place);
                value->defined = true;
                continue;
            }
            if (ready == nullptr)
                throw std::runtime_error(
                    "initialization left a pending value before parallel execution");
            auto value = define_parallel_value(item.first, ready->place);
            value->value.emplace(ready->value);
            value->defined = true;
        }
        store_ = ValueStore<Api>{};
    }

    std::mutex prepare_read_mutex_;
    std::unique_ptr<WaterlineLedger> waterline_;
    std::vector<std::uint64_t> waterline_allocator_capacity_;
    std::unordered_map<ValueId, ResourceUsage> waterline_storage_;
    std::unordered_map<ValueId, std::size_t> waterline_remaining_;
    std::unordered_map<ValueId, bool> waterline_released_;

    void reset_parallel_execution_state() {
        waterline_.reset();
        waterline_allocator_capacity_.clear();
        waterline_storage_.clear(); waterline_remaining_.clear(); waterline_released_.clear();
        parallel_values_.clear();
        parallel_groups_.clear();
        coordinator_groups_.clear();
        parallel_dependency_dependents_.clear();
        parallel_dependency_queues_.clear();
        parallel_failed_.store(false);
        parallel_failure_ = nullptr;
        {
            std::lock_guard<std::mutex> lock(parallel_progress_mutex_);
            parallel_progress_epoch_ = 0;
        }
    }

    void record_parallel_failure(std::exception_ptr failure) {
        {
            ThreadTraceLockGuard lock(parallel_failure_mutex_, "runtime.parallel_failure");
            if (parallel_failed_.load()) return;
            parallel_failure_ = std::move(failure);
            parallel_failed_.store(true);
        }
        for (const auto &item : parallel_values_) {
            // Pair cancellation with the waiter's mutex so a notification
            // cannot be lost between checking the predicate and sleeping.
            ThreadTraceLockGuard lock(item.second->mutex, "runtime.parallel_value");
            item.second->condition.notify_all();
        }
        notify_parallel_progress();
        for (const auto &queue : parallel_dependency_queues_)
            queue->condition.notify_all();
    }

    void notify_parallel_progress() {
        if (device_execution_mode_ !=
            DeviceExecutionMode::PerDeviceReadyWorkers)
            return;
        {
            std::lock_guard<std::mutex> lock(parallel_progress_mutex_);
            ++parallel_progress_epoch_;
        }
        parallel_progress_condition_.notify_all();
    }

    void enqueue_dependency_task(
        const std::shared_ptr<DependencyTask> &task) {
        auto queue = task->queue.lock();
        if (!queue)
            throw std::runtime_error(
                "dependency task lost its execution queue");
        {
            std::lock_guard<std::mutex> lock(queue->mutex);
            queue->ready.push(task);
        }
        queue->condition.notify_one();
    }

    // Called once a value has data.  The dependency workers queue a task when
    // its last missing input arrives; the other executors have nothing to do.
    void activate_parallel_dependents(ValueId id) {
        if (device_execution_mode_ !=
            DeviceExecutionMode::PerDeviceDependencyWorkers)
            return;
        const auto found = parallel_dependency_dependents_.find(id);
        if (found == parallel_dependency_dependents_.end()) return;
        for (const auto &task : found->second) {
            const std::size_t previous =
                task->pending_inputs.fetch_sub(1, std::memory_order_acq_rel);
            if (previous == 0)
                throw std::runtime_error(
                    "dependency task input was activated twice");
            if (previous == 1) enqueue_dependency_task(task);
        }
    }

    ParallelTaskReadiness parallel_task_readiness(
        const ParallelTask &task) const {
        ParallelTaskReadiness readiness = ParallelTaskReadiness::Ready;
        for (ValueId id : task.inputs) {
            auto value = parallel_value(id);
            std::lock_guard<std::mutex> lock(value->mutex);
            if (!value->defined) return ParallelTaskReadiness::Blocked;
            if (!value->value)
                readiness = ParallelTaskReadiness::PendingCommunication;
        }
        return readiness;
    }

    void finish_parallel_group(
        const std::shared_ptr<ParallelGroup> &group,
        std::optional<std::uint64_t> consumer_ordinal) {
        ThreadTraceUniqueLock lock(
            group->mutex, "runtime.parallel_communication_group");
        if (group->completed) return;
        if (!group->posted)
            throw std::runtime_error(
                "parallel communication was waited before it was posted");

        const auto wait_start = std::chrono::steady_clock::now();
        const std::uint64_t trace_wait_start =
            ThreadTrace::enabled() ? ThreadTrace::timestamp_ns() : 0;
        auto outputs = api_.wait(group->handle);
        if (ThreadTrace::enabled())
            ThreadTrace::record_duration(
                "runtime.communication_wait", group.get(), trace_wait_start,
                ThreadTrace::timestamp_ns());
        const auto wait_elapsed = elapsed_nanoseconds(
            wait_start, std::chrono::steady_clock::now());
        {
            ThreadTraceLockGuard timing_lock(
                parallel_timing_mutex_, "runtime.parallel_timing");
            ++timing_.communication_wait_calls;
            timing_.communication_wait_nanoseconds += wait_elapsed;
        }
        trace_event("comm_wait", group->action.id,
                    group->instruction_ordinal, consumer_ordinal,
                    to_string(group->action.kind), wait_elapsed);
        if (outputs.size() != group->action.outputs.size())
            throw std::runtime_error("Api wait returned wrong parallel output count");
        for (std::size_t slot = 0; slot < outputs.size(); ++slot)
            if (group->action.destinations[slot].rank != rank_ && outputs[slot])
                throw std::runtime_error("Api wait returned a remote parallel output");

        std::vector<ValueId> published;
        for (const auto &local_output : group->local_outputs) {
            const ValueId id = local_output.id;
            if (local_output.action_slot >= group->action.outputs.size() ||
                group->action.outputs[local_output.action_slot] != id)
                throw std::runtime_error(
                    "parallel communication output slot mapping is invalid");
            const auto &expected_desc = desc(id);
            auto &output = outputs[local_output.action_slot];
            if (output) api_.validate_value(*output, expected_desc);
            const auto &value = local_output.value;
            {
                ThreadTraceLockGuard value_lock(
                    value->mutex, "runtime.parallel_value");
                if (!value->defined)
                    throw std::runtime_error(
                        "parallel communication output state is invalid");
                if (!value->group) {
                    if (output)
                        throw std::runtime_error("Api delivered a parallel communication output twice");
                } else {
                    if (!output)
                        throw std::runtime_error("Api wait omitted a pending parallel communication output");
                    if (value->group.get() != group.get())
                        throw std::runtime_error(
                            "pending communication output group is invalid");
                    if(waterline_)value->completion.emplace(*output);
                    if (!can_clear(value->uses))
                        value->value.emplace(std::move(*output));
                    output.reset();
                    value->group.reset();
                    published.push_back(id);
                }
            }
            value->condition.notify_all();
        }
        group->completed = true;
        lock.unlock();
        for (ValueId id : published) activate_parallel_dependents(id);
        notify_parallel_progress();
    }

    Value resolve_parallel_value(
        ValueId id, const Place &expected_place,
        std::optional<std::uint64_t> consumer_ordinal) {
        auto value = parallel_value(id);
        for (;;) {
            std::shared_ptr<ParallelGroup> group;
            {
                ThreadTraceUniqueLock lock(
                    value->mutex, "runtime.parallel_value");
                lock.wait(value->condition, [&] {
                    return value->defined || parallel_failed_.load();
                }, "runtime.value_dependency_wait");
                if (parallel_failed_.load())
                    throw std::runtime_error("parallel execution was cancelled");
                if (value->place != expected_place)
                    throw std::runtime_error(
                        "parallel value Place mismatch for " +
                        std::to_string(id));
                if (value->value) return *value->value;
                if (can_clear(value->uses))
                    throw std::runtime_error("parallel ValueId was released: " + std::to_string(id));
                group = value->group;
            }
            if (!group)
                throw std::runtime_error(
                    "parallel value is defined without data or communication");
            if (group->cross_rank) {
                ThreadTraceUniqueLock lock(
                    value->mutex, "runtime.remote_value_wait");
                lock.wait(value->condition, [&] {
                    return value->value.has_value() ||
                           parallel_failed_.load();
                }, "runtime.remote_value_wait");
                if (parallel_failed_.load())
                    throw std::runtime_error(
                        "parallel execution was cancelled");
                continue;
            }
            finish_parallel_group(group, consumer_ordinal);
        }
    }

    void execute_parallel_compute(
        ComputeOp op, std::uint64_t instruction_ordinal,
        const std::shared_ptr<ParallelValue> &output_value) {
        std::vector<Value> inputs;
        inputs.reserve(op.inputs.size());
        for (ValueId id : op.inputs)
            inputs.push_back(resolve_parallel_value(
                id, op.place, instruction_ordinal));

        const auto start = std::chrono::steady_clock::now();
        const std::uint64_t trace_compute_start =
            ThreadTrace::enabled() ? ThreadTrace::timestamp_ns() : 0;
        if (op.reuse_input) {
            const auto &old = parallel_values_.at(op.inputs.front());
            ThreadTraceLockGuard lock(old->mutex, "runtime.parallel_value");
            old->value.reset();
            old->uses.release_requested = true;
        }
        Value output = compute_output(op, inputs);
        const auto finish = std::chrono::steady_clock::now();
        inputs.clear();
        submitted_parallel_uses(op.inputs);
        if (ThreadTrace::enabled())
            ThreadTrace::record_duration(
                "runtime.compute", output_value.get(), trace_compute_start,
                ThreadTrace::timestamp_ns());
        const auto elapsed = elapsed_nanoseconds(
            start, finish);
        trace_event("compute", op.output, instruction_ordinal,
                    std::nullopt, to_string(op.kind), elapsed);
        {
            ThreadTraceLockGuard lock(
                parallel_timing_mutex_, "runtime.parallel_timing");
            ++timing_.compute_calls;
            timing_.compute_including_boot_nanoseconds += elapsed;
            if (op.kind == ComputeKind::Boot) {
                ++timing_.boot_calls;
                timing_.boot_nanoseconds += elapsed;
            } else if (op.kind == ComputeKind::BootStep) {
                ++timing_.boot_step_calls;
                timing_.boot_step_nanoseconds += elapsed;
            }
        }

        const auto &output_desc = desc(op.output);
        api_.validate_value(output, output_desc);
        {
            ThreadTraceLockGuard lock(
                output_value->mutex, "runtime.parallel_value");
            if(waterline_)output_value->completion.emplace(output);
            if (!can_clear(output_value->uses))
                output_value->value.emplace(std::move(output));
            output_value->defined = true;
        }
        output_value->condition.notify_all();
        activate_parallel_dependents(op.output);
        notify_parallel_progress();
    }

    void execute_parallel_communication(
        CommAction action, std::uint64_t instruction_ordinal,
        const std::shared_ptr<ParallelGroup> &group) {
        std::vector<Value> local_inputs;
        for (std::size_t i = 0; i < action.inputs.size(); ++i) {
            if (action.sources[i].rank != rank_) continue;
            local_inputs.push_back(resolve_parallel_value(
                action.inputs[i], action.sources[i], instruction_ordinal));
        }

        const auto post_start = std::chrono::steady_clock::now();
        CommHandle handle = api_.communicate_async(
            action, local_inputs, communication_output_descs(action));
        auto posted = posted_outputs(action, handle);
        const auto post_elapsed = elapsed_nanoseconds(
            post_start, std::chrono::steady_clock::now());
        {
            ThreadTraceLockGuard lock(
                parallel_timing_mutex_, "runtime.parallel_timing");
            ++timing_.communication_post_calls;
            timing_.communication_post_nanoseconds += post_elapsed;
        }
        trace_event("comm_post", action.id, instruction_ordinal,
                    std::nullopt, to_string(action.kind), post_elapsed);

        {
            ThreadTraceLockGuard lock(
                group->mutex, "runtime.parallel_communication_group");
            group->handle = std::move(handle);
            group->posted = true;
        }

        bool all_posted = true;
        std::vector<ValueId> published;
        for (const auto &local_output : group->local_outputs) {
            const auto &value = local_output.value;
            auto &output = posted.at(local_output.action_slot);
            if (group->cross_rank &&
                detail::UsesBackgroundCommunicationIssuer<Api>::value &&
                !output)
                throw std::runtime_error(
                    "background communication did not publish a local output");
            {
                ThreadTraceLockGuard lock(
                    value->mutex, "runtime.parallel_value");
                if (output) {
                    const auto &expected_desc = desc(local_output.id);
                    api_.validate_value(*output, expected_desc);
                    if(waterline_)value->completion.emplace(*output);
                    if (!can_clear(value->uses))
                        value->value.emplace(std::move(*output));
                    output.reset();
                    published.push_back(local_output.id);
                } else {
                    value->group = group;
                    all_posted = false;
                }
                value->defined = true;
            }
            value->condition.notify_all();
        }
        local_inputs.clear();
        submitted_parallel_uses(group->local_input_ids);
        for (ValueId id : published) activate_parallel_dependents(id);
        if (device_execution_mode_ ==
            DeviceExecutionMode::PerDeviceDependencyWorkers) {
            // Waiting here holds this communication worker until the copy has
            // landed, and holds the consumers until then too, although their
            // kernels already wait on the copy's completion event on the
            // device.  Where the wait has nothing to reclaim -- a peer or
            // same-device copy -- the consumers already have their outputs and
            // the wait moves to the end of the phase.  A Host-staged copy is
            // still waited at once, so its pinned buffer is freed as soon as
            // it can be.
            bool defer = false;
            if constexpr (detail::CanDeferCommunicationWait<Api>::value) {
                ThreadTraceLockGuard lock(
                    group->mutex, "runtime.parallel_communication_group");
                defer = all_posted && api_.wait_can_be_deferred(group->handle);
            }
            if (!defer) finish_parallel_group(group, std::nullopt);
        }
        notify_parallel_progress();
    }

    bool is_cross_rank_action(const CommAction &action) const {
        if (action.sources.empty())
            throw std::runtime_error("parallel communication has no source");
        const int source_rank = action.sources.front().rank;
        return std::any_of(
            action.destinations.begin(), action.destinations.end(),
            [source_rank](const Place &place) {
                return place.rank != source_rank;
            });
    }

    std::shared_ptr<ParallelGroup> make_parallel_group(
        CommAction action, std::uint64_t instruction_ordinal,
        bool cross_rank) {
        auto group = std::make_shared<ParallelGroup>();
        group->action = std::move(action);
        group->instruction_ordinal = instruction_ordinal;
        group->cross_rank = cross_rank;
        for (std::size_t i = 0; i < group->action.inputs.size(); ++i)
            if (group->action.sources[i].rank == rank_)
                group->local_input_ids.push_back(group->action.inputs[i]);
        for (std::size_t i = 0; i < group->action.outputs.size(); ++i) {
            if (group->action.destinations[i].rank != rank_) continue;
            group->local_outputs.push_back(ParallelOutput{
                group->action.outputs[i], i,
                define_parallel_value(group->action.outputs[i],
                                      group->action.destinations[i])});
        }
        parallel_groups_.push_back(group);
        return group;
    }

    int local_communication_worker(const CommAction &action) const {
        if (action.outputs.size() != 1 || action.destinations.size() != 1 ||
            action.sources.size() != 1)
            throw std::runtime_error(
                "local parallel communication must have one destination");
        const Place &destination = action.destinations.front();
        const Place &source = action.sources.front();
        // Device-to-Device copies follow the producer so the destination
        // worker can keep executing until it consumes the copied value.
        const Place *device_place =
            source.kind == PlaceKind::Device ? &source :
            destination.kind == PlaceKind::Device ? &destination : nullptr;
        if (device_place == nullptr || device_place->rank != rank_ ||
            device_place->index < 0 || device_place->index >= local_devices_)
            throw std::runtime_error(
                "local parallel communication has no available device worker");
        return device_place->index;
    }

    // Destination-following selector kept for PerDeviceDependencyWorkers.
    //
    // local_communication_worker() above follows the *source* for D2D copies,
    // which is what the multi-rank ready path wants: the destination worker
    // stays free until it consumes the value.  The dependency-driven path
    // schedules on the consumer instead, so it wants the destination, and it
    // accepts multi-output actions rather than one pre-sliced slot.  The two
    // modes genuinely differ here; sharing one selector would silently change
    // whichever mode did not pick the policy.
    int parallel_communication_worker(const CommAction &action) const {
        std::optional<int> worker;
        const auto select = [&](const Place &place) {
            if (place.rank != rank_ || place.kind != PlaceKind::Device) return;
            if (place.index < 0 || place.index >= local_devices_)
                throw std::runtime_error(
                    "parallel communication uses an unavailable device");
            if (worker && *worker != place.index)
                throw std::runtime_error(
                    "parallel communication cannot own outputs on multiple devices");
            worker = place.index;
        };
        for (const Place &destination : action.destinations) select(destination);
        if (!worker)
            for (const Place &source : action.sources) select(source);
        if (!worker)
            throw std::runtime_error(
                "parallel communication has no local device worker");
        return *worker;
    }

    CommAction local_communication_slice(const CommAction &action,
                                         std::size_t slot) const {
        if (action.inputs.size() != 1 || action.sources.size() != 1 ||
            slot >= action.outputs.size() ||
            slot >= action.destinations.size() ||
            slot >= action.output_types.size())
            throw std::runtime_error(
                "local parallel communication mapping is invalid");
        CommAction result;
        result.id = action.id;
        result.kind = CommKind::Transfer;
        result.hint = action.hint;
        result.inputs = action.inputs;
        result.sources = action.sources;
        result.outputs = {action.outputs[slot]};
        result.destinations = {action.destinations[slot]};
        result.output_types = {action.output_types[slot]};
        return result;
    }

    CommAction cross_rank_communication_slice(
        const CommAction &action) const {
        if (action.inputs.size() != 1 || action.sources.size() != 1 ||
            action.outputs.size() != action.destinations.size() ||
            action.outputs.size() != action.output_types.size())
            throw std::runtime_error(
                "cross-rank parallel communication mapping is invalid");
        CommAction result;
        result.id = action.id;
        result.hint = action.hint;
        result.inputs = action.inputs;
        result.sources = action.sources;
        const int source_rank = action.sources.front().rank;
        for (std::size_t slot = 0; slot < action.outputs.size(); ++slot) {
            if (action.destinations[slot].rank == source_rank) continue;
            result.outputs.push_back(action.outputs[slot]);
            result.destinations.push_back(action.destinations[slot]);
            result.output_types.push_back(action.output_types[slot]);
        }
        if (result.outputs.empty())
            throw std::runtime_error(
                "cross-rank parallel communication has no remote destination");
        result.kind = result.outputs.size() == 1
            ? CommKind::Transfer : CommKind::Replicate;
        return result;
    }

    void compile_parallel_phase(
        const std::vector<Instruction> &instructions,
        std::vector<std::vector<ParallelTask>> &tasks,
        std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            const auto &instruction = instructions[i];
            if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) {
                if (op->place.rank != rank_) continue;
                if (local_devices_ > 0 && op->place.kind == PlaceKind::Host && plan_->format_version < 3)
                    throw std::runtime_error("per-device worker Host compute requires V3");
                if (op->place.kind == PlaceKind::Device &&
                    (op->place.index < 0 || op->place.index >= local_devices_))
                    throw std::runtime_error(
                        "per-device worker execution requires Device compute operations");
                auto output = define_parallel_value(
                    op->output, desc(op->output).place);
                const std::size_t worker =
                    op->place.kind == PlaceKind::Host ? tasks.size() - 1 :
                    static_cast<std::size_t>(op->place.index) % (tasks.size() - 1);
                tasks[worker].push_back(ParallelTask{
                    [this, op = *op, ordinal = instruction.ordinal, output] {
                        execute_parallel_compute(op, ordinal, output);
                    }, &instruction, op->inputs, {op->output}});
                continue;
            }
            if (const auto *encode = std::get_if<EncodeOp>(&instruction.body)) {
                if (desc(encode->output).place.rank != rank_) continue;
                auto output = define_parallel_value(encode->output, desc(encode->output).place);
                tasks.back().push_back(ParallelTask{[this, op = *encode, ordinal = instruction.ordinal, output] {
                    auto encoded = encode_value(op, ordinal);
                    {
                        ThreadTraceLockGuard lock(output->mutex, "runtime.parallel_value");
                        if(waterline_)output->completion.emplace(encoded);
                        if (!can_clear(output->uses)) output->value.emplace(std::move(encoded));
                        output->defined = true;
                    }
                    output->condition.notify_all();
                }, &instruction, {}, {encode->output}});
                continue;
            }
            if (const auto *prepare = std::get_if<PreparePlaintextOp>(&instruction.body)) {
                // Registration has no allocation or eager preparation. A
                // segment adds this producer when its first consumer needs it.
                continue;
            }
            if (const auto *release = std::get_if<ReleaseOp>(&instruction.body)) {
                const auto &place = desc(release->value).place;
                if (place.rank != rank_) continue;
                const std::size_t worker = place.kind == PlaceKind::Device
                    ? static_cast<std::size_t>(place.index) % (tasks.size() - 1) : tasks.size() - 1;
                tasks[worker].push_back(ParallelTask{[this, id = release->value, ordinal = instruction.ordinal] {
                    execute_release(id, ordinal, true);
                }, &instruction, {release->value}, {}});
                continue;
            }

            const auto action = std::get<CommAction>(instruction.body);
            const bool has_cross_rank_destination =
                is_cross_rank_action(action);
            if (has_cross_rank_destination) {
                const auto same_backend = [&](const Place &place) {
                    return place.kind == action.sources.front().kind;
                };
                if (!std::all_of(action.sources.begin(), action.sources.end(),
                                 same_backend) ||
                    !std::all_of(action.destinations.begin(),
                                 action.destinations.end(), same_backend))
                    throw std::runtime_error(
                        "cross-rank workers require Host-to-Host or Device-to-Device communication");
                auto cross_rank_action =
                    cross_rank_communication_slice(action);
                coordinator_groups_.push_back(make_parallel_group(
                    std::move(cross_rank_action), instruction.ordinal, true));
            }
            if (action.sources.empty() ||
                action.sources.front().rank != rank_) continue;
            for (std::size_t slot = 0; slot < action.outputs.size(); ++slot) {
                if (has_cross_rank_destination &&
                    action.destinations[slot].rank != rank_)
                    continue;
                CommAction local = local_communication_slice(action, slot);
                auto group = make_parallel_group(
                    local, instruction.ordinal, false);
                const int device = local_communication_worker(local);
                const std::size_t worker =
                    static_cast<std::size_t>(device) % (tasks.size() - 1);
                tasks[worker].push_back(ParallelTask{
                    [this, local = std::move(local),
                     ordinal = instruction.ordinal, group] {
                        execute_parallel_communication(
                            local, ordinal, group);
                    }, &instruction, group->action.inputs, group->action.outputs});
            }
        }
    }

    // The two executors carried over from the measured search backend run the
    // online phases as they are: Compute, Transfer and Release.  An online
    // Encode, a lazy Prepare, a Fence or a Host compute needs the shared
    // dispatcher of PerDeviceWorkers; refusing them here is cheaper than a
    // value that is never produced.
    void require_worker_executable(const char *mode) const {
        if (plan_->waterline)
            throw std::runtime_error(std::string(mode) + " does not run Waterline plans");
        for (const auto *phase : {&plan_->execution, &plan_->finalization})
            for (const auto &instruction : *phase) {
                const char *what = nullptr;
                if (std::holds_alternative<EncodeOp>(instruction.body)) what = "online Encode";
                else if (std::holds_alternative<PreparePlaintextOp>(instruction.body)) what = "PreparePlaintext";
                else if (std::holds_alternative<FenceOp>(instruction.body)) what = "Fence";
                else if (const auto *op = std::get_if<ComputeOp>(&instruction.body);
                         op && op->place.rank == rank_ && op->place.kind != PlaceKind::Device)
                    what = "Host compute";
                if (what)
                    throw std::runtime_error(std::string(mode) + " does not support " + what +
                                             " (instruction #" +
                                             std::to_string(instruction.ordinal) + ")");
            }
    }

    std::size_t parallel_worker_count() const {
        const std::size_t worker_count =
            device_worker_count_ == 0
                ? static_cast<std::size_t>(local_devices_)
                : device_worker_count_;
        if (worker_count == 0 ||
            worker_count > static_cast<std::size_t>(local_devices_))
            throw std::runtime_error(
                "device worker count must be between one and local device count");
        return worker_count;
    }

    void append_dependency_task(
        const std::shared_ptr<DependencyTaskQueue> &queue,
        std::uint64_t ordinal, std::vector<ValueId> input_ids,
        std::string op_label, std::function<void()> execute) {
        auto task = std::make_shared<DependencyTask>();
        task->ordinal = ordinal;
        task->input_ids = std::move(input_ids);
        task->execute = std::move(execute);
        task->queue = queue;
        task->queue_position = queue->tasks.size();
        task->op_label = std::move(op_label);
        queue->tasks.push_back(std::move(task));
    }

    void compile_dependency_phase(
        const std::vector<Instruction> &instructions,
        const std::vector<std::shared_ptr<DependencyTaskQueue>> &compute_queues,
        const std::vector<std::shared_ptr<DependencyTaskQueue>> &communication_queues) {
        for (const auto &instruction : instructions) {
            if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) {
                if (op->place.rank != rank_) continue;
                if (op->place.kind != PlaceKind::Device ||
                    op->place.index < 0 || op->place.index >= local_devices_)
                    throw std::runtime_error(
                        "dependency worker execution requires Device compute operations");
                auto output = define_parallel_value(
                    op->output, desc(op->output).place);
                const std::size_t worker =
                    static_cast<std::size_t>(op->place.index) %
                    compute_queues.size();
                append_dependency_task(
                    compute_queues[worker], instruction.ordinal, op->inputs,
                    to_string(op->kind),
                    [this, op = *op, ordinal = instruction.ordinal, output] {
                        execute_parallel_compute(op, ordinal, output);
                    });
                value_producers_[op->output] = ValueProducer{
                    "compute", worker, instruction.ordinal, to_string(op->kind)};
                continue;
            }
            if (const auto *release = std::get_if<ReleaseOp>(&instruction.body)) {
                // Bookkeeping only: it marks the value releasable once it has
                // data, and the last submitted consumer drops it.  It runs on
                // the queue of the device that holds the value.
                const auto &place = desc(release->value).place;
                if (place.rank != rank_) continue;
                const std::size_t worker =
                    place.kind == PlaceKind::Device
                        ? static_cast<std::size_t>(place.index) % compute_queues.size()
                        : 0;
                append_dependency_task(
                    compute_queues[worker], instruction.ordinal, {release->value},
                    "Release",
                    [this, id = release->value, ordinal = instruction.ordinal] {
                        execute_release(id, ordinal, true);
                    });
                continue;
            }
            const auto *found = std::get_if<CommAction>(&instruction.body);
            if (found == nullptr)
                throw std::runtime_error(
                    "dependency worker execution met an unsupported instruction #" +
                    std::to_string(instruction.ordinal));
            const auto &action = *found;
            if (action.sources.empty() || action.sources.front().rank != rank_ ||
                is_cross_rank_action(action))
                throw std::runtime_error(
                    "dependency worker execution supports single-rank communication only");
            // One task per destination, as the other parallel executors do: a
            // Replicate to several devices lands each copy on the queue of the
            // device that consumes it.
            for (std::size_t slot = 0; slot < action.outputs.size(); ++slot) {
                CommAction local = local_communication_slice(action, slot);
                // Built by the factory rather than by hand: local_outputs is a
                // vector<ParallelOutput>, and the factory is what registers
                // the group in parallel_groups_.
                auto group =
                    make_parallel_group(local, instruction.ordinal, false);
                const int device = parallel_communication_worker(local);
                const std::size_t worker =
                    static_cast<std::size_t>(device) % communication_queues.size();
                value_producers_[local.outputs.front()] = ValueProducer{
                    "communication", worker, instruction.ordinal,
                    to_string(action.kind)};
                append_dependency_task(
                    communication_queues[worker], instruction.ordinal,
                    group->local_input_ids, to_string(action.kind),
                    [this, local = std::move(local), ordinal = instruction.ordinal, group] {
                        execute_parallel_communication(local, ordinal, group);
                    });
            }
        }
    }

    void initialize_dependency_tasks() {
        for (const auto &queue : parallel_dependency_queues_) {
            for (const auto &task : queue->tasks) {
                std::size_t pending = 0;
                for (ValueId id : task->input_ids) {
                    auto value = parallel_value(id);
                    bool materialized = false;
                    {
                        std::lock_guard<std::mutex> lock(value->mutex);
                        materialized = value->value.has_value();
                    }
                    if (materialized) continue;
                    ++pending;
                    parallel_dependency_dependents_[id].push_back(task);
                }
                task->pending_inputs.store(pending, std::memory_order_release);
            }
        }
        for (const auto &queue : parallel_dependency_queues_)
            for (const auto &task : queue->tasks)
                if (task->pending_inputs.load(std::memory_order_acquire) == 0)
                    enqueue_dependency_task(task);
    }

    void execute_dependency_task_queue(
        const std::shared_ptr<DependencyTaskQueue> &queue) {
        for (;;) {
            std::shared_ptr<DependencyTask> task;
            bool backfilled = false;
            std::uint64_t wait_elapsed = 0;
            std::optional<WaitTraceRecord> wait_trace;
            {
                std::unique_lock<std::mutex> lock(queue->mutex);
                if (queue->ready.empty() &&
                    queue->completed != queue->tasks.size() &&
                    !parallel_failed_.load()) {
                    if (wait_trace_enabled_)
                        wait_trace = sample_wait_trace(queue);
                    const auto wait_start = std::chrono::steady_clock::now();
                    queue->condition.wait(lock, [&] {
                        return parallel_failed_.load() ||
                               !queue->ready.empty() ||
                               queue->completed == queue->tasks.size();
                    });
                    wait_elapsed = elapsed_nanoseconds(
                        wait_start, std::chrono::steady_clock::now());
                    if (wait_trace) {
                        wait_trace->duration_ns = wait_elapsed;
                        wait_trace->wake_reason =
                            queue->completed == queue->tasks.size()
                                ? "completed"
                                : (parallel_failed_.load() ? "failed"
                                                           : "task_enqueued");
                        record_wait_trace(std::move(*wait_trace));
                    }
                }
                if (parallel_failed_.load()) return;
                if (queue->completed == queue->tasks.size()) return;
                if (queue->ready.empty())
                    throw std::runtime_error(
                        "dependency task queue woke without a ready task");

                task = queue->ready.top();
                queue->ready.pop();
                backfilled = task->queue_position != queue->next_unfinished;
                task->completed = true;
                ++queue->completed;
                while (queue->next_unfinished < queue->tasks.size() &&
                       queue->tasks[queue->next_unfinished]->completed)
                    ++queue->next_unfinished;
            }

            if (wait_elapsed != 0) {
                // Count both compute and communication queue waits so the
                // metric matches the total time workers spent blocked.
                std::lock_guard<std::mutex> lock(parallel_timing_mutex_);
                ++timing_.device_ready_wait_calls;
                timing_.device_ready_wait_nanoseconds += wait_elapsed;
            }
            if (backfilled) {
                std::lock_guard<std::mutex> lock(parallel_timing_mutex_);
                ++timing_.device_backfill_tasks;
            }
            task->execute();
        }
    }

    void execute_dependency_parallel_phases() {
        prepare_parallel_values();
        require_worker_executable("PerDeviceDependencyWorkers");
        const std::size_t worker_count = parallel_worker_count();

        std::vector<std::shared_ptr<DependencyTaskQueue>> compute_queues;
        std::vector<std::shared_ptr<DependencyTaskQueue>> communication_queues;
        compute_queues.reserve(worker_count);
        communication_queues.reserve(worker_count);
        parallel_dependency_queues_.reserve(worker_count * 2);
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            auto compute_queue = std::make_shared<DependencyTaskQueue>();
            compute_queue->compute_queue = true;
            compute_queue->index = worker;
            compute_queues.push_back(compute_queue);
            parallel_dependency_queues_.push_back(std::move(compute_queue));

            auto communication_queue = std::make_shared<DependencyTaskQueue>();
            communication_queue->index = worker;
            communication_queues.push_back(communication_queue);
            parallel_dependency_queues_.push_back(
                std::move(communication_queue));
        }

        wait_trace_worker_count_ = worker_count;
        value_producers_.clear();
        compile_dependency_phase(
            plan_->execution, compute_queues, communication_queues);
        compile_dependency_phase(
            plan_->finalization, compute_queues, communication_queues);
        initialize_dependency_tasks();

        std::vector<std::thread> workers;
        workers.reserve(parallel_dependency_queues_.size());
        for (const auto &queue : parallel_dependency_queues_) {
            workers.emplace_back([this, queue] {
                ThreadTrace::set_thread_name(
                    "rank-" + std::to_string(rank_) +
                    (queue->compute_queue ? "-compute-worker-" : "-communication-worker-") +
                    std::to_string(queue->index));
                try {
                    execute_dependency_task_queue(queue);
                } catch (...) {
                    record_parallel_failure(std::current_exception());
                }
            });
        }
        for (auto &worker : workers) worker.join();
        write_wait_trace();
        if (parallel_failed_.load()) {
            std::lock_guard<std::mutex> lock(parallel_failure_mutex_);
            std::rethrow_exception(parallel_failure_);
        }

        for (ValueId id : plan_->final_outputs) {
            const auto &output_desc = desc(id);
            if (output_desc.place.rank != rank_) continue;
            Value output = resolve_parallel_value(
                id, output_desc.place, std::nullopt);
            store_.define_ready(id, output_desc.place, std::move(output));
        }
        // The waits put off at post time.  Every copy has long landed by now,
        // so this only confirms completion.
        for (const auto &group : parallel_groups_)
            if (!group->completed) finish_parallel_group(group, std::nullopt);
        for (const auto &group : parallel_groups_)
            if (!group->completed)
                throw std::runtime_error(
                    "dependency communication did not complete");
        parallel_groups_.clear();
        parallel_dependency_queues_.clear();
        parallel_dependency_dependents_.clear();
    }

    void execute_ready_parallel_tasks(std::vector<ParallelTask> &tasks) {
        std::vector<bool> completed(tasks.size(), false);
        std::size_t remaining = tasks.size();
        while (remaining != 0) {
            if (parallel_failed_.load()) return;

            std::uint64_t observed_epoch = 0;
            {
                std::lock_guard<std::mutex> lock(parallel_progress_mutex_);
                observed_epoch = parallel_progress_epoch_;
            }

            std::optional<std::size_t> first_unfinished;
            std::optional<std::size_t> pending_communication;
            std::optional<std::size_t> selected;
            for (std::size_t index = 0; index < tasks.size(); ++index) {
                if (completed[index]) continue;
                if (!first_unfinished) first_unfinished = index;
                const auto readiness = parallel_task_readiness(tasks[index]);
                if (readiness == ParallelTaskReadiness::Ready) {
                    selected = index;
                    break;
                }
                if (readiness == ParallelTaskReadiness::PendingCommunication &&
                    !pending_communication)
                    pending_communication = index;
            }
            if (!selected) selected = pending_communication;

            if (selected) {
                if (first_unfinished && *selected != *first_unfinished) {
                    std::lock_guard<std::mutex> lock(parallel_timing_mutex_);
                    ++timing_.device_backfill_tasks;
                }
                completed[*selected] = true;
                tasks[*selected].execute();
                --remaining;
                continue;
            }

            const auto wait_start = std::chrono::steady_clock::now();
            {
                std::unique_lock<std::mutex> lock(parallel_progress_mutex_);
                parallel_progress_condition_.wait(lock, [&] {
                    return parallel_failed_.load() ||
                           parallel_progress_epoch_ != observed_epoch;
                });
            }
            const auto wait_elapsed = elapsed_nanoseconds(
                wait_start, std::chrono::steady_clock::now());
            {
                std::lock_guard<std::mutex> lock(parallel_timing_mutex_);
                ++timing_.device_ready_wait_calls;
                timing_.device_ready_wait_nanoseconds += wait_elapsed;
            }
        }
    }

    // Per-device lists, as PerDeviceWorkers ran them before the shared
    // dispatcher: each worker takes the earliest task whose inputs are ready,
    // and only waits when none is.
    void execute_ready_parallel_phases() {
        prepare_parallel_values();
        require_worker_executable("PerDeviceReadyWorkers");
        const std::size_t worker_count = parallel_worker_count();
        // compile_parallel_phase keeps its last queue for the CPU worker.  The
        // check above leaves only Releases of Host values there; they are
        // bookkeeping, so worker 0 takes them in program order.
        std::vector<std::vector<ParallelTask>> tasks(worker_count + 1);
        compile_parallel_phase(plan_->execution, tasks, 0, plan_->execution.size());
        compile_parallel_phase(plan_->finalization, tasks, 0, plan_->finalization.size());
        for (auto &task : tasks.back()) {
            if (!std::holds_alternative<ReleaseOp>(task.instruction->body))
                throw std::runtime_error("PerDeviceReadyWorkers produced Host-side tasks");
            tasks.front().push_back(std::move(task));
        }
        tasks.pop_back();
        std::stable_sort(tasks.front().begin(), tasks.front().end(),
                         [](const ParallelTask &left, const ParallelTask &right) {
                             return left.instruction->ordinal < right.instruction->ordinal;
                         });

        std::vector<std::thread> workers;
        workers.reserve(tasks.size());
        for (std::size_t device = 0; device < tasks.size(); ++device) {
            workers.emplace_back([this, &tasks, device] {
                ThreadTrace::set_thread_name(
                    "rank-" + std::to_string(rank_) + "-device-worker-" +
                    std::to_string(device));
                try {
                    execute_ready_parallel_tasks(tasks[device]);
                } catch (...) {
                    record_parallel_failure(std::current_exception());
                }
            });
        }
        std::thread communication_issuer;
        if constexpr (detail::UsesBackgroundCommunicationIssuer<Api>::value) {
            communication_issuer = std::thread([this] {
                ThreadTrace::set_thread_name(
                    "rank-" + std::to_string(rank_) +
                    "-communication-issuer");
                try {
                    for (const auto &group : coordinator_groups_) {
                        if (parallel_failed_.load()) return;
                        execute_parallel_communication(
                            group->action, group->instruction_ordinal, group);
                    }
                } catch (...) {
                    record_parallel_failure(std::current_exception());
                }
            });
        } else {
            try {
                for (const auto &group : coordinator_groups_) {
                    if (parallel_failed_.load()) break;
                    execute_parallel_communication(
                        group->action, group->instruction_ordinal, group);
                    finish_parallel_group(group, std::nullopt);
                }
            } catch (...) {
                record_parallel_failure(std::current_exception());
            }
        }
        for (auto &worker : workers) worker.join();
        if (communication_issuer.joinable()) communication_issuer.join();
        if (parallel_failed_.load()) {
            ThreadTraceLockGuard lock(
                parallel_failure_mutex_, "runtime.parallel_failure");
            std::rethrow_exception(parallel_failure_);
        }

        for (const auto &group : parallel_groups_)
            if (!group->completed)
                finish_parallel_group(group, std::nullopt);

        for (ValueId id : plan_->final_outputs) {
            const auto &output_desc = desc(id);
            if (output_desc.place.rank != rank_) continue;
            Value output = resolve_parallel_value(
                id, output_desc.place, std::nullopt);
            store_.define_ready(id, output_desc.place, std::move(output));
        }
        parallel_groups_.clear();
        coordinator_groups_.clear();
    }

    void execute_device_parallel_phases() {
        prepare_parallel_values();
        if constexpr (detail::HasWaterlineApi<Api>::value) {
            if (plan_->waterline) {
                waterline_ = std::make_unique<WaterlineLedger>(*plan_->waterline,waterline_allocator_capacity_);
                waterline_storage_.clear(); waterline_remaining_.clear(); waterline_released_.clear();
                // Reusable Host/pinned preparation buffers stay charged across
                // Ready completion and explicit Fence boundaries for this run.
                if constexpr(detail::HasWaterlineFixedResources<Api>::value)
                    waterline_->reserve(api_.waterline_fixed_resources());
                for (const auto &[id, value] : parallel_values_) {
                    if (!value->value) continue;
                    auto storage = api_.value_resources(desc(id));
                    waterline_->reserve(storage); waterline_storage_.emplace(id, storage);
                }
                ResourceUsage resident_pack(local_devices_);resident_pack.host=timing_.bundle_resident_bytes;
                waterline_->reserve(resident_pack);
                for (const auto &[id, uses] : uses_) waterline_remaining_[id] = uses.remaining;
            }
        }
        const std::size_t device_workers =
            device_worker_count_ == 0
                ? static_cast<std::size_t>(local_devices_)
                : device_worker_count_;
        if (device_workers > static_cast<std::size_t>(local_devices_))
            throw std::runtime_error(
                "device worker count cannot exceed local device count");
        std::vector<const std::vector<Instruction> *> phases;
        if (worker_initialization()) phases.push_back(&plan_->initialization);
        phases.push_back(&plan_->execution);phases.push_back(&plan_->finalization);
        std::unordered_map<ValueId,const Instruction *> preparation_instructions;
        if(waterline_)for(const auto *phase:phases)for(const auto &instruction:*phase)
            if(const auto *op=std::get_if<PreparePlaintextOp>(&instruction.body))
                preparation_instructions.emplace(op->output,&instruction);
        // Returning a lazy plaintext is also a demand. Give each returned
        // Prepare its own terminal group, after the plan's instructions, so
        // both preflight and admission include its complete preparation peak.
        std::vector<std::pair<ValueId, Instruction>> final_output_demands;
        std::size_t terminal_ordinal = plan_->initialization.size() +
            plan_->execution.size() + plan_->finalization.size();
        for (auto id : plan_->final_outputs)
            if (preparation_instructions.count(id))
                final_output_demands.emplace_back(id, Instruction{terminal_ordinal++, FenceOp{}});
        const auto add_final_output_demands = [&](std::vector<std::vector<ParallelTask>> &tasks) {
            for (const auto &[id, instruction] : final_output_demands)
                tasks.back().push_back(ParallelTask{[] {}, &instruction, {id}, {},
                    false, PrepareBackend::Cpu, true});
        };
        if (waterline_) {
            // Metadata only: validate the entire graph before any initialization
            // Encode, transfer or consumer is submitted.
            std::vector<std::vector<ParallelTask>> baseline(2);
            for (const auto *phase : phases) for (const auto &instruction : *phase) {
                if (const auto *op=std::get_if<PreparePlaintextOp>(&instruction.body)) {
                    baseline.back().push_back(ParallelTask{{},&instruction,{}, {op->output},true});
                } else if (const auto *op=std::get_if<EncodeOp>(&instruction.body)) {
                    if(desc(op->output).place.rank==rank_)baseline.back().push_back(ParallelTask{{},&instruction,{}, {op->output}});
                } else if (const auto *op=std::get_if<ComputeOp>(&instruction.body)) {
                    if(op->place.rank==rank_)baseline.back().push_back(ParallelTask{{},&instruction,op->inputs,{op->output}});
                } else if (const auto *op=std::get_if<ReleaseOp>(&instruction.body)) {
                    if(desc(op->value).place.rank==rank_)baseline.back().push_back(ParallelTask{{},&instruction,{op->value},{}});
                } else if (const auto *op=std::get_if<CommAction>(&instruction.body)) {
                    if(op->sources.front().rank==rank_)
                        for(std::size_t slot=0;slot<op->outputs.size();++slot)
                            if(op->destinations[slot].rank==rank_)baseline.back().push_back(ParallelTask{{},&instruction,op->inputs,{op->outputs[slot]}});
                }
            }
            add_final_output_demands(baseline);
            run_parallel_tasks(baseline,true);
        }
        for (const auto *phase : phases) {
            const auto phase_start=std::chrono::steady_clock::now();
            std::size_t begin = 0;
            for (std::size_t end = 0; end <= phase->size(); ++end) {
                if (end != phase->size() && !std::holds_alternative<FenceOp>((*phase)[end].body)) continue;
                // The last queue is always Host; zero GPUs needs no fake device.
                std::vector<std::vector<ParallelTask>> tasks(device_workers + 1);
                compile_parallel_phase(*phase, tasks, begin, end);
                if (phase == &plan_->finalization && end == phase->size())
                    add_final_output_demands(tasks);
                if(waterline_) {
                    std::set<ValueId> needed;
                    for(const auto &queue:tasks)for(const auto &task:queue)for(auto id:task.inputs)
                        if(preparation_instructions.count(id) && !parallel_values_.count(id))needed.insert(id);
                    for(auto id:needed) {
                        define_parallel_value(id,desc(id).place);
                        tasks.back().push_back(ParallelTask{{},preparation_instructions.at(id),{}, {id},true});
                    }
                }
                const auto start = std::chrono::steady_clock::now();
                run_parallel_tasks(tasks);
                if (end != phase->size()) {
                    api_.drain();
                    record_fence((*phase)[end].ordinal, start);
                }
                begin = end + 1;
            }
            if (phase == &plan_->initialization) timing_.initialization_nanoseconds = elapsed_nanoseconds(phase_start,std::chrono::steady_clock::now());
        }
        if (waterline_) {
            for (std::size_t d=0; d<waterline_->peak.front.size(); ++d)
                std::cerr << "waterline GPU" << d << " front_peak=" << waterline_->peak.front[d]
                          << " plain_peak=" << waterline_->peak.plain[d] << " remaining_front=" << waterline_->used.front[d]
                          << " remaining_plain=" << waterline_->used.plain[d] << '\n';
            std::cerr << "waterline shared host_peak=" << waterline_->peak.host << " pinned_peak=" << waterline_->peak.pinned
                      << " raw_peak=" << waterline_->peak.raw << '\n';
        }
        for (ValueId id : plan_->final_outputs) {
            const auto &output_desc = desc(id);
            if (output_desc.place.rank != rank_) continue;
            Value output = resolve_parallel_value(id, output_desc.place, std::nullopt);
            store_.define_ready(id, output_desc.place, std::move(output));
        }
    }

    void run_resident_tasks(std::vector<std::vector<ParallelTask>> &queues) {
        std::vector<std::thread> workers;
        for (std::size_t w = 0; w < queues.size(); ++w) {
            if (queues[w].empty()) continue;
            workers.emplace_back([&, w] {
                ThreadTrace::set_thread_name("rank-" + std::to_string(rank_) +
                    (w + 1 == queues.size() ? "-cpu-worker" : "-device-worker-" + std::to_string(w)));
                try {
                    for (auto &task : queues[w]) {
                        if (parallel_failed_.load()) return;
                        task.execute();
                    }
                } catch (...) { record_parallel_failure(std::current_exception()); }
            });
        }
        const auto communicate = [&] {
            try {
                for (const auto &group : coordinator_groups_) {
                    if (parallel_failed_.load()) return;
                    execute_parallel_communication(group->action, group->instruction_ordinal, group);
                    if constexpr (!detail::UsesBackgroundCommunicationIssuer<Api>::value)
                        finish_parallel_group(group, std::nullopt);
                }
            } catch (...) { record_parallel_failure(std::current_exception()); }
        };
        std::thread issuer;
        if constexpr (detail::UsesBackgroundCommunicationIssuer<Api>::value) {
            if (!coordinator_groups_.empty()) issuer = std::thread(communicate);
        } else communicate(); // MPI_THREAD_FUNNELED stays on the calling thread.
        for (auto &worker : workers) worker.join();
        if (issuer.joinable()) issuer.join();
        if (parallel_failed_.load()) std::rethrow_exception(parallel_failure_);
        for (const auto &group : parallel_groups_)
            if (!group->completed) finish_parallel_group(group, std::nullopt);
        parallel_groups_.clear(); coordinator_groups_.clear();
    }

    void run_parallel_tasks(std::vector<std::vector<ParallelTask>> &queues, bool baseline_only=false) {
        // Resident execution needs neither admission metadata nor a central
        // handoff between successive submissions on the same worker.
        if (!waterline_) {
            if (!baseline_only) run_resident_tasks(queues);
            return;
        }
        struct TaskState {
            ParallelTask task;
            std::size_t worker = 0;
            bool admitted = false, started = false, completed = false;
            ResourceUsage temporary;
        };
        const std::size_t device_workers=queues.size()-1, cpu_worker=device_workers;
        const std::size_t devices=static_cast<std::size_t>(local_devices_);
        std::vector<TaskState> tasks;
        for(std::size_t w=0;w<queues.size();++w)
            for(auto &t:queues[w]) tasks.push_back({std::move(t),w,false,false,false,ResourceUsage(devices)});
        std::stable_sort(tasks.begin(),tasks.end(),[](const auto &a,const auto &b){return a.task.instruction->ordinal<b.task.instruction->ordinal;});
        std::map<std::size_t,std::vector<std::size_t>> groups;
        std::unordered_map<ValueId,std::size_t> preparations;
        for(std::size_t i=0;i<tasks.size();++i) {
            if(tasks[i].task.preparation) preparations.emplace(tasks[i].task.outputs.front(),i);
            else groups[tasks[i].task.instruction->ordinal].push_back(i);
        }
        struct Ticket { ResourceUsage temporary; std::size_t remaining=0; };
        std::map<std::size_t,Ticket> tickets;
        const auto storage=[&](ValueId id) {
            if constexpr(detail::HasWaterlineApi<Api>::value) return api_.value_resources(desc(id));
            else return ResourceUsage(devices);
        };
        const auto prepare_cost=[&](std::size_t i,PrepareBackend backend) {
            if constexpr(detail::HasWaterlineApi<Api>::value) return api_.prepare_resources(desc(tasks[i].task.outputs.front()),backend);
            else throw std::runtime_error("Api lacks PreparePlaintext");
            return PrepareResources{ResourceUsage(devices),ResourceUsage(devices),false};
        };
        const auto workspace=[&](const Instruction &instruction) {
            if constexpr(detail::HasWaterlineApi<Api>::value) return api_.instruction_workspace(instruction);
            else return ResourceUsage(devices);
        };
        const auto external=[&](ValueId id) {return std::find(plan_->external_inputs.begin(),plan_->external_inputs.end(),id)!=plan_->external_inputs.end();};
        // The same topological, one-group-at-a-time baseline is checked with
        // backend resource queries before submitting any online work.
        if(waterline_ && baseline_only) {
            WaterlineLedger baseline(*plan_->waterline,waterline_allocator_capacity_); baseline.used=waterline_->used;
            auto resident=waterline_storage_; auto remaining=waterline_remaining_;auto released=waterline_released_;
            std::unordered_map<ValueId,bool> seen;
            for(const auto &[ordinal,indices]:groups) {
                const auto &first = tasks[indices.front()].task;
                ResourceUsage delta(devices), transient = first.final_output
                    ? ResourceUsage(devices) : workspace(*first.instruction);
                std::vector<std::pair<std::size_t,PrepareResources>> needed;
                delta+=transient;
                for(auto i:indices) {
                    for(ValueId id:tasks[i].task.inputs) if(preparations.count(id) && !seen[id]) {
                        const auto pi=preparations.at(id);
                        auto backend=plan_->waterline->backend==PrepareBackend::Cpu?PrepareBackend::Cpu:PrepareBackend::Gpu;
                        auto cost=prepare_cost(pi,backend);
                        if(plan_->waterline->backend==PrepareBackend::Auto && !cost.supported) {
                            backend=PrepareBackend::Cpu;cost=prepare_cost(pi,backend);
                        }
                        if(!cost.supported) throw std::runtime_error("unsupported complete Prepare backend at ValueId "+std::to_string(id));
                        delta+=cost.peak;auto temp=cost.peak;temp-=cost.resident;transient+=temp;
                        tasks[pi].task.backend=backend;needed.emplace_back(pi,cost);
                        resident.emplace(id,cost.resident);seen[id]=true;
                    }
                    for(ValueId id:tasks[i].task.outputs) {auto cost=storage(id);delta+=cost;resident.emplace(id,cost);}
                }
                if(plan_->waterline->backend==PrepareBackend::Auto && !baseline.fits(delta)) {
                    for(auto &[pi,cost]:needed)if(tasks[pi].task.backend==PrepareBackend::Gpu) {
                        auto cpu=prepare_cost(pi,PrepareBackend::Cpu);
                        if(!cpu.supported)continue;
                        delta-=cost.peak;delta+=cpu.peak;
                        auto old_temp=cost.peak;old_temp-=cost.resident;transient-=old_temp;
                        auto new_temp=cpu.peak;new_temp-=cpu.resident;transient+=new_temp;
                        resident[tasks[pi].task.outputs.front()]=cpu.resident;
                        cost=std::move(cpu);tasks[pi].task.backend=PrepareBackend::Cpu;
                    }
                }
                if(!baseline.fits(delta)) throw std::runtime_error("Waterline single-group baseline infeasible at instruction #"+std::to_string(ordinal)+baseline.describe(delta));
                baseline.reserve(delta);baseline.release(transient);
                for(auto i:indices) {
                    const auto &task=tasks[i].task;
                    if(const auto *op=std::get_if<ReleaseOp>(&task.instruction->body)) released[op->value]=true;
                    else if (!task.final_output) {
                        for(auto id:task.inputs) {if(!remaining.at(id))throw std::logic_error("Waterline baseline use underflow");--remaining.at(id);}
                        if(const auto *op=std::get_if<ComputeOp>(&task.instruction->body);op && op->reuse_input)released[op->inputs.front()]=true;
                    }
                }
                for(auto i:indices) for(auto id:tasks[i].task.inputs) {
                    const auto found=resident.find(id);
                    if(found!=resident.end() && released[id] && !remaining.at(id) && !external(id)) {
                        baseline.release(found->second);resident.erase(found);
                    }
                }
            }
        }
        if(baseline_only)return;
        struct Completion { std::size_t task; std::vector<Value> values; };
        struct Worker {
            std::mutex mutex; std::condition_variable condition;
            std::deque<std::size_t> assigned; bool stop=false, busy=false;
            std::size_t preparations_remaining=0;
            std::deque<Completion> submitted;
        };
        // Existing foreground/CPU workers, plus one independent preparation lane per GPU.
        std::vector<std::unique_ptr<Worker>> workers;
        for(std::size_t i=0;i<queues.size()+devices;++i) workers.push_back(std::make_unique<Worker>());
        std::vector<std::deque<std::size_t>> pending(workers.size());
        std::vector<std::deque<Completion>> completing(workers.size());
        std::set<ValueId> producing;
        std::deque<std::size_t> cpu_preparations;
        for (std::size_t i=0;i<tasks.size();++i) if(!tasks[i].task.preparation)pending[tasks[i].worker].push_back(i);
        std::vector<std::thread> threads;
        std::mutex dispatcher_mutex;
        std::condition_variable dispatcher_condition;
        bool submission_changed = false;
        const auto notify_dispatcher = [&] {
            {std::lock_guard<std::mutex> lock(dispatcher_mutex);submission_changed = true;}
            dispatcher_condition.notify_one();
        };
        const auto stop_workers=[&] {
            for(auto &w:workers) {{std::lock_guard<std::mutex> lock(w->mutex);w->stop=true;}w->condition.notify_all();}
            for(auto &thread:threads) if(thread.joinable())thread.join();
        };
        for(std::size_t w=0;w<workers.size();++w) threads.emplace_back([&,w] {
            ThreadTrace::set_thread_name("rank-"+std::to_string(rank_)+"-worker-"+std::to_string(w));
            auto &worker=*workers[w];
            try {
                for(;;) {
                    std::deque<std::size_t> batch;
                    {std::unique_lock<std::mutex> lock(worker.mutex);worker.condition.wait(lock,[&]{return worker.stop || !worker.assigned.empty();});
                     if(worker.stop)return;
                     batch.swap(worker.assigned);}
                    std::deque<Completion> submitted;
                    for (auto i : batch) {
                    if (parallel_failed_.load()) return;
                    Completion completion{i,{}};
                    auto &task=tasks[i].task;
                    if(task.preparation) {
                        if constexpr(detail::HasWaterlineApi<Api>::value) {
                            const auto &op=std::get<PreparePlaintextOp>(task.instruction->body);
                            auto slots=read_encode_slots(op.payload,op.output,task.instruction->ordinal);
                            // Read ahead while the preceding Encode is on the
                            // GPU. Wait only when its staging is about to be
                            // reused. Both preparations already own full credit.
                            if(!submitted.empty())for(auto &value:submitted.back().values)
                                api_.synchronize(value);
                            const auto start=std::chrono::steady_clock::now();
                            Value result=api_.prepare_plaintext_async(desc(op.output),slots,task.backend);
                            api_.validate_value(result,desc(op.output));
                            const auto elapsed=elapsed_nanoseconds(start,std::chrono::steady_clock::now());
                            {std::lock_guard<std::mutex> lock(parallel_timing_mutex_);++timing_.encode_calls;timing_.encode_nanoseconds+=elapsed;}
                            trace_event(task.backend==PrepareBackend::Cpu?"prepare_cpu":"prepare_gpu",op.output,task.instruction->ordinal,std::nullopt,"PreparePlaintext",elapsed);
                            auto output=parallel_value(op.output);
                            {
                                std::lock_guard<std::mutex> lock(output->mutex);
                                output->completion.emplace(result);
                                if(!can_clear(output->uses))output->value.emplace(std::move(result));
                                output->defined=true;
                            }
                            output->condition.notify_all();
                        } else throw std::runtime_error("Api lacks PreparePlaintext");
                    } else task.execute();
                    for(auto id:task.outputs) {
                        auto output=parallel_value(id);
                        std::shared_ptr<ParallelGroup> group;
                        {std::lock_guard<std::mutex> lock(output->mutex);group=output->group;}
                        if(group && waterline_)finish_parallel_group(group,std::nullopt);
                        {std::lock_guard<std::mutex> lock(output->mutex);
                         if(output->completion) {completion.values.push_back(std::move(*output->completion));output->completion.reset();}}
                    }
                    submitted.push_back(std::move(completion));
                    }
                    {std::lock_guard<std::mutex> lock(worker.mutex);
                     for (auto &completion : submitted) worker.submitted.push_back(std::move(completion));}
                    notify_dispatcher();
                }
            } catch(...) {record_parallel_failure(std::current_exception());notify_dispatcher();}
        });
        // Only APIs that opt into a background issuer may communicate off the
        // caller thread. MPI_THREAD_FUNNELED backends advance below, after
        // dispatching the local producers they need. V4 Waterline is single-rank.
        std::thread issuer;
        if constexpr (detail::UsesBackgroundCommunicationIssuer<Api>::value) {
            if(!coordinator_groups_.empty()) issuer=std::thread([this] {
                try {for(const auto &group:coordinator_groups_) {if(parallel_failed_.load())return;
                    execute_parallel_communication(group->action,group->instruction_ordinal,group);}}
                catch(...) {record_parallel_failure(std::current_exception());}
            });
        }
        std::size_t next_coordinator_group = 0;
        auto next=groups.begin();std::size_t finished=0,started=0,active_groups=0;
        const auto ready=[&](const std::vector<ValueId> &inputs) {
            for(auto id:inputs) {auto value=parallel_value(id);std::lock_guard<std::mutex> lock(value->mutex);if(!value->defined)return false;}
            return true;
        };
        const auto retire=[&](const std::vector<ValueId> &ids) {
            for(auto id:ids) {
                const auto it=waterline_storage_.find(id);
                if(it!=waterline_storage_.end() && !producing.count(id) && waterline_released_[id] && !waterline_remaining_.at(id) && !external(id)) {
                    waterline_->release(it->second);waterline_storage_.erase(it);
                }
            }
        };
        try {
            while(finished<tasks.size() ||
                  (!detail::UsesBackgroundCommunicationIssuer<Api>::value &&
                   next_coordinator_group<coordinator_groups_.size())) {
                if(parallel_failed_.load())std::rethrow_exception(parallel_failure_);
                {std::lock_guard<std::mutex> lock(dispatcher_mutex);submission_changed=false;}
                bool progress=false;
                api_.collect_completed();
                for(std::size_t w=0;w<workers.size();++w) {
                    auto &worker=*workers[w];
                    std::deque<Completion> submitted;
                    {std::lock_guard<std::mutex> lock(worker.mutex);submitted.swap(worker.submitted);}
                    if(submitted.empty())continue;
                    // The Api has retained the inputs and published Ready handles.
                    // Foreground submission can continue while the GPU is running;
                    // its completion ticket still owns all resource credit.
                    // A preparation lane instead keeps its reusable staging buffer
                    // exclusive until the actual upload/encode has completed.
                    if(!tasks[submitted.front().task].task.preparation)worker.busy=false;
                    for(auto &completion:submitted)completing[w].push_back(std::move(completion));
                    progress=true;
                }
                for(std::size_t w=0;w<completing.size();++w) {
                  auto &queue=completing[w];
                  while(!queue.empty()) {
                    auto &completion=queue.front();
                    auto &state=tasks[completion.task];auto &task=state.task;
                    bool complete=true;
                    if(waterline_) if constexpr(detail::HasWaterlineApi<Api>::value)
                        for(auto &v:completion.values) if(!api_.is_ready(v))complete=false;
                    if(!complete)break;
                    completion.values.clear();state.completed=true;++finished;progress=true;
                    if(waterline_) {
                        for(auto id:task.outputs)producing.erase(id);
                        if(task.preparation)waterline_->release(state.temporary);
                        else {
                            auto &ticket=tickets.at(task.instruction->ordinal);
                            if(--ticket.remaining==0) {waterline_->release(ticket.temporary);--active_groups;tickets.erase(task.instruction->ordinal);}
                            if(const auto *op=std::get_if<ReleaseOp>(&task.instruction->body))waterline_released_[op->value]=true;
                            else if (!task.final_output) {
                                for(auto id:task.inputs) {if(!waterline_remaining_.at(id))throw std::logic_error("Waterline completed use underflow");--waterline_remaining_.at(id);}
                                if(const auto *op=std::get_if<ComputeOp>(&task.instruction->body);op && op->reuse_input)waterline_released_[op->inputs.front()]=true;
                            }
                        }
                        retire(task.inputs);
                        retire(task.outputs);
                    }
                    if(task.preparation && --workers[w]->preparations_remaining==0)workers[w]->busy=false;
                    queue.pop_front();
                  }
                }
                while(next!=groups.end() && (!waterline_ || active_groups<plan_->waterline->lookahead_groups)) {
                    const auto &indices=next->second;
                    ResourceUsage delta(devices),temp(devices);
                    std::vector<std::pair<std::size_t,PrepareResources>> needed;
                    if(waterline_) {
                        const auto &first = tasks[indices.front()].task;
                        if (!first.final_output) temp=workspace(*first.instruction);
                        delta+=temp;
                        for(auto i:indices) {
                            for(auto id:tasks[i].task.inputs) if(preparations.count(id) && !tasks[preparations.at(id)].admitted) {
                                const auto pi=preparations.at(id);
                                if(std::any_of(needed.begin(),needed.end(),[&](const auto &n){return n.first==pi;}))continue;
                                auto backend=plan_->waterline->backend;
                                if(backend==PrepareBackend::Auto) {
                                    backend=PrepareBackend::Gpu;
                                    if(!prepare_cost(pi,backend).supported)backend=PrepareBackend::Cpu;
                                }
                                auto cost=prepare_cost(pi,backend);
                                if(!cost.supported)throw std::runtime_error("unsupported Prepare backend at ValueId "+std::to_string(id));
                                tasks[pi].task.backend=backend;
                                needed.emplace_back(pi,cost);delta+=cost.peak;
                            }
                            for(auto id:tasks[i].task.outputs)delta+=storage(id);
                        }
                        // Temporary pressure waits for already admitted work.
                        // CPU is chosen only when the GPU group still cannot
                        // fit after those groups have completed.
                        if(plan_->waterline->backend==PrepareBackend::Auto && active_groups==0 && started==finished && !waterline_->fits(delta)) {
                            for(auto &[pi,cost]:needed)if(tasks[pi].task.backend==PrepareBackend::Gpu) {
                                auto cpu=prepare_cost(pi,PrepareBackend::Cpu);
                                if(cpu.supported) {delta-=cost.peak;delta+=cpu.peak;cost=std::move(cpu);tasks[pi].task.backend=PrepareBackend::Cpu;}
                            }
                        }
                        if(!waterline_->fits(delta))break;
                        waterline_->reserve(delta);
                        tickets.emplace(next->first,Ticket{temp,indices.size()});++active_groups;
                        for(const auto &[pi,cost]:needed) {
                            auto &state=tasks[pi];state.admitted=true;state.temporary=cost.peak;state.temporary-=cost.resident;
                            const auto id=state.task.outputs.front();waterline_storage_.emplace(id,cost.resident);
                            producing.insert(id);
                            state.worker=state.task.backend==PrepareBackend::Cpu?cpu_worker:queues.size()+desc(id).place.index;
                            if(state.task.backend==PrepareBackend::Cpu)cpu_preparations.push_back(pi);
                            else pending[state.worker].push_back(pi);
                        }
                        for(auto i:indices)for(auto id:tasks[i].task.outputs) {waterline_storage_.emplace(id,storage(id));producing.insert(id);}
                        trace_event("admit",0,next->first,std::nullopt,"CompleteGroup",0);
                    }
                    for(auto i:indices)tasks[i].admitted=true;
                    ++next;progress=true;
                }
                for(std::size_t w=0;w<workers.size();++w) {
                    auto &worker=*workers[w];
                    const bool foreground=w<device_workers;
                    const bool cpu=w==cpu_worker;
                    if(!foreground && worker.busy)continue;
                    std::deque<std::size_t> selected;
                    while(!pending[w].empty()) {
                        const auto i=pending[w].front();const auto &state=tasks[i];
                        if(!state.admitted || (!foreground && !ready(state.task.inputs)))break;
                        selected.push_back(i);pending[w].pop_front();
                        if(cpu)break;
                    }
                    if(selected.empty() && w==cpu_worker && !cpu_preparations.empty()) {
                        selected.push_back(cpu_preparations.front());cpu_preparations.pop_front();
                    }
                    if(!selected.empty()) {
                        for(auto i:selected) {tasks[i].started=true;++started;}
                        if(tasks[selected.front()].task.preparation)worker.preparations_remaining=selected.size();
                        {std::lock_guard<std::mutex> lock(worker.mutex);
                         for(auto i:selected)worker.assigned.push_back(i);}
                        worker.busy=true;worker.condition.notify_one();progress=true;
                    }
                }
                if constexpr (!detail::UsesBackgroundCommunicationIssuer<Api>::value) {
                    if (next_coordinator_group < coordinator_groups_.size()) {
                        const auto &group = coordinator_groups_[next_coordinator_group];
                        if (ready(group->local_input_ids)) {
                            execute_parallel_communication(group->action, group->instruction_ordinal, group);
                            finish_parallel_group(group, std::nullopt);
                            ++next_coordinator_group;
                            progress = true;
                        }
                    }
                }
                if(!progress) {
                    const bool in_flight=started!=finished;
                    if(!in_flight && (!detail::UsesBackgroundCommunicationIssuer<Api>::value || coordinator_groups_.empty())) {
                        if(next!=groups.end())throw std::runtime_error("Waterline cannot advance at instruction #"+std::to_string(next->first)+ (waterline_?waterline_->describe(ResourceUsage(devices)):""));
                        throw std::runtime_error("executor has no runnable producer: dependency/order invariant violated");
                    }
                    // Host submissions wake the dispatcher immediately. Only
                    // GPU completions (without a host notifier) need timed polls;
                    // the interval bounds how late a completion-gated admission
                    // starts (POSEIDON_RUNTIME_DISPATCH_POLL_US, default 100;
                    // 20 measured no different on the 4090).
                    static const auto poll_interval = [] {
                        const char *raw = std::getenv("POSEIDON_RUNTIME_DISPATCH_POLL_US");
                        long value = raw == nullptr || *raw == '\0' ? 100 : std::strtol(raw, nullptr, 10);
                        return std::chrono::microseconds(value > 0 ? value : 100);
                    }();
                    std::unique_lock<std::mutex> lock(dispatcher_mutex);
                    dispatcher_condition.wait_for(lock,poll_interval,[&] {
                        return submission_changed || parallel_failed_.load();
                    });
                }
            }
            stop_workers();if(issuer.joinable())issuer.join();
            if (parallel_failed_.load()) std::rethrow_exception(parallel_failure_);
        } catch(...) {
            record_parallel_failure(std::current_exception());stop_workers();if(issuer.joinable())issuer.join();throw;
        }
        for(const auto &group:parallel_groups_)if(!group->completed)finish_parallel_group(group,std::nullopt);
        parallel_groups_.clear();coordinator_groups_.clear();
    }

    void finish_all_groups() {
        for (std::size_t i = 0; i < groups_.size(); ++i) finish_group(i);
        // finish_group replaces every Pending entry before indices are reused.
        groups_.clear();
    }

    // An upload publishes its output as soon as the copy is submitted, so when
    // the initialization phase ends its transfers can still be in flight: the
    // CUDA events are unsynchronized and the pinned staging buffers are still
    // allocated.  Whoever drops the last reference pays for both, and that was
    // the end of the online phase -- 122 cudaFreeHost calls, ~170 ms of a
    // 1,470 ms single-card measurement, none of it computation.  Draining the
    // transfers here charges them to initialization, where the work belongs.
    // The ids are collected first because ensure_ready() may touch the store;
    // a value already released has nothing left to drain.
    void drain_initialization_transfers() {
        std::vector<std::pair<ValueId, Place>> initialized;
        initialized.reserve(store_.entries().size());
        for (const auto &item : store_.entries())
            if (const auto *ready =
                    std::get_if<typename ValueStore<Api>::Ready>(&item.second))
                initialized.emplace_back(item.first, ready->place);
        for (const auto &item : initialized)
            api_.synchronize(ensure_ready(item.first, item.second));
    }

    void synchronize_final_outputs() {
        for (ValueId id : plan_->final_outputs) {
            const auto &output_desc = desc(id);
            if (output_desc.place.rank != rank_) continue;
            current_value_ = id;
            api_.synchronize(ensure_ready(id, output_desc.place));
        }
        current_value_.reset();
    }

    RunArtifact<Value> make_artifact(DiffMode mode) {
        RunArtifact<Value> result;
        result.timing = timing_;
        if (mode == DiffMode::AllValuesAfterRun) {
            for (const auto &item : store_.entries())
                if (const auto *ready = std::get_if<typename ValueStore<Api>::Ready>(&item.second))
                    result.values.emplace(item.first, ArtifactValue<Value>{ready->place, ready->value});
        } else {
            for (ValueId id : plan_->final_outputs) {
                const auto &output_desc = desc(id);
                if (output_desc.place.rank != rank_) continue;
                Value &value = ensure_ready(id, output_desc.place);
                result.values.emplace(id, ArtifactValue<Value>{output_desc.place, value});
            }
        }
        return result;
    }

    std::string format_error(const std::string &reason) const {
        std::ostringstream out;
        out << "fatal runtime error: reason=" << reason
            << " plan_id=" << (plan_ ? plan_->plan_id : 0)
            << " plan_source_sha256=" << plan_source_sha256_
            << " local_rank=" << rank_ << " api=" << api_.name();
        if (current_value_) out << " value_id=" << *current_value_;
        if (current_) {
            out << " op_ordinal=" << current_->ordinal;
            if (const auto *encode = std::get_if<EncodeOp>(&current_->body))
                out << " op=Encode output=" << encode->output;
            else if (const auto *op = std::get_if<ComputeOp>(&current_->body))
                out << " op=" << to_string(op->kind) << " output=" << op->output << " place=" << to_string(op->place);
            else if (std::holds_alternative<FenceOp>(current_->body))
                out << " op=Fence";
            else if (const auto *release = std::get_if<ReleaseOp>(&current_->body))
                out << " op=Release value_id=" << release->value;
            else if (const auto *prepare = std::get_if<PreparePlaintextOp>(&current_->body))
                out << " op=PreparePlaintext output=" << prepare->output;
            else {
                const auto &action = std::get<CommAction>(current_->body);
                out << " op=" << to_string(action.kind) << " transfer_id=" << action.id;
            }
        }
        return out.str();
    }

    int rank_;
    int world_size_;
    int local_devices_;
    Api &api_;
    DeviceExecutionMode device_execution_mode_;
    std::size_t device_worker_count_ = 0;
    const RuntimePlan *plan_ = nullptr;
    std::string plan_source_sha256_;
    const Instruction *current_ = nullptr;
    std::optional<ValueId> current_value_;
    ValueStore<Api> store_;
    std::vector<PendingGroup> groups_;
    std::unordered_map<ValueId, ValueUses> uses_;
    std::map<std::string, std::vector<double>> bundle_slots_;
    std::optional<PlaintextBundleLoader> bundle_index_;
    std::unordered_map<ValueId, const ValueDesc *> descriptors_;
    std::unordered_map<ValueId, std::shared_ptr<ParallelValue>> parallel_values_;
    std::vector<std::shared_ptr<ParallelGroup>> parallel_groups_;
    std::vector<std::shared_ptr<ParallelGroup>> coordinator_groups_;
    std::unordered_map<
        ValueId, std::vector<std::shared_ptr<DependencyTask>>>
        parallel_dependency_dependents_;
    std::vector<std::shared_ptr<DependencyTaskQueue>>
        parallel_dependency_queues_;
    RuntimeTiming timing_;
    std::unique_ptr<std::ofstream> runtime_trace_;
    std::mutex trace_mutex_;
    bool wait_trace_enabled_ = false;
    std::string wait_trace_path_;
    std::size_t wait_trace_worker_count_ = 0;
    std::vector<WaitTraceRecord> wait_trace_records_;
    std::mutex wait_trace_mutex_;
    std::unordered_map<ValueId, ValueProducer> value_producers_;
    std::mutex parallel_timing_mutex_;
    std::atomic<bool> parallel_failed_{false};
    std::mutex parallel_failure_mutex_;
    std::exception_ptr parallel_failure_;
    std::mutex parallel_progress_mutex_;
    std::condition_variable parallel_progress_condition_;
    std::uint64_t parallel_progress_epoch_ = 0;
    bool return_all_values_ = false;
};

} // namespace fhegpu
