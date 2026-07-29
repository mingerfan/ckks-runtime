#pragma once

#include "runtime/plaintext_bundle.hpp"
#include "runtime/verifier.hpp"

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
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace fhegpu {

enum class DiffMode { FinalOnly, AllValuesAfterRun };
enum class DeviceExecutionMode {
    Sequential,
    PerDeviceWorkers,
    PerDeviceReadyWorkers,
    PerDeviceDependencyWorkers
};

struct RuntimeResources {
    const LoadedOperatorSpec &operator_spec;
    std::optional<std::filesystem::path> plaintext_bundle_dir;
    bool skip_artifact_digest_checks = false;
};

template <class Value>
struct ArtifactValue { Place place; Value value; };

// Wall time inside Api::compute. Asynchronous APIs report submission time.
struct RuntimeTiming {
    std::size_t compute_calls = 0;
    std::size_t boot_calls = 0;
    std::uint64_t compute_including_boot_nanoseconds = 0;
    std::uint64_t boot_nanoseconds = 0;
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
    using Entry = std::variant<Ready, Pending>;

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
    void erase(ValueId id) {
        if (entries_.erase(id) != 1)
            throw std::runtime_error("ValueId cannot be released from ValueStore: " +
                                     std::to_string(id));
    }

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
        bundle_slots_.clear();
        retain_all_values_ = diff_mode == DiffMode::AllValuesAfterRun;
        remaining_uses_.clear();
        timing_ = RuntimeTiming{};
        open_trace();
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
            initialize_use_counts();
            load_bundle(resources);
            api_.preflight(loaded_plan.source_sha256,
                           resources.skip_artifact_digest_checks,
                           plan_->target, resources.operator_spec.spec, requirements);
            bind_inputs(local_inputs);

            const auto initialization_start = std::chrono::steady_clock::now();
            timing_.setup_nanoseconds = elapsed_nanoseconds(
                setup_start, initialization_start);
            execute_phase(plan_->initialization);
            finish_all_groups();

            const auto online_execution_start = std::chrono::steady_clock::now();
            timing_.initialization_nanoseconds = elapsed_nanoseconds(
                initialization_start, online_execution_start);
            if (device_execution_mode_ ==
                DeviceExecutionMode::PerDeviceDependencyWorkers) {
                execute_dependency_parallel_phases();
            } else if (device_execution_mode_ != DeviceExecutionMode::Sequential) {
                execute_device_parallel_phases();
            } else {
                execute_phase(plan_->execution);
                execute_phase(plan_->finalization);
                finish_all_groups();
            }
            synchronize_final_outputs();
            timing_.online_execution_nanoseconds = elapsed_nanoseconds(
                online_execution_start, std::chrono::steady_clock::now());
            return make_artifact(diff_mode);
        } catch (const std::exception &error) {
            const std::string diagnostic = format_error(error.what());
            std::cerr << diagnostic << std::endl;
            std::cerr.flush();
            api_.abort_all(1, diagnostic);
        }
    }

private:
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
        *runtime_trace_
            << "rank,event,id,ordinal,consumer_ordinal,op,duration_ns\n";
        runtime_trace_->flush();
    }

    void trace_event(const char *event, std::uint64_t id,
                     std::uint64_t ordinal,
                     std::optional<std::uint64_t> consumer_ordinal,
                     std::string_view op,
                     std::uint64_t duration_nanoseconds) {
        if (!runtime_trace_) return;
        std::lock_guard<std::mutex> lock(trace_mutex_);
        *runtime_trace_ << rank_ << ',' << event << ',' << id << ','
                        << ordinal << ',';
        if (consumer_ordinal) *runtime_trace_ << *consumer_ordinal;
        *runtime_trace_ << ',' << op << ',' << duration_nanoseconds << '\n';
        runtime_trace_->flush();
    }

    struct PendingGroup {
        CommAction action;
        std::vector<ValueId> local_input_ids;
        std::vector<ValueId> local_output_ids;
        CommHandle handle;
        std::uint64_t instruction_ordinal = 0;
        bool completed = false;
    };

    struct ParallelGroup;

    struct ParallelValue {
        Place place;
        std::mutex mutex;
        std::condition_variable condition;
        bool defined = false;
        std::optional<Value> value;
        std::shared_ptr<ParallelGroup> group;
        std::size_t local_slot = 0;
    };

    struct ParallelGroup {
        CommAction action;
        std::uint64_t instruction_ordinal = 0;
        std::vector<ValueId> local_input_ids;
        std::vector<std::shared_ptr<ParallelValue>> local_outputs;
        CommHandle handle;
        std::mutex mutex;
        bool completed = false;
    };

    enum class ParallelTaskReadiness {
        Blocked,
        PendingCommunication,
        Ready
    };

    struct ParallelTask {
        std::uint64_t ordinal = 0;
        std::vector<ValueId> input_ids;
        std::function<void()> execute;
        bool completed = false;
    };

    struct DependencyTaskQueue;

    struct DependencyTask {
        std::uint64_t ordinal = 0;
        std::vector<ValueId> input_ids;
        std::function<void()> execute;
        std::weak_ptr<DependencyTaskQueue> queue;
        std::size_t queue_position = 0;
        std::atomic<std::size_t> pending_inputs{0};
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
    };

    const ValueDesc &desc(ValueId id) const {
        for (const auto &value : plan_->values) if (value.id == id) return value;
        throw std::runtime_error("missing value descriptor for " + std::to_string(id));
    }

    void load_bundle(const RuntimeResources &resources) {
        if (!plan_->plaintext_bundle) return;
        if (!resources.plaintext_bundle_dir)
            throw std::runtime_error("RuntimePlan requires a plaintext bundle directory");
        std::vector<std::string> local_contents;
        for (const auto &instruction : plan_->initialization) {
            const auto *encode = std::get_if<EncodeOp>(&instruction.body);
            if (!encode || desc(encode->output).place.rank != rank_) continue;
            if (const auto *payload = std::get_if<BundleEncodePayload>(&encode->payload))
                local_contents.push_back(payload->content);
        }
        auto bundle = PlaintextBundleLoader::load(*resources.plaintext_bundle_dir,
                                                   *plan_->plaintext_bundle,
                                                   local_contents,
                                                   resources.operator_spec.spec.poly_degree / 2,
                                                   resources.skip_artifact_digest_checks);
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
            else execute_communication(std::get<CommAction>(instruction.body));
        }
        current_ = nullptr;
    }

    void execute_encode(const EncodeOp &op) {
        const auto &output_desc = desc(op.output);
        if (output_desc.place.rank != rank_) return;
        std::vector<double> slots;
        if (const auto *inline_payload = std::get_if<InlineEncodePayload>(&op.payload))
            slots = inline_payload->values;
        else {
            const auto &content = std::get<BundleEncodePayload>(op.payload).content;
            slots = bundle_slots_.at(content);
        }
        for (double &value : slots) if (value == 0.0) value = 0.0;
        Value output = api_.encode_plaintext(output_desc, slots);
        api_.validate_value(output, output_desc);
        store_.define_ready(op.output, output_desc.place, std::move(output));
    }

    Value &ensure_ready(ValueId id, const Place &expected_place) {
        auto &entry = store_.lookup(id);
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
        Value output = api_.compute(op, inputs);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start);
        const auto elapsed_nanoseconds =
            static_cast<std::uint64_t>(elapsed.count());
        trace_event("compute", op.output, current_->ordinal, std::nullopt,
                    to_string(op.kind), elapsed_nanoseconds);
        ++timing_.compute_calls;
        timing_.compute_including_boot_nanoseconds += elapsed_nanoseconds;
        if (op.kind == ComputeKind::Boot) {
            ++timing_.boot_calls;
            timing_.boot_nanoseconds += elapsed_nanoseconds;
        }
        const auto &output_desc = desc(op.output);
        api_.validate_value(output, output_desc);
        store_.define_ready(op.output, op.place, std::move(output));
        for (ValueId id : op.inputs) release_after_use(id);
    }

    void execute_communication(const CommAction &action) {
        std::vector<Value> local_inputs;
        for (std::size_t i = 0; i < action.inputs.size(); ++i)
            if (action.sources[i].rank == rank_) local_inputs.push_back(ensure_ready(action.inputs[i], action.sources[i]));
        PendingGroup group;
        group.action = action;
        group.instruction_ordinal = current_->ordinal;
        for (std::size_t i = 0; i < action.inputs.size(); ++i)
            if (action.sources[i].rank == rank_)
                group.local_input_ids.push_back(action.inputs[i]);
        const auto post_start = std::chrono::steady_clock::now();
        group.handle = api_.communicate_async(action, local_inputs);
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
            store_.define_pending(action.outputs[i], action.destinations[i], group_id, local_slot);
        }
        groups_.push_back(std::move(group));
    }

    void finish_group(std::size_t group_id) {
        auto &group = groups_.at(group_id);
        if (group.completed) return;
        const auto wait_start = std::chrono::steady_clock::now();
        auto outputs = api_.wait(group.handle);
        const auto wait_elapsed = elapsed_nanoseconds(
            wait_start, std::chrono::steady_clock::now());
        ++timing_.communication_wait_calls;
        timing_.communication_wait_nanoseconds += wait_elapsed;
        trace_event("comm_wait", group.action.id, group.instruction_ordinal,
                    current_ ? std::optional<std::uint64_t>(current_->ordinal)
                             : std::nullopt,
                    to_string(group.action.kind), wait_elapsed);
        if (outputs.size() != group.local_output_ids.size()) throw std::runtime_error("Api wait returned wrong output count");
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            const ValueId id = group.local_output_ids[i];
            const auto &expected_desc = desc(id);
            api_.validate_value(outputs[i], expected_desc);
            store_.lookup(id) = typename ValueStore<Api>::Ready{expected_desc.place, std::move(outputs[i])};
        }
        for (ValueId id : group.local_input_ids) release_after_use(id);
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
        if (!parallel_values_.emplace(id, value).second)
            throw std::runtime_error(
                "ValueId defined twice in parallel ValueStore: " +
                std::to_string(id));
        return value;
    }

    void prepare_parallel_values() {
        if (retain_all_values_)
            throw std::runtime_error(
                "per-device worker execution does not support AllValuesAfterRun");
        if (world_size_ != 1 || local_devices_ <= 0)
            throw std::runtime_error(
                "per-device worker execution requires one rank and at least one device");

        parallel_values_.clear();
        parallel_groups_.clear();
        parallel_dependency_dependents_.clear();
        parallel_dependency_queues_.clear();
        parallel_remaining_uses_ = remaining_uses_;
        parallel_failed_.store(false);
        parallel_failure_ = nullptr;
        {
            std::lock_guard<std::mutex> lock(parallel_progress_mutex_);
            parallel_progress_epoch_ = 0;
        }

        for (const auto &item : store_.entries()) {
            const auto *ready =
                std::get_if<typename ValueStore<Api>::Ready>(&item.second);
            if (ready == nullptr)
                throw std::runtime_error(
                    "initialization left a pending value before parallel execution");
            auto value = define_parallel_value(item.first, ready->place);
            value->value.emplace(ready->value);
            value->defined = true;
        }
        store_ = ValueStore<Api>{};
    }

    void record_parallel_failure(std::exception_ptr failure) {
        bool expected = false;
        if (!parallel_failed_.compare_exchange_strong(expected, true)) return;
        {
            std::lock_guard<std::mutex> lock(parallel_failure_mutex_);
            parallel_failure_ = std::move(failure);
        }
        for (const auto &item : parallel_values_)
            item.second->condition.notify_all();
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
        for (ValueId id : task.input_ids) {
            auto value = parallel_value(id);
            std::lock_guard<std::mutex> lock(value->mutex);
            if (!value->defined) return ParallelTaskReadiness::Blocked;
            if (!value->value)
                readiness = ParallelTaskReadiness::PendingCommunication;
        }
        return readiness;
    }

    void release_parallel_value_after_use(ValueId id) {
        bool release = false;
        {
            std::lock_guard<std::mutex> lock(parallel_use_mutex_);
            const auto found = parallel_remaining_uses_.find(id);
            if (found == parallel_remaining_uses_.end() || found->second == 0)
                throw std::runtime_error(
                    "ValueId has no remaining parallel use: " +
                    std::to_string(id));
            release = --found->second == 0;
        }
        if (!release) return;

        std::optional<Value> released;
        auto value = parallel_value(id);
        {
            std::lock_guard<std::mutex> lock(value->mutex);
            if (!value->defined || !value->value)
                throw std::runtime_error(
                    "parallel ValueId cannot be released before it is ready: " +
                    std::to_string(id));
            released.emplace(std::move(*value->value));
            value->value.reset();
        }
    }

    void finish_parallel_group(
        const std::shared_ptr<ParallelGroup> &group,
        std::optional<std::uint64_t> consumer_ordinal) {
        std::unique_lock<std::mutex> lock(group->mutex);
        if (group->completed) return;

        const auto wait_start = std::chrono::steady_clock::now();
        auto outputs = api_.wait(group->handle);
        const auto wait_elapsed = elapsed_nanoseconds(
            wait_start, std::chrono::steady_clock::now());
        {
            std::lock_guard<std::mutex> timing_lock(parallel_timing_mutex_);
            ++timing_.communication_wait_calls;
            timing_.communication_wait_nanoseconds += wait_elapsed;
        }
        trace_event("comm_wait", group->action.id,
                    group->instruction_ordinal, consumer_ordinal,
                    to_string(group->action.kind), wait_elapsed);
        if (outputs.size() != group->local_outputs.size())
            throw std::runtime_error("Api wait returned wrong parallel output count");

        for (std::size_t i = 0; i < outputs.size(); ++i) {
            const ValueId id = group->action.outputs[i];
            const auto &expected_desc = desc(id);
            api_.validate_value(outputs[i], expected_desc);
            auto &value = group->local_outputs[i];
            {
                std::lock_guard<std::mutex> value_lock(value->mutex);
                if (!value->defined || value->group.get() != group.get())
                    throw std::runtime_error(
                        "parallel communication output state is invalid");
                value->value.emplace(std::move(outputs[i]));
                value->group.reset();
            }
            value->condition.notify_all();
            activate_parallel_dependents(id);
        }
        notify_parallel_progress();
        group->completed = true;
        const auto input_ids = group->local_input_ids;
        lock.unlock();
        for (ValueId id : input_ids) release_parallel_value_after_use(id);
    }

    Value resolve_parallel_value(
        ValueId id, const Place &expected_place,
        std::optional<std::uint64_t> consumer_ordinal) {
        auto value = parallel_value(id);
        for (;;) {
            std::shared_ptr<ParallelGroup> group;
            {
                std::unique_lock<std::mutex> lock(value->mutex);
                value->condition.wait(lock, [&] {
                    return value->defined || parallel_failed_.load();
                });
                if (parallel_failed_.load())
                    throw std::runtime_error("parallel execution was cancelled");
                if (value->place != expected_place)
                    throw std::runtime_error(
                        "parallel value Place mismatch for " +
                        std::to_string(id));
                if (value->value) return *value->value;
                group = value->group;
            }
            if (!group)
                throw std::runtime_error(
                    "parallel value is defined without data or communication");
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
        Value output = api_.compute(op, inputs);
        const auto elapsed = elapsed_nanoseconds(
            start, std::chrono::steady_clock::now());
        trace_event("compute", op.output, instruction_ordinal,
                    std::nullopt, to_string(op.kind), elapsed);
        {
            std::lock_guard<std::mutex> lock(parallel_timing_mutex_);
            ++timing_.compute_calls;
            timing_.compute_including_boot_nanoseconds += elapsed;
            if (op.kind == ComputeKind::Boot) {
                ++timing_.boot_calls;
                timing_.boot_nanoseconds += elapsed;
            }
        }

        const auto &output_desc = desc(op.output);
        api_.validate_value(output, output_desc);
        {
            std::lock_guard<std::mutex> lock(output_value->mutex);
            output_value->value.emplace(std::move(output));
            output_value->defined = true;
        }
        output_value->condition.notify_all();
        activate_parallel_dependents(op.output);
        notify_parallel_progress();
        for (ValueId id : op.inputs) release_parallel_value_after_use(id);
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
        group->handle = api_.communicate_async(action, local_inputs);
        const auto post_elapsed = elapsed_nanoseconds(
            post_start, std::chrono::steady_clock::now());
        {
            std::lock_guard<std::mutex> lock(parallel_timing_mutex_);
            ++timing_.communication_post_calls;
            timing_.communication_post_nanoseconds += post_elapsed;
        }
        trace_event("comm_post", action.id, instruction_ordinal,
                    std::nullopt, to_string(action.kind), post_elapsed);

        for (std::size_t i = 0; i < group->local_outputs.size(); ++i) {
            auto &value = group->local_outputs[i];
            {
                std::lock_guard<std::mutex> lock(value->mutex);
                value->group = group;
                value->local_slot = i;
                value->defined = true;
            }
            value->condition.notify_all();
        }
        if (device_execution_mode_ ==
            DeviceExecutionMode::PerDeviceDependencyWorkers) {
            finish_parallel_group(group, std::nullopt);
        } else {
            notify_parallel_progress();
        }
    }

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

    void compile_parallel_phase(
        const std::vector<Instruction> &instructions,
        std::vector<std::vector<ParallelTask>> &tasks) {
        for (const auto &instruction : instructions) {
            if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) {
                if (op->place.rank != rank_) continue;
                if (op->place.kind != PlaceKind::Device ||
                    op->place.index < 0 || op->place.index >= local_devices_)
                    throw std::runtime_error(
                        "per-device worker execution requires Device compute operations");
                auto output = define_parallel_value(
                    op->output, desc(op->output).place);
                const std::size_t worker =
                    static_cast<std::size_t>(op->place.index) % tasks.size();
                tasks[worker].push_back(ParallelTask{
                    instruction.ordinal,
                    op->inputs,
                    [this, op = *op, ordinal = instruction.ordinal, output] {
                        execute_parallel_compute(op, ordinal, output);
                    }});
                continue;
            }
            if (std::holds_alternative<EncodeOp>(instruction.body))
                throw std::runtime_error(
                    "per-device worker execution does not support online Encode");

            const auto action = std::get<CommAction>(instruction.body);
            auto group = std::make_shared<ParallelGroup>();
            group->action = action;
            group->instruction_ordinal = instruction.ordinal;
            for (std::size_t i = 0; i < action.inputs.size(); ++i)
                if (action.sources[i].rank == rank_)
                    group->local_input_ids.push_back(action.inputs[i]);
            for (std::size_t i = 0; i < action.outputs.size(); ++i) {
                if (action.destinations[i].rank != rank_) continue;
                group->local_outputs.push_back(define_parallel_value(
                    action.outputs[i], action.destinations[i]));
            }
            parallel_groups_.push_back(group);
            const int device = parallel_communication_worker(action);
            const std::size_t worker =
                static_cast<std::size_t>(device) % tasks.size();
            tasks[worker].push_back(ParallelTask{
                instruction.ordinal,
                group->local_input_ids,
                [this, action, ordinal = instruction.ordinal, group] {
                    execute_parallel_communication(action, ordinal, group);
                }});
        }
    }

    void append_dependency_task(
        const std::shared_ptr<DependencyTaskQueue> &queue,
        std::uint64_t ordinal, std::vector<ValueId> input_ids,
        std::function<void()> execute) {
        auto task = std::make_shared<DependencyTask>();
        task->ordinal = ordinal;
        task->input_ids = std::move(input_ids);
        task->execute = std::move(execute);
        task->queue = queue;
        task->queue_position = queue->tasks.size();
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
                    [this, op = *op, ordinal = instruction.ordinal, output] {
                        execute_parallel_compute(op, ordinal, output);
                    });
                continue;
            }
            if (std::holds_alternative<EncodeOp>(instruction.body))
                throw std::runtime_error(
                    "dependency worker execution does not support online Encode");

            const auto action = std::get<CommAction>(instruction.body);
            auto group = std::make_shared<ParallelGroup>();
            group->action = action;
            group->instruction_ordinal = instruction.ordinal;
            for (std::size_t i = 0; i < action.inputs.size(); ++i)
                if (action.sources[i].rank == rank_)
                    group->local_input_ids.push_back(action.inputs[i]);
            for (std::size_t i = 0; i < action.outputs.size(); ++i) {
                if (action.destinations[i].rank != rank_) continue;
                group->local_outputs.push_back(define_parallel_value(
                    action.outputs[i], action.destinations[i]));
            }
            parallel_groups_.push_back(group);
            const int device = parallel_communication_worker(action);
            const std::size_t worker =
                static_cast<std::size_t>(device) % communication_queues.size();
            append_dependency_task(
                communication_queues[worker], instruction.ordinal,
                group->local_input_ids,
                [this, action, ordinal = instruction.ordinal, group] {
                    execute_parallel_communication(action, ordinal, group);
                });
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
            {
                std::unique_lock<std::mutex> lock(queue->mutex);
                if (queue->ready.empty() &&
                    queue->completed != queue->tasks.size() &&
                    !parallel_failed_.load()) {
                    const auto wait_start = std::chrono::steady_clock::now();
                    queue->condition.wait(lock, [&] {
                        return parallel_failed_.load() ||
                               !queue->ready.empty() ||
                               queue->completed == queue->tasks.size();
                    });
                    wait_elapsed = elapsed_nanoseconds(
                        wait_start, std::chrono::steady_clock::now());
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

            if (wait_elapsed != 0 && queue->compute_queue) {
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
        const std::size_t worker_count =
            device_worker_count_ == 0
                ? static_cast<std::size_t>(local_devices_)
                : device_worker_count_;
        if (worker_count == 0 ||
            worker_count > static_cast<std::size_t>(local_devices_))
            throw std::runtime_error(
                "dependency worker count must be between one and local device count");

        std::vector<std::shared_ptr<DependencyTaskQueue>> compute_queues;
        std::vector<std::shared_ptr<DependencyTaskQueue>> communication_queues;
        compute_queues.reserve(worker_count);
        communication_queues.reserve(worker_count);
        parallel_dependency_queues_.reserve(worker_count * 2);
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            auto compute_queue = std::make_shared<DependencyTaskQueue>();
            compute_queue->compute_queue = true;
            compute_queues.push_back(compute_queue);
            parallel_dependency_queues_.push_back(std::move(compute_queue));

            auto communication_queue = std::make_shared<DependencyTaskQueue>();
            communication_queues.push_back(communication_queue);
            parallel_dependency_queues_.push_back(
                std::move(communication_queue));
        }

        compile_dependency_phase(
            plan_->execution, compute_queues, communication_queues);
        compile_dependency_phase(
            plan_->finalization, compute_queues, communication_queues);
        initialize_dependency_tasks();

        std::vector<std::thread> workers;
        workers.reserve(parallel_dependency_queues_.size());
        for (const auto &queue : parallel_dependency_queues_) {
            workers.emplace_back([this, queue] {
                try {
                    execute_dependency_task_queue(queue);
                } catch (...) {
                    record_parallel_failure(std::current_exception());
                }
            });
        }
        for (auto &worker : workers) worker.join();
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
        for (const auto &group : parallel_groups_)
            if (!group->completed)
                throw std::runtime_error(
                    "dependency communication did not complete");
        parallel_dependency_queues_.clear();
        parallel_dependency_dependents_.clear();
    }

    void execute_ready_parallel_tasks(std::vector<ParallelTask> &tasks) {
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
                auto &task = tasks[index];
                if (task.completed) continue;
                if (!first_unfinished) first_unfinished = index;
                const auto readiness = parallel_task_readiness(task);
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
                auto &task = tasks[*selected];
                if (first_unfinished && *selected != *first_unfinished) {
                    std::lock_guard<std::mutex> lock(parallel_timing_mutex_);
                    ++timing_.device_backfill_tasks;
                }
                task.completed = true;
                task.execute();
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

    void execute_device_parallel_phases() {
        prepare_parallel_values();
        const std::size_t worker_count =
            device_worker_count_ == 0
                ? static_cast<std::size_t>(local_devices_)
                : device_worker_count_;
        if (worker_count == 0 ||
            worker_count > static_cast<std::size_t>(local_devices_))
            throw std::runtime_error(
                "device worker count must be between one and local device count");
        std::vector<std::vector<ParallelTask>> tasks(
            worker_count);
        compile_parallel_phase(plan_->execution, tasks);
        compile_parallel_phase(plan_->finalization, tasks);

        std::vector<std::thread> workers;
        workers.reserve(tasks.size());
        for (std::size_t device = 0; device < tasks.size(); ++device) {
            workers.emplace_back([this, &tasks, device] {
                try {
                    if (device_execution_mode_ ==
                        DeviceExecutionMode::PerDeviceReadyWorkers) {
                        execute_ready_parallel_tasks(tasks[device]);
                    } else {
                        for (auto &task : tasks[device]) {
                            if (parallel_failed_.load()) return;
                            task.execute();
                        }
                    }
                } catch (...) {
                    record_parallel_failure(std::current_exception());
                }
            });
        }
        for (auto &worker : workers) worker.join();
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
        for (const auto &group : parallel_groups_)
            if (!group->completed)
                throw std::runtime_error(
                    "parallel communication output was never consumed");
    }

    void initialize_use_counts() {
        if (retain_all_values_) return;
        const auto count_phase = [&](const std::vector<Instruction> &instructions) {
            for (const auto &instruction : instructions) {
                if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) {
                    if (op->place.rank != rank_) continue;
                    for (ValueId id : op->inputs) ++remaining_uses_[id];
                } else if (const auto *action =
                               std::get_if<CommAction>(&instruction.body)) {
                    for (std::size_t i = 0; i < action->inputs.size(); ++i)
                        if (action->sources[i].rank == rank_)
                            ++remaining_uses_[action->inputs[i]];
                }
            }
        };
        count_phase(plan_->initialization);
        count_phase(plan_->execution);
        count_phase(plan_->finalization);
        for (ValueId id : plan_->final_outputs)
            if (desc(id).place.rank == rank_) ++remaining_uses_[id];
    }

    void release_after_use(ValueId id) {
        if (retain_all_values_) return;
        auto found = remaining_uses_.find(id);
        if (found == remaining_uses_.end() || found->second == 0)
            throw std::runtime_error("ValueId has no remaining local use: " +
                                     std::to_string(id));
        if (--found->second == 0) store_.erase(id);
    }

    void finish_all_groups() {
        for (std::size_t i = 0; i < groups_.size(); ++i) finish_group(i);
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
    std::map<std::string, std::vector<double>> bundle_slots_;
    std::unordered_map<ValueId, std::size_t> remaining_uses_;
    std::unordered_map<ValueId, std::shared_ptr<ParallelValue>> parallel_values_;
    std::vector<std::shared_ptr<ParallelGroup>> parallel_groups_;
    std::unordered_map<ValueId, std::size_t> parallel_remaining_uses_;
    std::unordered_map<
        ValueId, std::vector<std::shared_ptr<DependencyTask>>>
        parallel_dependency_dependents_;
    std::vector<std::shared_ptr<DependencyTaskQueue>>
        parallel_dependency_queues_;
    RuntimeTiming timing_;
    std::unique_ptr<std::ofstream> runtime_trace_;
    std::mutex trace_mutex_;
    std::mutex parallel_timing_mutex_;
    std::mutex parallel_use_mutex_;
    std::atomic<bool> parallel_failed_{false};
    std::mutex parallel_failure_mutex_;
    std::exception_ptr parallel_failure_;
    std::mutex parallel_progress_mutex_;
    std::condition_variable parallel_progress_condition_;
    std::uint64_t parallel_progress_epoch_ = 0;
    bool retain_all_values_ = false;
};

} // namespace fhegpu
