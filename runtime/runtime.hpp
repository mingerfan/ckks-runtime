#pragma once

#include "runtime/plaintext_bundle.hpp"
#include "runtime/verifier.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <variant>

namespace fhegpu {

enum class DiffMode { FinalOnly, AllValuesAfterRun };

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

    SequentialRuntime(int rank, int world_size, int local_devices, Api &api)
        : rank_(rank), world_size_(world_size), local_devices_(local_devices), api_(api) {}

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
            execute_phase(plan_->execution);
            execute_phase(plan_->finalization);
            finish_all_groups();
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

    struct PendingGroup {
        CommAction action;
        std::vector<ValueId> local_input_ids;
        std::vector<ValueId> local_output_ids;
        CommHandle handle;
        bool completed = false;
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
        for (std::size_t i = 0; i < action.inputs.size(); ++i)
            if (action.sources[i].rank == rank_)
                group.local_input_ids.push_back(action.inputs[i]);
        group.handle = api_.communicate_async(action, local_inputs);
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
        auto outputs = api_.wait(group.handle);
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
    const RuntimePlan *plan_ = nullptr;
    std::string plan_source_sha256_;
    const Instruction *current_ = nullptr;
    std::optional<ValueId> current_value_;
    ValueStore<Api> store_;
    std::vector<PendingGroup> groups_;
    std::map<std::string, std::vector<double>> bundle_slots_;
    std::unordered_map<ValueId, std::size_t> remaining_uses_;
    RuntimeTiming timing_;
    bool retain_all_values_ = false;
};

} // namespace fhegpu
