#pragma once

#include "runtime/verifier.hpp"

#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <variant>

namespace fhegpu {

enum class DiffMode { FinalOnly, AllValuesAfterRun };

template <class Value>
struct ArtifactValue {
    Place place;
    Value value;
};

template <class Value>
struct RunArtifact {
    std::map<ValueId, ArtifactValue<Value>> values;
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

    RunArtifact<Value> run(const RuntimePlan &plan,
                           const std::unordered_map<ValueId, Value> &local_inputs,
                           DiffMode diff_mode = DiffMode::FinalOnly) {
        plan_ = &plan;
        current_ = nullptr;
        current_value_.reset();
        try {
            PlanVerifier::verify(plan);
            PlanVerifier::verify_runtime_target(plan, rank_, world_size_, local_devices_);
            api_.preflight(plan.fingerprint());
            bind_inputs(local_inputs);
            execute_phase(plan.initialization);
            finish_all_groups();
            execute_phase(plan.execution);
            execute_phase(plan.finalization);
            finish_all_groups();
            synchronize_final_outputs();
            return make_artifact(diff_mode);
        } catch (const std::exception &error) {
            const std::string diagnostic = format_error(error.what());
            std::cerr << diagnostic << std::endl;
            std::cerr.flush();
            api_.abort_all(1, diagnostic);
        }
    }

private:
    struct PendingGroup {
        CommAction action;
        std::vector<ValueId> local_output_ids;
        CommHandle handle;
        bool completed = false;
    };

    const ValueDesc &desc(ValueId id) const {
        for (const auto &d : plan_->values) if (d.id == id) return d;
        throw std::runtime_error("missing value descriptor for " + std::to_string(id));
    }

    void bind_inputs(const std::unordered_map<ValueId, Value> &inputs) {
        std::size_t expected = 0;
        for (ValueId id : plan_->external_inputs) {
            const auto &d = desc(id);
            if (d.place.rank != rank_) continue;
            ++expected;
            auto it = inputs.find(id);
            if (it == inputs.end()) throw std::runtime_error("missing local external input " + std::to_string(id));
            if (api_.kind(it->second) != d.kind) throw std::runtime_error("external input kind mismatch for " + std::to_string(id));
            store_.define_ready(id, d.place, it->second);
        }
        if (inputs.size() != expected) throw std::runtime_error("unexpected local external input binding");
    }

    void execute_phase(const std::vector<Instruction> &instructions) {
        for (const auto &instruction : instructions) {
            current_ = &instruction;
            if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) execute_compute(*op);
            else execute_communication(std::get<CommAction>(instruction.body));
        }
        current_ = nullptr;
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
        inputs.reserve(op.inputs.size());
        for (ValueId id : op.inputs) inputs.push_back(ensure_ready(id, op.place));
        Value output = api_.compute(op, inputs);
        const auto &out_desc = desc(op.output);
        if (api_.kind(output) != out_desc.kind) throw std::runtime_error("Api returned wrong compute output kind");
        store_.define_ready(op.output, op.place, std::move(output));
    }

    void execute_communication(const CommAction &action) {
        std::vector<Value> local_inputs;
        for (std::size_t i = 0; i < action.inputs.size(); ++i)
            if (action.sources[i].rank == rank_) local_inputs.push_back(ensure_ready(action.inputs[i], action.sources[i]));
        PendingGroup group;
        group.action = action;
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
            const auto &d = desc(id);
            if (api_.kind(outputs[i]) != d.kind) throw std::runtime_error("Api wait returned wrong output kind");
            store_.lookup(id) = typename ValueStore<Api>::Ready{d.place, std::move(outputs[i])};
        }
        group.completed = true;
    }

    void finish_all_groups() {
        for (std::size_t i = 0; i < groups_.size(); ++i) finish_group(i);
    }

    void synchronize_final_outputs() {
        for (ValueId id : plan_->final_outputs) {
            const auto &d = desc(id);
            if (d.place.rank != rank_) continue;
            current_value_ = id;
            api_.synchronize(ensure_ready(id, d.place));
        }
        current_value_.reset();
    }

    RunArtifact<Value> make_artifact(DiffMode mode) {
        RunArtifact<Value> result;
        if (mode == DiffMode::AllValuesAfterRun) {
            for (const auto &item : store_.entries()) {
                if (const auto *ready = std::get_if<typename ValueStore<Api>::Ready>(&item.second))
                    result.values.emplace(item.first, ArtifactValue<Value>{ready->place, ready->value});
            }
        } else {
            for (ValueId id : plan_->final_outputs) {
                const auto &d = desc(id);
                if (d.place.rank != rank_) continue;
                Value &value = ensure_ready(id, d.place);
                result.values.emplace(id, ArtifactValue<Value>{d.place, value});
            }
        }
        return result;
    }

    std::string format_error(const std::string &reason) const {
        std::ostringstream os;
        os << "fatal runtime error: reason=" << reason
           << " plan_id=" << (plan_ ? plan_->plan_id : 0)
           << " fingerprint=" << (plan_ ? plan_->fingerprint() : 0)
           << " local_rank=" << rank_ << " api=" << api_.name();
        if (current_value_) os << " value_id=" << *current_value_;
        if (current_) {
            os << " op_ordinal=" << current_->ordinal;
            if (const auto *op = std::get_if<ComputeOp>(&current_->body)) {
                os << " op=" << to_string(op->kind) << " output=" << op->output << " place=" << to_string(op->place);
            } else {
                const auto &a = std::get<CommAction>(current_->body);
                os << " op=" << to_string(a.kind) << " transfer_id=" << a.id;
                if (!a.sources.empty()) os << " source=" << to_string(a.sources[0]);
                if (!a.destinations.empty()) os << " destination=" << to_string(a.destinations[0]);
            }
        }
        return os.str();
    }

    int rank_;
    int world_size_;
    int local_devices_;
    Api &api_;
    const RuntimePlan *plan_ = nullptr;
    const Instruction *current_ = nullptr;
    std::optional<ValueId> current_value_;
    ValueStore<Api> store_;
    std::vector<PendingGroup> groups_;
};

} // namespace fhegpu
