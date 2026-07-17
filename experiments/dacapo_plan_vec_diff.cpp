#include "runtime/json_plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/utils/sha256.hpp"
#include "testing/testing.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace fhegpu;

namespace {

struct IndexedValue {
    Place place;
    VecValue value;
};

using ArtifactIndex = std::map<ValueId, IndexedValue>;

struct Comparison {
    std::size_t mismatch_slots = 0;
    std::optional<std::size_t> first_mismatch;
    double expected_at_first = 0.0;
    double actual_at_first = 0.0;
    double max_abs_diff = 0.0;
    bool metadata_matches = true;
};

struct Summary {
    std::size_t compared_instructions = 0;
    std::size_t mismatched_instructions = 0;
    std::size_t compared_final_outputs = 0;
    std::size_t mismatched_final_outputs = 0;
    std::size_t mismatch_slots = 0;
    double max_abs_diff = 0.0;
    std::vector<std::string> final_sha256;
};

const ValueDesc &value_desc(const RuntimePlan &plan, ValueId id) {
    for (const ValueDesc &value : plan.values)
        if (value.id == id) return value;
    throw std::runtime_error("missing ValueDesc " + std::to_string(id));
}

bool same_value_metadata(const ValueDesc &lhs, const ValueDesc &rhs) {
    return lhs.kind == rhs.kind && lhs.context == rhs.context &&
           lhs.level == rhs.level && lhs.scale_log2 == rhs.scale_log2 &&
           lhs.ntt == rhs.ntt && lhs.components == rhs.components;
}

bool same_compute_attrs(const ComputeAttrs &lhs, const ComputeAttrs &rhs) {
    if (lhs.index() != rhs.index()) return false;
    if (std::holds_alternative<std::monostate>(lhs)) return true;
    if (const auto *value = std::get_if<RotateAttrs>(&lhs))
        return value->steps == std::get<RotateAttrs>(rhs).steps;
    if (const auto *value = std::get_if<RescaleAttrs>(&lhs)) {
        const auto other = std::get<RescaleAttrs>(rhs);
        return value->target_level == other.target_level &&
               value->target_scale_log2 == other.target_scale_log2;
    }
    if (const auto *value = std::get_if<ModSwitchAttrs>(&lhs))
        return value->target_level ==
               std::get<ModSwitchAttrs>(rhs).target_level;
    const auto value = std::get<BootAttrs>(lhs);
    const auto other = std::get<BootAttrs>(rhs);
    return value.target_level == other.target_level &&
           value.target_scale_log2 == other.target_scale_log2 &&
           value.target_components == other.target_components &&
           value.operator_profile == other.operator_profile &&
           value.implementation == other.implementation;
}

bool same_encode_payload(const EncodePayload &lhs, const EncodePayload &rhs) {
    if (lhs.index() != rhs.index()) return false;
    if (const auto *value = std::get_if<InlineEncodePayload>(&lhs))
        return value->values == std::get<InlineEncodePayload>(rhs).values;
    return std::get<BundleEncodePayload>(lhs).content ==
           std::get<BundleEncodePayload>(rhs).content;
}

std::vector<const Instruction *> instructions(const RuntimePlan &plan) {
    std::vector<const Instruction *> result;
    const auto append = [&](const std::vector<Instruction> &phase) {
        for (const Instruction &instruction : phase)
            result.push_back(&instruction);
    };
    append(plan.initialization);
    append(plan.execution);
    append(plan.finalization);
    return result;
}

ValueId instruction_output(const Instruction &instruction) {
    if (const auto *encode = std::get_if<EncodeOp>(&instruction.body))
        return encode->output;
    if (const auto *op = std::get_if<ComputeOp>(&instruction.body))
        return op->output;
    const auto &action = std::get<CommAction>(instruction.body);
    if (action.outputs.size() != 1)
        throw std::runtime_error(
            "Dacapo point-to-point diff requires one communication output");
    return action.outputs.front();
}

std::string instruction_name(const Instruction &instruction) {
    if (std::holds_alternative<EncodeOp>(instruction.body)) return "Encode";
    if (const auto *op = std::get_if<ComputeOp>(&instruction.body))
        return to_string(op->kind);
    return to_string(std::get<CommAction>(instruction.body).kind);
}

class PlanAlignment {
public:
    PlanAlignment(const RuntimePlan &reference, const RuntimePlan &distributed)
        : reference_(reference), distributed_(distributed) {}

    void verify() {
        verify_targets();
        index_reference();
        index_transfers();
        verify_values();
        verify_instructions();
        verify_outputs();
    }

    ValueId reference_value(ValueId distributed_value) const {
        std::set<ValueId> active;
        return resolve(distributed_value, active);
    }

private:
    void verify_targets() const {
        if (reference_.target.world_size != 1 ||
            reference_.target.device_counts != std::vector<int>{0})
            throw std::runtime_error(
                "reference RuntimePlan must be single-rank Host-only");
        if (reference_.target.target_id != distributed_.target.target_id ||
            reference_.target.capability_version !=
                distributed_.target.capability_version ||
            reference_.target.operator_spec.id !=
                distributed_.target.operator_spec.id ||
            reference_.target.operator_spec.version !=
                distributed_.target.operator_spec.version ||
            reference_.target.operator_spec.source_sha256 !=
                distributed_.target.operator_spec.source_sha256)
            throw std::runtime_error(
                "reference and distributed targets use different OperatorSpecs");
        if (reference_.external_inputs != distributed_.external_inputs)
            throw std::runtime_error(
                "reference and distributed external inputs differ");
        for (ValueId id : distributed_.external_inputs)
            if (value_desc(distributed_, id).place !=
                Place{PlaceKind::Host, 0, 0})
                throw std::runtime_error(
                    "distributed external inputs must originate on Host rank 0");
    }

    void index_reference() {
        for (const ValueDesc &value : reference_.values)
            reference_values_.insert(value.id);
        for (const Instruction *instruction : instructions(reference_)) {
            if (std::holds_alternative<CommAction>(instruction->body))
                throw std::runtime_error(
                    "reference RuntimePlan must not contain communication");
            const ValueId output = instruction_output(*instruction);
            if (!reference_producers_.emplace(output, instruction).second)
                throw std::runtime_error(
                    "reference instruction output is defined twice");
        }
    }

    void index_transfers() {
        for (const Instruction *instruction : instructions(distributed_)) {
            const auto *action = std::get_if<CommAction>(&instruction->body);
            if (!action) continue;
            if (action->kind != CommKind::Transfer ||
                action->hint != CommHint::PointToPoint ||
                action->inputs.size() != 1 || action->outputs.size() != 1 ||
                action->sources.size() != 1 ||
                action->destinations.size() != 1 ||
                action->output_types.size() != 1)
                throw std::runtime_error(
                    "distributed plan contains a non-point-to-point transfer");
            const ValueId output = action->outputs.front();
            if (reference_values_.count(output) != 0)
                throw std::runtime_error(
                    "transfer output reuses a reference logical ValueId");
            if (!transfer_sources_.emplace(output, action->inputs.front()).second)
                throw std::runtime_error("transfer output is defined twice");
        }
        for (const auto &entry : transfer_sources_)
            static_cast<void>(reference_value(entry.first));
    }

    void verify_values() const {
        std::set<ValueId> distributed_ids;
        for (const ValueDesc &value : distributed_.values)
            distributed_ids.insert(value.id);
        for (ValueId id : reference_values_) {
            if (distributed_ids.count(id) == 0)
                throw std::runtime_error(
                    "distributed plan is missing reference ValueId " +
                    std::to_string(id));
            if (!same_value_metadata(value_desc(reference_, id),
                                     value_desc(distributed_, id)))
                throw std::runtime_error(
                    "logical ValueDesc metadata differs for ValueId " +
                    std::to_string(id));
        }
        for (const auto &[output, source] : transfer_sources_) {
            const ValueId logical = reference_value(source);
            if (!same_value_metadata(value_desc(reference_, logical),
                                     value_desc(distributed_, output)))
                throw std::runtime_error(
                    "transfer output metadata differs from its logical source");
        }
    }

    void verify_instructions() const {
        std::set<ValueId> seen_logical_outputs;
        for (const Instruction *instruction : instructions(distributed_)) {
            if (std::holds_alternative<CommAction>(instruction->body)) continue;
            const ValueId output = instruction_output(*instruction);
            auto found = reference_producers_.find(output);
            if (found == reference_producers_.end())
                throw std::runtime_error(
                    "distributed compute/Encode has no reference instruction");
            if (!seen_logical_outputs.insert(output).second)
                throw std::runtime_error(
                    "distributed logical output is defined twice");
            const Instruction &expected = *found->second;
            if (const auto *encode =
                    std::get_if<EncodeOp>(&instruction->body)) {
                const auto *reference_encode =
                    std::get_if<EncodeOp>(&expected.body);
                if (!reference_encode ||
                    !same_encode_payload(encode->payload,
                                         reference_encode->payload))
                    throw std::runtime_error(
                        "distributed Encode differs from reference");
                continue;
            }
            const auto &op = std::get<ComputeOp>(instruction->body);
            const auto *reference_op = std::get_if<ComputeOp>(&expected.body);
            if (!reference_op || op.kind != reference_op->kind ||
                !same_compute_attrs(op.attrs, reference_op->attrs) ||
                op.inputs.size() != reference_op->inputs.size())
                throw std::runtime_error(
                    "distributed compute differs from reference");
            for (std::size_t i = 0; i < op.inputs.size(); ++i)
                if (reference_value(op.inputs[i]) != reference_op->inputs[i])
                    throw std::runtime_error(
                        "distributed compute operand lineage differs from reference");
        }
        if (seen_logical_outputs.size() != reference_producers_.size())
            throw std::runtime_error(
                "distributed plan does not contain every reference instruction");
    }

    void verify_outputs() const {
        if (reference_.final_outputs.size() !=
            distributed_.final_outputs.size())
            throw std::runtime_error(
                "reference and distributed final output counts differ");
        for (std::size_t i = 0; i < reference_.final_outputs.size(); ++i)
            if (reference_value(distributed_.final_outputs[i]) !=
                reference_.final_outputs[i])
                throw std::runtime_error(
                    "distributed final output lineage differs from reference");
    }

    ValueId resolve(ValueId id, std::set<ValueId> &active) const {
        if (reference_values_.count(id) != 0) return id;
        auto found = transfer_sources_.find(id);
        if (found == transfer_sources_.end())
            throw std::runtime_error(
                "ValueId has no reference or transfer lineage: " +
                std::to_string(id));
        if (!active.insert(id).second)
            throw std::runtime_error("cycle in transfer lineage");
        const ValueId result = resolve(found->second, active);
        active.erase(id);
        return result;
    }

    const RuntimePlan &reference_;
    const RuntimePlan &distributed_;
    std::set<ValueId> reference_values_;
    std::map<ValueId, const Instruction *> reference_producers_;
    std::map<ValueId, ValueId> transfer_sources_;
};

std::unordered_map<ValueId, VecValue>
make_inputs(const RuntimePlan &plan, const LoadedOperatorSpec &operator_spec) {
    const std::size_t slot_count =
        static_cast<std::size_t>(operator_spec.spec.poly_degree / 2);
    std::unordered_map<ValueId, VecValue> inputs;
    for (ValueId id : plan.external_inputs) {
        const ValueDesc &desc = value_desc(plan, id);
        std::vector<double> slots(slot_count, 0.0);
        const std::size_t active_slots = std::min<std::size_t>(784, slot_count);
        for (std::size_t i = 0; i < active_slots; ++i)
            slots[i] = static_cast<double>((i * 17 + id * 31 + 3) % 256) /
                       255.0;
        VecValue value = desc.kind == ValueKind::Plaintext
            ? make_plain(std::move(slots), desc.context,
                         operator_spec.spec.poly_degree, desc.level,
                         desc.scale_log2, desc.ntt)
            : make_cipher(std::move(slots), desc.context,
                          operator_spec.spec.poly_degree, desc.level,
                          desc.scale_log2, desc.ntt, desc.components);
        inputs.emplace(id, std::move(value));
    }
    return inputs;
}

std::vector<std::optional<std::filesystem::path>>
bundle_paths(const RuntimePlan &plan,
             const std::optional<std::filesystem::path> &path) {
    if (!plan.plaintext_bundle) return {};
    if (!path)
        throw std::runtime_error(
            "RuntimePlan requires a plaintext bundle directory");
    return std::vector<std::optional<std::filesystem::path>>(
        static_cast<std::size_t>(plan.target.world_size), path);
}

HarnessResult run_plan(const RuntimePlan &plan,
                       const LoadedOperatorSpec &operator_spec,
                       const std::optional<std::filesystem::path> &bundle,
                       bool async) {
    MockClusterConfig cluster;
    cluster.world_size = plan.target.world_size;
    const VecExecConfig executor{
        async ? VecExecMode::Async : VecExecMode::Sync, 17, async ? 1 : 0};
    return run_mock_cluster(
        plan, operator_spec, make_inputs(plan, operator_spec), cluster, executor,
        DiffMode::AllValuesAfterRun, false, bundle_paths(plan, bundle));
}

ArtifactIndex index_artifacts(const RuntimePlan &plan,
                              const HarnessResult &result) {
    ArtifactIndex index;
    for (std::size_t rank = 0; rank < result.artifacts.size(); ++rank) {
        for (const auto &[id, artifact] : result.artifacts[rank].values) {
            const ValueDesc &desc = value_desc(plan, id);
            if (artifact.place != desc.place ||
                artifact.place.rank != static_cast<int>(rank))
                throw std::runtime_error(
                    "RunArtifact value is owned by the wrong Place");
            if (!index.emplace(id, IndexedValue{artifact.place, artifact.value})
                     .second)
                throw std::runtime_error(
                    "RunArtifact ValueId appears on more than one rank");
        }
    }
    if (index.size() != plan.values.size())
        throw std::runtime_error(
            "AllValuesAfterRun did not return every RuntimePlan value");
    return index;
}

Comparison compare(const VecPayload &expected, const VecPayload &actual) {
    Comparison result;
    result.metadata_matches = expected.kind == actual.kind &&
                              expected.metadata == actual.metadata;
    if (expected.slots.size() != actual.slots.size()) {
        result.mismatch_slots =
            std::max(expected.slots.size(), actual.slots.size());
        result.max_abs_diff = std::numeric_limits<double>::infinity();
        return result;
    }
    for (std::size_t i = 0; i < expected.slots.size(); ++i) {
        const double expected_value = expected.slots[i];
        const double actual_value = actual.slots[i];
        const bool equal = expected_value == actual_value ||
                           (std::isnan(expected_value) &&
                            std::isnan(actual_value));
        if (equal) continue;
        if (!result.first_mismatch) {
            result.first_mismatch = i;
            result.expected_at_first = expected_value;
            result.actual_at_first = actual_value;
        }
        ++result.mismatch_slots;
        const double difference = std::abs(expected_value - actual_value);
        if (std::isnan(difference))
            result.max_abs_diff = std::numeric_limits<double>::infinity();
        else
            result.max_abs_diff =
                std::max(result.max_abs_diff, difference);
    }
    return result;
}

bool mismatched(const Comparison &comparison) {
    return comparison.mismatch_slots != 0 ||
           !comparison.metadata_matches;
}

void record(std::ostream &report, std::string_view source,
            const Instruction *instruction, ValueId distributed_id,
            ValueId reference_id, const Place &place,
            const Comparison &comparison) {
    report << "source=" << source;
    if (instruction)
        report << " ordinal=" << instruction->ordinal
               << " op=" << instruction_name(*instruction);
    report << " distributed_value=" << distributed_id
           << " reference_value=" << reference_id
           << " place=" << to_string(place)
           << " metadata="
           << (comparison.metadata_matches ? "MATCH" : "MISMATCH")
           << " mismatch_slots=" << comparison.mismatch_slots
           << " max_abs_diff=" << std::setprecision(17)
           << comparison.max_abs_diff;
    if (comparison.first_mismatch)
        report << " first_mismatch_slot=" << *comparison.first_mismatch
               << " expected=" << comparison.expected_at_first
               << " actual=" << comparison.actual_at_first;
    report << '\n';
}

std::string slots_sha256(const VecPayload &payload) {
    const auto *bytes = reinterpret_cast<const char *>(payload.slots.data());
    return sha256_hex(
        std::string_view(bytes, payload.slots.size() * sizeof(double)));
}

Summary diff(const RuntimePlan &reference, const RuntimePlan &distributed,
             const PlanAlignment &alignment,
             const ArtifactIndex &reference_values,
             const ArtifactIndex &distributed_values, std::ostream &report) {
    Summary summary;
    for (const Instruction *instruction : instructions(distributed)) {
        const ValueId distributed_id = instruction_output(*instruction);
        const ValueId reference_id =
            alignment.reference_value(distributed_id);
        const auto &actual = distributed_values.at(distributed_id);
        const Comparison comparison = compare(
            reference_values.at(reference_id).value.materialize(),
            actual.value.materialize());
        record(report, "instruction", instruction, distributed_id,
               reference_id, actual.place, comparison);
        ++summary.compared_instructions;
        if (mismatched(comparison)) ++summary.mismatched_instructions;
        summary.mismatch_slots += comparison.mismatch_slots;
        summary.max_abs_diff =
            std::max(summary.max_abs_diff, comparison.max_abs_diff);
    }
    for (std::size_t i = 0; i < distributed.final_outputs.size(); ++i) {
        const ValueId distributed_id = distributed.final_outputs[i];
        const ValueId reference_id = reference.final_outputs[i];
        const auto &actual = distributed_values.at(distributed_id);
        const VecPayload actual_payload = actual.value.materialize();
        const Comparison comparison = compare(
            reference_values.at(reference_id).value.materialize(),
            actual_payload);
        record(report, "final_output", nullptr, distributed_id, reference_id,
               actual.place, comparison);
        ++summary.compared_final_outputs;
        if (mismatched(comparison)) ++summary.mismatched_final_outputs;
        summary.mismatch_slots += comparison.mismatch_slots;
        summary.max_abs_diff =
            std::max(summary.max_abs_diff, comparison.max_abs_diff);
        summary.final_sha256.push_back(slots_sha256(actual_payload));
    }
    return summary;
}

std::optional<std::filesystem::path> optional_path(const char *argument) {
    if (std::string_view(argument) == "-") return std::nullopt;
    return std::filesystem::path(argument);
}

std::size_t count_transfers(const RuntimePlan &plan) {
    std::size_t count = 0;
    for (const Instruction *instruction : instructions(plan))
        if (std::holds_alternative<CommAction>(instruction->body)) ++count;
    return count;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 7) {
        std::cerr
            << "usage: dacapo_plan_vec_diff REFERENCE_PLAN DISTRIBUTED_PLAN "
               "OPERATOR_SPEC REFERENCE_BUNDLE DISTRIBUTED_BUNDLE REPORT\n"
            << "use - for a plan without a plaintext bundle\n";
        return 2;
    }

    try {
        const LoadedRuntimePlan reference =
            RuntimePlanJsonReader::read_file(argv[1]);
        const LoadedRuntimePlan distributed =
            RuntimePlanJsonReader::read_file(argv[2]);
        const LoadedOperatorSpec operator_spec =
            OperatorSpecReader::read_file(argv[3]);
        PlanAlignment alignment(reference.plan, distributed.plan);
        alignment.verify();

        const HarnessResult reference_result = run_plan(
            reference.plan, operator_spec, optional_path(argv[4]), false);
        const ArtifactIndex reference_values =
            index_artifacts(reference.plan, reference_result);
        const HarnessResult distributed_result = run_plan(
            distributed.plan, operator_spec, optional_path(argv[5]), true);
        const ArtifactIndex distributed_values =
            index_artifacts(distributed.plan, distributed_result);

        const std::filesystem::path report_path = argv[6];
        if (!report_path.parent_path().empty())
            std::filesystem::create_directories(report_path.parent_path());
        std::ofstream report(report_path);
        if (!report) throw std::runtime_error("cannot open diff report");
        const Summary summary =
            diff(reference.plan, distributed.plan, alignment, reference_values,
                 distributed_values, report);

        const bool passed = summary.mismatched_instructions == 0 &&
                            summary.mismatched_final_outputs == 0;
        std::cout << (passed ? "PASS" : "FAIL")
                  << " world_size=" << distributed.plan.target.world_size
                  << " device_counts=";
        for (std::size_t i = 0;
             i < distributed.plan.target.device_counts.size(); ++i) {
            if (i != 0) std::cout << 'x';
            std::cout << distributed.plan.target.device_counts[i];
        }
        std::cout << " instructions=" << summary.compared_instructions
                  << " transfers=" << count_transfers(distributed.plan)
                  << " instruction_mismatches="
                  << summary.mismatched_instructions
                  << " final_outputs=" << summary.compared_final_outputs
                  << " final_mismatches="
                  << summary.mismatched_final_outputs
                  << " mismatch_slots=" << summary.mismatch_slots
                  << " max_abs_diff=" << std::setprecision(17)
                  << summary.max_abs_diff << '\n';
        for (std::size_t i = 0; i < summary.final_sha256.size(); ++i)
            std::cout << "final_output[" << i
                      << "]_sha256=" << summary.final_sha256[i] << '\n';
        std::cout << "diff_report=" << report_path << '\n';
        return passed ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
