#include "runtime/verifier.hpp"

#include <cmath>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace fhegpu {
namespace {

[[noreturn]] void fail(const std::string &message) { throw std::runtime_error("plan verification failed: " + message); }

struct DescMap {
    std::unordered_map<ValueId, const ValueDesc *> values;
};

void check_place(const RuntimePlan &plan, const Place &p) {
    if (p.rank < 0 || p.rank >= plan.target.world_size) fail("place rank is outside target: " + to_string(p));
    if (p.kind == PlaceKind::Host) {
        if (p.index != 0) fail("Host index must be zero: " + to_string(p));
    } else {
        if (p.index < 0 || p.index >= plan.target.device_counts.at(static_cast<std::size_t>(p.rank)))
            fail("device index is outside target: " + to_string(p));
    }
}

std::vector<ValueKind> expected_inputs(ComputeKind kind) {
    using V = ValueKind;
    switch (kind) {
    case ComputeKind::AddCC: case ComputeKind::SubCC: case ComputeKind::MulCC: return {V::Ciphertext, V::Ciphertext};
    case ComputeKind::AddCP: case ComputeKind::SubCP: case ComputeKind::MulCP: return {V::Ciphertext, V::Plaintext};
    default: return {V::Ciphertext};
    }
}

void check_attrs(const ComputeOp &op) {
    const bool none = std::holds_alternative<std::monostate>(op.attrs);
    switch (op.kind) {
    case ComputeKind::Rotate:
        if (!std::holds_alternative<RotateAttrs>(op.attrs)) fail("Rotate requires RotateAttrs");
        return;
    case ComputeKind::Rescale: {
        const auto *a = std::get_if<RescaleAttrs>(&op.attrs);
        if (!a || a->target_level < 0 || !std::isfinite(a->scale_divisor) || a->scale_divisor <= 0.0) fail("invalid RescaleAttrs");
        return;
    }
    case ComputeKind::ModSwitch: {
        const auto *a = std::get_if<ModSwitchAttrs>(&op.attrs);
        if (!a || a->target_level < 0) fail("invalid ModSwitchAttrs");
        return;
    }
    case ComputeKind::Boot: {
        const auto *a = std::get_if<BootAttrs>(&op.attrs);
        if (!a || a->level < 0 || !std::isfinite(a->scale) || a->scale <= 0.0 || a->components < 2) fail("invalid BootAttrs");
        return;
    }
    default:
        if (!none) fail(to_string(op.kind) + " does not accept attributes");
    }
}

} // namespace

void PlanVerifier::verify(const RuntimePlan &plan) {
    if (plan.format_version != 1) fail("unsupported format version");
    if (plan.plan_id == 0) fail("plan id must be nonzero");
    if (plan.target.target_id.empty()) fail("target id is empty");
    if (plan.target.world_size <= 0) fail("world size must be positive");
    if (plan.target.device_counts.size() != static_cast<std::size_t>(plan.target.world_size)) fail("device count table does not match world size");
    for (int n : plan.target.device_counts) if (n < 0) fail("negative device count");

    DescMap desc;
    for (const auto &v : plan.values) {
        check_place(plan, v.place);
        if (!desc.values.emplace(v.id, &v).second) fail("duplicate ValueId " + std::to_string(v.id));
    }

    std::unordered_set<ValueId> defined;
    std::unordered_set<ValueId> external;
    for (ValueId id : plan.external_inputs) {
        if (!desc.values.count(id)) fail("external input has no value descriptor: " + std::to_string(id));
        if (!external.insert(id).second) fail("duplicate external input " + std::to_string(id));
        defined.insert(id);
    }

    std::unordered_set<TransferId> transfers;
    std::size_t expected_ordinal = 0;
    const auto check_list = [&](const std::vector<Instruction> &list) {
        for (const auto &inst : list) {
            if (inst.ordinal != expected_ordinal++) fail("instruction ordinals must be contiguous and stable");
            if (const auto *op = std::get_if<ComputeOp>(&inst.body)) {
                check_place(plan, op->place);
                if (op->place.kind != PlaceKind::Device) fail("compute must execute on Device");
                const auto types = expected_inputs(op->kind);
                if (op->inputs.size() != types.size()) fail(to_string(op->kind) + " has wrong input count");
                for (std::size_t i = 0; i < op->inputs.size(); ++i) {
                    const ValueId id = op->inputs[i];
                    if (!defined.count(id)) fail("undefined or use-before-definition ValueId " + std::to_string(id));
                    const auto &d = *desc.values.at(id);
                    if (d.kind != types[i]) fail(to_string(op->kind) + " has wrong input type");
                    if (d.place != op->place) fail("implicit cross-Place compute operand " + std::to_string(id));
                }
                auto out = desc.values.find(op->output);
                if (out == desc.values.end()) fail("compute output has no descriptor");
                if (out->second->kind != ValueKind::Ciphertext) fail("compute output must be Ciphertext");
                if (out->second->place != op->place) fail("compute output is not at compute Place");
                if (!defined.insert(op->output).second) fail("duplicate definition of ValueId " + std::to_string(op->output));
                check_attrs(*op);
            } else {
                const auto &a = std::get<CommAction>(inst.body);
                if (!transfers.insert(a.id).second) fail("duplicate TransferId " + std::to_string(a.id));
                if (a.kind != CommKind::Transfer && a.kind != CommKind::Replicate) fail("communication kind is not implemented");
                if (a.hint != CommHint::Auto && a.hint != CommHint::PointToPoint && a.hint != CommHint::Broadcast) fail("communication hint is not implemented");
                if (a.kind == CommKind::Transfer && a.hint == CommHint::Broadcast) fail("Broadcast hint is only valid for Replicate");
                if (a.inputs.size() != 1 || a.sources.size() != 1) fail(to_string(a.kind) + " requires one input and source");
                if (!defined.count(a.inputs[0])) fail("communication source is undefined or used before definition");
                const auto &source_desc = *desc.values.at(a.inputs[0]);
                check_place(plan, a.sources[0]);
                if (source_desc.place != a.sources[0]) fail("communication source Place mismatch");
                const std::size_t n = a.outputs.size();
                if (n == 0 || a.destinations.size() != n || a.output_types.size() != n) fail("communication outputs/destinations/types mapping mismatch");
                if (a.kind == CommKind::Transfer && n != 1) fail("Transfer requires exactly one output");
                std::set<Place> destinations;
                for (std::size_t i = 0; i < n; ++i) {
                    check_place(plan, a.destinations[i]);
                    if (!destinations.insert(a.destinations[i]).second) fail("duplicate communication destination");
                    auto out = desc.values.find(a.outputs[i]);
                    if (out == desc.values.end()) fail("communication output has no descriptor");
                    if (out->second->place != a.destinations[i]) fail("communication output/destination index mapping mismatch");
                    if (out->second->kind != a.output_types[i] || a.output_types[i] != source_desc.kind) fail("Transfer/Replicate changes value type");
                    if (!defined.insert(a.outputs[i]).second) fail("duplicate definition of communication output");
                }
            }
        }
    };
    check_list(plan.initialization);
    check_list(plan.execution);
    check_list(plan.finalization);
    for (ValueId id : plan.final_outputs) if (!defined.count(id)) fail("final output is undefined");
}

void PlanVerifier::verify_runtime_target(const RuntimePlan &plan, int rank, int world_size, int local_devices) {
    if (world_size != plan.target.world_size) fail("runtime world size does not match target");
    if (rank < 0 || rank >= world_size) fail("local rank is outside runtime world");
    if (local_devices != plan.target.device_counts.at(static_cast<std::size_t>(rank))) fail("local device count does not match target");
}

} // namespace fhegpu
