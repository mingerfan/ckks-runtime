#include "runtime/verifier.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace fhegpu {
namespace {

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error("plan verification failed: " + message);
}

using DescMap = std::unordered_map<ValueId, const ValueDesc *>;

void check_place(const RuntimePlan &plan, const Place &place) {
    if (place.rank < 0 || place.rank >= plan.target.world_size)
        fail("place rank is outside target: " + to_string(place));
    if (place.kind == PlaceKind::Host) {
        if (place.index != 0) fail("Host index must be zero: " + to_string(place));
    } else if (place.kind == PlaceKind::Device) {
        if (place.index < 0 || place.index >= plan.target.device_counts.at(static_cast<std::size_t>(place.rank)))
            fail("device index is outside target: " + to_string(place));
    } else {
        fail("unknown Place kind");
    }
}

const ValueDesc &lookup(const DescMap &descs, ValueId id, std::string_view where) {
    const auto found = descs.find(id);
    if (found == descs.end()) fail(std::string(where) + " has no ValueDesc: " + std::to_string(id));
    return *found->second;
}

bool same_metadata(const ValueDesc &a, const ValueDesc &b) {
    return a.kind == b.kind && a.context == b.context && a.level == b.level &&
           a.scale_log2 == b.scale_log2 && a.ntt == b.ntt && a.components == b.components;
}

void require_same_base(const ValueDesc &input, const ValueDesc &output, const std::string &op) {
    if (input.context != output.context || input.ntt != output.ntt)
        fail(op + " changes context or NTT state");
}

std::vector<ValueKind> expected_inputs(ComputeKind kind) {
    using V = ValueKind;
    switch (kind) {
    case ComputeKind::AddCC: case ComputeKind::SubCC: case ComputeKind::MulCC:
        return {V::Ciphertext, V::Ciphertext};
    case ComputeKind::AddCP: case ComputeKind::SubCP: case ComputeKind::MulCP:
        return {V::Ciphertext, V::Plaintext};
    case ComputeKind::Negate: case ComputeKind::Rotate: case ComputeKind::Rescale:
    case ComputeKind::ModSwitch: case ComputeKind::Relinearize: case ComputeKind::Boot:
        return {V::Ciphertext};
    }
    fail("unknown compute operation");
}

void verify_compute_metadata(const ComputeOp &op, const DescMap &descs) {
    const auto types = expected_inputs(op.kind);
    if (op.inputs.size() != types.size()) fail(to_string(op.kind) + " has wrong input count");
    std::vector<const ValueDesc *> inputs;
    for (std::size_t i = 0; i < op.inputs.size(); ++i) {
        const auto &input = lookup(descs, op.inputs[i], "compute input");
        if (input.kind != types[i]) fail(to_string(op.kind) + " has wrong input kind");
        if (input.place != op.place) fail("implicit cross-Place compute operand " + std::to_string(input.id));
        inputs.push_back(&input);
    }
    const auto &output = lookup(descs, op.output, "compute output");
    if (output.kind != ValueKind::Ciphertext) fail("compute output must be ciphertext");
    if (output.place != op.place) fail("compute output Place mismatch");
    require_same_base(*inputs[0], output, to_string(op.kind));
    if (inputs.size() == 2) {
        if (inputs[0]->context != inputs[1]->context || inputs[0]->ntt != inputs[1]->ntt ||
            inputs[0]->level != inputs[1]->level)
            fail(to_string(op.kind) + " inputs have incompatible metadata");
    }

    const bool no_attrs = std::holds_alternative<std::monostate>(op.attrs);
    switch (op.kind) {
    case ComputeKind::AddCC: case ComputeKind::SubCC:
        if (!no_attrs) fail(to_string(op.kind) + " does not accept attrs");
        if (inputs[0]->scale_log2 != inputs[1]->scale_log2 || inputs[0]->components != inputs[1]->components ||
            !same_metadata(*inputs[0], output)) fail(to_string(op.kind) + " metadata rule failed");
        break;
    case ComputeKind::AddCP: case ComputeKind::SubCP:
        if (!no_attrs) fail(to_string(op.kind) + " does not accept attrs");
        if (inputs[0]->scale_log2 != inputs[1]->scale_log2 ||
            output.level != inputs[0]->level || output.scale_log2 != inputs[0]->scale_log2 ||
            output.components != inputs[0]->components)
            fail(to_string(op.kind) + " metadata rule failed");
        break;
    case ComputeKind::MulCC:
        if (!no_attrs) fail("MulCC does not accept attrs");
        if (output.level != inputs[0]->level ||
            static_cast<long long>(output.scale_log2) != static_cast<long long>(inputs[0]->scale_log2) + inputs[1]->scale_log2 ||
            static_cast<long long>(output.components) != static_cast<long long>(inputs[0]->components) + inputs[1]->components - 1)
            fail("MulCC metadata rule failed");
        break;
    case ComputeKind::MulCP:
        if (!no_attrs) fail("MulCP does not accept attrs");
        if (output.level != inputs[0]->level ||
            static_cast<long long>(output.scale_log2) != static_cast<long long>(inputs[0]->scale_log2) + inputs[1]->scale_log2 ||
            output.components != inputs[0]->components)
            fail("MulCP metadata rule failed");
        break;
    case ComputeKind::Negate:
        if (!no_attrs || !same_metadata(*inputs[0], output)) fail("Negate metadata rule failed");
        break;
    case ComputeKind::Rotate: {
        const auto *attrs = std::get_if<RotateAttrs>(&op.attrs);
        if (!attrs || attrs->steps == 0 || inputs[0]->components != 2 || !same_metadata(*inputs[0], output))
            fail("Rotate metadata rule failed");
        break;
    }
    case ComputeKind::Rescale: {
        const auto *attrs = std::get_if<RescaleAttrs>(&op.attrs);
        if (!attrs || attrs->target_level >= inputs[0]->level || output.level != attrs->target_level ||
            output.scale_log2 != attrs->target_scale_log2 || output.components != inputs[0]->components)
            fail("Rescale metadata rule failed");
        break;
    }
    case ComputeKind::ModSwitch: {
        const auto *attrs = std::get_if<ModSwitchAttrs>(&op.attrs);
        if (!attrs || attrs->target_level >= inputs[0]->level || output.level != attrs->target_level ||
            output.scale_log2 != inputs[0]->scale_log2 || output.components != inputs[0]->components)
            fail("ModSwitch metadata rule failed");
        break;
    }
    case ComputeKind::Relinearize:
        if (!no_attrs || inputs[0]->components != 3 || output.level != inputs[0]->level ||
            output.scale_log2 != inputs[0]->scale_log2 || output.components != 2)
            fail("Relinearize metadata rule failed");
        break;
    case ComputeKind::Boot: {
        const auto *attrs = std::get_if<BootAttrs>(&op.attrs);
        if (!attrs || attrs->operator_profile.empty() || output.level != attrs->target_level ||
            output.scale_log2 != attrs->target_scale_log2 || output.components != attrs->target_components)
            fail("Boot metadata rule failed");
        if (attrs->implementation == BootImplementation::DecryptReencrypt && op.place.kind != PlaceKind::Host)
            fail("decrypt_reencrypt Boot must execute on Host");
        break;
    }
    }
}

const BootProfile &find_boot_profile(const OperatorSpec &spec, const std::string &id) {
    for (const auto &profile : spec.boot_profiles) if (profile.profile_id == id) return profile;
    fail("Boot references unknown operator profile: " + id);
}

int normalized_rotation_step(int steps, std::uint64_t slot_count) {
    if (slot_count == 0 || slot_count > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        fail("OperatorSpec slot count is outside supported rotation range");
    const long long modulus = static_cast<long long>(slot_count);
    long long normalized = static_cast<long long>(steps) % modulus;
    if (normalized < 0) normalized += modulus;
    if (normalized == 0) fail("Rotate step becomes zero after slot-count normalization");
    return static_cast<int>(normalized);
}

} // namespace

PlanRequirements PlanVerifier::verify(const RuntimePlan &plan,
                                      const LoadedOperatorSpec &loaded_spec,
                                      bool skip_artifact_digest_checks) {
    const auto &spec = loaded_spec.spec;
    if (plan.format_version != 1) fail("unsupported format version");
    if (spec.format_version != 1) fail("unsupported OperatorSpec format version");
    if (plan.target.target_id.empty()) fail("target id is empty");
    if (plan.target.capability_version <= 0) fail("capability version must be positive");
    if (plan.target.world_size <= 0) fail("world size must be positive");
    if (plan.target.device_counts.size() != static_cast<std::size_t>(plan.target.world_size))
        fail("device count table does not match world size");
    for (int count : plan.target.device_counts) if (count < 0) fail("negative device count");
    if (plan.target.operator_spec.id != spec.id || plan.target.operator_spec.version != spec.version)
        fail("OperatorSpec id/version mismatch");
    if (!skip_artifact_digest_checks && plan.target.operator_spec.source_sha256 != loaded_spec.source_sha256)
        fail("OperatorSpec source SHA-256 mismatch");
    if (plan.target.target_id != spec.target_id) fail("target_id does not match OperatorSpec");

    DescMap descs;
    for (const auto &value : plan.values) {
        check_place(plan, value.place);
        if (!descs.emplace(value.id, &value).second) fail("duplicate ValueId " + std::to_string(value.id));
        if (value.context != spec.context_id) fail("value context does not match OperatorSpec");
        if (value.level < spec.level_lower_bound || value.level > spec.level_upper_bound)
            fail("value level is outside OperatorSpec range");
        long long modulus_budget = 0;
        for (int level = 0; level <= value.level; ++level) modulus_budget += spec.rns_moduli_log2.at(static_cast<std::size_t>(level));
        if (value.scale_log2 >= modulus_budget) fail("value scale exceeds modulus budget");
        if (value.kind == ValueKind::Plaintext && value.components != 1)
            fail("plaintext components must be one");
        if (value.kind == ValueKind::Ciphertext && value.components < 2)
            fail("ciphertext components must be at least two");
    }

    std::unordered_set<ValueId> defined;
    std::unordered_set<ValueId> mentioned;
    for (ValueId id : plan.external_inputs) {
        const auto &value = lookup(descs, id, "external input");
        mentioned.insert(id);
        if (value.place.kind != PlaceKind::Host) fail("external input must be placed on Host (IO-2)");
        if (!defined.insert(id).second) fail("duplicate external input " + std::to_string(id));
    }

    std::set<RequiredCapability> capabilities;
    std::set<KeyRequirement> keys;
    std::unordered_set<TransferId> transfers;
    bool has_bundle_encode = false;
    std::size_t expected_ordinal = 0;

    const auto check_list = [&](const std::vector<Instruction> &list, Phase phase) {
        for (const auto &instruction : list) {
            if (instruction.ordinal != expected_ordinal++) fail("instruction ordinals must be contiguous and stable");
            if (const auto *encode = std::get_if<EncodeOp>(&instruction.body)) {
                if (phase != Phase::Initialization) fail("Encode is only allowed in initialization");
                const auto &output = lookup(descs, encode->output, "Encode output");
                mentioned.insert(output.id);
                if (output.kind != ValueKind::Plaintext || output.place.kind != PlaceKind::Host || output.components != 1)
                    fail("Encode output must be Host plaintext");
                if (!defined.insert(output.id).second) fail("duplicate definition of ValueId " + std::to_string(output.id));
                capabilities.insert(RequiredCapability::Encode);
                if (const auto *inline_payload = std::get_if<InlineEncodePayload>(&encode->payload)) {
                    if (inline_payload->values.empty()) fail("inline Encode payload is empty");
                    if (inline_payload->values.size() > spec.poly_degree / 2) fail("inline Encode exceeds CKKS slot capacity");
                } else if (std::holds_alternative<BundleEncodePayload>(encode->payload)) {
                    has_bundle_encode = true;
                } else {
                    fail("unknown Encode payload");
                }
            } else if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) {
                check_place(plan, op->place);
                for (ValueId id : op->inputs) {
                    mentioned.insert(id);
                    if (!defined.count(id)) fail("undefined or use-before-definition ValueId " + std::to_string(id));
                }
                mentioned.insert(op->output);
                if (!defined.insert(op->output).second) fail("duplicate definition of ValueId " + std::to_string(op->output));
                verify_compute_metadata(*op, descs);
                const auto support = spec.operators.find(op->kind);
                if (support == spec.operators.end() || !support->second.supported)
                    fail(to_string(op->kind) + " is unsupported by OperatorSpec");
                if (op->place.kind == PlaceKind::Host) capabilities.insert(RequiredCapability::HostCompute);
                if (op->kind == ComputeKind::Rotate) {
                    const int step = normalized_rotation_step(std::get<RotateAttrs>(op->attrs).steps, spec.poly_degree / 2);
                    keys.insert(KeyRequirement{KeyKind::Galois, op->place, step});
                } else if (op->kind == ComputeKind::Relinearize) {
                    keys.insert(KeyRequirement{KeyKind::Relin, op->place, std::nullopt});
                } else if (op->kind == ComputeKind::Rescale) {
                    const auto attrs = std::get<RescaleAttrs>(op->attrs);
                    const auto &input = lookup(descs, op->inputs[0], "Rescale input");
                    if (!support->second.max_levels_per_op || input.level - attrs.target_level > *support->second.max_levels_per_op)
                        fail("Rescale level drop exceeds OperatorSpec max_levels_per_op");
                } else if (op->kind == ComputeKind::Boot) {
                    const auto attrs = std::get<BootAttrs>(op->attrs);
                    const auto &profile = find_boot_profile(spec, attrs.operator_profile);
                    const auto &input = lookup(descs, op->inputs[0], "Boot input");
                    if (profile.implementation != attrs.implementation || input.level < profile.input_level_min ||
                        input.level > profile.input_level_max || input.components != profile.input_components ||
                        attrs.target_level != profile.output_level || attrs.target_scale_log2 != profile.output_scale_log2 ||
                        attrs.target_components != profile.output_components)
                        fail("Boot instruction does not match OperatorSpec profile");
                    if (attrs.implementation == BootImplementation::Native)
                        capabilities.insert(RequiredCapability::BootNative);
                    else {
                        capabilities.insert(RequiredCapability::BootDecryptReencrypt);
                        keys.insert(KeyRequirement{KeyKind::Secret, op->place, std::nullopt});
                    }
                }
            } else if (const auto *action = std::get_if<CommAction>(&instruction.body)) {
                if (!transfers.insert(action->id).second) fail("duplicate TransferId " + std::to_string(action->id));
                if (action->kind == CommKind::Transfer) capabilities.insert(RequiredCapability::Transfer);
                else if (action->kind == CommKind::Replicate) capabilities.insert(RequiredCapability::Replicate);
                else fail("unknown communication kind");
                switch (action->hint) {
                case CommHint::Auto: case CommHint::PointToPoint: case CommHint::Broadcast:
                case CommHint::Tree: case CommHint::Ring: case CommHint::HostStaged: break;
                default: fail("unknown communication hint");
                }
                if (action->inputs.size() != 1 || action->sources.size() != 1)
                    fail(to_string(action->kind) + " requires one input and source");
                mentioned.insert(action->inputs[0]);
                if (!defined.count(action->inputs[0])) fail("communication source is undefined or used before definition");
                const auto &source = lookup(descs, action->inputs[0], "communication source");
                check_place(plan, action->sources[0]);
                if (source.place != action->sources[0]) fail("communication source Place mismatch");
                const std::size_t count = action->outputs.size();
                if (action->destinations.size() != count || action->output_types.size() != count)
                    fail("communication outputs/destinations/types mapping mismatch");
                if (action->kind == CommKind::Transfer && count != 1) fail("Transfer requires exactly one output");
                if (action->kind == CommKind::Replicate && count < 2) fail("Replicate requires at least two outputs");
                std::set<Place> destinations;
                for (std::size_t i = 0; i < count; ++i) {
                    check_place(plan, action->destinations[i]);
                    if (action->destinations[i] == source.place) fail("communication destination equals source Place");
                    if (!destinations.insert(action->destinations[i]).second) fail("duplicate communication destination");
                    const auto &output = lookup(descs, action->outputs[i], "communication output");
                    mentioned.insert(output.id);
                    if (output.place != action->destinations[i]) fail("communication output/destination mapping mismatch");
                    if (action->output_types[i] != source.kind || !same_metadata(source, output))
                        fail("communication changes kind or CKKS metadata");
                    if (!defined.insert(output.id).second) fail("duplicate definition of communication output");
                }
            } else {
                fail("unknown instruction body");
            }
        }
    };
    check_list(plan.initialization, Phase::Initialization);
    check_list(plan.execution, Phase::Execution);
    check_list(plan.finalization, Phase::Finalization);

    if (has_bundle_encode != plan.plaintext_bundle.has_value())
        fail("plaintext_bundle must be present exactly when bundle Encode is used");
    if (plan.final_outputs.empty()) fail("final_outputs must not be empty");
    std::unordered_set<ValueId> final_ids;
    for (ValueId id : plan.final_outputs) {
        lookup(descs, id, "final output");
        mentioned.insert(id);
        if (!defined.count(id)) fail("final output is undefined");
        if (!final_ids.insert(id).second) fail("duplicate final output");
    }
    if (mentioned.size() != descs.size()) fail("values contains an unused ValueDesc");
    for (const auto &entry : descs) if (!mentioned.count(entry.first)) fail("values contains an unused ValueDesc");

    PlanRequirements result;
    result.capabilities.assign(capabilities.begin(), capabilities.end());
    result.keys.assign(keys.begin(), keys.end());
    return result;
}

void PlanVerifier::verify_runtime_target(const RuntimePlan &plan, int rank,
                                         int world_size, int local_devices) {
    if (world_size != plan.target.world_size) fail("runtime world size does not match target");
    if (rank < 0 || rank >= world_size) fail("local rank is outside runtime world");
    if (local_devices != plan.target.device_counts.at(static_cast<std::size_t>(rank)))
        fail("local device count does not match target");
}

} // namespace fhegpu
