#include "runtime/plan.hpp"

#include <sstream>
#include <stdexcept>
#include <tuple>

namespace fhegpu {

bool operator==(const Place &a, const Place &b) {
    return a.kind == b.kind && a.rank == b.rank && a.index == b.index;
}
bool operator!=(const Place &a, const Place &b) { return !(a == b); }
bool operator<(const Place &a, const Place &b) {
    return std::tie(a.kind, a.rank, a.index) < std::tie(b.kind, b.rank, b.index);
}

bool operator==(const KeyRequirement &a, const KeyRequirement &b) {
    return a.kind == b.kind && a.place == b.place && a.rotation_step == b.rotation_step;
}
bool operator<(const KeyRequirement &a, const KeyRequirement &b) {
    return std::tie(a.kind, a.place, a.rotation_step) <
           std::tie(b.kind, b.place, b.rotation_step);
}

std::string to_string(const Place &p) {
    std::ostringstream os;
    os << (p.kind == PlaceKind::Host ? "Host" : "Device") << "(rank=" << p.rank;
    if (p.kind == PlaceKind::Device) os << ",index=" << p.index;
    os << ')';
    return os.str();
}

std::string to_string(ValueKind k) { return k == ValueKind::Plaintext ? "Plaintext" : "Ciphertext"; }

std::string to_string(ComputeKind k) {
    switch (k) {
    case ComputeKind::AddCC: return "AddCC"; case ComputeKind::AddCP: return "AddCP";
    case ComputeKind::SubCC: return "SubCC"; case ComputeKind::SubCP: return "SubCP";
    case ComputeKind::MulCC: return "MulCC"; case ComputeKind::MulCP: return "MulCP";
    case ComputeKind::Negate: return "Negate"; case ComputeKind::Rotate: return "Rotate";
    case ComputeKind::Rescale: return "Rescale"; case ComputeKind::ModSwitch: return "ModSwitch";
    case ComputeKind::Relinearize: return "Relinearize"; case ComputeKind::Boot: return "Boot";
    }
    throw std::runtime_error("unknown compute kind");
}

std::string to_string(CommKind k) {
    switch (k) { case CommKind::Transfer: return "Transfer"; case CommKind::Replicate: return "Replicate"; }
    throw std::runtime_error("unknown communication kind");
}
std::string to_string(CommHint h) {
    switch (h) {
    case CommHint::Auto: return "Auto"; case CommHint::PointToPoint: return "PointToPoint";
    case CommHint::Broadcast: return "Broadcast"; case CommHint::Tree: return "Tree";
    case CommHint::Ring: return "Ring"; case CommHint::HostStaged: return "HostStaged";
    }
    throw std::runtime_error("unknown communication hint");
}
std::string to_string(Phase p) {
    switch (p) { case Phase::Initialization: return "Initialization"; case Phase::Execution: return "Execution"; case Phase::Finalization: return "Finalization"; }
    throw std::runtime_error("unknown phase");
}
std::string to_string(RescaleMode mode) {
    switch (mode) { case RescaleMode::Eager: return "eager"; case RescaleMode::Lazy: return "lazy"; }
    throw std::runtime_error("unknown rescale mode");
}
std::string to_string(BootImplementation implementation) {
    switch (implementation) {
    case BootImplementation::Native: return "native";
    case BootImplementation::DecryptReencrypt: return "decrypt_reencrypt";
    }
    throw std::runtime_error("unknown boot implementation");
}
std::string to_string(RequiredCapability capability) {
    switch (capability) {
    case RequiredCapability::Encode: return "encode";
    case RequiredCapability::Transfer: return "transfer";
    case RequiredCapability::Replicate: return "replicate";
    case RequiredCapability::HostCompute: return "host_compute";
    case RequiredCapability::BootNative: return "boot_native";
    case RequiredCapability::BootDecryptReencrypt: return "boot_decrypt_reencrypt";
    }
    throw std::runtime_error("unknown required capability");
}
std::string to_string(KeyKind kind) {
    switch (kind) { case KeyKind::Secret: return "secret"; case KeyKind::Relin: return "relin"; case KeyKind::Galois: return "galois"; }
    throw std::runtime_error("unknown key kind");
}

void RuntimePlan::print(std::ostream &out) const {
    out << "RuntimePlan(version=" << format_version << ", id=" << plan_id
        << ", target=" << target.target_id << ", capability=" << target.capability_version
        << ", operator_spec=" << target.operator_spec.id << '@' << target.operator_spec.version
        << ", world=" << target.world_size << ")\n";
    if (plaintext_bundle)
        out << "  plaintext_bundle=" << plaintext_bundle->id << '@' << plaintext_bundle->version << '\n';
    for (const auto &v : values) {
        out << "  %" << v.id << " : " << to_string(v.kind) << " @ " << to_string(v.place)
            << " context=" << v.context << " level=" << v.level
            << " scale_log2=" << v.scale_log2 << " ntt=" << v.ntt
            << " components=" << v.components << '\n';
    }
    const auto print_phase = [&](const char *name, const std::vector<Instruction> &list) {
        out << name << ":\n";
        for (const auto &inst : list) {
            out << "  #" << inst.ordinal << ' ';
            if (const auto *encode = std::get_if<EncodeOp>(&inst.body)) {
                out << '%' << encode->output << " = Encode(";
                if (const auto *bundle = std::get_if<BundleEncodePayload>(&encode->payload)) out << bundle->content;
                else out << "inline";
                out << ")\n";
            } else if (const auto *op = std::get_if<ComputeOp>(&inst.body)) {
                out << '%' << op->output << " = " << to_string(op->kind) << '(';
                for (std::size_t i = 0; i < op->inputs.size(); ++i) { if (i) out << ','; out << '%' << op->inputs[i]; }
                out << ") @ " << to_string(op->place) << '\n';
            } else {
                const auto &a = std::get<CommAction>(inst.body);
                out << to_string(a.kind) << " transfer=" << a.id << " hint=" << to_string(a.hint) << " outputs=";
                for (std::size_t i = 0; i < a.outputs.size(); ++i) out << (i ? "," : "") << '%' << a.outputs[i] << '@' << to_string(a.destinations[i]);
                out << '\n';
            }
        }
    };
    print_phase("Initialization", initialization);
    print_phase("Execution", execution);
    print_phase("Finalization", finalization);
}

} // namespace fhegpu
