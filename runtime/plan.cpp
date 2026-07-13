#include "runtime/plan.hpp"

#include <iomanip>
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
    switch (h) { case CommHint::Auto: return "Auto"; case CommHint::PointToPoint: return "PointToPoint";
    case CommHint::Broadcast: return "Broadcast"; case CommHint::Tree: return "Tree"; case CommHint::Ring: return "Ring"; case CommHint::HostStaged: return "HostStaged"; }
    throw std::runtime_error("unknown communication hint");
}
std::string to_string(Phase p) {
    switch (p) { case Phase::Initialization: return "Initialization"; case Phase::Execution: return "Execution"; case Phase::Finalization: return "Finalization"; }
    throw std::runtime_error("unknown phase");
}

static void append_place(std::ostringstream &os, const Place &p) {
    os << static_cast<int>(p.kind) << ':' << p.rank << ':' << p.index;
}

static void append_instruction(std::ostringstream &os, const Instruction &inst) {
    os << '#' << inst.ordinal << ':';
    if (const auto *op = std::get_if<ComputeOp>(&inst.body)) {
        os << 'C' << static_cast<int>(op->kind) << ':' << op->output << ':';
        append_place(os, op->place);
        for (auto id : op->inputs) os << ':' << id;
        if (const auto *a = std::get_if<RotateAttrs>(&op->attrs)) os << ":r" << a->steps;
        if (const auto *a = std::get_if<RescaleAttrs>(&op->attrs)) os << ":s" << a->target_level << ':' << std::setprecision(17) << a->scale_divisor;
        if (const auto *a = std::get_if<ModSwitchAttrs>(&op->attrs)) os << ":m" << a->target_level;
        if (const auto *a = std::get_if<BootAttrs>(&op->attrs)) os << ":b" << a->level << ':' << std::setprecision(17) << a->scale << ':' << a->components;
    } else {
        const auto &a = std::get<CommAction>(inst.body);
        os << 'M' << a.id << ':' << static_cast<int>(a.kind) << ':' << static_cast<int>(a.hint);
        for (auto id : a.inputs) os << ":i" << id;
        for (auto id : a.outputs) os << ":o" << id;
        for (const auto &p : a.sources) { os << ":s"; append_place(os, p); }
        for (const auto &p : a.destinations) { os << ":d"; append_place(os, p); }
        for (auto t : a.output_types) os << ":t" << static_cast<int>(t);
    }
    os << ';';
}

std::uint64_t RuntimePlan::fingerprint() const {
    std::ostringstream os;
    os << format_version << ':' << plan_id << ':' << target.target_id << ':' << target.world_size;
    for (int n : target.device_counts) os << ':' << n;
    for (const auto &v : values) { os << "|v" << v.id << ':' << static_cast<int>(v.kind) << ':'; append_place(os, v.place); }
    for (auto id : external_inputs) os << "|e" << id;
    for (const auto &i : initialization) { os << "|I"; append_instruction(os, i); }
    for (const auto &i : execution) { os << "|E"; append_instruction(os, i); }
    for (const auto &i : finalization) { os << "|F"; append_instruction(os, i); }
    for (auto id : final_outputs) os << "|o" << id;
    const std::string bytes = os.str();
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : bytes) { hash ^= c; hash *= 1099511628211ULL; }
    return hash;
}

void RuntimePlan::print(std::ostream &out) const {
    out << "RuntimePlan(version=" << format_version << ", id=" << plan_id
        << ", target=" << target.target_id << ", world=" << target.world_size
        << ", fingerprint=0x" << std::hex << fingerprint() << std::dec << ")\n";
    for (const auto &v : values) out << "  %" << v.id << " : " << to_string(v.kind) << " @ " << to_string(v.place) << '\n';
    const auto print_phase = [&](const char *name, const std::vector<Instruction> &list) {
        out << name << ":\n";
        for (const auto &inst : list) {
            out << "  #" << inst.ordinal << ' ';
            if (const auto *op = std::get_if<ComputeOp>(&inst.body)) {
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
