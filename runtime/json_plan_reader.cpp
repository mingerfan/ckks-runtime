#include "runtime/json_plan_reader.hpp"
#include "runtime/json_utils.hpp"

#include <istream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace fhegpu {
namespace {

using namespace json_utils;
constexpr const char *doc = "runtime plan";

Place read_place(const Json &value, const std::string &path) {
    if (!value.is_object()) fail(doc, path, "expected object");
    if (!value.contains("kind")) fail(doc, path, "missing required field 'kind'");
    const std::string kind = read_string(value.at("kind"), doc, path + ".kind");
    if (kind == "host") {
        require_members(value, doc, path, {"kind", "rank"});
        return {PlaceKind::Host, read_nonnegative_int(value.at("rank"), doc, path + ".rank"), 0};
    }
    if (kind == "device") {
        require_members(value, doc, path, {"kind", "rank", "index"});
        return {PlaceKind::Device,
                read_nonnegative_int(value.at("rank"), doc, path + ".rank"),
                read_nonnegative_int(value.at("index"), doc, path + ".index")};
    }
    fail(doc, path + ".kind", "unknown place kind '" + kind + "'");
}

ValueKind read_value_kind(const Json &value, const std::string &path) {
    const std::string kind = read_string(value, doc, path);
    if (kind == "plaintext") return ValueKind::Plaintext;
    if (kind == "ciphertext") return ValueKind::Ciphertext;
    fail(doc, path, "unknown value kind '" + kind + "'");
}

BootImplementation read_boot_implementation(const Json &value, const std::string &path) {
    const std::string implementation = read_string(value, doc, path);
    if (implementation == "native") return BootImplementation::Native;
    if (implementation == "decrypt_reencrypt") return BootImplementation::DecryptReencrypt;
    fail(doc, path, "unknown boot implementation '" + implementation + "'");
}

ComputeKind read_compute_kind(const Json &value, const std::string &path) {
    const std::string kind = read_string(value, doc, path);
    if (kind == "add_cc") return ComputeKind::AddCC;
    if (kind == "add_cp") return ComputeKind::AddCP;
    if (kind == "sub_cc") return ComputeKind::SubCC;
    if (kind == "sub_cp") return ComputeKind::SubCP;
    if (kind == "mul_cc") return ComputeKind::MulCC;
    if (kind == "mul_cp") return ComputeKind::MulCP;
    if (kind == "negate") return ComputeKind::Negate;
    if (kind == "rotate") return ComputeKind::Rotate;
    if (kind == "rescale") return ComputeKind::Rescale;
    if (kind == "mod_switch") return ComputeKind::ModSwitch;
    if (kind == "relinearize") return ComputeKind::Relinearize;
    if (kind == "boot") return ComputeKind::Boot;
    fail(doc, path, "unknown compute operation '" + kind + "'");
}

CommHint read_comm_hint(const Json &value, const std::string &path) {
    const std::string hint = read_string(value, doc, path);
    if (hint == "auto") return CommHint::Auto;
    if (hint == "point_to_point") return CommHint::PointToPoint;
    if (hint == "broadcast") return CommHint::Broadcast;
    if (hint == "tree") return CommHint::Tree;
    if (hint == "ring") return CommHint::Ring;
    if (hint == "host_staged") return CommHint::HostStaged;
    fail(doc, path, "unknown communication hint '" + hint + "'");
}

OperatorSpecRef read_operator_spec(const Json &value, const std::string &path) {
    require_members(value, doc, path, {"id", "version", "source_sha256"});
    return {read_string(value.at("id"), doc, path + ".id"),
            read_positive_int(value.at("version"), doc, path + ".version"),
            read_sha256(value.at("source_sha256"), doc, path + ".source_sha256")};
}

PlaintextBundleRef read_bundle_ref(const Json &value, const std::string &path) {
    require_members(value, doc, path, {"id", "version", "manifest_sha256"});
    return {read_string(value.at("id"), doc, path + ".id"),
            read_positive_int(value.at("version"), doc, path + ".version"),
            read_sha256(value.at("manifest_sha256"), doc, path + ".manifest_sha256")};
}

TargetConfig read_target(const Json &value, const std::string &path) {
    require_members(value, doc, path,
                    {"target_id", "capability_version", "operator_spec",
                     "world_size", "device_counts"});
    TargetConfig target;
    target.target_id = read_string(value.at("target_id"), doc, path + ".target_id");
    target.capability_version = read_positive_int(value.at("capability_version"), doc, path + ".capability_version");
    target.operator_spec = read_operator_spec(value.at("operator_spec"), path + ".operator_spec");
    target.world_size = read_positive_int(value.at("world_size"), doc, path + ".world_size");
    const auto &counts = value.at("device_counts");
    if (!counts.is_array()) fail(doc, path + ".device_counts", "expected array");
    for (std::size_t i = 0; i < counts.size(); ++i)
        target.device_counts.push_back(read_nonnegative_int(counts[i], doc, item_path(path + ".device_counts", i)));
    if (target.device_counts.size() != static_cast<std::size_t>(target.world_size))
        fail(doc, path + ".device_counts", "length must equal world_size");
    return target;
}

ValueDesc read_value_desc(const Json &value, const std::string &path) {
    require_members(value, doc, path,
                    {"id", "kind", "place", "context", "level", "scale_log2", "ntt", "components"});
    return {read_id(value.at("id"), doc, path + ".id"),
            read_value_kind(value.at("kind"), path + ".kind"),
            read_place(value.at("place"), path + ".place"),
            read_string(value.at("context"), doc, path + ".context"),
            read_nonnegative_int(value.at("level"), doc, path + ".level"),
            read_nonnegative_int(value.at("scale_log2"), doc, path + ".scale_log2"),
            read_bool(value.at("ntt"), doc, path + ".ntt"),
            read_positive_int(value.at("components"), doc, path + ".components")};
}

std::vector<ValueId> read_ids(const Json &value, const std::string &path) {
    if (!value.is_array()) fail(doc, path, "expected array");
    std::vector<ValueId> result;
    for (std::size_t i = 0; i < value.size(); ++i)
        result.push_back(read_id(value[i], doc, item_path(path, i)));
    return result;
}

std::vector<Place> read_places(const Json &value, const std::string &path) {
    if (!value.is_array()) fail(doc, path, "expected array");
    std::vector<Place> result;
    for (std::size_t i = 0; i < value.size(); ++i)
        result.push_back(read_place(value[i], item_path(path, i)));
    return result;
}

std::vector<ValueKind> read_value_kinds(const Json &value, const std::string &path) {
    if (!value.is_array()) fail(doc, path, "expected array");
    std::vector<ValueKind> result;
    for (std::size_t i = 0; i < value.size(); ++i)
        result.push_back(read_value_kind(value[i], item_path(path, i)));
    return result;
}

bool compute_requires_attrs(ComputeKind kind) {
    return kind == ComputeKind::Rotate || kind == ComputeKind::Rescale ||
           kind == ComputeKind::ModSwitch || kind == ComputeKind::Boot;
}

ComputeAttrs read_compute_attrs(ComputeKind kind, const Json &value, const std::string &path) {
    switch (kind) {
    case ComputeKind::Rotate:
        require_members(value, doc, path, {"steps"});
        return RotateAttrs{read_int(value.at("steps"), doc, path + ".steps",
                                    std::numeric_limits<int>::min(), std::numeric_limits<int>::max())};
    case ComputeKind::Rescale:
        require_members(value, doc, path, {"target_level", "target_scale_log2"});
        return RescaleAttrs{read_nonnegative_int(value.at("target_level"), doc, path + ".target_level"),
                            read_nonnegative_int(value.at("target_scale_log2"), doc, path + ".target_scale_log2")};
    case ComputeKind::ModSwitch:
        require_members(value, doc, path, {"target_level"});
        return ModSwitchAttrs{read_nonnegative_int(value.at("target_level"), doc, path + ".target_level")};
    case ComputeKind::Boot:
        require_members(value, doc, path,
                        {"target_level", "target_scale_log2", "target_components",
                         "operator_profile", "implementation"});
        return BootAttrs{read_nonnegative_int(value.at("target_level"), doc, path + ".target_level"),
                         read_nonnegative_int(value.at("target_scale_log2"), doc, path + ".target_scale_log2"),
                         read_positive_int(value.at("target_components"), doc, path + ".target_components"),
                         read_string(value.at("operator_profile"), doc, path + ".operator_profile"),
                         read_boot_implementation(value.at("implementation"), path + ".implementation")};
    default:
        fail(doc, path, "operation does not accept attrs");
    }
}

Instruction read_encode(const Json &value, const std::string &path) {
    require_members(value, doc, path, {"ordinal", "kind", "payload", "output"});
    const auto &payload = value.at("payload");
    if (!payload.is_object() || !payload.contains("kind"))
        fail(doc, path + ".payload", "missing required field 'kind'");
    const std::string kind = read_string(payload.at("kind"), doc, path + ".payload.kind");
    EncodePayload parsed;
    if (kind == "inline") {
        require_members(payload, doc, path + ".payload", {"kind", "values"});
        const auto &values = payload.at("values");
        if (!values.is_array() || values.empty()) fail(doc, path + ".payload.values", "expected non-empty array");
        InlineEncodePayload inline_payload;
        for (std::size_t i = 0; i < values.size(); ++i)
            inline_payload.values.push_back(read_finite_double(values[i], doc, item_path(path + ".payload.values", i)));
        parsed = std::move(inline_payload);
    } else if (kind == "bundle") {
        require_members(payload, doc, path + ".payload", {"kind", "content"});
        parsed = BundleEncodePayload{read_sha256(payload.at("content"), doc, path + ".payload.content")};
    } else {
        fail(doc, path + ".payload.kind", "unknown encode payload kind '" + kind + "'");
    }
    return {static_cast<std::size_t>(read_nonnegative_int(value.at("ordinal"), doc, path + ".ordinal")),
            EncodeOp{std::move(parsed), read_id(value.at("output"), doc, path + ".output")}};
}

Instruction read_compute(const Json &value, const std::string &path) {
    if (!value.contains("op")) fail(doc, path, "missing required field 'op'");
    const ComputeKind kind = read_compute_kind(value.at("op"), path + ".op");
    if (compute_requires_attrs(kind))
        require_members(value, doc, path, {"ordinal", "kind", "op", "place", "inputs", "output", "attrs"});
    else
        require_members(value, doc, path, {"ordinal", "kind", "op", "place", "inputs", "output"});
    ComputeOp op;
    op.kind = kind;
    op.place = read_place(value.at("place"), path + ".place");
    op.inputs = read_ids(value.at("inputs"), path + ".inputs");
    op.output = read_id(value.at("output"), doc, path + ".output");
    op.attrs = compute_requires_attrs(kind) ? read_compute_attrs(kind, value.at("attrs"), path + ".attrs")
                                            : ComputeAttrs{std::monostate{}};
    return {static_cast<std::size_t>(read_nonnegative_int(value.at("ordinal"), doc, path + ".ordinal")), std::move(op)};
}

Instruction read_comm(const Json &value, const std::string &path, CommKind kind) {
    require_members(value, doc, path,
                    {"ordinal", "kind", "transfer_id", "hint", "inputs", "outputs",
                     "sources", "destinations", "output_kinds"});
    CommAction action;
    action.kind = kind;
    action.id = read_id(value.at("transfer_id"), doc, path + ".transfer_id");
    action.hint = read_comm_hint(value.at("hint"), path + ".hint");
    action.inputs = read_ids(value.at("inputs"), path + ".inputs");
    action.outputs = read_ids(value.at("outputs"), path + ".outputs");
    action.sources = read_places(value.at("sources"), path + ".sources");
    action.destinations = read_places(value.at("destinations"), path + ".destinations");
    action.output_types = read_value_kinds(value.at("output_kinds"), path + ".output_kinds");
    return {static_cast<std::size_t>(read_nonnegative_int(value.at("ordinal"), doc, path + ".ordinal")),
            std::move(action)};
}

Instruction read_instruction(const Json &value, const std::string &path) {
    if (!value.is_object() || !value.contains("kind")) fail(doc, path, "missing required field 'kind'");
    const std::string kind = read_string(value.at("kind"), doc, path + ".kind");
    if (kind == "encode") return read_encode(value, path);
    if (kind == "compute") return read_compute(value, path);
    if (kind == "transfer") return read_comm(value, path, CommKind::Transfer);
    if (kind == "replicate") return read_comm(value, path, CommKind::Replicate);
    fail(doc, path + ".kind", "unknown instruction kind '" + kind + "'");
}

std::vector<Instruction> read_instructions(const Json &value, const std::string &path) {
    if (!value.is_array()) fail(doc, path, "expected array");
    std::vector<Instruction> result;
    for (std::size_t i = 0; i < value.size(); ++i)
        result.push_back(read_instruction(value[i], item_path(path, i)));
    return result;
}

RuntimePlan read_document(const Json &root) {
    require_members(root, doc, "$",
                    {"format_version", "plan_id", "target", "values", "external_inputs",
                     "initialization", "execution", "finalization", "final_outputs"},
                    {"plaintext_bundle"});
    RuntimePlan plan;
    plan.format_version = static_cast<std::uint32_t>(read_nonnegative_int(root.at("format_version"), doc, "$.format_version"));
    if (plan.format_version != 1) fail(doc, "$.format_version", "unsupported format version");
    plan.plan_id = read_id(root.at("plan_id"), doc, "$.plan_id");
    plan.target = read_target(root.at("target"), "$.target");
    if (root.contains("plaintext_bundle")) plan.plaintext_bundle = read_bundle_ref(root.at("plaintext_bundle"), "$.plaintext_bundle");
    const auto &values = root.at("values");
    if (!values.is_array()) fail(doc, "$.values", "expected array");
    for (std::size_t i = 0; i < values.size(); ++i)
        plan.values.push_back(read_value_desc(values[i], item_path("$.values", i)));
    plan.external_inputs = read_ids(root.at("external_inputs"), "$.external_inputs");
    plan.initialization = read_instructions(root.at("initialization"), "$.initialization");
    plan.execution = read_instructions(root.at("execution"), "$.execution");
    plan.finalization = read_instructions(root.at("finalization"), "$.finalization");
    plan.final_outputs = read_ids(root.at("final_outputs"), "$.final_outputs");
    return plan;
}

} // namespace

LoadedRuntimePlan RuntimePlanJsonReader::read(std::istream &input) {
    const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    return read_text(text);
}

LoadedRuntimePlan RuntimePlanJsonReader::read_text(std::string_view text) {
    return {read_document(json_utils::parse(text, doc)), json_utils::source_sha256(text)};
}

LoadedRuntimePlan RuntimePlanJsonReader::read_file(const std::string &path) {
    return read_text(json_utils::read_file_bytes(path));
}

} // namespace fhegpu
