#include "runtime/json_plan_reader.hpp"
#include "runtime/utils/sha256.hpp"

#include <nlohmann/json.hpp>

#include <charconv>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace fhegpu {
namespace {

using Json = nlohmann::json;

[[noreturn]] void fail(const std::string &path, const std::string &message) {
    throw std::runtime_error("runtime plan JSON error at " + path + ": " + message);
}

std::string item_path(const std::string &path, std::size_t index) {
    return path + '[' + std::to_string(index) + ']';
}

bool contains_name(std::initializer_list<const char *> names, const std::string &name) {
    for (const char *candidate : names) if (name == candidate) return true;
    return false;
}

void require_members(const Json &value, const std::string &path,
                     std::initializer_list<const char *> names,
                     std::initializer_list<const char *> optional_names = {}) {
    if (!value.is_object()) fail(path, "expected object");
    for (const char *name : names) {
        if (!value.contains(name)) fail(path, "missing required field '" + std::string(name) + "'");
    }
    for (const auto &item : value.items()) {
        if (!contains_name(names, item.key()) && !contains_name(optional_names, item.key()))
            fail(path, "unknown field '" + item.key() + "'");
    }
}

std::string read_string(const Json &value, const std::string &path, bool allow_empty = false) {
    if (!value.is_string()) fail(path, "expected string");
    const std::string result = value.get<std::string>();
    if (!allow_empty && result.empty()) fail(path, "string must not be empty");
    return result;
}

bool read_bool(const Json &value, const std::string &path) {
    if (!value.is_boolean()) fail(path, "expected boolean");
    return value.get<bool>();
}

int read_int(const Json &value, const std::string &path, int minimum, int maximum) {
    if (value.is_number_unsigned()) {
        const auto number = value.get<std::uint64_t>();
        if (number > static_cast<std::uint64_t>(maximum)) fail(path, "integer is outside the supported range");
        return static_cast<int>(number);
    }
    if (!value.is_number_integer()) fail(path, "expected integer");
    const auto number = value.get<std::int64_t>();
    if (number < minimum || number > maximum) fail(path, "integer is outside the supported range");
    return static_cast<int>(number);
}

int read_nonnegative_int(const Json &value, const std::string &path) {
    return read_int(value, path, 0, std::numeric_limits<int>::max());
}

int read_positive_int(const Json &value, const std::string &path) {
    return read_int(value, path, 1, std::numeric_limits<int>::max());
}

std::uint64_t read_id(const Json &value, const std::string &path) {
    const std::string text = read_string(value, path);
    if (text != "0") {
        if (text.front() < '1' || text.front() > '9') fail(path, "identifier must be canonical unsigned decimal");
        for (char character : text) {
            if (character < '0' || character > '9') fail(path, "identifier must be canonical unsigned decimal");
        }
    }
    std::uint64_t result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        fail(path, "identifier is outside uint64 range");
    return result;
}

std::string read_fingerprint(const Json &value, const std::string &path) {
    const std::string fingerprint = read_string(value, path);
    if (fingerprint.size() != 71 || fingerprint.compare(0, 7, "sha256:") != 0)
        fail(path, "expected sha256 followed by 64 lowercase hexadecimal digits");
    for (std::size_t index = 7; index < fingerprint.size(); ++index) {
        const char character = fingerprint[index];
        if (!((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f')))
            fail(path, "expected sha256 followed by 64 lowercase hexadecimal digits");
    }
    return fingerprint;
}

Json parse_json(std::string_view text) {
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xef &&
        static_cast<unsigned char>(text[1]) == 0xbb &&
        static_cast<unsigned char>(text[2]) == 0xbf)
        fail("$", "UTF-8 BOM is not allowed");

    std::vector<std::set<std::string>> object_keys;
    const auto callback = [&](int, Json::parse_event_t event, Json &parsed) {
        switch (event) {
        case Json::parse_event_t::object_start:
            object_keys.emplace_back();
            break;
        case Json::parse_event_t::key: {
            if (object_keys.empty()) fail("$", "invalid parser object state");
            const std::string key = parsed.get<std::string>();
            if (!object_keys.back().insert(key).second) fail("$", "duplicate object key '" + key + "'");
            break;
        }
        case Json::parse_event_t::object_end:
            if (object_keys.empty()) fail("$", "invalid parser object state");
            object_keys.pop_back();
            break;
        case Json::parse_event_t::value:
            if (parsed.is_number_float()) fail("$", "floating-point numbers are not allowed");
            break;
        case Json::parse_event_t::array_start:
        case Json::parse_event_t::array_end:
            break;
        }
        return true;
    };

    try {
        return Json::parse(text.begin(), text.end(), callback, true, false);
    } catch (const Json::exception &error) {
        fail("$", std::string("JSON parsing failed: ") + error.what());
    }
}

std::string calculate_fingerprint(Json document) {
    document.erase("fingerprint");
    return "sha256:" + sha256_hex(
        document.dump(-1, ' ', false, Json::error_handler_t::strict));
}

Place read_place(const Json &value, const std::string &path) {
    if (!value.is_object()) fail(path, "expected object");
    if (!value.contains("kind")) fail(path, "missing required field 'kind'");
    const std::string kind = read_string(value.at("kind"), path + ".kind");
    if (kind == "host") {
        require_members(value, path, {"kind", "rank"});
        return Place{PlaceKind::Host, read_nonnegative_int(value.at("rank"), path + ".rank"), 0};
    }
    if (kind == "device") {
        require_members(value, path, {"kind", "rank", "index"});
        return Place{PlaceKind::Device,
                     read_nonnegative_int(value.at("rank"), path + ".rank"),
                     read_nonnegative_int(value.at("index"), path + ".index")};
    }
    fail(path + ".kind", "unknown place kind '" + kind + "'");
}

ValueKind read_value_kind(const Json &value, const std::string &path) {
    const std::string kind = read_string(value, path);
    if (kind == "plaintext") return ValueKind::Plaintext;
    if (kind == "ciphertext") return ValueKind::Ciphertext;
    fail(path, "unknown value kind '" + kind + "'");
}

RescaleMode read_rescale_mode(const Json &value, const std::string &path) {
    const std::string mode = read_string(value, path);
    if (mode == "eager") return RescaleMode::Eager;
    if (mode == "lazy") return RescaleMode::Lazy;
    fail(path, "unknown rescale mode '" + mode + "'");
}

BootMode read_boot_mode(const Json &value, const std::string &path) {
    const std::string mode = read_string(value, path);
    if (mode == "native") return BootMode::Native;
    if (mode == "decrypt_reencrypt") return BootMode::DecryptReencrypt;
    fail(path, "unknown boot mode '" + mode + "'");
}

BootImplementation read_boot_implementation(const Json &value, const std::string &path) {
    const std::string implementation = read_string(value, path);
    if (implementation == "native") return BootImplementation::Native;
    if (implementation == "decrypt_reencrypt") return BootImplementation::DecryptReencrypt;
    fail(path, "unknown boot implementation '" + implementation + "'");
}

RequiredCapability read_capability(const Json &value, const std::string &path) {
    const std::string capability = read_string(value, path);
    if (capability == "transfer") return RequiredCapability::Transfer;
    if (capability == "replicate") return RequiredCapability::Replicate;
    if (capability == "host_compute") return RequiredCapability::HostCompute;
    if (capability == "boot_native") return RequiredCapability::BootNative;
    if (capability == "boot_decrypt_reencrypt") return RequiredCapability::BootDecryptReencrypt;
    fail(path, "unknown required capability '" + capability + "'");
}

KeyKind read_key_kind(const Json &value, const std::string &path) {
    const std::string kind = read_string(value, path);
    if (kind == "secret") return KeyKind::Secret;
    if (kind == "relin") return KeyKind::Relin;
    if (kind == "galois") return KeyKind::Galois;
    fail(path, "unknown key kind '" + kind + "'");
}

ComputeKind read_compute_kind(const Json &value, const std::string &path) {
    const std::string kind = read_string(value, path);
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
    fail(path, "unknown compute operation '" + kind + "'");
}

CommHint read_comm_hint(const Json &value, const std::string &path) {
    const std::string hint = read_string(value, path);
    if (hint == "auto") return CommHint::Auto;
    if (hint == "point_to_point") return CommHint::PointToPoint;
    if (hint == "broadcast") return CommHint::Broadcast;
    if (hint == "tree") return CommHint::Tree;
    if (hint == "ring") return CommHint::Ring;
    if (hint == "host_staged") return CommHint::HostStaged;
    fail(path, "unknown communication hint '" + hint + "'");
}

OperatorSpecRef read_operator_spec(const Json &value, const std::string &path) {
    require_members(value, path, {"id", "version", "fingerprint"});
    return OperatorSpecRef{
        read_string(value.at("id"), path + ".id"),
        read_positive_int(value.at("version"), path + ".version"),
        read_fingerprint(value.at("fingerprint"), path + ".fingerprint")};
}

PlaintextBundleRef read_plaintext_bundle(const Json &value, const std::string &path) {
    require_members(value, path, {"id", "version", "fingerprint"});
    return PlaintextBundleRef{
        read_string(value.at("id"), path + ".id"),
        read_positive_int(value.at("version"), path + ".version"),
        read_fingerprint(value.at("fingerprint"), path + ".fingerprint")};
}

TargetConfig read_target(const Json &value, const std::string &path) {
    require_members(value, path,
        {"target_id", "capability_version", "operator_spec", "rescale_mode",
         "boot_mode", "world_size", "device_counts", "required_capabilities"});

    TargetConfig target;
    target.target_id = read_string(value.at("target_id"), path + ".target_id");
    target.capability_version = read_positive_int(value.at("capability_version"), path + ".capability_version");
    target.operator_spec = read_operator_spec(value.at("operator_spec"), path + ".operator_spec");
    target.rescale_mode = read_rescale_mode(value.at("rescale_mode"), path + ".rescale_mode");
    target.boot_mode = read_boot_mode(value.at("boot_mode"), path + ".boot_mode");
    target.world_size = read_positive_int(value.at("world_size"), path + ".world_size");

    const Json &device_counts = value.at("device_counts");
    if (!device_counts.is_array() || device_counts.empty()) fail(path + ".device_counts", "expected non-empty array");
    target.device_counts.reserve(device_counts.size());
    for (std::size_t index = 0; index < device_counts.size(); ++index)
        target.device_counts.push_back(read_nonnegative_int(device_counts[index], item_path(path + ".device_counts", index)));
    if (target.device_counts.size() != static_cast<std::size_t>(target.world_size))
        fail(path + ".device_counts", "length must equal world_size");

    const Json &capabilities = value.at("required_capabilities");
    if (!capabilities.is_array()) fail(path + ".required_capabilities", "expected array");
    std::set<RequiredCapability> seen_capabilities;
    target.required_capabilities.reserve(capabilities.size());
    for (std::size_t index = 0; index < capabilities.size(); ++index) {
        const RequiredCapability capability =
            read_capability(capabilities[index], item_path(path + ".required_capabilities", index));
        if (!seen_capabilities.insert(capability).second)
            fail(item_path(path + ".required_capabilities", index), "duplicate required capability");
        target.required_capabilities.push_back(capability);
    }
    return target;
}

ValueDesc read_value_desc(const Json &value, const std::string &path) {
    require_members(value, path,
        {"id", "kind", "place", "context", "level", "scale_log2", "ntt", "components"});
    return ValueDesc{
        read_id(value.at("id"), path + ".id"),
        read_value_kind(value.at("kind"), path + ".kind"),
        read_place(value.at("place"), path + ".place"),
        read_string(value.at("context"), path + ".context"),
        read_nonnegative_int(value.at("level"), path + ".level"),
        read_nonnegative_int(value.at("scale_log2"), path + ".scale_log2"),
        read_bool(value.at("ntt"), path + ".ntt"),
        read_positive_int(value.at("components"), path + ".components")};
}

KeyRequirement read_key_requirement(const Json &value, const std::string &path) {
    require_members(value, path, {"kind", "place"});
    return KeyRequirement{read_key_kind(value.at("kind"), path + ".kind"),
                          read_place(value.at("place"), path + ".place")};
}

std::vector<ValueId> read_ids(const Json &value, const std::string &path) {
    if (!value.is_array()) fail(path, "expected array");
    std::vector<ValueId> ids;
    ids.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index)
        ids.push_back(read_id(value[index], item_path(path, index)));
    return ids;
}

std::vector<Place> read_places(const Json &value, const std::string &path) {
    if (!value.is_array()) fail(path, "expected array");
    std::vector<Place> places;
    places.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index)
        places.push_back(read_place(value[index], item_path(path, index)));
    return places;
}

std::vector<ValueKind> read_value_kinds(const Json &value, const std::string &path) {
    if (!value.is_array()) fail(path, "expected array");
    std::vector<ValueKind> kinds;
    kinds.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index)
        kinds.push_back(read_value_kind(value[index], item_path(path, index)));
    return kinds;
}

bool compute_requires_attrs(ComputeKind kind) {
    return kind == ComputeKind::Rotate || kind == ComputeKind::Rescale ||
           kind == ComputeKind::ModSwitch || kind == ComputeKind::Boot;
}

std::size_t compute_input_count(ComputeKind kind) {
    switch (kind) {
    case ComputeKind::AddCC: case ComputeKind::AddCP:
    case ComputeKind::SubCC: case ComputeKind::SubCP:
    case ComputeKind::MulCC: case ComputeKind::MulCP:
        return 2;
    default:
        return 1;
    }
}

ComputeAttrs read_compute_attrs(ComputeKind kind, const Json &value, const std::string &path) {
    switch (kind) {
    case ComputeKind::Rotate:
        require_members(value, path, {"steps"});
        return RotateAttrs{read_int(value.at("steps"), path + ".steps",
                                    std::numeric_limits<int>::min(), std::numeric_limits<int>::max())};
    case ComputeKind::Rescale:
        require_members(value, path, {"target_level", "target_scale_log2"});
        return RescaleAttrs{
            read_nonnegative_int(value.at("target_level"), path + ".target_level"),
            read_nonnegative_int(value.at("target_scale_log2"), path + ".target_scale_log2")};
    case ComputeKind::ModSwitch:
        require_members(value, path, {"target_level"});
        return ModSwitchAttrs{read_nonnegative_int(value.at("target_level"), path + ".target_level")};
    case ComputeKind::Boot:
        require_members(value, path,
            {"target_level", "target_scale_log2", "target_components",
             "operator_profile", "implementation"});
        return BootAttrs{
            read_nonnegative_int(value.at("target_level"), path + ".target_level"),
            read_nonnegative_int(value.at("target_scale_log2"), path + ".target_scale_log2"),
            read_positive_int(value.at("target_components"), path + ".target_components"),
            read_string(value.at("operator_profile"), path + ".operator_profile"),
            read_boot_implementation(value.at("implementation"), path + ".implementation")};
    default:
        fail(path, "operation does not accept attrs");
    }
}

Instruction read_compute_instruction(const Json &value, const std::string &path) {
    if (!value.contains("op")) fail(path, "missing required field 'op'");
    const ComputeKind kind = read_compute_kind(value.at("op"), path + ".op");
    if (compute_requires_attrs(kind))
        require_members(value, path, {"ordinal", "kind", "op", "place", "inputs", "output", "attrs"});
    else
        require_members(value, path, {"ordinal", "kind", "op", "place", "inputs", "output"});

    ComputeOp operation;
    operation.kind = kind;
    operation.place = read_place(value.at("place"), path + ".place");
    operation.inputs = read_ids(value.at("inputs"), path + ".inputs");
    if (operation.inputs.size() != compute_input_count(kind))
        fail(path + ".inputs", "wrong number of inputs for operation");
    operation.output = read_id(value.at("output"), path + ".output");
    operation.attrs = compute_requires_attrs(kind)
        ? read_compute_attrs(kind, value.at("attrs"), path + ".attrs")
        : ComputeAttrs{std::monostate{}};

    if (const auto *rotate = std::get_if<RotateAttrs>(&operation.attrs); rotate && rotate->steps == 0)
        fail(path + ".attrs.steps", "rotate steps must be nonzero");

    return Instruction{
        static_cast<std::size_t>(read_nonnegative_int(value.at("ordinal"), path + ".ordinal")),
        std::move(operation)};
}

Instruction read_comm_instruction(const Json &value, const std::string &path,
                                  const std::string &kind_text) {
    require_members(value, path,
        {"ordinal", "kind", "transfer_id", "hint", "inputs", "outputs",
         "sources", "destinations", "output_kinds"});

    CommAction action;
    action.kind = kind_text == "transfer" ? CommKind::Transfer : CommKind::Replicate;
    action.id = read_id(value.at("transfer_id"), path + ".transfer_id");
    action.hint = read_comm_hint(value.at("hint"), path + ".hint");
    action.inputs = read_ids(value.at("inputs"), path + ".inputs");
    action.outputs = read_ids(value.at("outputs"), path + ".outputs");
    action.sources = read_places(value.at("sources"), path + ".sources");
    action.destinations = read_places(value.at("destinations"), path + ".destinations");
    action.output_types = read_value_kinds(value.at("output_kinds"), path + ".output_kinds");

    if (action.inputs.size() != 1) fail(path + ".inputs", "communication requires exactly one input");
    if (action.sources.size() != 1) fail(path + ".sources", "communication requires exactly one source");
    const std::size_t output_count = action.outputs.size();
    if (output_count == 0 || action.destinations.size() != output_count ||
        action.output_types.size() != output_count)
        fail(path, "outputs, destinations and output_kinds must have the same nonzero length");
    if (action.kind == CommKind::Transfer && output_count != 1)
        fail(path, "transfer requires exactly one output");
    if (action.kind == CommKind::Replicate && output_count < 2)
        fail(path, "replicate requires at least two outputs");

    return Instruction{
        static_cast<std::size_t>(read_nonnegative_int(value.at("ordinal"), path + ".ordinal")),
        std::move(action)};
}

Instruction read_instruction(const Json &value, const std::string &path) {
    if (!value.is_object()) fail(path, "expected object");
    if (!value.contains("kind")) fail(path, "missing required field 'kind'");
    const std::string kind = read_string(value.at("kind"), path + ".kind");
    if (kind == "compute") return read_compute_instruction(value, path);
    if (kind == "transfer" || kind == "replicate") return read_comm_instruction(value, path, kind);
    fail(path + ".kind", "unknown instruction kind '" + kind + "'");
}

std::vector<Instruction> read_instruction_list(const Json &value, const std::string &path) {
    if (!value.is_array()) fail(path, "expected array");
    std::vector<Instruction> instructions;
    instructions.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index)
        instructions.push_back(read_instruction(value[index], item_path(path, index)));
    return instructions;
}

RuntimePlan convert_plan(const Json &document) {
    require_members(document, "$",
        {"format_version", "plan_id", "fingerprint", "target", "values",
         "external_inputs", "required_keys", "initialization", "execution",
         "finalization", "final_outputs"},
        {"plaintext_bundle"});

    RuntimePlan plan;
    const int format_version = read_nonnegative_int(document.at("format_version"), "$.format_version");
    if (format_version != 1) fail("$.format_version", "unsupported format version");
    plan.format_version = static_cast<std::uint32_t>(format_version);
    plan.plan_id = read_id(document.at("plan_id"), "$.plan_id");
    plan.fingerprint_sha256 = read_fingerprint(document.at("fingerprint"), "$.fingerprint");
    plan.target = read_target(document.at("target"), "$.target");
    if (document.contains("plaintext_bundle"))
        plan.plaintext_bundle = read_plaintext_bundle(document.at("plaintext_bundle"), "$.plaintext_bundle");

    const Json &values = document.at("values");
    if (!values.is_array()) fail("$.values", "expected array");
    plan.values.reserve(values.size());
    for (std::size_t index = 0; index < values.size(); ++index)
        plan.values.push_back(read_value_desc(values[index], item_path("$.values", index)));

    plan.external_inputs = read_ids(document.at("external_inputs"), "$.external_inputs");

    const Json &required_keys = document.at("required_keys");
    if (!required_keys.is_array()) fail("$.required_keys", "expected array");
    plan.required_keys.reserve(required_keys.size());
    for (std::size_t index = 0; index < required_keys.size(); ++index)
        plan.required_keys.push_back(read_key_requirement(required_keys[index], item_path("$.required_keys", index)));

    plan.initialization = read_instruction_list(document.at("initialization"), "$.initialization");
    plan.execution = read_instruction_list(document.at("execution"), "$.execution");
    plan.finalization = read_instruction_list(document.at("finalization"), "$.finalization");
    plan.final_outputs = read_ids(document.at("final_outputs"), "$.final_outputs");
    if (plan.final_outputs.empty()) fail("$.final_outputs", "must not be empty");

    const std::string calculated = calculate_fingerprint(document);
    if (calculated != plan.fingerprint_sha256)
        fail("$.fingerprint", "fingerprint mismatch: expected " + calculated +
                              ", file contains " + plan.fingerprint_sha256);
    return plan;
}

} // namespace

RuntimePlan RuntimePlanJsonReader::read(std::istream &input) {
    const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (input.bad()) throw std::runtime_error("failed to read RuntimePlan JSON stream");
    return read_text(text);
}

RuntimePlan RuntimePlanJsonReader::read_text(std::string_view text) {
    return convert_plan(parse_json(text));
}

RuntimePlan RuntimePlanJsonReader::read_file(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("failed to open RuntimePlan JSON file: " + path);
    return read(input);
}

} // namespace fhegpu
