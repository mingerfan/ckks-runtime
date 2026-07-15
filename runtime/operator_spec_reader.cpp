#include "runtime/operator_spec_reader.hpp"
#include "runtime/json_utils.hpp"

#include <istream>
#include <iterator>
#include <limits>
#include <set>

namespace fhegpu {
namespace {

using namespace json_utils;
constexpr const char *doc = "operator spec";

RescaleMode read_rescale_mode(const Json &value, const std::string &path) {
    const std::string mode = read_string(value, doc, path);
    if (mode == "eager") return RescaleMode::Eager;
    if (mode == "lazy") return RescaleMode::Lazy;
    fail(doc, path, "unknown rescale mode '" + mode + "'");
}

BootImplementation read_boot_implementation(const Json &value, const std::string &path) {
    const std::string implementation = read_string(value, doc, path);
    if (implementation == "native") return BootImplementation::Native;
    if (implementation == "decrypt_reencrypt") return BootImplementation::DecryptReencrypt;
    fail(doc, path, "unknown boot implementation '" + implementation + "'");
}

std::optional<std::vector<std::uint64_t>> read_latency(const Json &value,
                                                        const std::string &path,
                                                        int expected_length) {
    if (value.is_null()) return std::nullopt;
    if (!value.is_array()) fail(doc, path, "expected array or null");
    if (value.size() != static_cast<std::size_t>(expected_length))
        fail(doc, path, "length must equal levels.upper_bound + 1");
    std::vector<std::uint64_t> result;
    for (std::size_t i = 0; i < value.size(); ++i)
        result.push_back(read_safe_uint(value[i], doc, item_path(path, i), 0,
                                        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())));
    return result;
}

std::optional<std::vector<double>> read_noise(const Json &value,
                                              const std::string &path,
                                              int expected_length) {
    if (value.is_null()) return std::nullopt;
    if (!value.is_array()) fail(doc, path, "expected array or null");
    if (value.size() != static_cast<std::size_t>(expected_length))
        fail(doc, path, "length must equal levels.upper_bound + 1");
    std::vector<double> result;
    for (std::size_t i = 0; i < value.size(); ++i) {
        const double noise = read_finite_double(value[i], doc, item_path(path, i));
        if (noise < 0.0) fail(doc, item_path(path, i), "noise must be nonnegative");
        result.push_back(noise);
    }
    return result;
}

OperatorSupport read_operator(const Json &value, const std::string &path,
                              int table_length, bool rescale, int format_version) {
    if (rescale)
        require_members(value, doc, path,
                        {"supported", "max_levels_per_op", "latency_us_by_level", "noise_by_level"});
    else
        require_members(value, doc, path,
                        {"supported", "latency_us_by_level", "noise_by_level"});
    OperatorSupport result;
    result.supported = read_bool(value.at("supported"), doc, path + ".supported");
    result.latency_us_by_level = read_latency(value.at("latency_us_by_level"), path + ".latency_us_by_level", table_length);
    if (format_version == 1) {
        if (!value.at("noise_by_level").is_null())
            fail(doc, path + ".noise_by_level", "must be null in V1");
    } else {
        result.noise_by_level = read_noise(value.at("noise_by_level"), path + ".noise_by_level", table_length);
    }
    if (rescale)
        result.max_levels_per_op = read_positive_int(value.at("max_levels_per_op"), doc, path + ".max_levels_per_op");
    return result;
}

BootProfile read_boot_profile(const Json &value, const std::string &path,
                              int format_version, int table_length) {
    if (format_version == 1)
        require_members(value, doc, path,
                        {"profile_id", "implementation", "input_level_min", "input_level_max",
                         "input_components", "output_level", "output_scale_log2", "output_components",
                         "latency_us", "host_requirements"});
    else
        require_members(value, doc, path,
                        {"profile_id", "implementation", "input_level_min", "input_level_max",
                         "input_components", "output_level", "output_scale_log2", "output_components",
                         "latency_us_by_input_level", "noise_by_input_level", "host_requirements"});
    const auto &host = value.at("host_requirements");
    require_members(host, doc, path + ".host_requirements",
                    {"needs_secret_key", "needs_host_compute"});
    BootProfile result;
    result.profile_id = read_string(value.at("profile_id"), doc, path + ".profile_id");
    result.implementation = read_boot_implementation(value.at("implementation"), path + ".implementation");
    result.input_level_min = read_nonnegative_int(value.at("input_level_min"), doc, path + ".input_level_min");
    result.input_level_max = read_nonnegative_int(value.at("input_level_max"), doc, path + ".input_level_max");
    result.input_components = read_positive_int(value.at("input_components"), doc, path + ".input_components");
    result.output_level = read_nonnegative_int(value.at("output_level"), doc, path + ".output_level");
    result.output_scale_log2 = read_nonnegative_int(value.at("output_scale_log2"), doc, path + ".output_scale_log2");
    result.output_components = read_positive_int(value.at("output_components"), doc, path + ".output_components");
    if (format_version == 1)
        result.latency_us = read_safe_uint(value.at("latency_us"), doc, path + ".latency_us", 0,
                                           static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()));
    else {
        result.latency_us_by_input_level = read_latency(
            value.at("latency_us_by_input_level"), path + ".latency_us_by_input_level", table_length);
        result.noise_by_input_level = read_noise(
            value.at("noise_by_input_level"), path + ".noise_by_input_level", table_length);
    }
    result.needs_secret_key = read_bool(host.at("needs_secret_key"), doc, path + ".host_requirements.needs_secret_key");
    result.needs_host_compute = read_bool(host.at("needs_host_compute"), doc, path + ".host_requirements.needs_host_compute");
    return result;
}

OperatorSpec read_document(const Json &root) {
    if (!root.is_object()) fail(doc, "$", "expected object");
    if (!root.contains("spec_format_version")) fail(doc, "$", "missing required field 'spec_format_version'");
    OperatorSpec spec;
    spec.format_version = read_nonnegative_int(root.at("spec_format_version"), doc, "$.spec_format_version");
    if (spec.format_version == 1)
        require_members(root, doc, "$",
                        {"spec_format_version", "spec_id", "version", "status", "target_id",
                         "rescale_mode", "context", "levels", "operators", "boot_profiles"});
    else if (spec.format_version == 2)
        require_members(root, doc, "$",
                        {"spec_format_version", "spec_id", "version", "status", "target_id",
                         "rescale_mode", "noise_unit", "provenance", "context", "levels",
                         "operators", "boot_profiles"});
    else
        fail(doc, "$.spec_format_version", "unsupported format version");
    spec.id = read_string(root.at("spec_id"), doc, "$.spec_id");
    spec.version = read_positive_int(root.at("version"), doc, "$.version");
    spec.status = read_string(root.at("status"), doc, "$.status");
    if (spec.format_version == 1) {
        if (spec.status != "placeholder" && spec.status != "validated")
            fail(doc, "$.status", "unknown status");
    } else {
        if (spec.status != "imported" && spec.status != "validated")
            fail(doc, "$.status", "unknown status");
        if (!root.at("noise_unit").is_null())
            spec.noise_unit = read_string(root.at("noise_unit"), doc, "$.noise_unit");
        const auto &provenance = root.at("provenance");
        require_members(provenance, doc, "$.provenance",
                        {"kind", "repository", "revision", "path", "source_sha256"});
        OperatorSpecProvenance source;
        source.kind = read_string(provenance.at("kind"), doc, "$.provenance.kind");
        source.repository = read_string(provenance.at("repository"), doc, "$.provenance.repository");
        source.revision = read_string(provenance.at("revision"), doc, "$.provenance.revision");
        source.path = read_string(provenance.at("path"), doc, "$.provenance.path");
        source.source_sha256 = read_sha256(provenance.at("source_sha256"), doc, "$.provenance.source_sha256");
        spec.provenance = std::move(source);
    }
    spec.target_id = read_string(root.at("target_id"), doc, "$.target_id");
    spec.rescale_mode = read_rescale_mode(root.at("rescale_mode"), "$.rescale_mode");

    const auto &context = root.at("context");
    require_members(context, doc, "$.context",
                    {"context_id", "poly_degree", "rns_moduli_log2", "max_modulus_log2", "default_scale_log2"});
    spec.context_id = read_string(context.at("context_id"), doc, "$.context.context_id");
    spec.poly_degree = read_safe_uint(context.at("poly_degree"), doc, "$.context.poly_degree", 2,
                                      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()));
    if ((spec.poly_degree & (spec.poly_degree - 1)) != 0) fail(doc, "$.context.poly_degree", "must be a power of two");
    const auto &moduli = context.at("rns_moduli_log2");
    if (!moduli.is_array() || moduli.empty()) fail(doc, "$.context.rns_moduli_log2", "expected non-empty array");
    for (std::size_t i = 0; i < moduli.size(); ++i)
        spec.rns_moduli_log2.push_back(read_int(moduli[i], doc, item_path("$.context.rns_moduli_log2", i), 1, 64));
    spec.max_modulus_log2 = read_int(context.at("max_modulus_log2"), doc, "$.context.max_modulus_log2", 1, 64);
    for (int bits : spec.rns_moduli_log2) if (bits > spec.max_modulus_log2)
        fail(doc, "$.context.rns_moduli_log2", "modulus exceeds max_modulus_log2");
    spec.default_scale_log2 = read_nonnegative_int(context.at("default_scale_log2"), doc, "$.context.default_scale_log2");

    const auto &levels = root.at("levels");
    require_members(levels, doc, "$.levels", {"lower_bound", "upper_bound"});
    spec.level_lower_bound = read_nonnegative_int(levels.at("lower_bound"), doc, "$.levels.lower_bound");
    spec.level_upper_bound = read_nonnegative_int(levels.at("upper_bound"), doc, "$.levels.upper_bound");
    if (spec.level_lower_bound > spec.level_upper_bound) fail(doc, "$.levels", "lower_bound exceeds upper_bound");
    if (spec.level_upper_bound >= static_cast<int>(spec.rns_moduli_log2.size()))
        fail(doc, "$.levels.upper_bound", "must be inside the modulus chain");

    const auto &operators = root.at("operators");
    require_members(operators, doc, "$.operators",
                    {"add_cc", "add_cp", "sub_cc", "sub_cp", "mul_cc", "mul_cp",
                     "negate", "rotate", "rescale", "mod_switch", "relinearize", "boot"});
    const std::vector<std::pair<const char *, ComputeKind>> ordinary = {
        {"add_cc", ComputeKind::AddCC}, {"add_cp", ComputeKind::AddCP},
        {"sub_cc", ComputeKind::SubCC}, {"sub_cp", ComputeKind::SubCP},
        {"mul_cc", ComputeKind::MulCC}, {"mul_cp", ComputeKind::MulCP},
        {"negate", ComputeKind::Negate}, {"rotate", ComputeKind::Rotate},
        {"rescale", ComputeKind::Rescale}, {"mod_switch", ComputeKind::ModSwitch},
        {"relinearize", ComputeKind::Relinearize}};
    for (const auto &entry : ordinary)
        spec.operators.emplace(entry.second, read_operator(operators.at(entry.first),
            "$.operators." + std::string(entry.first), spec.level_upper_bound + 1,
            entry.second == ComputeKind::Rescale, spec.format_version));
    const auto &boot = operators.at("boot");
    require_members(boot, doc, "$.operators.boot", {"supported"});
    OperatorSupport boot_support;
    boot_support.supported = read_bool(boot.at("supported"), doc, "$.operators.boot.supported");
    spec.operators.emplace(ComputeKind::Boot, std::move(boot_support));

    const auto &profiles = root.at("boot_profiles");
    if (!profiles.is_array()) fail(doc, "$.boot_profiles", "expected array");
    std::set<std::string> profile_ids;
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        BootProfile profile = read_boot_profile(profiles[i], item_path("$.boot_profiles", i),
                                                spec.format_version, spec.level_upper_bound + 1);
        if (!profile_ids.insert(profile.profile_id).second) fail(doc, item_path("$.boot_profiles", i), "duplicate profile_id");
        if (profile.input_level_min > profile.input_level_max) fail(doc, item_path("$.boot_profiles", i), "input level range is empty");
        if (profile.input_level_min < spec.level_lower_bound || profile.input_level_max > spec.level_upper_bound ||
            profile.output_level < spec.level_lower_bound || profile.output_level > spec.level_upper_bound)
            fail(doc, item_path("$.boot_profiles", i), "boot level is outside levels range");
        if (profile.input_components < 2 || profile.output_components < 2)
            fail(doc, item_path("$.boot_profiles", i), "Boot components must describe ciphertexts");
        long long output_budget = 0;
        for (int level = 0; level <= profile.output_level; ++level)
            output_budget += spec.rns_moduli_log2.at(static_cast<std::size_t>(level));
        if (profile.output_scale_log2 >= output_budget)
            fail(doc, item_path("$.boot_profiles", i), "Boot output scale exceeds modulus budget");
        if (profile.implementation == BootImplementation::DecryptReencrypt &&
            (!profile.needs_secret_key || !profile.needs_host_compute))
            fail(doc, item_path("$.boot_profiles", i), "decrypt_reencrypt must require secret key and host compute");
        if (profile.implementation == BootImplementation::Native &&
            (profile.needs_secret_key || profile.needs_host_compute))
            fail(doc, item_path("$.boot_profiles", i), "native boot cannot declare host requirements");
        spec.boot_profiles.push_back(std::move(profile));
    }
    if (spec.operators.at(ComputeKind::Boot).supported != !spec.boot_profiles.empty())
        fail(doc, "$.boot_profiles", "boot supported flag must match profile availability");
    if (spec.format_version == 2) {
        bool has_noise = false;
        for (const auto &entry : spec.operators)
            has_noise = has_noise || entry.second.noise_by_level.has_value();
        for (const auto &profile : spec.boot_profiles)
            has_noise = has_noise || profile.noise_by_input_level.has_value();
        if (has_noise != spec.noise_unit.has_value())
            fail(doc, "$.noise_unit", "must be set exactly when a V2 noise table is present");
    }
    return spec;
}

} // namespace

LoadedOperatorSpec OperatorSpecReader::read(std::istream &input) {
    const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    return read_text(text);
}

LoadedOperatorSpec OperatorSpecReader::read_text(std::string_view text) {
    return {read_document(json_utils::parse(text, doc)), json_utils::source_sha256(text)};
}

LoadedOperatorSpec OperatorSpecReader::read_file(const std::string &path) {
    return read_text(json_utils::read_file_bytes(path));
}

} // namespace fhegpu
