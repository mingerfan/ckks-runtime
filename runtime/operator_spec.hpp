#pragma once

#include "runtime/plan.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace fhegpu {

struct OperatorSupport {
    bool supported = false;
    std::optional<std::vector<std::uint64_t>> latency_us_by_level;
    std::optional<int> max_levels_per_op;
};

struct BootProfile {
    std::string profile_id;
    BootImplementation implementation = BootImplementation::Native;
    int input_level_min = 0;
    int input_level_max = 0;
    int input_components = 2;
    int output_level = 0;
    int output_scale_log2 = 0;
    int output_components = 2;
    std::uint64_t latency_us = 0;
    bool needs_secret_key = false;
    bool needs_host_compute = false;
};

struct OperatorSpec {
    int format_version = 1;
    std::string id;
    int version = 0;
    std::string status;
    std::string target_id;
    RescaleMode rescale_mode = RescaleMode::Eager;
    std::string context_id;
    std::uint64_t poly_degree = 0;
    std::vector<int> rns_moduli_log2;
    int max_modulus_log2 = 0;
    int default_scale_log2 = 0;
    int level_lower_bound = 0;
    int level_upper_bound = 0;
    std::map<ComputeKind, OperatorSupport> operators;
    std::vector<BootProfile> boot_profiles;
};

struct LoadedOperatorSpec {
    OperatorSpec spec;
    std::string source_sha256;
};

} // namespace fhegpu
