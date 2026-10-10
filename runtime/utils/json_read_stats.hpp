#pragma once

#include <cstdint>

namespace fhegpu {

struct JsonReadStats {
    std::uint64_t source_bytes = 0;
    double read_seconds = 0;
    double hash_seconds = 0;
    double parse_build_seconds = 0;
};

} // namespace fhegpu
