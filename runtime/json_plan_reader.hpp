#pragma once

#include "runtime/plan.hpp"
#include "runtime/utils/json_read_stats.hpp"

#include <iosfwd>
#include <string>
#include <string_view>

namespace fhegpu {

class RuntimePlanJsonReader {
public:
    static LoadedRuntimePlan read(std::istream &input);
    static LoadedRuntimePlan read_text(std::string_view text);
    static LoadedRuntimePlan read_file(const std::string &path, JsonReadStats *stats = nullptr);
    // Binary containers keep only metadata here; all six record arrays must be empty.
    static RuntimePlan read_binary_metadata_text(std::string_view text);
};

} // namespace fhegpu
