#pragma once

#include "runtime/plan.hpp"

#include <iosfwd>
#include <string>
#include <string_view>

namespace fhegpu {

class RuntimePlanJsonReader {
public:
    static LoadedRuntimePlan read(std::istream &input);
    static LoadedRuntimePlan read_text(std::string_view text);
    static LoadedRuntimePlan read_file(const std::string &path);
};

} // namespace fhegpu
