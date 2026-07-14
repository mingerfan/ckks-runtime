#pragma once

#include "runtime/plan.hpp"

#include <iosfwd>
#include <string>
#include <string_view>

namespace fhegpu {

class RuntimePlanJsonReader {
public:
    static RuntimePlan read(std::istream &input);
    static RuntimePlan read_text(std::string_view text);
    static RuntimePlan read_file(const std::string &path);
};

} // namespace fhegpu
