#pragma once

#include "runtime/execution_config.hpp"

#include <iosfwd>
#include <string>
#include <string_view>

namespace fhegpu {

class ExecutionConfigJsonReader {
public:
    static LoadedExecutionConfig read(std::istream &input);
    static LoadedExecutionConfig read_text(std::string_view text);
    static LoadedExecutionConfig read_file(const std::string &path);
};

} // namespace fhegpu
