#pragma once

#include "runtime/operator_spec.hpp"

#include <iosfwd>
#include <string>
#include <string_view>

namespace fhegpu {

class OperatorSpecReader {
public:
    static LoadedOperatorSpec read(std::istream &input);
    static LoadedOperatorSpec read_text(std::string_view text);
    static LoadedOperatorSpec read_file(const std::string &path);
};

} // namespace fhegpu
