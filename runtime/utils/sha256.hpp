#pragma once

#include <string>
#include <string_view>

namespace fhegpu {

std::string sha256_hex(std::string_view bytes);

} // namespace fhegpu
