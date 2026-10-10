#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace fhegpu {

class Sha256 {
public:
    void update(std::string_view bytes);
    // A snapshot; callers may continue updating or request the digest again.
    std::string hex_digest() const;
private:
    std::array<std::uint32_t, 8> state_ = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::array<unsigned char, 64> buffer_{};
    std::size_t buffered_ = 0;
    std::uint64_t bytes_ = 0;
};

std::string sha256_hex(std::string_view bytes);

} // namespace fhegpu
