#include "runtime/utils/sha256.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace fhegpu {
namespace {

constexpr std::array<std::uint32_t, 64> round_constants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

constexpr std::uint32_t rotate_right(std::uint32_t value, unsigned int bits) {
    return (value >> bits) | (value << (32U - bits));
}

std::uint32_t load_big_endian(const unsigned char *bytes) {
    return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
           (static_cast<std::uint32_t>(bytes[1]) << 16U) |
           (static_cast<std::uint32_t>(bytes[2]) << 8U) |
           static_cast<std::uint32_t>(bytes[3]);
}

void transform(std::array<std::uint32_t, 8> &state, const unsigned char *block) {
    std::array<std::uint32_t, 64> schedule{};
    for (std::size_t index = 0; index < 16; ++index)
        schedule[index] = load_big_endian(block + index * 4);
    for (std::size_t index = 16; index < schedule.size(); ++index) {
        const std::uint32_t s0 = rotate_right(schedule[index - 15], 7) ^
                                 rotate_right(schedule[index - 15], 18) ^
                                 (schedule[index - 15] >> 3U);
        const std::uint32_t s1 = rotate_right(schedule[index - 2], 17) ^
                                 rotate_right(schedule[index - 2], 19) ^
                                 (schedule[index - 2] >> 10U);
        schedule[index] = schedule[index - 16] + s0 + schedule[index - 7] + s1;
    }

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];

    for (std::size_t index = 0; index < schedule.size(); ++index) {
        const std::uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const std::uint32_t choose = (e & f) ^ (~e & g);
        const std::uint32_t temp1 = h + sum1 + choose + round_constants[index] + schedule[index];
        const std::uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = sum0 + majority;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

} // namespace

std::string sha256_hex(std::string_view bytes) {
    if (bytes.size() > std::numeric_limits<std::uint64_t>::max() / 8U)
        throw std::length_error("SHA-256 input is too large");

    std::array<std::uint32_t, 8> state = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };

    const auto *data = reinterpret_cast<const unsigned char *>(bytes.data());
    const std::size_t full_blocks = bytes.size() / 64;
    for (std::size_t index = 0; index < full_blocks; ++index)
        transform(state, data + index * 64);

    const std::size_t remainder = bytes.size() % 64;
    std::array<unsigned char, 128> tail{};
    if (remainder != 0) std::memcpy(tail.data(), data + full_blocks * 64, remainder);
    tail[remainder] = 0x80;

    const std::size_t tail_size = remainder < 56 ? 64 : 128;
    const std::uint64_t bit_length = static_cast<std::uint64_t>(bytes.size()) * 8U;
    for (std::size_t index = 0; index < 8; ++index)
        tail[tail_size - 1 - index] = static_cast<unsigned char>(bit_length >> (index * 8U));
    transform(state, tail.data());
    if (tail_size == 128) transform(state, tail.data() + 64);

    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (std::uint32_t word : state) {
        for (int shift = 28; shift >= 0; shift -= 4)
            result.push_back(hex[(word >> static_cast<unsigned int>(shift)) & 0x0fU]);
    }
    return result;
}

} // namespace fhegpu
