#pragma once

// Version 1 wire format: explicit little-endian fields, independent of ABI.
#include "runtime/plan.hpp"
#include "runtime/json_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <unordered_map>

namespace fhegpu::binary_io {
using json_utils::Json;
static_assert(sizeof(int) == 4 && sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "binary format requires int32 and IEEE float64");

struct Strings {
    std::vector<std::string> values;
    std::unordered_map<std::string, std::uint32_t> indices;
    void add(const std::string &value) {
        if (indices.count(value))
            return;
        if (values.size() >= std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("too many strings");
        indices.emplace(value, static_cast<std::uint32_t>(values.size()));
        values.push_back(value);
    }
};

template <bool Reading> class Archive {
    using Stream = std::conditional_t<Reading, std::istream, std::ostream>;
    Stream *stream;
    std::uint64_t remaining;
    const char *cursor_ = nullptr;
    const Strings *shared_strings_ = nullptr;

  public:
    Strings strings;
    explicit Archive(Stream &stream, std::uint64_t size = 0) : stream(&stream), remaining(size) {}
    explicit Archive(std::string_view bytes, const Strings *strings = nullptr)
        : stream(nullptr), remaining(bytes.size()), cursor_(bytes.data()), shared_strings_(strings) {
        static_assert(Reading, "memory cursor is read-only");
    }
    const char *cursor() const { return cursor_; }
    std::uint64_t available() const { return remaining; }
    void skip(std::uint64_t size) {
        static_assert(Reading, "skip is read-only");
        if (!cursor_ || size > remaining) throw std::runtime_error("truncated binary record");
        cursor_ += size;
        remaining -= size;
    }
    void bytes(char *data, std::size_t size) {
        if constexpr (Reading) {
            if (size > remaining)
                throw std::runtime_error("truncated binary record");
            if (cursor_) { std::memcpy(data, cursor_, size); cursor_ += size; }
            else stream->read(data, size);
            remaining -= size;
        } else
            stream->write(data, size);
        if (stream && !*stream)
            throw std::runtime_error("binary I/O failed");
    }
    template <class T> void number(T &value) {
        static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>);
        using U = std::make_unsigned_t<T>;
        unsigned char data[sizeof(T)];
        U bits = 0;
        if constexpr (!Reading) {
            std::memcpy(&bits, &value, sizeof(T));
            for (std::size_t i = 0; i < sizeof(T); ++i)
                data[i] = static_cast<unsigned char>(bits >> (8 * i));
        }
        bytes(reinterpret_cast<char *>(data), sizeof(T));
        if constexpr (Reading) {
            for (std::size_t i = 0; i < sizeof(T); ++i)
                bits |= U(data[i]) << (8 * i);
            std::memcpy(&value, &bits, sizeof(T));
        }
    }
    void boolean(bool &value) {
        std::uint8_t encoded = value ? 1 : 0;
        number(encoded);
        if (encoded > 1)
            throw std::runtime_error("invalid binary boolean");
        if constexpr (Reading)
            value = encoded != 0;
    }
    template <class E> void enumeration(E &value, std::uint8_t maximum) {
        std::uint8_t encoded = static_cast<std::uint8_t>(value);
        number(encoded);
        if (encoded > maximum)
            throw std::runtime_error("invalid binary enum");
        if constexpr (Reading)
            value = static_cast<E>(encoded);
    }
    void real(double &value) {
        std::uint64_t bits = 0;
        if constexpr (!Reading)
            std::memcpy(&bits, &value, 8);
        number(bits);
        if constexpr (Reading)
            std::memcpy(&value, &bits, 8);
        if (!std::isfinite(value))
            throw std::runtime_error("nonfinite binary float64");
    }
    void text(std::string &value) {
        if (value.size() > 1024 * 1024)
            throw std::runtime_error("binary string exceeds 1 MiB");
        std::uint32_t size = static_cast<std::uint32_t>(value.size());
        number(size);
        if constexpr (Reading) {
            if (size > remaining || size > 1024 * 1024)
                throw std::runtime_error("invalid binary string length");
            value.resize(size);
        }
        bytes(value.data(), size);
    }
    void string_ref(std::string &value) {
        std::uint32_t index = 0;
        if constexpr (!Reading)
            index = strings.indices.at(value);
        number(index);
        const auto &table = shared_strings_ ? *shared_strings_ : strings;
        if (index >= table.values.size())
            throw std::runtime_error("invalid binary string reference");
        if constexpr (Reading)
            value = table.values[index];
    }
    template <class T, class Visit> void sequence(std::vector<T> &values, std::size_t minimum_bytes, Visit visit) {
        std::uint64_t size = values.size();
        number(size);
        if constexpr (Reading) {
            // The reader bounds individual typed arrays to 8 GiB and
            // rejects impossible counts before allocating them.
            if (size > remaining / minimum_bytes || size > (8ULL << 30) / sizeof(T))
                throw std::runtime_error("invalid binary array length");
            values.resize(static_cast<std::size_t>(size));
        }
        for (auto &value : values)
            visit(value);
    }
    void check_count(std::uint64_t count, std::size_t minimum_bytes) const {
        if constexpr (Reading)
            if (count > remaining / minimum_bytes)
                throw std::runtime_error("invalid binary record count");
    }
    void magic(const char *expected) {
        char value[8];
        std::memcpy(value, expected, 8);
        bytes(value, 8);
        if (std::memcmp(value, expected, 8))
            throw std::runtime_error("unknown binary format");
        std::uint32_t version = 1;
        number(version);
        if (version != 1)
            throw std::runtime_error("unsupported binary version");
    }
    void finish() {
        if constexpr (Reading) {
            if (remaining || (stream && stream->peek() != std::char_traits<char>::eof()))
                throw std::runtime_error("trailing binary bytes");
        } else if (!*stream)
            throw std::runtime_error("binary write failed");
    }
};


} // namespace fhegpu::binary_io
