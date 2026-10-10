#pragma once

#include "runtime/utils/sha256.hpp"

#include <nlohmann/json.hpp>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace fhegpu::json_utils {

using Json = nlohmann::json;

[[noreturn]] inline void fail(const std::string &document, const std::string &path,
                              const std::string &message) {
    throw std::runtime_error(document + " JSON error at " + path + ": " + message);
}

inline std::string item_path(const std::string &path, std::size_t index) {
    return path + '[' + std::to_string(index) + ']';
}

inline bool contains_name(std::initializer_list<const char *> names, const std::string &name) {
    for (const char *candidate : names) if (name == candidate) return true;
    return false;
}

inline void require_members(const Json &value, const std::string &document,
                            const std::string &path,
                            std::initializer_list<const char *> names,
                            std::initializer_list<const char *> optional_names = {}) {
    if (!value.is_object()) fail(document, path, "expected object");
    for (const char *name : names)
        if (!value.contains(name)) fail(document, path, "missing required field '" + std::string(name) + "'");
    for (const auto &item : value.items())
        if (!contains_name(names, item.key()) && !contains_name(optional_names, item.key()))
            fail(document, path, "unknown field '" + item.key() + "'");
}

inline std::string read_string(const Json &value, const std::string &document,
                               const std::string &path, bool allow_empty = false) {
    if (!value.is_string()) fail(document, path, "expected string");
    const std::string result = value.get<std::string>();
    if (!allow_empty && result.empty()) fail(document, path, "string must not be empty");
    return result;
}

inline bool read_bool(const Json &value, const std::string &document, const std::string &path) {
    if (!value.is_boolean()) fail(document, path, "expected boolean");
    return value.get<bool>();
}

inline int read_int(const Json &value, const std::string &document, const std::string &path,
                    int minimum, int maximum) {
    if (value.is_number_unsigned()) {
        const auto number = value.get<std::uint64_t>();
        if (number > static_cast<std::uint64_t>(maximum)) fail(document, path, "integer is outside the supported range");
        return static_cast<int>(number);
    }
    if (!value.is_number_integer()) fail(document, path, "expected integer");
    const auto number = value.get<std::int64_t>();
    if (number < minimum || number > maximum) fail(document, path, "integer is outside the supported range");
    return static_cast<int>(number);
}

inline int read_nonnegative_int(const Json &value, const std::string &document,
                                const std::string &path) {
    return read_int(value, document, path, 0, std::numeric_limits<int>::max());
}

inline int read_positive_int(const Json &value, const std::string &document,
                             const std::string &path) {
    return read_int(value, document, path, 1, std::numeric_limits<int>::max());
}

inline std::uint64_t read_id(const Json &value, const std::string &document,
                             const std::string &path) {
    const std::string encoded = read_string(value, document, path);
    if (encoded != "0") {
        if (encoded.front() < '1' || encoded.front() > '9')
            fail(document, path, "identifier must be canonical unsigned decimal");
        for (char c : encoded) if (c < '0' || c > '9')
            fail(document, path, "identifier must be canonical unsigned decimal");
    }
    std::uint64_t result = 0;
    const auto parsed = std::from_chars(encoded.data(), encoded.data() + encoded.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != encoded.data() + encoded.size())
        fail(document, path, "identifier is outside uint64 range");
    return result;
}

inline std::uint64_t read_safe_uint(const Json &value, const std::string &document,
                                    const std::string &path, std::uint64_t minimum,
                                    std::uint64_t maximum) {
    if (!value.is_number_unsigned() && !value.is_number_integer())
        fail(document, path, "expected integer");
    if (value.is_number_integer() && value.get<std::int64_t>() < 0)
        fail(document, path, "integer is outside the supported range");
    const auto number = value.get<std::uint64_t>();
    if (number < minimum || number > maximum)
        fail(document, path, "integer is outside the supported range");
    return number;
}

inline double read_finite_double(const Json &value, const std::string &document,
                                 const std::string &path) {
    if (!value.is_number()) fail(document, path, "expected number");
    double result = 0.0;
    try { result = value.get<double>(); }
    catch (const Json::exception &) { fail(document, path, "number is outside float64 range"); }
    if (!std::isfinite(result)) fail(document, path, "number must be finite float64");
    return result == 0.0 ? 0.0 : result;
}

inline std::string read_sha256(const Json &value, const std::string &document,
                               const std::string &path) {
    const std::string digest = read_string(value, document, path);
    if (digest.size() != 71 || digest.compare(0, 7, "sha256:") != 0)
        fail(document, path, "expected sha256 followed by 64 lowercase hexadecimal digits");
    for (std::size_t i = 7; i < digest.size(); ++i) {
        const char c = digest[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            fail(document, path, "expected sha256 followed by 64 lowercase hexadecimal digits");
    }
    return digest;
}

// Use the public SAX interface. The DOM callback parser scans the complete
// parent array at every object_end, even when the callback discards nothing.
class StrictJsonSax : public nlohmann::json_sax<Json> {
public:
    explicit StrictJsonSax(const std::string &document) : document_(document) {}

    bool null() override { return value(nullptr); }
    bool boolean(bool v) override { return value(v); }
    bool number_integer(Json::number_integer_t v) override { return value(v); }
    bool number_unsigned(Json::number_unsigned_t v) override { return value(v); }
    bool number_float(Json::number_float_t v, const std::string &) override { return value(v); }
    bool string(std::string &v) override { return value(std::move(v)); }
    bool binary(Json::binary_t &v) override { return value(std::move(v)); }
    bool start_object(std::size_t) override {
        frames_.push_back({Json::object(), {}});
        return true;
    }
    bool start_array(std::size_t) override {
        frames_.push_back({Json::array(), {}});
        return true;
    }
    bool key(std::string &v) override {
        auto &frame = frames_.back();
        if (frame.container.contains(v))
            fail(document_, "$", "duplicate object key '" + v + "'");
        frame.key = std::move(v);
        return true;
    }
    bool end_object() override { return end_container(); }
    bool end_array() override { return end_container(); }
    bool parse_error(std::size_t position, const std::string &,
                     const Json::exception &error) override {
        fail(document_, "byte " + std::to_string(position), error.what());
    }
    Json take_result() { return std::move(result_); }

private:
    struct Frame { Json container; std::string key; };
    bool value(Json v) {
        if (frames_.empty()) result_ = std::move(v);
        else {
            auto &frame = frames_.back();
            if (frame.container.is_array()) frame.container.push_back(std::move(v));
            else frame.container.emplace(std::move(frame.key), std::move(v));
        }
        return true;
    }
    bool end_container() {
        Json v = std::move(frames_.back().container);
        frames_.pop_back();
        return value(std::move(v));
    }
    std::string document_;
    std::vector<Frame> frames_;
    Json result_;
};

inline Json parse(std::string_view text, const std::string &document) {
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xef &&
        static_cast<unsigned char>(text[1]) == 0xbb &&
        static_cast<unsigned char>(text[2]) == 0xbf)
        fail(document, "$", "UTF-8 BOM is not allowed");
    StrictJsonSax sax(document);
    Json::sax_parse(text.begin(), text.end(), &sax);
    return sax.take_result();
}

inline std::string read_file_bytes(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open file: " + path);
    std::string result{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (input.bad()) throw std::runtime_error("failed to read file: " + path);
    return result;
}

inline std::string source_sha256(std::string_view bytes) {
    return "sha256:" + sha256_hex(bytes);
}

} // namespace fhegpu::json_utils
