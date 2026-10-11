#pragma once
#include "runtime/binary_archive.hpp"

namespace fhegpu::binary_io {
struct ManifestEntry {
    std::uint64_t offset, length;
};
struct Manifest {
    Json metadata;
    std::unordered_map<std::string, ManifestEntry> entries;
};

inline void validate_manifest(const Manifest &m) {
    using namespace json_utils;
    require_members(m.metadata, "manifest", "$",
                    {"bundle_format_version", "bundle_id", "version", "pack_byte_length", "blobs"});
    if (read_nonnegative_int(m.metadata.at("bundle_format_version"), "manifest", "version") != 2)
        throw std::runtime_error("binary manifest requires pack storage V2");
    read_string(m.metadata.at("bundle_id"), "manifest", "id");
    read_positive_int(m.metadata.at("version"), "manifest", "version");
    if (!m.metadata.at("blobs").is_array() || !m.metadata.at("blobs").empty())
        throw std::runtime_error("binary manifest metadata must have an empty blobs placeholder");
    const auto total = read_safe_uint(m.metadata.at("pack_byte_length"), "manifest", "length", 0, (1ULL << 53) - 1);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    ranges.reserve(m.entries.size());
    for (const auto &[content, e] : m.entries) {
        if (e.length < 8 || e.length % 8 || e.offset % 8 || e.offset > total || e.length > total - e.offset)
            throw std::runtime_error("invalid binary manifest range");
        ranges.emplace_back(e.offset, e.length);
    }
    std::sort(ranges.begin(), ranges.end());
    std::uint64_t end = 0;
    for (const auto &[offset, length] : ranges) {
        if (offset != end)
            throw std::runtime_error("manifest gaps or overlap");
        end += length;
    }
    if (end != total)
        throw std::runtime_error("manifest length mismatch");
}

inline Manifest read_json_manifest(std::istream &in) {
    using namespace json_utils;
    Manifest m;
    StrictJsonSax sax("binary manifest", {"blobs"}, [&](const std::string &, std::size_t, Json &&e) {
        require_members(e, "manifest", "$", {"content", "offset", "byte_length"});
        auto content = read_sha256(e.at("content"), "manifest", "content");
        const auto offset = read_safe_uint(e.at("offset"), "manifest", "offset", 0, (1ULL << 53) - 1);
        const auto length = read_safe_uint(e.at("byte_length"), "manifest", "length", 8, (1ULL << 53) - 1);
        if (!m.entries.emplace(std::move(content), ManifestEntry{offset, length}).second)
            throw std::runtime_error("duplicate manifest content");
    });
    Json::sax_parse(in, &sax);
    m.metadata = sax.take_result();
    validate_manifest(m);
    return m;
}

template <bool R> void manifest_records(Archive<R> &a, Manifest &m, const char *magic = "CKKSMF01") {
    a.magic(magic);
    std::string root;
    if constexpr (!R)
        root = m.metadata.dump();
    a.text(root);
    if constexpr (R)
        m.metadata = json_utils::parse(root, "binary manifest metadata");
    std::uint64_t count = m.entries.size();
    a.number(count);
    a.check_count(count, 48);
    if constexpr (R)
        m.entries.reserve(static_cast<std::size_t>(count));
    std::vector<std::string> keys;
    if constexpr (!R) {
        for (const auto &[content, e] : m.entries)
            keys.push_back(content);
        std::sort(keys.begin(), keys.end());
    }
    constexpr char hex[] = "0123456789abcdef";
    for (std::uint64_t i = 0; i < count; ++i) {
        char raw[32];
        std::string content = "sha256:";
        content.reserve(71);
        ManifestEntry entry{};
        if constexpr (!R) {
            content = keys[i];
            entry = m.entries.at(content);
            for (int j = 0; j < 32; ++j) {
                const auto digit = [&](char c) {
                    const auto *p = std::strchr(hex, c);
                    if (!p)
                        throw std::runtime_error("invalid digest");
                    return p - hex;
                };
                raw[j] = static_cast<char>(digit(content[7 + 2 * j]) * 16 + digit(content[8 + 2 * j]));
            }
        }
        a.bytes(raw, 32);
        a.number(entry.offset);
        a.number(entry.length);
        if constexpr (R) {
            for (unsigned char c : raw) {
                content += hex[c >> 4];
                content += hex[c & 15];
            }
            if (!m.entries.emplace(std::move(content), entry).second)
                throw std::runtime_error("duplicate binary content");
        }
    }
    a.finish();
    validate_manifest(m);
}
} // namespace fhegpu::binary_io
