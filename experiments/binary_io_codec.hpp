#pragma once

// Experimental, versioned wire encoding. This is not a production file format.
#include "runtime/json_plan_reader.hpp"
#include "runtime/json_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <unordered_map>

namespace fhegpu::binary_io_experiment {
using json_utils::Json;
static_assert(sizeof(int) == 4 && sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "experiment requires int32 and IEEE float64");

inline Json metadata(const RuntimePlan &p) {
    Json result{{"format_version", p.format_version},
                {"plan_id", std::to_string(p.plan_id)},
                {"target",
                 {{"target_id", p.target.target_id},
                  {"world_size", p.target.world_size},
                  {"device_counts", p.target.device_counts},
                  {"capability_version", p.target.capability_version},
                  {"operator_spec",
                   {{"id", p.target.operator_spec.id},
                    {"version", p.target.operator_spec.version},
                    {"source_sha256", p.target.operator_spec.source_sha256}}}}}};
    if (p.plaintext_bundle)
        result["plaintext_bundle"] = {{"id", p.plaintext_bundle->id},
                                      {"version", p.plaintext_bundle->version},
                                      {"manifest_sha256", p.plaintext_bundle->manifest_sha256}};
    for (const char *key :
         {"values", "external_inputs", "initialization", "execution", "finalization", "final_outputs"})
        result[key] = Json::array();
    return result;
}

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
    Stream &stream;
    std::uint64_t remaining;

  public:
    Strings strings;
    explicit Archive(Stream &stream, std::uint64_t size = 0) : stream(stream), remaining(size) {}
    void bytes(char *data, std::size_t size) {
        if constexpr (Reading) {
            if (size > remaining)
                throw std::runtime_error("truncated binary record");
            stream.read(data, size);
            remaining -= size;
        } else
            stream.write(data, size);
        if (!stream)
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
            throw std::runtime_error("experiment string exceeds 1 MiB");
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
        if (index >= strings.values.size())
            throw std::runtime_error("invalid binary string reference");
        if constexpr (Reading)
            value = strings.values[index];
    }
    template <class T, class Visit> void sequence(std::vector<T> &values, std::size_t minimum_bytes, Visit visit) {
        std::uint64_t size = values.size();
        number(size);
        if constexpr (Reading) {
            // This experiment bounds individual typed arrays to 8 GiB and
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
            throw std::runtime_error("unknown experiment binary format");
        std::uint32_t version = 1;
        number(version);
        if (version != 1)
            throw std::runtime_error("unsupported experiment binary version");
    }
    void finish() {
        if constexpr (Reading) {
            if (remaining || stream.peek() != std::char_traits<char>::eof())
                throw std::runtime_error("trailing binary bytes");
        } else if (!stream)
            throw std::runtime_error("binary write failed");
    }
};

template <bool R> void place(Archive<R> &a, Place &p) {
    a.enumeration(p.kind, 1);
    a.number(p.rank);
    a.number(p.index);
    if (p.rank < 0 || p.index < 0 || (p.kind == PlaceKind::Host && p.index != 0))
        throw std::runtime_error("invalid binary place");
}

template <bool R> void instruction(Archive<R> &a, Instruction &i, std::uint32_t version) {
    std::uint64_t ordinal = i.ordinal;
    a.number(ordinal);
    if (ordinal > std::numeric_limits<int>::max())
        throw std::runtime_error("binary ordinal out of range");
    i.ordinal = static_cast<std::size_t>(ordinal);
    std::uint8_t tag = static_cast<std::uint8_t>(i.body.index());
    a.number(tag);
    switch (tag) {
    case 0: {
        if constexpr (R)
            i.body = EncodeOp{};
        auto &op = std::get<EncodeOp>(i.body);
        a.number(op.output);
        std::uint8_t payload = static_cast<std::uint8_t>(op.payload.index());
        a.number(payload);
        if (payload == 0) {
            if constexpr (R)
                op.payload = InlineEncodePayload{};
            auto &values = std::get<InlineEncodePayload>(op.payload).values;
            a.sequence(values, 8, [&](double &v) { a.real(v); });
            if (values.empty())
                throw std::runtime_error("empty inline binary payload");
        } else if (payload == 1) {
            if constexpr (R)
                op.payload = BundleEncodePayload{};
            auto &content = std::get<BundleEncodePayload>(op.payload).content;
            a.string_ref(content);
            json_utils::read_sha256(Json(content), "binary plan", "content");
        } else
            throw std::runtime_error("unknown binary payload tag");
        break;
    }
    case 1: {
        if constexpr (R)
            i.body = ComputeOp{};
        auto &op = std::get<ComputeOp>(i.body);
        a.enumeration(op.kind, 11);
        a.sequence(op.inputs, 8, [&](ValueId &id) { a.number(id); });
        a.number(op.output);
        place(a, op.place);
        std::uint8_t attrs = static_cast<std::uint8_t>(op.attrs.index());
        a.number(attrs);
        const std::uint8_t expected = op.kind == ComputeKind::Rotate      ? 1
                                      : op.kind == ComputeKind::Rescale   ? 2
                                      : op.kind == ComputeKind::ModSwitch ? 3
                                      : op.kind == ComputeKind::Boot      ? 4
                                                                          : 0;
        if (attrs != expected)
            throw std::runtime_error("binary compute attrs mismatch");
        switch (attrs) {
        case 0:
            if constexpr (R)
                op.attrs = std::monostate{};
            break;
        case 1: {
            if constexpr (R)
                op.attrs = RotateAttrs{};
            a.number(std::get<RotateAttrs>(op.attrs).steps);
            break;
        }
        case 2: {
            if constexpr (R)
                op.attrs = RescaleAttrs{};
            auto &v = std::get<RescaleAttrs>(op.attrs);
            a.number(v.target_level);
            a.number(v.target_scale_log2);
            if (v.target_level < 0 || v.target_scale_log2 < 0)
                throw std::runtime_error("invalid binary rescale attrs");
            break;
        }
        case 3: {
            if constexpr (R)
                op.attrs = ModSwitchAttrs{};
            auto &v = std::get<ModSwitchAttrs>(op.attrs);
            a.number(v.target_level);
            if (v.target_level < 0)
                throw std::runtime_error("invalid binary modswitch attrs");
            break;
        }
        case 4: {
            if constexpr (R)
                op.attrs = BootAttrs{};
            auto &v = std::get<BootAttrs>(op.attrs);
            a.number(v.target_level);
            a.number(v.target_scale_log2);
            a.number(v.target_components);
            a.string_ref(v.operator_profile);
            a.enumeration(v.implementation, 1);
            if (v.target_level < 0 || v.target_scale_log2 < 0 || v.target_components < 1)
                throw std::runtime_error("invalid binary boot attrs");
            break;
        }
        }
        bool reuse = op.reuse_input.has_value();
        a.boolean(reuse);
        if (reuse) {
            if (version < 2)
                throw std::runtime_error("binary reuse requires plan V2");
            std::uint64_t index = op.reuse_input.value_or(0);
            a.number(index);
            if (index > std::numeric_limits<int>::max())
                throw std::runtime_error("binary reuse index out of range");
            op.reuse_input = static_cast<std::size_t>(index);
        }
        break;
    }
    case 2: {
        if constexpr (R)
            i.body = CommAction{};
        auto &op = std::get<CommAction>(i.body);
        a.number(op.id);
        a.enumeration(op.kind, 1);
        a.enumeration(op.hint, 5);
        a.sequence(op.inputs, 8, [&](ValueId &v) { a.number(v); });
        a.sequence(op.outputs, 8, [&](ValueId &v) { a.number(v); });
        a.sequence(op.sources, 9, [&](Place &v) { place(a, v); });
        a.sequence(op.destinations, 9, [&](Place &v) { place(a, v); });
        a.sequence(op.output_types, 1, [&](ValueKind &v) { a.enumeration(v, 1); });
        break;
    }
    case 3:
        if (version < 2)
            throw std::runtime_error("binary Release requires plan V2");
        if constexpr (R)
            i.body = ReleaseOp{};
        a.number(std::get<ReleaseOp>(i.body).value);
        break;
    case 4:
        if (version < 3)
            throw std::runtime_error("binary Fence requires plan V3");
        if constexpr (R)
            i.body = FenceOp{};
        break;
    default:
        throw std::runtime_error("unknown binary instruction tag");
    }
}

template <bool R> void plan_records(Archive<R> &a, RuntimePlan &p) {
    a.sequence(p.values, 35, [&](ValueDesc &v) {
        a.number(v.id);
        a.enumeration(v.kind, 1);
        place(a, v.place);
        a.string_ref(v.context);
        a.number(v.level);
        a.number(v.scale_log2);
        a.boolean(v.ntt);
        a.number(v.components);
        if (v.level < 0 || v.scale_log2 < 0 || v.components < 1)
            throw std::runtime_error("invalid binary value metadata");
    });
    a.sequence(p.external_inputs, 8, [&](ValueId &v) { a.number(v); });
    for (auto *phase : {&p.initialization, &p.execution, &p.finalization})
        a.sequence(*phase, 9, [&](Instruction &i) { instruction(a, i, p.format_version); });
    a.sequence(p.final_outputs, 8, [&](ValueId &v) { a.number(v); });
}

inline Strings collect_strings(const RuntimePlan &p) {
    Strings strings;
    for (const auto &v : p.values)
        strings.add(v.context);
    for (const auto *phase : {&p.initialization, &p.execution, &p.finalization})
        for (const auto &i : *phase) {
            if (const auto *op = std::get_if<EncodeOp>(&i.body)) {
                if (const auto *payload = std::get_if<BundleEncodePayload>(&op->payload))
                    strings.add(payload->content);
            } else if (const auto *op = std::get_if<ComputeOp>(&i.body)) {
                if (const auto *attrs = std::get_if<BootAttrs>(&op->attrs))
                    strings.add(attrs->operator_profile);
            }
        }
    return strings;
}

inline void write_plan(std::ostream &out, RuntimePlan &p) {
    Archive<false> a(out);
    a.magic("PLNEXP01");
    auto root = metadata(p).dump();
    a.text(root);
    a.strings = collect_strings(p);
    a.sequence(a.strings.values, 4, [&](std::string &s) { a.text(s); });
    plan_records(a, p);
    a.finish();
}

inline RuntimePlan read_plan(std::istream &in, std::uint64_t size) {
    Archive<true> a(in, size);
    a.magic("PLNEXP01");
    std::string root;
    a.text(root);
    auto p = RuntimePlanJsonReader::read_text(root).plan;
    a.sequence(a.strings.values, 4, [&](std::string &s) { a.text(s); });
    plan_records(a, p);
    a.finish();
    return p;
}

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
        throw std::runtime_error("experiment manifest requires pack storage V2");
    read_string(m.metadata.at("bundle_id"), "manifest", "id");
    read_positive_int(m.metadata.at("version"), "manifest", "version");
    if (!m.metadata.at("blobs").is_array() || !m.metadata.at("blobs").empty())
        throw std::runtime_error("experiment manifest metadata must have an empty blobs placeholder");
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
    StrictJsonSax sax("experiment manifest", {"blobs"}, [&](const std::string &, std::size_t, Json &&e) {
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

template <bool R> void manifest_records(Archive<R> &a, Manifest &m) {
    a.magic("MNFEXP01");
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
} // namespace fhegpu::binary_io_experiment
