#pragma once
#include "runtime/binary_archive.hpp"
#include "runtime/json_plan_reader.hpp"

namespace fhegpu::binary_io {
inline Json metadata(const RuntimePlan &p) {
    if (p.format_version != 1 && p.format_version != 2 && p.format_version != 3 &&
        p.format_version != 4 && p.format_version != 5 && p.format_version != 7 && p.format_version != 8)
        throw std::runtime_error("unsupported plan format version");
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
    if (p.waterline) {
        const auto &c = *p.waterline;
        result["waterline"] = {{"device_capacity_bytes", c.device_capacity_bytes},
            {"device_reserve_bytes", c.device_reserve_bytes}, {"device_fractions", c.device_fractions},
            {"host_budget_bytes", c.host_budget_bytes}, {"pinned_budget_bytes", c.pinned_budget_bytes},
            {"raw_budget_bytes", c.raw_budget_bytes}, {"lookahead_groups", c.lookahead_groups},
            {"backend", c.backend == PrepareBackend::Cpu ? "cpu" : c.backend == PrepareBackend::Gpu ? "gpu" : "auto"}};
    }
    if (p.plaintext_bundle) {
        result["plaintext_bundle"] = {{"id", p.plaintext_bundle->id},
                                      {"version", p.plaintext_bundle->version},
                                      {"manifest_sha256", p.plaintext_bundle->manifest_sha256}};
        if (p.plaintext_bundle->manifest_format == "binary") {
            result["plaintext_bundle"].erase("manifest_sha256");
            result["plaintext_bundle"]["manifest_format"] = "binary";
        }
    }
    for (const char *key :
         {"values", "external_inputs", "initialization", "execution", "finalization", "final_outputs"})
        result[key] = Json::array();
    return result;
}

template <bool R> void scale_record(Archive<R> &a, double &scale, std::uint32_t version) {
    if (version >= 5) a.real(scale);
    else {
        if constexpr (!R)
            if (!std::isfinite(scale) || scale < 0 || scale > std::numeric_limits<int>::max() || std::floor(scale) != scale)
                throw std::runtime_error("fractional scales require plan V5");
        int nominal = static_cast<int>(scale);
        a.number(nominal);
        scale = nominal;
    }
    if (!std::isfinite(scale) || scale < 0 || scale > 1023)
        throw std::runtime_error("invalid binary scale");
}

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
    case 5:
        if (version < 4) throw std::runtime_error("binary PreparePlaintext requires plan V4");
        [[fallthrough]];
    case 0: {
        if constexpr (!R) {
            if (tag == 5) {
                const auto &prepare = std::get<PreparePlaintextOp>(i.body);
                Instruction temporary{i.ordinal, EncodeOp{prepare.payload, prepare.output}};
                // Emit the shared payload record below, with the original tag.
                auto &op = std::get<EncodeOp>(temporary.body);
                a.number(op.output);
                std::uint8_t payload = static_cast<std::uint8_t>(op.payload.index());
                a.number(payload);
                if (payload == 0) {
                    auto &values = std::get<InlineEncodePayload>(op.payload).values;
                    a.sequence(values, 8, [&](double &v) { a.real(v); });
                } else a.string_ref(std::get<BundleEncodePayload>(op.payload).content);
                break;
            }
        }
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
        if (tag == 5) {
            auto encoded = std::get<EncodeOp>(std::move(i.body));
            i.body = PreparePlaintextOp{std::move(encoded.payload), encoded.output};
        }
        break;
    }
    case 1: {
        if constexpr (R)
            i.body = ComputeOp{};
        auto &op = std::get<ComputeOp>(i.body);
        a.enumeration(op.kind, 13);
        if (static_cast<int>(op.kind) == 12)
            throw std::runtime_error("reserved binary compute kind 12");
        if (op.kind == ComputeKind::ApiCall && version < 7)
            throw std::runtime_error("binary ApiCall requires plan V7");
        a.sequence(op.inputs, 8, [&](ValueId &id) { a.number(id); });
        a.number(op.output);
        place(a, op.place);
        // Attribute wire tag 5 is reserved for the separate V6 BootStep protocol.
        std::uint8_t attrs = std::holds_alternative<ApiCallAttrs>(op.attrs)
                                 ? 6 : static_cast<std::uint8_t>(op.attrs.index());
        a.number(attrs);
        const std::uint8_t expected = op.kind == ComputeKind::Rotate      ? 1
                                      : op.kind == ComputeKind::Rescale   ? 2
                                      : op.kind == ComputeKind::ModSwitch ? 3
                                      : op.kind == ComputeKind::Boot      ? 4
                                      : op.kind == ComputeKind::ApiCall   ? 6
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
            scale_record(a, v.target_scale_log2, version);
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
        case 6: {
            if constexpr (R)
                op.attrs = ApiCallAttrs{};
            auto &v = std::get<ApiCallAttrs>(op.attrs);
            a.string_ref(v.callee);
            a.number(v.call_id);
            if (v.callee.empty()) throw std::runtime_error("binary ApiCall callee must not be empty");
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
        throw std::runtime_error("unknown binary instruction tag " + std::to_string(tag) +
                                 " at ordinal " + std::to_string(i.ordinal));
    }
}

inline std::size_t value_record_width(std::uint32_t version) {
    return version >= 8 ? 51 : version >= 5 ? 39 : 35;
}

template <bool R> void value_record(Archive<R> &a, ValueDesc &v, std::uint32_t version) {
    if constexpr (!R)
        if (v.kind == ValueKind::Buffer && version < 8)
            throw std::runtime_error("binary Buffer requires plan V8");
    a.number(v.id);
    a.enumeration(v.kind, version >= 8 ? 2 : 1);
    place(a, v.place);
    a.string_ref(v.context);
    a.number(v.level);
    scale_record(a, v.scale_log2, version);
    a.boolean(v.ntt);
    a.number(v.components);
    if (version >= 8) {
        a.number(v.buffer_bytes);
        a.string_ref(v.buffer_layout);
    }
    if (v.kind == ValueKind::Buffer) {
        if (v.level != 0 || v.scale_log2 != 0 || v.ntt || v.components != 0 || v.buffer_layout.empty())
            throw std::runtime_error("invalid binary Buffer metadata: expected zero crypto placeholders and non-empty buffer_layout");
    } else {
        if (v.level < 0 || v.scale_log2 < 0 || v.components < 1)
            throw std::runtime_error("invalid binary value metadata");
        if (v.buffer_bytes != 0 || !v.buffer_layout.empty())
            throw std::runtime_error("non-Buffer value has buffer metadata");
    }
}

template <bool R> void plan_records(Archive<R> &a, RuntimePlan &p) {
    a.sequence(p.values, value_record_width(p.format_version), [&](ValueDesc &v) { value_record(a, v, p.format_version); });
    a.sequence(p.external_inputs, 8, [&](ValueId &v) { a.number(v); });
    for (auto *phase : {&p.initialization, &p.execution, &p.finalization})
        a.sequence(*phase, 9, [&](Instruction &i) { instruction(a, i, p.format_version); });
    a.sequence(p.final_outputs, 8, [&](ValueId &v) { a.number(v); });
}

inline Strings collect_strings(const RuntimePlan &p) {
    Strings strings;
    for (const auto &v : p.values) {
        strings.add(v.context);
        if (p.format_version >= 8) strings.add(v.buffer_layout);
    }
    for (const auto *phase : {&p.initialization, &p.execution, &p.finalization})
        for (const auto &i : *phase) {
            if (const auto *op = std::get_if<EncodeOp>(&i.body)) {
                if (const auto *payload = std::get_if<BundleEncodePayload>(&op->payload))
                    strings.add(payload->content);
            } else if (const auto *op = std::get_if<PreparePlaintextOp>(&i.body)) {
                if (const auto *payload = std::get_if<BundleEncodePayload>(&op->payload)) strings.add(payload->content);
            } else if (const auto *op = std::get_if<ComputeOp>(&i.body)) {
                if (const auto *attrs = std::get_if<BootAttrs>(&op->attrs))
                    strings.add(attrs->operator_profile);
                else if (const auto *attrs = std::get_if<ApiCallAttrs>(&op->attrs))
                    strings.add(attrs->callee);
            }
        }
    return strings;
}

inline void check_v8_version(const RuntimePlan &p) {
    const bool requires_v8 = plan_requires_v8(p);
    if (requires_v8 && p.format_version != 8) {
        if (std::any_of(p.values.begin(), p.values.end(), [](const ValueDesc &v) { return v.kind == ValueKind::Buffer; }))
            throw std::runtime_error("binary Buffer requires plan V8");
        throw std::runtime_error("Device ciphertext external inputs require plan V8");
    }
    if (!requires_v8 && p.format_version == 8)
        throw std::runtime_error("format version 8 requires Buffer values or external Device ciphertext");
}

inline void write_plan(std::ostream &out, RuntimePlan &p, const char *magic = "CKKSPL01") {
    check_v8_version(p);
    Archive<false> a(out);
    a.magic(magic);
    auto root = metadata(p).dump();
    a.text(root);
    a.strings = collect_strings(p);
    a.sequence(a.strings.values, 4, [&](std::string &s) { a.text(s); });
    plan_records(a, p);
    a.finish();
}

inline RuntimePlan read_plan(std::istream &in, std::uint64_t size, const char *magic = "CKKSPL01") {
    Archive<true> a(in, size);
    a.magic(magic);
    std::string root;
    a.text(root);
    auto p = RuntimePlanJsonReader::read_binary_metadata_text(root);
    a.sequence(a.strings.values, 4, [&](std::string &s) { a.text(s); });
    plan_records(a, p);
    a.finish();
    check_v8_version(p);
    return p;
}


} // namespace fhegpu::binary_io
