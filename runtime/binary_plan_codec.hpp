#pragma once
#include "runtime/binary_archive.hpp"
#include "runtime/json_plan_reader.hpp"

namespace fhegpu::binary_io {
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

template <bool R> void value_record(Archive<R> &a, ValueDesc &v) {
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
}

template <bool R> void plan_records(Archive<R> &a, RuntimePlan &p) {
    a.sequence(p.values, 35, [&](ValueDesc &v) { value_record(a, v); });
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

inline void write_plan(std::ostream &out, RuntimePlan &p, const char *magic = "CKKSPL01") {
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
    auto p = RuntimePlanJsonReader::read_text(root).plan;
    a.sequence(a.strings.values, 4, [&](std::string &s) { a.text(s); });
    plan_records(a, p);
    a.finish();
    return p;
}


} // namespace fhegpu::binary_io
