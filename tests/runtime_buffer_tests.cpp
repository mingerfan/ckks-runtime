#include "runtime/binary_plan_parallel.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/plan_reader.hpp"
#include "runtime/runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace fhegpu;

namespace {
using Json = json_utils::Json;
void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
template <class Function>
void expect_error(Function function, const std::string &expected) {
    try { function(); }
    catch (const std::exception &error) {
        require(std::string(error.what()).find(expected) != std::string::npos,
                "wrong error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected error: " + expected);
}
Place device() { return {PlaceKind::Device, 0, 0}; }
Place host() { return {PlaceKind::Host, 0, 0}; }
LoadedOperatorSpec operator_spec() {
    LoadedOperatorSpec loaded;
    auto &spec = loaded.spec;
    spec.id = "buffer-api";
    spec.version = 1;
    spec.target_id = "buffer-device";
    spec.context_id = "buffer-context";
    spec.poly_degree = 8;
    spec.rns_moduli_log2 = {40, 40, 40, 40};
    spec.level_lower_bound = 0;
    spec.level_upper_bound = 3;
    loaded.source_sha256 = "sha256:" + std::string(64, '0');
    return loaded;
}
RuntimePlan fixture() {
    RuntimePlan plan;
    plan.format_version = 8;
    plan.plan_id = 8;
    plan.target = {"buffer-device", 1, {1}, 1, {"buffer-api", 1, operator_spec().source_sha256}};
    plan.values = {
        {0, ValueKind::Ciphertext, device(), "buffer-context", 1, 20.25, true, 2, 0, {}},
        {1, ValueKind::Buffer, device(), "buffer-context", 0, 0, false, 0, 4096, "test-hoisted-v1"},
        {2, ValueKind::Ciphertext, device(), "buffer-context", 3, 22.5, true, 2, 0, {}},
        {3, ValueKind::Ciphertext, device(), "buffer-context", 2, 13.75, true, 2, 0, {}},
        {4, ValueKind::Ciphertext, device(), "buffer-context", 0, 8.25, true, 2, 0, {}}};
    plan.execution = {
        {0, ComputeOp{ComputeKind::ApiCall, {}, 0, device(), ApiCallAttrs{"stem", 0}}},
        {1, ComputeOp{ComputeKind::ApiCall, {0}, 1, device(), ApiCallAttrs{"hoist", 1}}},
        {2, ComputeOp{ComputeKind::ApiCall, {0, 1}, 2, device(), ApiCallAttrs{"suffix_left", 2}}},
        {3, ReleaseOp{0}},
        {4, ComputeOp{ComputeKind::ApiCall, {1}, 3, device(), ApiCallAttrs{"suffix_right", 3}}},
        {5, ReleaseOp{1}},
        {6, ComputeOp{ComputeKind::ApiCall, {2, 3}, 4, device(), ApiCallAttrs{"head", 4}}},
        {7, ReleaseOp{2}},
        {8, ReleaseOp{3}}};
    plan.final_outputs = {4};
    return plan;
}
RuntimePlan no_buffer_fixture() {
    auto plan = fixture();
    plan.format_version = 7;
    plan.values[1] = plan.values[0];
    plan.values[1].id = 1;
    return plan;
}
RuntimePlan codec_fixture() {
    auto plan = fixture();
    // Readers do not verify graph uses. Include PT to check its V8 tail too.
    plan.values.push_back({5, ValueKind::Plaintext, device(), "buffer-context", 1, 20.25, true, 1, 0, {}});
    return plan;
}
void renumber(RuntimePlan &plan) {
    std::size_t ordinal = 0;
    for (auto *phase : {&plan.initialization, &plan.execution, &plan.finalization})
        for (auto &instruction : *phase) instruction.ordinal = ordinal++;
}
Json json_fixture(const RuntimePlan &plan) {
    auto root = binary_io::metadata(plan);
    const auto place_json = [](const Place &place) {
        Json result{{"kind", place.kind == PlaceKind::Host ? "host" : "device"}, {"rank", place.rank}};
        if (place.kind == PlaceKind::Device) result["index"] = place.index;
        return result;
    };
    for (const auto &value : plan.values) {
        const char *kind = value.kind == ValueKind::Buffer ? "buffer" :
                           value.kind == ValueKind::Plaintext ? "plaintext" : "ciphertext";
        Json record{{"id", std::to_string(value.id)}, {"kind", kind}, {"place", place_json(value.place)},
                    {"context", value.context}, {"level", value.level}, {"scale_log2", value.scale_log2},
                    {"ntt", value.ntt}, {"components", value.components}};
        if (value.kind == ValueKind::Buffer) {
            record["buffer_bytes"] = std::to_string(value.buffer_bytes);
            record["buffer_layout"] = value.buffer_layout;
        }
        root["values"].push_back(std::move(record));
    }
    for (const auto &instruction : plan.execution) {
        if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) {
            const auto &attrs = std::get<ApiCallAttrs>(op->attrs);
            Json inputs = Json::array();
            for (auto id : op->inputs) inputs.push_back(std::to_string(id));
            root["execution"].push_back({{"ordinal", instruction.ordinal}, {"kind", "compute"},
                {"op", "ApiCall"}, {"inputs", inputs}, {"output", std::to_string(op->output)},
                {"place", place_json(op->place)}, {"attrs", {{"callee", attrs.callee}, {"call_id", std::to_string(attrs.call_id)}}}});
        } else {
            root["execution"].push_back({{"ordinal", instruction.ordinal}, {"kind", "release"},
                {"value", std::to_string(std::get<ReleaseOp>(instruction.body).value)}});
        }
    }
    for (auto id : plan.external_inputs) root["external_inputs"].push_back(std::to_string(id));
    for (auto id : plan.final_outputs) root["final_outputs"].push_back(std::to_string(id));
    return root;
}
std::string binary(RuntimePlan plan) {
    std::ostringstream output(std::ios::binary);
    binary_io::write_plan(output, plan);
    return output.str();
}
void expect_binary_error(const std::string &bytes, const std::string &expected) {
    expect_error([&] {
        std::istringstream input(bytes, std::ios::binary);
        binary_io::read_plan(input, bytes.size());
    }, expected);
    for (unsigned threads : {1U, 4U})
        expect_error([&] { binary_io::read_plan_parallel(bytes, threads); }, expected);
}
struct WireLayout {
    std::size_t values_offset;
    std::uint64_t count;
    std::vector<std::string> strings;
};
WireLayout wire_layout(const std::string &bytes) {
    binary_io::Archive<true> a(bytes);
    a.magic("CKKSPL01");
    std::string metadata;
    a.text(metadata);
    WireLayout result{};
    a.sequence(result.strings, 4, [&](std::string &text) { a.text(text); });
    a.number(result.count);
    result.values_offset = static_cast<std::size_t>(a.cursor() - bytes.data());
    return result;
}
std::uint64_t number_at(const std::string &bytes, std::size_t offset, unsigned width) {
    std::uint64_t result = 0;
    for (unsigned byte = 0; byte < width; ++byte)
        result |= std::uint64_t(static_cast<unsigned char>(bytes.at(offset + byte))) << (8 * byte);
    return result;
}
void set_number(std::string &bytes, std::size_t offset, unsigned width, std::uint64_t value) {
    for (unsigned byte = 0; byte < width; ++byte)
        bytes.at(offset + byte) = static_cast<char>(value >> (8 * byte));
}
std::uint32_t string_ref(const WireLayout &layout, const std::string &text) {
    const auto found = std::find(layout.strings.begin(), layout.strings.end(), text);
    require(found != layout.strings.end(), "missing test string reference");
    return static_cast<std::uint32_t>(found - layout.strings.begin());
}
void check_value_records(const std::string &bytes, const RuntimePlan &plan, std::size_t width) {
    const auto layout = wire_layout(bytes);
    require(layout.count == plan.values.size(), "wrong wire value count");
    for (std::size_t index = 0; index < plan.values.size(); ++index) {
        const auto offset = layout.values_offset + width * index;
        const auto &value = plan.values[index];
        require(number_at(bytes, offset, 8) == value.id &&
                number_at(bytes, offset + 8, 1) == static_cast<unsigned>(value.kind),
                "wrong fixed-width ValueDesc prefix");
        if (width == 51) {
            require(number_at(bytes, offset + 39, 8) == value.buffer_bytes &&
                    layout.strings.at(number_at(bytes, offset + 47, 4)) == value.buffer_layout,
                    "wrong V8 ValueDesc tail");
        }
    }
    // Independently use the specified width, not the codec's width helper.
    binary_io::Archive<true> tail(std::string_view(bytes).substr(layout.values_offset + width * layout.count));
    std::vector<ValueId> ids;
    tail.sequence(ids, 8, [&](ValueId &id) { tail.number(id); });
    require(ids == plan.external_inputs, "wrong boundary after ValueDesc records");
    for (const auto *phase : {&plan.initialization, &plan.execution, &plan.finalization}) {
        std::uint64_t count = 0;
        tail.number(count);
        require(count == phase->size(), "wrong instruction boundary after ValueDesc records");
        for (std::uint64_t index = 0; index < count; ++index)
            binary_io::skip_instruction(tail, plan.format_version);
    }
    tail.sequence(ids, 8, [&](ValueId &id) { tail.number(id); });
    require(ids == plan.final_outputs, "wrong final output boundary");
    tail.finish();
}
Json binary_metadata(const std::string &bytes) {
    binary_io::Archive<true> a(bytes);
    a.magic("CKKSPL01");
    std::string text;
    a.text(text);
    return Json::parse(text);
}
std::string replace_binary_metadata(const std::string &bytes, const Json &metadata) {
    binary_io::Archive<true> a(bytes);
    a.magic("CKKSPL01");
    std::string old_text;
    a.text(old_text);
    std::ostringstream output(std::ios::binary);
    binary_io::Archive<false> b(output);
    b.magic("CKKSPL01");
    auto text = metadata.dump();
    b.text(text);
    output.write(a.cursor(), static_cast<std::streamsize>(a.available()));
    b.finish();
    return output.str();
}
struct TemporaryFile {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ckks-buffer-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
    ~TemporaryFile() { std::error_code error; std::filesystem::remove(path, error); }
};
void test_roundtrip() {
    require(static_cast<unsigned>(ValueKind::Buffer) == 2, "Buffer wire kind is not 2");
    TemporaryFile file;
    for (auto bytes_count : {std::uint64_t{0}, std::numeric_limits<std::uint64_t>::max()}) {
        auto plan = codec_fixture();
        plan.values[1].buffer_bytes = bytes_count;
        const auto json = json_fixture(plan);
        const auto from_json = RuntimePlanJsonReader::read_text(json.dump()).plan;
        require(json_fixture(from_json) == json, "V8 JSON fields changed");
        const auto bytes = binary(plan);
        require(binary(from_json) == bytes, "V8 JSON decoding changed binary records");
        check_value_records(bytes, plan, 51);
        std::istringstream input(bytes, std::ios::binary);
        require(binary(binary_io::read_plan(input, bytes.size())) == bytes, "V8 stream roundtrip changed");
        for (unsigned threads : {1U, 4U})
            require(binary(binary_io::read_plan_parallel(bytes, threads)) == bytes, "V8 parallel roundtrip changed");
        {
            std::ofstream output(file.path, std::ios::binary);
            output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            output.close();
            require(bool(output), "cannot write test plan");
        }
        for (unsigned threads : {1U, 4U})
            require(binary(RuntimePlanReader::read_file(file.path.string(), {threads}).plan) == bytes,
                    "V8 production file reader changed records");
    }
    const auto old_plan = no_buffer_fixture();
    PlanVerifier::verify(old_plan, operator_spec());
    const auto old_bytes = binary(old_plan);
    check_value_records(old_bytes, old_plan, 39);
    require(binary(RuntimePlanJsonReader::read_text(json_fixture(old_plan).dump()).plan) == old_bytes,
            "no-Buffer V7 JSON changed");
    std::istringstream input(old_bytes, std::ios::binary);
    require(binary(binary_io::read_plan(input, old_bytes.size())) == old_bytes, "V7 stream roundtrip changed");
    for (unsigned threads : {1U, 4U})
        require(binary(binary_io::read_plan_parallel(old_bytes, threads)) == old_bytes, "V7 parallel roundtrip changed");
    std::ostringstream printed;
    fixture().print(printed);
    require(printed.str().find("Buffer") != std::string::npos &&
            printed.str().find("buffer_bytes=4096 buffer_layout=test-hoisted-v1") != std::string::npos,
            "plan print omitted Buffer metadata");
}
void test_json_errors() {
    const auto original = json_fixture(codec_fixture());
    for (const char *field : {"id", "kind", "place", "context", "level", "scale_log2", "ntt", "components", "buffer_bytes", "buffer_layout"}) {
        auto bad = original;
        bad["values"][1].erase(field);
        expect_error([&] { RuntimePlanJsonReader::read_text(bad.dump()); }, "missing required field");
    }
    for (const char *field : {"level", "scale_log2", "ntt", "components"}) {
        auto bad = original;
        bad["values"][1][field] = std::string(field) == "ntt" ? Json(true) : Json(1);
        expect_error([&] { RuntimePlanJsonReader::read_text(bad.dump()); }, "zero crypto placeholders");
    }
    for (const char *text : {"", "-1", "+1", "01", "00", " 1", "1 ", "1.0", "1e3", "18446744073709551616"}) {
        auto bad = original;
        bad["values"][1]["buffer_bytes"] = text;
        expect_error([&] { RuntimePlanJsonReader::read_text(bad.dump()); },
                     std::string(text).empty() ? "string must not be empty" : "identifier");
    }
    for (const Json &value : {Json(0), Json(std::numeric_limits<std::uint64_t>::max()), Json(false), Json(nullptr)}) {
        auto bad = original;
        bad["values"][1]["buffer_bytes"] = value;
        expect_error([&] { RuntimePlanJsonReader::read_text(bad.dump()); }, "expected string");
    }
    for (const Json &value : {Json(""), Json(0), Json(nullptr)}) {
        auto bad = original;
        bad["values"][1]["buffer_layout"] = value;
        expect_error([&] { RuntimePlanJsonReader::read_text(bad.dump()); }, "string");
    }
    for (std::size_t index : {0U, 5U}) {
        for (const char *field : {"buffer_bytes", "buffer_layout"}) {
            auto bad = original;
            bad["values"][index][field] = std::string(field) == "buffer_bytes" ? Json("0") : Json("");
            expect_error([&] { RuntimePlanJsonReader::read_text(bad.dump()); }, "unknown field");
        }
    }
    auto bad = original;
    bad["execution"][2] = {{"ordinal", 2}, {"kind", "transfer"}, {"transfer_id", "0"}, {"hint", "auto"},
        {"inputs", {"1"}}, {"outputs", {"2"}}, {"sources", {original["values"][1]["place"]}},
        {"destinations", {{{"kind", "host"}, {"rank", 0}}}}, {"output_kinds", {"buffer"}}};
    expect_error([&] { RuntimePlanJsonReader::read_text(bad.dump()); }, "Buffer cannot be communicated");
}
void test_binary_errors() {
    const auto bytes = binary(codec_fixture());
    const auto layout = wire_layout(bytes);
    const auto buffer = layout.values_offset + 51;
    for (const auto &field : std::array<std::pair<std::size_t, unsigned>, 4>{{{22, 4}, {26, 8}, {34, 1}, {35, 4}}}) {
        auto bad = bytes;
        set_number(bad, buffer + field.first, field.second,
                   field.first == 26 ? 0x3ff0000000000000ULL : 1);
        expect_binary_error(bad, "Buffer metadata");
    }
    auto bad = bytes;
    set_number(bad, buffer + 47, 4, string_ref(layout, ""));
    expect_binary_error(bad, "Buffer metadata");
    bad = bytes;
    set_number(bad, buffer + 47, 4, std::numeric_limits<std::uint32_t>::max());
    expect_binary_error(bad, "string reference");
    bad = bytes;
    set_number(bad, buffer + 8, 1, 3);
    expect_binary_error(bad, "enum");
    bad = bytes;
    set_number(bad, buffer + 34, 1, 2);
    expect_binary_error(bad, "boolean");
    bad = bytes;
    set_number(bad, buffer + 26, 8, 0x7ff8000000000000ULL);
    expect_binary_error(bad, "nonfinite");
    for (std::size_t index : {0U, 5U}) {
        bad = bytes;
        set_number(bad, layout.values_offset + 51 * index + 39, 8, 1);
        expect_binary_error(bad, "non-Buffer value has buffer metadata");
        bad = bytes;
        set_number(bad, layout.values_offset + 51 * index + 47, 4, string_ref(layout, "test-hoisted-v1"));
        expect_binary_error(bad, "non-Buffer value has buffer metadata");
    }
    for (std::size_t length : {layout.values_offset + 1, buffer + 50, bytes.size() - 1})
        expect_binary_error(bytes.substr(0, length), "");
    auto metadata = binary_metadata(bytes);
    require(RuntimePlanJsonReader::read_binary_metadata_text(metadata.dump()).format_version == 8,
            "V8 binary metadata skeleton must be accepted");
    for (const char *field : {"values", "external_inputs", "initialization", "execution", "finalization", "final_outputs"}) {
        auto bad_metadata = metadata;
        bad_metadata[field].push_back(nullptr);
        expect_binary_error(replace_binary_metadata(bytes, bad_metadata), "record array must be empty");
    }
}
void test_versions() {
    const auto spec = operator_spec();
    for (unsigned version : {1U, 2U, 3U, 4U, 5U, 7U}) {
        auto plan = fixture();
        plan.format_version = version;
        plan.values = {plan.values[1]};
        plan.execution.clear();
        plan.final_outputs.clear();
        auto json = json_fixture(plan);
        json["values"][0]["scale_log2"] = 0;
        expect_error([&] { RuntimePlanJsonReader::read_text(json.dump()); }, "Buffer requires format version 8");
        expect_error([&] { binary(plan); }, "Buffer requires plan V8");
        expect_error([&] { PlanVerifier::verify(plan, spec); }, "Buffer requires format version 8");
        // Real old-width records with wire kind 2 must fail, not just spoofed V8 framing.
        plan.values = {{0, ValueKind::Ciphertext, device(), "buffer-context", 0, 0, true, 2, 0, {}}};
        auto bytes = binary(plan);
        set_number(bytes, wire_layout(bytes).values_offset + 8, 1, 2);
        expect_binary_error(bytes, "enum");
    }
    auto plan = no_buffer_fixture();
    plan.format_version = 8;
    expect_error([&] { RuntimePlanJsonReader::read_text(json_fixture(plan).dump()); }, "format version 8 requires Buffer values");
    expect_error([&] { binary(plan); }, "format version 8 requires Buffer values");
    expect_error([&] { PlanVerifier::verify(plan, spec); }, "format version 8 requires Buffer values");
    auto bytes = binary(fixture());
    const auto layout = wire_layout(bytes);
    const auto buffer = layout.values_offset + 51;
    set_number(bytes, buffer + 8, 1, 1);
    set_number(bytes, buffer + 34, 1, 1);
    set_number(bytes, buffer + 35, 4, 2);
    set_number(bytes, buffer + 39, 8, 0);
    set_number(bytes, buffer + 47, 4, string_ref(layout, ""));
    expect_binary_error(bytes, "format version 8 requires Buffer values");
    const auto v8 = binary(fixture());
    auto metadata = binary_metadata(v8);
    metadata["format_version"] = 7;
    expect_binary_error(replace_binary_metadata(v8, metadata), "");
    const auto v7 = binary(no_buffer_fixture());
    metadata = binary_metadata(v7);
    metadata["format_version"] = 8;
    expect_binary_error(replace_binary_metadata(v7, metadata), "");
    metadata["format_version"] = 6;
    expect_binary_error(replace_binary_metadata(v7, metadata), "unsupported format version");
    plan = no_buffer_fixture();
    auto reserved_json = json_fixture(plan);
    reserved_json["format_version"] = 6;
    expect_error([&] { RuntimePlanJsonReader::read_text(reserved_json.dump()); }, "unsupported format version");
    plan.format_version = 6;
    expect_error([&] { binary(plan); }, "unsupported plan format version");
    expect_error([&] { PlanVerifier::verify(plan, spec); }, "unsupported format version");
}
void test_verifier() {
    const auto spec = operator_spec();
    const auto plan = fixture();
    const auto requirements = PlanVerifier::verify(plan, spec);
    require(requirements.capabilities.empty() && requirements.keys.empty(), "Buffer ApiCalls acquired arithmetic requirements");
    for (auto bytes : {std::uint64_t{0}, std::numeric_limits<std::uint64_t>::max()}) {
        auto valid = plan;
        valid.values[1].buffer_bytes = bytes;
        PlanVerifier::verify(valid, spec);
    }
    auto bounded_spec = spec;
    bounded_spec.spec.level_lower_bound = 1;
    auto bounded_plan = plan;
    bounded_plan.values[4].level = 1;
    PlanVerifier::verify(bounded_plan, bounded_spec); // Buffer's level=0 is not a CKKS level.
    auto bad = plan;
    // ApiCall can also produce a Buffer directly with no inputs, or from another Buffer.
    bad.values[0] = bad.values[1];
    bad.values[0].id = 0;
    PlanVerifier::verify(bad, spec);
    for (const char *field : {"level", "scale", "ntt", "components", "layout"}) {
        bad = plan;
        auto &desc = bad.values[1];
        const std::string name(field);
        if (name == "level") desc.level = 1;
        if (name == "scale") desc.scale_log2 = 1;
        if (name == "ntt") desc.ntt = true;
        if (name == "components") desc.components = 1;
        if (name == "layout") desc.buffer_layout.clear();
        expect_error([&] { PlanVerifier::verify(bad, spec); }, "Buffer requires zero crypto placeholders");
        expect_error([&] { binary(bad); }, "Buffer metadata");
    }
    bad = plan;
    bad.values[1].scale_log2 = std::numeric_limits<double>::quiet_NaN();
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "Buffer requires zero crypto placeholders");
    for (auto kind : {ValueKind::Ciphertext, ValueKind::Plaintext}) {
        for (bool bytes_field : {false, true}) {
            bad = plan;
            auto &desc = bad.values[0];
            desc.kind = kind;
            desc.components = kind == ValueKind::Plaintext ? 1 : 2;
            if (bytes_field) desc.buffer_bytes = 1; else desc.buffer_layout = "not-a-buffer";
            expect_error([&] { PlanVerifier::verify(bad, spec); }, "non-Buffer value has buffer metadata");
        }
    }
    for (std::size_t index : {1U, 4U}) {
        bad = plan;
        std::get<ComputeOp>(bad.execution[index].body).reuse_input = 0;
        expect_error([&] { PlanVerifier::verify(bad, spec); }, "ApiCall does not support reuse_input");
    }
    bad = plan;
    bad.values[1].place = host();
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "compute output Place mismatch");
    bad = plan;
    bad.values[3].place = host();
    std::get<ComputeOp>(bad.execution[4].body).place = host();
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "cross-Place ApiCall operand");
    bad = plan;
    bad.values[1].context = "other-context";
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "context");
    bad = plan;
    bad.values[2].kind = ValueKind::Plaintext;
    bad.values[2].components = 1;
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "compute output must be ciphertext");
    bad = plan;
    bad.values.push_back({5, ValueKind::Plaintext, host(), "buffer-context", 1, 20, true, 1, 0, {}});
    bad.external_inputs = {5};
    std::get<ComputeOp>(bad.execution[2].body).inputs[0] = 5;
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "ApiCall input must be ciphertext");
    bad = plan;
    bad.external_inputs = {1};
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "Buffer cannot be an external input");
    bad = plan;
    bad.execution.erase(bad.execution.begin() + 5);
    bad.final_outputs.push_back(1);
    renumber(bad);
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "Buffer cannot be a final output");
    bad = plan;
    bad.execution[1].body = EncodeOp{InlineEncodePayload{{1}}, 1};
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "Encode output must be Host plaintext");
    for (auto kind : {ComputeKind::AddCC, ComputeKind::AddCP, ComputeKind::SubCC, ComputeKind::SubCP,
                      ComputeKind::MulCC, ComputeKind::MulCP, ComputeKind::Negate, ComputeKind::Rotate,
                      ComputeKind::Rescale, ComputeKind::ModSwitch, ComputeKind::Relinearize, ComputeKind::Boot}) {
        bad = plan;
        const bool binary_op = kind == ComputeKind::AddCC || kind == ComputeKind::AddCP ||
            kind == ComputeKind::SubCC || kind == ComputeKind::SubCP || kind == ComputeKind::MulCC || kind == ComputeKind::MulCP;
        bad.execution[4].body = ComputeOp{kind, binary_op ? std::vector<ValueId>{1, 2} : std::vector<ValueId>{1},
                                         3, device(), std::monostate{}};
        expect_error([&] { PlanVerifier::verify(bad, spec); }, "wrong input kind");
    }
    bad = plan;
    bad.execution[1].body = ComputeOp{ComputeKind::Negate, {0}, 1, device(), std::monostate{}};
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "compute output must be ciphertext");
    for (auto kind : {CommKind::Transfer, CommKind::Replicate}) {
        const bool replicate = kind == CommKind::Replicate;
        bad = plan;
        bad.target.device_counts = {2};
        bad.values[2].place = host();
        bad.values[3].place = {PlaceKind::Device, 0, 1};
        CommAction action{0, kind, CommHint::Auto, {1}, {2}, {device()}, {host()}, {ValueKind::Ciphertext}};
        if (replicate) {
            action.outputs.push_back(3);
            action.destinations.push_back(bad.values[3].place);
            action.output_types.push_back(ValueKind::Ciphertext);
        }
        bad.execution[2].body = action;
        expect_error([&] { PlanVerifier::verify(bad, spec); }, "Buffer cannot be communicated");
        for (bool descriptor_is_buffer : {false, true}) {
            bad = plan;
            bad.target.device_counts = {2};
            bad.values[1].place = host();
            if (!descriptor_is_buffer) {
                bad.values[1] = bad.values[0];
                bad.values[1].id = 1;
                bad.values[1].place = host();
                // Keep a Buffer elsewhere so this remains a V8 plan.
                bad.values[3] = plan.values[1];
                bad.values[3].id = 3;
            }
            action = {0, kind, CommHint::Auto, {0}, {1}, {device()}, {host()},
                      {descriptor_is_buffer ? ValueKind::Ciphertext : ValueKind::Buffer}};
            if (replicate) {
                auto desc = plan.values[0];
                desc.id = 5;
                desc.place = {PlaceKind::Device, 0, 1};
                bad.values.push_back(desc);
                action.outputs.push_back(5);
                action.destinations.push_back(desc.place);
                action.output_types.push_back(ValueKind::Ciphertext);
            }
            bad.execution[1].body = action;
            expect_error([&] { PlanVerifier::verify(bad, spec); }, "Buffer cannot be communicated");
        }
    }
    bad = plan;
    std::swap(bad.execution[4], bad.execution[5]);
    renumber(bad);
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "already released");
    bad = plan;
    bad.execution.insert(bad.execution.begin() + 6, {0, ReleaseOp{1}});
    renumber(bad);
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "already released");
}

// This backend owns its opaque hoist representation. Neither the generic runtime
// nor a Vec/Mock backend needs to know how a Hoisted object is represented.
class HoistedApi {
public:
    struct Ciphertext { ValueDesc desc; int number; };
    struct Hoisted { ValueDesc desc; int number; };
    using Value = std::variant<std::shared_ptr<Ciphertext>, std::shared_ptr<Hoisted>>;
    struct CommHandle {};
    explicit HoistedApi(const RuntimePlan &plan) : plan_(plan) {}
    std::vector<std::uint64_t> calls;
    std::weak_ptr<Hoisted> hoisted;
    std::map<ValueId, std::weak_ptr<Ciphertext>> ciphertexts;
    unsigned suffix_reads = 0;
    std::string name() const { return "TestHoistedApi"; }
    void preflight(std::string_view, bool, const TargetConfig &, const OperatorSpec &,
                   const PlanRequirements &requirements) {
        require(requirements.capabilities.empty() && requirements.keys.empty(), "unexpected requirements");
    }
    Value compute(const ComputeOp &op, const std::vector<Value> &inputs) {
        require(op.kind == ComputeKind::ApiCall, "test backend only supports ApiCall");
        const auto &attrs = std::get<ApiCallAttrs>(op.attrs);
        const std::array<const char *, 5> callees{{"stem", "hoist", "suffix_left", "suffix_right", "head"}};
        require(attrs.callee == callees.at(attrs.call_id), "prepared callee/call_id mismatch");
        calls.push_back(attrs.call_id);
        const auto ct = [&](std::size_t index) { return std::get<std::shared_ptr<Ciphertext>>(inputs.at(index))->number; };
        const auto buffer = [&](std::size_t index) {
            const auto &value = std::get<std::shared_ptr<Hoisted>>(inputs.at(index));
            require(value == hoisted.lock(), "suffix did not receive the same shared Hoisted object");
            ++suffix_reads;
            return value->number;
        };
        switch (attrs.call_id) {
        case 0:
            require(inputs.empty(), "stem has operands");
            return ciphertext(op.output, 7);
        case 1: {
            require(inputs.size() == 1, "hoist needs one ciphertext");
            auto result = std::make_shared<Hoisted>(Hoisted{plan_.values.at(op.output), ct(0) * 2});
            hoisted = result;
            return result;
        }
        case 2:
            require(inputs.size() == 2 && !hoisted.expired(), "first suffix lost Buffer");
            return ciphertext(op.output, ct(0) + buffer(1));
        case 3:
            require(inputs.size() == 1 && ciphertexts.at(0).expired(), "Release retained stem ciphertext");
            require(!hoisted.expired(), "Buffer was released before its second suffix");
            return ciphertext(op.output, buffer(0) + 3);
        case 4:
            require(inputs.size() == 2 && hoisted.expired(), "Release retained Buffer at the next ApiCall");
            require(suffix_reads == 2, "shared Buffer was not consumed twice");
            return ciphertext(op.output, ct(0) + ct(1));
        }
        throw std::runtime_error("unknown prepared call");
    }
    void validate_value(const Value &value, const ValueDesc &desc) const {
        require((desc.kind == ValueKind::Buffer) == std::holds_alternative<std::shared_ptr<Hoisted>>(value),
                "backend variant kind mismatch");
        std::visit([&](const auto &payload) {
            const auto &actual = payload->desc;
            require(actual.id == desc.id && actual.kind == desc.kind && actual.place == desc.place &&
                    actual.context == desc.context && actual.level == desc.level && actual.scale_log2 == desc.scale_log2 &&
                    actual.ntt == desc.ntt && actual.components == desc.components &&
                    actual.buffer_bytes == desc.buffer_bytes && actual.buffer_layout == desc.buffer_layout,
                    "prepared output descriptor mismatch");
        }, value);
    }
    Value encode_plaintext(const ValueDesc &, const std::vector<double> &) {
        throw std::runtime_error("TestHoistedApi does not support Encode");
    }
    CommHandle communicate_async(const CommAction &, const std::vector<Value> &, const std::vector<ValueDesc> &) {
        throw std::runtime_error("TestHoistedApi does not support communication");
    }
    std::vector<std::optional<Value>> wait(CommHandle &) {
        throw std::runtime_error("TestHoistedApi does not support communication");
    }
    void collect_completed() {}
    void drain() {}
    void synchronize(Value &) {}
    [[noreturn]] void abort_all(int, const std::string &reason) { throw std::runtime_error(reason); }
private:
    const RuntimePlan &plan_;
    Value ciphertext(ValueId id, int number) {
        auto result = std::make_shared<Ciphertext>(Ciphertext{plan_.values.at(id), number});
        ciphertexts[id] = result;
        return result;
    }
};
void test_dispatch() {
    const auto plan = fixture();
    const auto spec = operator_spec();
    HoistedApi api(plan);
    SequentialRuntime<HoistedApi> runtime(0, 1, 1, api);
    const auto loaded = RuntimePlanJsonReader::read_text(json_fixture(plan).dump());
    const auto artifact = runtime.run(loaded, {spec, {}}, {});
    require(api.calls == std::vector<std::uint64_t>({0, 1, 2, 3, 4}), "prepared calls did not dispatch in order");
    require(artifact.values.size() == 1 &&
            std::get<std::shared_ptr<HoistedApi::Ciphertext>>(artifact.values.at(4).value)->number == 38,
            "wrong shared-hoist result");
    require(api.hoisted.expired() && api.ciphertexts.at(0).expired() && api.ciphertexts.at(2).expired() &&
            api.ciphertexts.at(3).expired(), "runtime retained released values");
    require(artifact.timing.compute_calls == 5 && artifact.timing.boot_calls == 0,
            "opaque prepared calls were counted as generic Boot");
}
class BoundPlanApi : public HoistedApi {
public:
    explicit BoundPlanApi(const RuntimePlan &plan) : HoistedApi(plan), expected_(&plan) {}
    const RuntimePlan *seen = nullptr;
    void validate_plan(const RuntimePlan &plan) {
        seen = &plan;
        require(&plan == expected_, "runtime plan identity mismatch");
    }
private:
    const RuntimePlan *expected_;
};
void test_plan_validation_hook() {
    const auto loaded = RuntimePlanJsonReader::read_text(json_fixture(fixture()).dump());
    const auto spec = operator_spec();
    BoundPlanApi api(loaded.plan);
    SequentialRuntime<BoundPlanApi> runtime(0, 1, 1, api);
    const auto artifact = runtime.run(loaded, {spec, {}}, {});
    require(api.seen == &loaded.plan && artifact.values.count(4), "plan hook did not receive actual run plan");
    const auto calls = api.calls;
    const auto other = loaded; // Same metadata and graph, but not the plan object bound to this API.
    expect_error([&] { runtime.run(other, {spec, {}}, {}); }, "runtime plan identity mismatch");
    require(api.seen == &other.plan && api.calls == calls, "different plan reached compute before hook rejection");
    // test_dispatch uses HoistedApi without this optional hook and executes the same graph successfully.
}
RuntimePlan resident_fixture(int device_index = 0, bool with_buffer = true) {
    auto plan = fixture();
    plan.target.device_counts = {2};
    const Place place{PlaceKind::Device, 0, device_index};
    plan.values = {
        {0, ValueKind::Ciphertext, place, "buffer-context", 1, 20.25, true, 2},
        {1, ValueKind::Ciphertext, place, "buffer-context", 1, 20.25, true, 2},
        {2, ValueKind::Buffer, place, "buffer-context", 0, 0, false, 0, 4096, "test-hoisted-v1"},
        {3, ValueKind::Ciphertext, place, "buffer-context", 1, 20.25, true, 2},
        {4, ValueKind::Ciphertext, place, "buffer-context", 1, 20.25, true, 2}};
    plan.external_inputs = {0, 1};
    plan.execution = {
        {0, ComputeOp{ComputeKind::ApiCall, {0}, 2, place, ApiCallAttrs{"resident_hoist", 10}}},
        {1, ComputeOp{ComputeKind::ApiCall, {0}, 3, place, ApiCallAttrs{"resident_identity", 11}}},
        {2, ComputeOp{ComputeKind::ApiCall, {1}, 4, place, ApiCallAttrs{"resident_identity", 12}}},
        {3, ReleaseOp{0}}, {4, ReleaseOp{1}}, {5, ReleaseOp{2}}};
    plan.final_outputs = {3, 4};
    if (!with_buffer) {
        plan.values[2] = plan.values[0];
        plan.values[2].id = 2;
        std::get<ApiCallAttrs>(std::get<ComputeOp>(plan.execution[0].body).attrs).callee = "resident_identity";
    }
    return plan;
}
class ResidentApi : public HoistedApi {
public:
    explicit ResidentApi(const RuntimePlan &plan) : HoistedApi(plan), plan_(plan) {}
    std::map<ValueId, std::weak_ptr<Ciphertext>> expected_inputs;
    Value compute(const ComputeOp &op, const std::vector<Value> &inputs) {
        require(op.kind == ComputeKind::ApiCall && inputs.size() == 1, "unexpected resident call");
        const auto &input = std::get<std::shared_ptr<Ciphertext>>(inputs.at(0));
        require(input == expected_inputs.at(op.inputs.at(0)).lock(), "runtime changed resident input identity");
        require(input->desc.place == op.place, "resident input reached wrong device");
        const auto &output = plan_.values.at(op.output);
        const auto &attrs = std::get<ApiCallAttrs>(op.attrs);
        if (attrs.callee == "resident_hoist")
            return std::make_shared<Hoisted>(Hoisted{output, input->number});
        require(attrs.callee == "resident_identity", "unknown resident call");
        return std::make_shared<Ciphertext>(Ciphertext{output, input->number});
    }
private:
    const RuntimePlan &plan_;
};
void test_device_external_inputs() {
    const auto spec = operator_spec();
    for (int device_index : {0, 1}) for (bool with_buffer : {false, true}) {
        const auto plan = resident_fixture(device_index, with_buffer);
        const auto requirements = PlanVerifier::verify(plan, spec);
        require(requirements.capabilities.empty(), "resident inputs required a transfer");
        for (auto mode : {DeviceExecutionMode::Sequential, DeviceExecutionMode::PerDeviceWorkers}) {
            ResidentApi api(plan);
            SequentialRuntime<ResidentApi> runtime(0, 1, 2, api, mode);
            auto first = std::make_shared<ResidentApi::Ciphertext>(ResidentApi::Ciphertext{plan.values[0], 17});
            auto second = std::make_shared<ResidentApi::Ciphertext>(ResidentApi::Ciphertext{plan.values[1], 29});
            api.expected_inputs = {{0, first}, {1, second}};
            const auto artifact = runtime.run({plan, "resident-test"}, {spec, {}}, {{0, first}, {1, second}});
            require(artifact.values.size() == 2 &&
                    std::get<std::shared_ptr<ResidentApi::Ciphertext>>(artifact.values.at(3).value)->number == 17 &&
                    std::get<std::shared_ptr<ResidentApi::Ciphertext>>(artifact.values.at(4).value)->number == 29 &&
                    artifact.values.at(3).place == plan.values[3].place &&
                    artifact.values.at(4).place == plan.values[4].place, "resident identity output binding mismatch");
            require(artifact.timing.communication_post_calls == 0 && artifact.timing.communication_wait_calls == 0,
                    "resident inputs introduced communication");
        }
    }
    auto bad = resident_fixture();
    bad.values[0].place.rank = 1;
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "place rank is outside target");
    bad = resident_fixture();
    bad.values[0].place.index = 2;
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "device index is outside target");
    bad = resident_fixture();
    bad.values[0].context = "wrong-context";
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "context does not match");
    bad = resident_fixture();
    bad.values[0].place.index = 1;
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "cross-Place ApiCall operand");
    bad = resident_fixture();
    bad.values[0].kind = ValueKind::Plaintext;
    bad.values[0].components = 1;
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "external input must be placed on Host");
    for (unsigned version : {1U, 2U, 3U, 4U, 5U, 7U}) {
        bad = no_buffer_fixture();
        bad.format_version = version;
        for (auto &value : bad.values) value.scale_log2 = 0;
        bad.external_inputs = {0};
        bad.execution.erase(bad.execution.begin());
        renumber(bad);
        expect_error([&] { PlanVerifier::verify(bad, spec); }, "external input must be placed on Host");
    }
    const auto plan = resident_fixture();
    ResidentApi api(plan);
    SequentialRuntime<ResidentApi> runtime(0, 1, 2, api);
    expect_error([&] { runtime.run({plan, "resident-test"}, {spec, {}}, {}); }, "missing local external input");
    auto wrong_desc = plan.values[0];
    wrong_desc.place = host();
    auto wrong_value = std::make_shared<ResidentApi::Ciphertext>(ResidentApi::Ciphertext{wrong_desc, 17});
    auto second = std::make_shared<ResidentApi::Ciphertext>(ResidentApi::Ciphertext{plan.values[1], 29});
    expect_error([&] { runtime.run({plan, "resident-test"}, {spec, {}}, {{0, wrong_value}, {1, second}}); },
                 "prepared output descriptor mismatch");
}
void test_device_external_versions() {
    const auto spec = operator_spec();
    const auto plan = resident_fixture(0, false);
    const auto bytes = binary(plan);
    require(plan_requires_v8(plan), "bufferless resident plan did not select V8");
    auto direct = plan;
    direct.values.resize(2);
    direct.execution.clear();
    direct.final_outputs = direct.external_inputs;
    ResidentApi api(direct);
    SequentialRuntime<ResidentApi> runtime(0, 1, 2, api);
    auto first = std::make_shared<ResidentApi::Ciphertext>(ResidentApi::Ciphertext{direct.values[0], 17});
    auto second = std::make_shared<ResidentApi::Ciphertext>(ResidentApi::Ciphertext{direct.values[1], 29});
    const auto identity = runtime.run({direct, "resident-identity"}, {spec, {}}, {{0, first}, {1, second}});
    require(std::get<std::shared_ptr<ResidentApi::Ciphertext>>(identity.values.at(0).value) == first &&
            std::get<std::shared_ptr<ResidentApi::Ciphertext>>(identity.values.at(1).value) == second &&
            identity.timing.compute_calls == 0 && identity.timing.communication_post_calls == 0,
            "bufferless external identity graph changed resident objects");
    require(binary(RuntimePlanJsonReader::read_text(json_fixture(plan).dump()).plan) == bytes,
            "bufferless Device external JSON changed");
    check_value_records(bytes, plan, 51);
    std::istringstream input(bytes, std::ios::binary);
    require(binary(binary_io::read_plan(input, bytes.size())) == bytes, "bufferless resident stream roundtrip changed");
    for (unsigned threads : {1U, 4U})
        require(binary(binary_io::read_plan_parallel(bytes, threads)) == bytes, "bufferless resident parallel roundtrip changed");
    TemporaryFile file;
    {
        std::ofstream output(file.path, std::ios::binary);
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        output.close();
        require(bool(output), "cannot write bufferless resident test plan");
    }
    for (unsigned threads : {1U, 4U})
        require(binary(RuntimePlanReader::read_file(file.path.string(), {threads}).plan) == bytes,
                "bufferless resident production reader changed");
    auto legacy = plan;
    legacy.format_version = 7;
    expect_error([&] { RuntimePlanJsonReader::read_text(json_fixture(legacy).dump()); }, "external inputs require format version 8");
    expect_error([&] { binary(legacy); }, "external inputs require plan V8");
    expect_error([&] { PlanVerifier::verify(legacy, spec); }, "external input must be placed on Host");

    auto host_plan = plan;
    for (auto &value : host_plan.values) value.place = host();
    for (auto &instruction : host_plan.execution)
        if (auto *op = std::get_if<ComputeOp>(&instruction.body)) op->place = host();
    require(!plan_requires_v8(host_plan), "Host external CT incorrectly selected V8");
    expect_error([&] { RuntimePlanJsonReader::read_text(json_fixture(host_plan).dump()); }, "external Device ciphertext");
    expect_error([&] { binary(host_plan); }, "external Device ciphertext");
    expect_error([&] { PlanVerifier::verify(host_plan, spec); }, "external Device ciphertext");
    // Structurally valid51-byte records must fail when the only V8 feature is removed.
    auto host_bytes = bytes;
    const auto layout = wire_layout(bytes);
    for (std::size_t index = 0; index < plan.values.size(); ++index)
        set_number(host_bytes, layout.values_offset + 51 * index + 9, 1, 0);
    expect_binary_error(host_bytes, "external Device ciphertext");
    host_plan.format_version = 7;
    PlanVerifier::verify(host_plan, spec);
    auto legacy_bytes = binary(host_plan);
    const auto legacy_layout = wire_layout(legacy_bytes);
    set_number(legacy_bytes, legacy_layout.values_offset + 9, 1, 1);
    expect_binary_error(legacy_bytes, "external inputs require plan V8");
    auto internal_device = host_plan;
    internal_device.format_version = 8;
    internal_device.values[2].place = device();
    require(!plan_requires_v8(internal_device), "internal Device CT incorrectly selected V8");
    expect_error([&] { RuntimePlanJsonReader::read_text(json_fixture(internal_device).dump()); }, "external Device ciphertext");
    expect_error([&] { binary(internal_device); }, "external Device ciphertext");
}
void test_compiler_output(const char *json_path, const char *binary_path, const char *spec_path) {
    const auto expected = RuntimePlanReader::read_file(json_path).plan;
    require(expected.format_version == 8, "compiler plan must be V8");
    std::size_t buffers = 0, calls = 0;
    for (const auto &value : expected.values) if (value.kind == ValueKind::Buffer) ++buffers;
    for (const auto *phase : {&expected.initialization, &expected.execution, &expected.finalization})
        for (const auto &instruction : *phase)
            if (const auto *op = std::get_if<ComputeOp>(&instruction.body); op && op->kind == ComputeKind::ApiCall) ++calls;
    require(plan_requires_v8(expected), "compiler fixture must contain a V8 feature");
    check_value_records(binary(expected), expected, 51);
    for (unsigned threads : {1U, 4U})
        require(binary(RuntimePlanReader::read_file(binary_path, {threads}).plan) == binary(expected),
                "compiler V8 JSON and binary records differ");
    if (spec_path) PlanVerifier::verify(expected, OperatorSpecReader::read_file(spec_path));
    std::cout << "PASS compiler V8 JSON/binary: buffers=" << buffers << " calls=" << calls
              << " readers=1,4 verified=" << bool(spec_path) << '\n';
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc == 1) {
            test_roundtrip();
            test_json_errors();
            test_binary_errors();
            test_versions();
            test_verifier();
            test_dispatch();
            test_plan_validation_hook();
            test_device_external_inputs();
            test_device_external_versions();
            std::cout << "PASS Buffer V8: JSON/binary, strict versions, verifier, variant backend, shared-hoist Release, resident Device CT inputs\n";
        } else if (argc == 3 || argc == 4) {
            test_compiler_output(argv[1], argv[2], argc == 4 ? argv[3] : nullptr);
        } else throw std::runtime_error("usage: runtime_buffer_tests [COMPILER_JSON COMPILER_BINARY [OPERATOR_SPEC]]");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
