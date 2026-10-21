#include "runtime/plan_reader.hpp"
#include "runtime/binary_plan_codec.hpp"
#include "runtime/binary_plan_parallel.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace fhegpu;

namespace {
void require(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}
std::string wire(RuntimePlan plan) {
    std::ostringstream out(std::ios::binary);
    binary_io::write_plan(out, plan);
    return out.str();
}
struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ckks-binary-plan-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDirectory() { std::filesystem::create_directory(path); }
    ~TemporaryDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
void write_file(const std::filesystem::path &path, const std::string &bytes) {
    std::ofstream output(path, std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
    require(bool(output), "failed to write binary test fixture");
}
RuntimePlan fixture(std::size_t groups) {
    RuntimePlan p;
    p.format_version = 4; p.plan_id = 404;
    p.target = {"test", 1, {1}, 1, {"test", 1, "sha256:" + std::string(64, '0')}};
    WaterlineConfig config;
    config.device_capacity_bytes = {1ULL << 30}; config.device_reserve_bytes = {0};
    config.device_fractions = {.5}; config.host_budget_bytes = 1ULL << 30;
    config.pinned_budget_bytes = 1ULL << 20; config.raw_budget_bytes = 1ULL << 20;
    config.lookahead_groups = 8; config.backend = PrepareBackend::Auto;
    p.waterline = config;
    const Place host{PlaceKind::Host, 0, 0}, device{PlaceKind::Device, 0, 0};
    const auto value = [&](ValueId id, ValueKind kind, Place place) {
        p.values.push_back({id, kind, place, "test-context", 1, 40, true,
                            kind == ValueKind::Plaintext ? 1 : 2});
    };
    value(0, ValueKind::Ciphertext, host); value(1, ValueKind::Ciphertext, device);
    p.external_inputs = {0};
    p.initialization = {{0, CommAction{0, CommKind::Transfer, CommHint::PointToPoint,
                                     {0}, {1}, {host}, {device}, {ValueKind::Ciphertext}}}};
    ValueId previous = 1;
    std::size_t ordinal = 1;
    for (std::size_t index = 0; index < groups; ++index) {
        const ValueId plain = 2 + index * 2, output = plain + 1;
        value(plain, ValueKind::Plaintext, device); value(output, ValueKind::Ciphertext, device);
        EncodePayload payload = index % 2 ? EncodePayload{BundleEncodePayload{"sha256:" + std::string(64, 'a')}} :
                                           EncodePayload{InlineEncodePayload{{.5, -1.0, 2.0}}};
        p.execution.push_back({ordinal++, PreparePlaintextOp{std::move(payload), plain}});
        p.execution.push_back({ordinal++, ComputeOp{ComputeKind::AddCP, {previous, plain}, output, device, {}}});
        p.execution.push_back({ordinal++, ReleaseOp{plain}});
        p.execution.push_back({ordinal++, ReleaseOp{previous}});
        previous = output;
    }
    p.execution.push_back({ordinal++, FenceOp{}});
    const ValueId output = 2 + groups * 2;
    value(output, ValueKind::Ciphertext, host);
    p.finalization = {{ordinal++, CommAction{1, CommKind::Transfer, CommHint::PointToPoint,
                                            {previous}, {output}, {device}, {host}, {ValueKind::Ciphertext}}},
                      {ordinal++, ReleaseOp{previous}}};
    p.final_outputs = {output};
    p.plaintext_bundle = PlaintextBundleRef{"test-bundle", 1, "sha256:" + std::string(64, 'b')};
    return p;
}
std::size_t first_execution_offset(const std::string &bytes) {
    binary_io::Archive<true> a(bytes);
    a.magic("CKKSPL01"); std::string root; a.text(root);
    std::vector<std::string> strings;
    a.sequence(strings, 4, [&](std::string &text) { a.text(text); });
    std::uint64_t count = 0; a.number(count); a.skip(count * 35);
    binary_io::skip_sequence(a, 8);
    a.number(count);
    for (std::uint64_t index = 0; index < count; ++index) binary_io::skip_instruction(a, 4);
    a.number(count);
    require(count > 0, "test fixture has no execution records");
    return static_cast<std::size_t>(a.cursor() - bytes.data());
}
void expect_read_error(const std::filesystem::path &path, const std::string &prefix, const std::string &reason) {
    try { RuntimePlanReader::read_file(path.string(), {4}); }
    catch (const std::exception &error) {
        const std::string message(error.what());
        require(message.find(prefix) != std::string::npos && message.find(reason) != std::string::npos,
                "binary error lacks record position/reason: " + message);
        return;
    }
    throw std::runtime_error("corrupt binary record was accepted");
}
void unit_tests() {
    TemporaryDirectory temp;
    for (const std::size_t groups : {2U, 8200U}) {
        const auto original = fixture(groups);
        const auto bytes = wire(original);
        const auto path = temp.path / "plan.bin";
        write_file(path, bytes);
        for (const unsigned threads : {1U, 4U}) {
            PlanReadStats stats;
            const auto read = RuntimePlanReader::read_file(path.string(), {threads}, &stats);
            require(stats.binary && stats.threads == threads && wire(read.plan) == bytes,
                    "production file reader changed V4 inline/bundle Prepare, phases, or other records");
        }
        const auto offset = first_execution_offset(bytes);
        const auto position = "binary execution record #0 at byte offset " + std::to_string(offset);
        auto bad = bytes; bad[offset + 8] = static_cast<char>(255);
        write_file(path, bad);
        expect_read_error(path, position, "unknown binary instruction tag 255");
        bad = bytes; bad[offset + 17] = static_cast<char>(255);
        write_file(path, bad);
        expect_read_error(path, position, "unknown binary payload tag");
        // A finite-width record can scan successfully but still fail typed decoding.
        bad = bytes;
        const std::uint64_t nan = 0x7ff8000000000000ULL;
        for (unsigned byte = 0; byte < 8; ++byte) bad[offset + 26 + byte] = static_cast<char>(nan >> (8 * byte));
        write_file(path, bad);
        expect_read_error(path, position, "nonfinite binary float64");
    }
    auto fractional = fixture(2);
    fractional.format_version = 5;
    for (auto &value : fractional.values) value.scale_log2 += 0.125;
    fractional.execution.push_back({fractional.execution.back().ordinal + 1,
        ComputeOp{ComputeKind::Rescale, {0}, 0, {PlaceKind::Device, 0, 0}, RescaleAttrs{1, 40.125}}});
    auto bytes = wire(fractional);
    auto path = temp.path / "fractional.bin";
    write_file(path, bytes);
    for (unsigned threads : {1U, 4U})
        require(wire(RuntimePlanReader::read_file(path.string(), {threads}).plan) == bytes,
                "V5 fractional scales or Rescale attrs changed in sequential/parallel decoding");
    std::cout << "PASS production V4 binary file reader: inline/bundle Prepare, 1/4 threads, "
                 "16384-record blocks, scan/decode corruption positions\n";
}
void compiler_comparison(const char *json, const char *binary) {
    auto expected = RuntimePlanReader::read_file(json).plan;
    require(expected.format_version == 4 && expected.waterline, "compiler fixture must be V4 Waterline");
    if (expected.plaintext_bundle) {
        expected.plaintext_bundle->manifest_format = "binary";
        expected.plaintext_bundle->manifest_sha256.clear();
    }
    std::size_t inline_count = 0, bundle_count = 0;
    for (const auto &i : expected.execution) if (const auto *prepare = std::get_if<PreparePlaintextOp>(&i.body)) {
        if (std::holds_alternative<InlineEncodePayload>(prepare->payload)) ++inline_count;
        else ++bundle_count;
    }
    require(inline_count && bundle_count, "compiler fixture must exercise both Prepare payload formats");
    for (const unsigned threads : {1U, 4U})
        require(wire(RuntimePlanReader::read_file(binary, {threads}).plan) == wire(expected),
                "compiler JSON and directly emitted binary records differ");
    std::cout << "PASS compiler JSON/direct binary all fields: V4 inline=" << inline_count << " bundle="
              << bundle_count << " readers=1,4\n";
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc == 1) unit_tests();
        else if (argc == 3) compiler_comparison(argv[1], argv[2]);
        else throw std::runtime_error("usage: runtime_binary_plan_tests [COMPILER_JSON COMPILER_BINARY]");
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
