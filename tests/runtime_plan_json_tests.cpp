#include "runtime/json_plan_reader.hpp"
#include "runtime/utils/sha256.hpp"
#include "runtime/verifier.hpp"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fhegpu;

namespace {

int tests_run = 0;

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Function>
void expect_throw(Function function, const std::string &needle) {
    try {
        function();
    } catch (const std::exception &error) {
        if (std::string(error.what()).find(needle) == std::string::npos)
            throw std::runtime_error("exception did not contain '" + needle + "': " + error.what());
        return;
    }
    throw std::runtime_error("expected exception containing '" + needle + "'");
}

void run_test(const char *name, const std::function<void()> &test) {
    test();
    ++tests_run;
    std::cout << "[PASS] " << name << '\n';
}

std::filesystem::path source_root() {
    return CKKS_RUNTIME_SOURCE_DIR;
}

std::filesystem::path testdata(const std::string &group, const std::string &name) {
    return source_root() / "docs/runtime-plan/v1/testdata" / group / name;
}

std::string read_text_file(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("failed to open test file: " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

RuntimePlan read_valid(const std::string &name) {
    return RuntimePlanJsonReader::read_file(testdata("valid", name).string());
}

void test_sha256_vectors() {
    require(sha256_hex("") ==
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "SHA-256 empty input vector failed");
    require(sha256_hex("abc") ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "SHA-256 abc vector failed");
    require(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            "SHA-256 two-block padding vector failed");

    const std::string full_block(64, 'a');
    require(sha256_hex(full_block) ==
            "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb",
            "SHA-256 full-block vector failed");

    const std::string binary_input("abc\0def", 7);
    require(sha256_hex(binary_input) ==
            "516a5e926ce20c5f4d80f00e1a01abdf14986def6588d6abeed9fce090bc660c",
            "SHA-256 binary input vector failed");

    const std::string million_as(1'000'000, 'a');
    require(sha256_hex(million_as) ==
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
            "SHA-256 million-a vector failed");
}

void test_valid_samples() {
    const std::vector<std::string> names = {
        "v001_minimal_single_device.json",
        "v002_mul_rescale_transfer.json",
        "v003_replicate_multi_rank.json",
        "v004_host_boot_emulation.json",
        "v005_plaintext_bundle.json",
    };
    for (const auto &name : names) {
        const RuntimePlan plan = read_valid(name);
        require(plan.format_version == 1, name + " format version was not read");
        require(plan.fingerprint_sha256.rfind("sha256:", 0) == 0, name + " fingerprint was not read");
        require(!plan.values.empty(), name + " values were not read");
        require(!plan.final_outputs.empty(), name + " final outputs were not read");
        require(!plan.initialization.empty(), name + " initialization uploads were not read");
    }

    const RuntimePlan gpu = read_valid("v002_mul_rescale_transfer.json");
    require(gpu.target.rescale_mode == RescaleMode::Lazy, "GPU rescale mode was not read");
    require(gpu.target.operator_spec.id == "poseidon-ckks-gpu-v1", "OperatorSpec reference was not read");
    require(!gpu.plaintext_bundle.has_value(), "absent plaintext_bundle must stay empty");
    require(gpu.values.at(2).scale_log2 == 80 && gpu.values.at(2).components == 3,
            "CKKS value metadata was not read");
    const auto &rescale = std::get<ComputeOp>(gpu.execution.at(2).body);
    const auto &rescale_attrs = std::get<RescaleAttrs>(rescale.attrs);
    require(rescale_attrs.target_level == 4 && rescale_attrs.target_scale_log2 == 40,
            "Rescale attrs were not read");

    const RuntimePlan boot = read_valid("v004_host_boot_emulation.json");
    require(boot.target.boot_mode == BootMode::DecryptReencrypt, "boot mode was not read");
    require(boot.required_keys.size() == 1 && boot.required_keys.front().kind == KeyKind::Secret,
            "required key was not read");
    const auto &boot_operation = std::get<ComputeOp>(boot.execution.at(1).body);
    const auto &boot_attrs = std::get<BootAttrs>(boot_operation.attrs);
    require(boot_operation.place.kind == PlaceKind::Host,
            "Host boot place was not read");
    require(boot_attrs.target_level == 12 && boot_attrs.target_scale_log2 == 40 &&
            boot_attrs.target_components == 2 &&
            boot_attrs.operator_profile == "poseidon-cpu-boot-emulation-v1" &&
            boot_attrs.implementation == BootImplementation::DecryptReencrypt,
            "Boot attrs were not read");

    const RuntimePlan bundled = read_valid("v005_plaintext_bundle.json");
    require(bundled.plaintext_bundle.has_value(), "plaintext_bundle was not read");
    require(bundled.plaintext_bundle->id == "v005-demo-weights" &&
            bundled.plaintext_bundle->version == 1 &&
            bundled.plaintext_bundle->fingerprint.rfind("sha256:", 0) == 0,
            "plaintext_bundle reference fields were not read");
    require(bundled.values.front().kind == ValueKind::Plaintext &&
            bundled.values.front().place.kind == PlaceKind::Host,
            "bundled plaintext input must be a Host plaintext value");
}

void test_external_input_host_only() {
    for (const auto &name : {std::string("v001_minimal_single_device.json"),
                             std::string("v005_plaintext_bundle.json")})
        PlanVerifier::verify(read_valid(name));

    const RuntimePlan bad = RuntimePlanJsonReader::read_file(
        testdata("invalid", "i013_device_external_input.json").string());
    expect_throw([&] { PlanVerifier::verify(bad); }, "IO-2");
}

void test_structural_invalid_samples() {
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"i001_unknown_format_version.json", "unsupported format version"},
        {"i002_float_scale_log2.json", "floating-point numbers are not allowed"},
        {"i004_comm_list_length_mismatch.json", "same nonzero length"},
        {"i008_bad_fingerprint.json", "fingerprint mismatch"},
        {"i009_unknown_field.json", "unknown field 'comment'"},
    };
    for (const auto &item : cases) {
        expect_throw([&] {
            RuntimePlanJsonReader::read_file(testdata("invalid", item.first).string());
        }, item.second);
    }
}

void test_strict_json_rules() {
    expect_throw([] {
        RuntimePlanJsonReader::read_text("{\"x\":1,\"x\":2}");
    }, "duplicate object key 'x'");

    const std::string bom = std::string("\xef\xbb\xbf") + "{}";
    expect_throw([&] { RuntimePlanJsonReader::read_text(bom); }, "BOM is not allowed");
    expect_throw([] { RuntimePlanJsonReader::read_text("{"); }, "JSON parsing failed");
    expect_throw([] { RuntimePlanJsonReader::read_text("{}"); }, "missing required field");

    std::string numeric_plan_id = read_text_file(testdata("valid", "v001_minimal_single_device.json"));
    const std::string encoded = "\"plan_id\": \"1\"";
    const auto position = numeric_plan_id.find(encoded);
    require(position != std::string::npos, "test fixture plan_id was not found");
    numeric_plan_id.replace(position, encoded.size(), "\"plan_id\": 1");
    expect_throw([&] { RuntimePlanJsonReader::read_text(numeric_plan_id); }, "expected string");
}

} // namespace

int main() {
    try {
        run_test("SHA-256 known-answer vectors", test_sha256_vectors);
        run_test("valid RuntimePlan V1 samples", test_valid_samples);
        run_test("external inputs are Host-only (IO-2)", test_external_input_host_only);
        run_test("structural invalid RuntimePlan samples", test_structural_invalid_samples);
        run_test("strict JSON rules", test_strict_json_rules);
        std::cout << "ALL " << tests_run << " JSON TEST GROUPS PASSED\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
