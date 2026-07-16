#include "runtime/json_plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/plaintext_bundle.hpp"
#include "runtime/utils/sha256.hpp"
#include "runtime/verifier.hpp"

#include <filesystem>
#include <fstream>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fhegpu;

namespace {

int tests_run = 0;
const std::filesystem::path source_dir = CKKS_RUNTIME_SOURCE_DIR;

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Function>
void expect_throw(Function function, const std::string &needle = {}) {
    try { function(); }
    catch (const std::exception &error) {
        if (!needle.empty() && std::string(error.what()).find(needle) == std::string::npos)
            throw std::runtime_error("exception did not contain '" + needle + "': " + error.what());
        return;
    }
    throw std::runtime_error("expected exception was not thrown");
}

void run_test(const char *name, const std::function<void()> &test) {
    test();
    ++tests_run;
    std::cout << "[PASS] " << name << '\n';
}

std::filesystem::path testdata(const char *group, const std::string &name) {
    return source_dir / "docs/runtime-plan/v1/testdata" / group / name;
}

LoadedOperatorSpec load_spec(const RuntimePlan &plan) {
    const std::string filename = plan.target.operator_spec.id == "poseidon-ckks-cpu-v1"
        ? "poseidon-ckks-cpu.v1.json" : "poseidon-ckks-gpu.v1.json";
    return OperatorSpecReader::read_file((source_dir / "docs/operator-spec/v1/profiles" / filename).string());
}

void test_sha256_and_raw_bytes() {
    require(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty SHA-256 failed");
    require(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "abc SHA-256 failed");
    const auto path = testdata("valid", "v001_inline_encode_host_compute.json");
    std::ifstream input(path, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const auto original = RuntimePlanJsonReader::read_text(bytes);
    const auto spaced = RuntimePlanJsonReader::read_text(bytes + "\n");
    require(original.source_sha256 != spaced.source_sha256, "raw-byte whitespace change did not change SHA-256");
}

void test_valid_samples() {
    const std::vector<std::string> names = {
        "v001_inline_encode_host_compute.json",
        "v002_device_mul_relin_rescale_rotate.json",
        "v003_bundle_multi_rank.json",
        "v004_host_boot.json",
        "v005_bundle_reuse.json",
        "v006_native_boot_device.json",
    };
    for (const auto &name : names) {
        const auto loaded = RuntimePlanJsonReader::read_file(testdata("valid", name).string());
        const auto spec = load_spec(loaded.plan);
        const auto requirements = PlanVerifier::verify(loaded.plan, spec);
        require(!requirements.capabilities.empty(), name + " did not derive capabilities");
    }
    const auto inline_plan = RuntimePlanJsonReader::read_file(testdata("valid", names[0]).string());
    const auto &encode = std::get<EncodeOp>(inline_plan.plan.initialization.front().body);
    const auto &slots = std::get<InlineEncodePayload>(encode.payload).values;
    require(slots.size() == 4 && slots[1] == 0.0 && !std::signbit(slots[1]), "inline Encode did not normalize zero");
}

void test_invalid_samples() {
    const std::vector<std::pair<std::string, std::string>> reader_cases = {
        {"i001_unknown_format_version.json", "unsupported format version"},
        {"i002_float_scale_log2.json", "expected integer"},
        {"i008_unknown_field.json", "unknown field 'comment'"},
    };
    for (const auto &item : reader_cases)
        expect_throw([&] { RuntimePlanJsonReader::read_file(testdata("invalid", item.first).string()); }, item.second);

    const std::vector<std::pair<std::string, std::string>> verifier_cases = {
        {"i003_duplicate_value_id.json", "duplicate ValueId"},
        {"i004_comm_list_length_mismatch.json", "mapping mismatch"},
        {"i005_unknown_boot_profile.json", "unknown operator profile"},
        {"i006_boot_target_desc_mismatch.json", "Boot metadata rule"},
        {"i007_use_before_define.json", "use-before-definition"},
        {"i009_place_mismatch.json", "cross-Place"},
        {"i010_device_external_input.json", "external input must be placed on Host"},
        {"i011_operator_spec_digest_mismatch.json", "source SHA-256 mismatch"},
        {"i013_encode_outside_initialization.json", "only allowed in initialization"},
        {"i014_rotate_normalizes_to_zero.json", "becomes zero"},
        {"i015_unused_value_desc.json", "unused ValueDesc"},
    };
    for (const auto &item : verifier_cases) {
        const auto loaded = RuntimePlanJsonReader::read_file(testdata("invalid", item.first).string());
        const auto spec = load_spec(loaded.plan);
        expect_throw([&] { PlanVerifier::verify(loaded.plan, spec); }, item.second);
    }
}

void test_strict_json_and_operator_spec() {
    expect_throw([] { RuntimePlanJsonReader::read_text("{\"x\":1,\"x\":2}"); }, "duplicate object key");
    expect_throw([] { RuntimePlanJsonReader::read_text(std::string("\xef\xbb\xbf") + "{}"); }, "BOM");
    const auto cpu_path = source_dir / "docs/operator-spec/v1/profiles/poseidon-ckks-cpu.v1.json";
    const auto spec = OperatorSpecReader::read_file(cpu_path.string());
    require(spec.spec.poly_degree == 32768 && spec.spec.boot_profiles.size() == 2, "OperatorSpec fields were not read");

    std::ifstream input(cpu_path);
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    auto replace_once = [&](const std::string &from, const std::string &to) {
        const auto position = text.find(from);
        require(position != std::string::npos, "fixture text not found: " + from);
        text.replace(position, from.size(), to);
    };
    replace_once("\"poly_degree\": 32768", "\"poly_degree\": 30000");
    expect_throw([&] { OperatorSpecReader::read_text(text); }, "power of two");

    std::ifstream duplicate_input(cpu_path);
    std::string duplicate_text((std::istreambuf_iterator<char>(duplicate_input)), std::istreambuf_iterator<char>());
    const std::string native_id = "poseidon-native-boot-v1";
    const std::string emulated_id = "poseidon-cpu-boot-emulation-v1";
    const auto duplicate_position = duplicate_text.find(emulated_id);
    require(duplicate_position != std::string::npos, "boot profile fixture was not found");
    duplicate_text.replace(duplicate_position, emulated_id.size(), native_id);
    expect_throw([&] { OperatorSpecReader::read_text(duplicate_text); }, "duplicate profile_id");

    std::ifstream level_input(cpu_path);
    std::string level_text((std::istreambuf_iterator<char>(level_input)), std::istreambuf_iterator<char>());
    const auto level_position = level_text.find("\"upper_bound\": 13");
    require(level_position != std::string::npos, "level fixture was not found");
    level_text.replace(level_position, std::string("\"upper_bound\": 13").size(), "\"upper_bound\": 14");
    expect_throw([&] { OperatorSpecReader::read_text(level_text); }, "modulus chain");

    std::ifstream v1_noise_input(cpu_path);
    std::string v1_noise_text((std::istreambuf_iterator<char>(v1_noise_input)),
                              std::istreambuf_iterator<char>());
    const std::string null_noise = "\"noise_by_level\": null";
    const auto noise_position = v1_noise_text.find(null_noise);
    require(noise_position != std::string::npos, "V1 noise fixture was not found");
    v1_noise_text.replace(noise_position, null_noise.size(), "\"noise_by_level\": [0]");
    expect_throw([&] { OperatorSpecReader::read_text(v1_noise_text); }, "must be null in V1");
}

void test_dacapo_operator_spec_v2_profiles() {
    const auto profile_dir = source_dir / "docs/operator-spec/v2/profiles";
    const auto cpu = OperatorSpecReader::read_file((profile_dir / "dacapo-heaan-cpu.v1.json").string());
    require(cpu.spec.format_version == 2 && cpu.spec.boot_profiles.at(0).output_level == 16,
            "Dacapo HEAAN CPU profile was not read");
    const auto gpu_path = profile_dir / "dacapo-heaan-gpu.v1.json";
    const auto gpu = OperatorSpecReader::read_file(gpu_path.string());
    require(gpu.spec.format_version == 2 && gpu.spec.status == "imported",
            "Dacapo HEAAN GPU profile version/status was not read");
    require(gpu.spec.poly_degree == 131072 && gpu.spec.level_lower_bound == 1 &&
                gpu.spec.level_upper_bound == 29,
            "Dacapo HEAAN GPU CKKS parameters were not migrated");
    require(gpu.spec.provenance &&
                gpu.spec.provenance->source_sha256 ==
                    "sha256:678e3d0225236e1f44fac4eb1bcc76513940ffabcfbe74d2c920b4362d08b23d",
            "Dacapo HEAAN GPU provenance was not read");
    const auto &gpu_add = gpu.spec.operators.at(ComputeKind::AddCC);
    require(gpu_add.latency_us_by_level && gpu_add.latency_us_by_level->at(2) == 25 &&
                gpu_add.latency_us_by_level->at(29) == 280,
            "Dacapo table indexing was not migrated with reader-compatible normalization");
    const auto &gpu_relinearize = gpu.spec.operators.at(ComputeKind::Relinearize);
    require(gpu_relinearize.supported && gpu_relinearize.latency_us_by_level &&
                gpu_relinearize.latency_us_by_level->at(3) == 1 &&
                !gpu_relinearize.noise_by_level,
            "Dacapo relinearize placeholder support was not migrated");
    const auto &gpu_boot = gpu.spec.boot_profiles.at(0);
    require(gpu_boot.input_level_min == 3 && gpu_boot.input_level_max == 16 &&
                gpu_boot.output_level == 16,
            "Dacapo Earth-to-CKKS bootstrap level conversion is wrong");
    require(gpu_boot.latency_us_by_input_level &&
                gpu_boot.latency_us_by_input_level->at(5) == 294928,
            "Dacapo per-level bootstrap latency was not migrated");
    RuntimePlan v2_plan;
    v2_plan.plan_id = 2001;
    v2_plan.target.target_id = gpu.spec.target_id;
    v2_plan.target.capability_version = 1;
    v2_plan.target.operator_spec = {gpu.spec.id, gpu.spec.version, gpu.source_sha256};
    v2_plan.target.world_size = 1;
    v2_plan.target.device_counts = {0};
    const Place host{PlaceKind::Host, 0, 0};
    v2_plan.values.push_back({1, ValueKind::Ciphertext, host,
                              gpu.spec.context_id, 3, 20, true, 2});
    v2_plan.values.push_back({2, ValueKind::Ciphertext, host,
                              gpu.spec.context_id, 3, 20, true, 2});
    v2_plan.values.push_back({3, ValueKind::Ciphertext, host,
                              gpu.spec.context_id, 3, 40, true, 3});
    v2_plan.values.push_back({4, ValueKind::Ciphertext, host,
                              gpu.spec.context_id, 3, 40, true, 2});
    v2_plan.external_inputs = {1, 2};
    v2_plan.execution.push_back({0, ComputeOp{ComputeKind::MulCC, {1, 2}, 3, host, {}}});
    v2_plan.execution.push_back({1, ComputeOp{ComputeKind::Relinearize, {3}, 4, host, {}}});
    v2_plan.final_outputs = {4};
    PlanVerifier::verify(v2_plan, gpu);

    const auto seal_path = profile_dir / "dacapo-seal-cpu.v1.json";
    const auto seal = OperatorSpecReader::read_file(seal_path.string());
    require(seal.spec.noise_unit && *seal.spec.noise_unit == "dacapo-legacy-estimator",
            "Dacapo SEAL noise unit was not read");
    const auto &seal_add = seal.spec.operators.at(ComputeKind::AddCC);
    require(seal_add.latency_us_by_level && seal_add.latency_us_by_level->at(1) == 85 &&
                seal_add.latency_us_by_level->at(13) == 3120,
            "Dacapo SEAL latency table was not migrated");
    const auto &seal_rotate = seal.spec.operators.at(ComputeKind::Rotate);
    require(seal_rotate.noise_by_level &&
                std::abs(seal_rotate.noise_by_level->at(1) - 1243767652.125024) < 1e-6 &&
                std::abs(seal_rotate.noise_by_level->at(13) - 19223819509.410683) < 1e-6,
            "Dacapo SEAL noise table was not migrated");
    require(seal.spec.boot_profiles.at(0).implementation == BootImplementation::DecryptReencrypt &&
                !seal.spec.boot_profiles.at(0).latency_us_by_input_level,
            "Dacapo SEAL Boot semantics were not migrated");

    std::ifstream input(seal_path);
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const std::string unit = "\"noise_unit\": \"dacapo-legacy-estimator\"";
    const auto position = text.find(unit);
    require(position != std::string::npos, "V2 noise unit fixture was not found");
    text.replace(position, unit.size(), "\"noise_unit\": null");
    expect_throw([&] { OperatorSpecReader::read_text(text); }, "must be set exactly");
}

void copy_fixture(const std::filesystem::path &source, const std::filesystem::path &destination) {
    std::filesystem::create_directories(destination.parent_path());
    std::filesystem::copy_file(source, destination, std::filesystem::copy_options::overwrite_existing);
}

void test_rank_local_bundle_loading() {
    const auto loaded = RuntimePlanJsonReader::read_file(testdata("valid", "v003_bundle_multi_rank.json").string());
    const auto bundle = source_dir / "docs/runtime-plan/v1/testdata/bundles/v005-demo";
    const auto &content = std::get<BundleEncodePayload>(std::get<EncodeOp>(loaded.plan.initialization.front().body).payload).content;
    const auto temp = std::filesystem::temp_directory_path() / "ckks-runtime-bundle-json-test";
    std::filesystem::remove_all(temp);
    copy_fixture(bundle / "manifest.json", temp / "manifest.json");
    copy_fixture(bundle / "data" / (content.substr(7) + ".bin"), temp / "data" / (content.substr(7) + ".bin"));
    const auto result = PlaintextBundleLoader::load(temp, *loaded.plan.plaintext_bundle, {content}, 16384, false);
    require(result.slots_by_content.at(content).size() == 4, "local bundle blob was not loaded");
    require(!std::signbit(result.slots_by_content.at(content)[1]), "bundle negative zero was not normalized");

    std::ifstream manifest(bundle / "manifest.json");
    const std::string manifest_text((std::istreambuf_iterator<char>(manifest)), std::istreambuf_iterator<char>());
    const auto marker = manifest_text.find("sha256:", manifest_text.find(content) + content.size());
    require(marker != std::string::npos, "second manifest content was not found");
    const std::string other = manifest_text.substr(marker, 71);
    expect_throw([&] { PlaintextBundleLoader::load(temp, *loaded.plan.plaintext_bundle, {other}, 16384, false); }, "cannot open file");

    const auto blob_path = temp / "data" / (content.substr(7) + ".bin");
    std::filesystem::remove(blob_path);
    expect_throw([&] { PlaintextBundleLoader::load(temp, *loaded.plan.plaintext_bundle, {content}, 16384, false); }, "cannot open file");
    copy_fixture(bundle / "data" / (content.substr(7) + ".bin"), blob_path);
    {
        std::ofstream output(blob_path, std::ios::binary | std::ios::trunc);
        output.write("short", 5);
    }
    expect_throw([&] { PlaintextBundleLoader::load(temp, *loaded.plan.plaintext_bundle, {content}, 16384, false); }, "byte length mismatch");
    copy_fixture(bundle / "data" / (content.substr(7) + ".bin"), blob_path);
    {
        std::fstream output(blob_path, std::ios::binary | std::ios::in | std::ios::out);
        char byte = 0;
        output.read(&byte, 1);
        byte ^= 1;
        output.seekp(0);
        output.write(&byte, 1);
    }
    expect_throw([&] { PlaintextBundleLoader::load(temp, *loaded.plan.plaintext_bundle, {content}, 16384, false); }, "content SHA-256 mismatch");
    expect_throw([&] { PlaintextBundleLoader::load(temp, *loaded.plan.plaintext_bundle, {content}, 16384, true); }, "content SHA-256 mismatch");

    copy_fixture(bundle / "data" / (content.substr(7) + ".bin"), blob_path);
    const auto bad_ref_plan = RuntimePlanJsonReader::read_file(testdata("invalid", "i012_manifest_digest_mismatch.json").string());
    expect_throw([&] { PlaintextBundleLoader::load(temp, *bad_ref_plan.plan.plaintext_bundle, {content}, 16384, false); }, "manifest SHA-256 mismatch");
    PlaintextBundleLoader::load(temp, *bad_ref_plan.plan.plaintext_bundle, {content}, 16384, true);
    auto wrong_id = *bad_ref_plan.plan.plaintext_bundle;
    wrong_id.id = "wrong-bundle";
    expect_throw([&] { PlaintextBundleLoader::load(temp, wrong_id, {content}, 16384, true); }, "does not match RuntimePlan reference");
    std::filesystem::remove_all(temp);
}

} // namespace

int main() {
    try {
        run_test("SHA-256 covers raw file bytes", test_sha256_and_raw_bytes);
        run_test("valid RuntimePlan V1 samples", test_valid_samples);
        run_test("invalid RuntimePlan V1 samples", test_invalid_samples);
        run_test("strict JSON and OperatorSpec constraints", test_strict_json_and_operator_spec);
        run_test("Dacapo OperatorSpec V2 profiles", test_dacapo_operator_spec_v2_profiles);
        run_test("rank-local bundle loading", test_rank_local_bundle_loading);
        std::cout << "ALL " << tests_run << " JSON TEST GROUPS PASSED\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
