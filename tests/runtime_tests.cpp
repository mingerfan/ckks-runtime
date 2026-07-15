#include "runtime/json_plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "testing/testing.hpp"

#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

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

std::filesystem::path plan_path(const char *group, const std::string &name) {
    return source_dir / "docs/runtime-plan/v1/testdata" / group / name;
}

LoadedOperatorSpec load_spec(const RuntimePlan &plan) {
    const std::string filename = plan.target.operator_spec.id == "poseidon-ckks-cpu-v1"
        ? "poseidon-ckks-cpu.v1.json" : "poseidon-ckks-gpu.v1.json";
    return OperatorSpecReader::read_file((source_dir / "docs/operator-spec/v1/profiles" / filename).string());
}

void copy_fixture(const std::filesystem::path &source, const std::filesystem::path &destination) {
    std::filesystem::create_directories(destination.parent_path());
    std::filesystem::copy_file(source, destination, std::filesystem::copy_options::overwrite_existing);
}

std::filesystem::path make_rank_bundle(const std::string &name, const std::vector<std::string> &contents) {
    const auto source = source_dir / "docs/runtime-plan/v1/testdata/bundles/v005-demo";
    const auto destination = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(destination);
    copy_fixture(source / "manifest.json", destination / "manifest.json");
    for (const auto &content : contents)
        copy_fixture(source / "data" / (content.substr(7) + ".bin"), destination / "data" / (content.substr(7) + ".bin"));
    return destination;
}

void test_requirements_and_vec_operations() {
    const auto loaded = RuntimePlanJsonReader::read_file(plan_path("valid", "v002_device_mul_relin_rescale_rotate.json").string());
    const auto spec = load_spec(loaded.plan);
    const auto requirements = PlanVerifier::verify(loaded.plan, spec);
    require(std::find(requirements.capabilities.begin(), requirements.capabilities.end(), RequiredCapability::Transfer) != requirements.capabilities.end(), "Transfer capability was not derived");
    require(std::find(requirements.keys.begin(), requirements.keys.end(), KeyRequirement{KeyKind::Relin, {PlaceKind::Device, 0, 0}, std::nullopt}) != requirements.keys.end(), "relin key was not derived");
    require(std::find(requirements.keys.begin(), requirements.keys.end(), KeyRequirement{KeyKind::Galois, {PlaceKind::Device, 0, 0}, 16383}) != requirements.keys.end(), "normalized Galois step was not derived");

    VecExecutor executor;
    const Place place{PlaceKind::Host, 0, 0};
    const auto ct = make_cipher({1, 2, 3, 4}, "ctx", 8192, 3, 1);
    const auto pt = make_plain({2, 3, 4, 5}, "ctx", 8192, 3, 1);
    const auto result = executor.compute(ComputeOp{ComputeKind::MulCP, {}, 1, place, {}}, {ct, pt}).materialize();
    require(result.slots == std::vector<double>({2, 6, 12, 20}) && result.metadata.scale_log2 == 2, "Vec MulCP failed");
}

void test_all_compute_metadata_rules() {
    const auto fixture = make_fanout_plan({1});
    const Place host{PlaceKind::Host, 0, 0};
    const auto verify_case = [&](ComputeKind kind, ValueDesc first, std::optional<ValueDesc> second,
                                 ValueDesc output, ComputeAttrs attrs = {}) {
        RuntimePlan plan;
        plan.plan_id = static_cast<std::uint64_t>(kind) + 1000;
        plan.target = fixture.plan.target;
        first.id = 1;
        output.id = 3;
        plan.values.push_back(first);
        plan.external_inputs.push_back(1);
        std::vector<ValueId> inputs{1};
        if (second) {
            second->id = 2;
            plan.values.push_back(*second);
            plan.external_inputs.push_back(2);
            inputs.push_back(2);
        }
        plan.values.push_back(output);
        plan.execution.push_back({0, ComputeOp{kind, inputs, 3, host, std::move(attrs)}});
        plan.final_outputs = {3};
        PlanVerifier::verify(plan, fixture.operator_spec);
    };
    const auto ct = ValueDesc{0, ValueKind::Ciphertext, host, "ctx", 3, 1, true, 2};
    const auto pt = ValueDesc{0, ValueKind::Plaintext, host, "ctx", 3, 1, true, 1};
    verify_case(ComputeKind::AddCC, ct, ct, ct);
    verify_case(ComputeKind::AddCP, ct, pt, ct);
    verify_case(ComputeKind::SubCC, ct, ct, ct);
    verify_case(ComputeKind::SubCP, ct, pt, ct);
    auto mul_cc = ct; mul_cc.scale_log2 = 2; mul_cc.components = 3;
    verify_case(ComputeKind::MulCC, ct, ct, mul_cc);
    auto mul_cp = ct; mul_cp.scale_log2 = 2;
    verify_case(ComputeKind::MulCP, ct, pt, mul_cp);
    verify_case(ComputeKind::Negate, ct, std::nullopt, ct);
    verify_case(ComputeKind::Rotate, ct, std::nullopt, ct, RotateAttrs{1});
    auto lowered = ct; lowered.level = 2;
    verify_case(ComputeKind::Rescale, ct, std::nullopt, lowered, RescaleAttrs{2, 1});
    verify_case(ComputeKind::ModSwitch, ct, std::nullopt, lowered, ModSwitchAttrs{2});
    auto unrelinearized = ct; unrelinearized.components = 3;
    verify_case(ComputeKind::Relinearize, unrelinearized, std::nullopt, ct);
    auto booted = ct; booted.level = 4;
    verify_case(ComputeKind::Boot, ct, std::nullopt, booted,
                BootAttrs{4, 1, 2, "test-boot", BootImplementation::DecryptReencrypt});

    auto bad_output = ct;
    bad_output.scale_log2 = 2;
    expect_throw([&] { verify_case(ComputeKind::AddCC, ct, ct, bad_output); }, "metadata rule failed");
}

void test_inline_encode_and_host_compute() {
    const auto loaded = RuntimePlanJsonReader::read_file(plan_path("valid", "v001_inline_encode_host_compute.json").string());
    const auto spec = load_spec(loaded.plan);
    const auto input = make_cipher({10, 20, 30, 40}, "ctx-main", 32768, 5, 40);
    const auto result = run_mock_cluster(loaded.plan, spec, {{1, input}}, {}, {}, DiffMode::AllValuesAfterRun);
    const auto encoded = result.artifacts[0].values.at(2).value.materialize();
    require(encoded.slots == std::vector<double>({1, 0, 3.5, -2}), "inline Encode slots are wrong");
    require(!std::signbit(encoded.slots[1]), "Encode did not normalize negative zero");
    require(result.artifacts[0].values.at(3).value.materialize().slots == std::vector<double>({11, 20, 33.5, 38}), "Host AddCP result is wrong");
    require(result.stats[0].compute_calls == 1, "Host compute was not executed");
}

void test_device_compute_and_value_validation() {
    const auto loaded = RuntimePlanJsonReader::read_file(plan_path("valid", "v002_device_mul_relin_rescale_rotate.json").string());
    const auto spec = load_spec(loaded.plan);
    const auto input = make_cipher({1, 2, 3, 4}, "ctx-main", 32768, 5, 40);
    const auto result = run_mock_cluster(loaded.plan, spec, {{1, input}}, {}, {}, DiffMode::FinalOnly);
    const auto output = result.artifacts[0].values.at(7).value.materialize();
    require(output.metadata.level == 4 && output.metadata.scale_log2 == 40 && output.metadata.components == 2, "Device compute metadata is wrong");

    MockClusterConfig corrupt;
    corrupt.corrupt_output_metadata.insert(100);
    expect_throw([&] { run_mock_cluster(loaded.plan, spec, {{1, input}}, corrupt, {}, DiffMode::FinalOnly); }, "metadata");
    expect_throw([&] {
        run_mock_cluster(loaded.plan, spec,
            {{1, make_cipher({1, 2, 3, 4}, "wrong", 32768, 5, 40)}}, {}, {}, DiffMode::FinalOnly);
    }, "metadata");
}

void test_bundle_reuse_and_rank_local_files() {
    const auto reuse = RuntimePlanJsonReader::read_file(plan_path("valid", "v005_bundle_reuse.json").string());
    const auto reuse_spec = load_spec(reuse.plan);
    const auto content = std::get<BundleEncodePayload>(std::get<EncodeOp>(reuse.plan.initialization[0].body).payload).content;
    const auto local = make_rank_bundle("ckks-runtime-reuse", {content});
    const auto reuse_result = run_mock_cluster(reuse.plan, reuse_spec, {}, {}, {}, DiffMode::FinalOnly, false, {local});
    const auto first = reuse_result.artifacts[0].values.at(1).value.materialize();
    const auto second = reuse_result.artifacts[0].values.at(2).value.materialize();
    require(first.slots == std::vector<double>({1, 0, 3.5, -2}) && first.slots == second.slots &&
            first.metadata.level == 5 && second.metadata.level == 4 &&
            first.metadata.scale_log2 == 40 && second.metadata.scale_log2 == 20,
            "one bundle content was not encoded with two ValueDesc settings");

    const auto multi = RuntimePlanJsonReader::read_file(plan_path("valid", "v003_bundle_multi_rank.json").string());
    const auto multi_spec = load_spec(multi.plan);
    const auto rank0 = make_rank_bundle("ckks-runtime-rank0", {});
    const auto rank1 = make_rank_bundle("ckks-runtime-rank1", {content});
    const auto input = make_cipher({2, 3, 4, 5}, "ctx-main", 32768, 5, 40);
    const auto result = run_mock_cluster(multi.plan, multi_spec, {{1, input}}, {}, {}, DiffMode::FinalOnly,
                                         false, {rank0, rank1});
    require(result.artifacts[1].values.at(7).value.materialize().slots == std::vector<double>({2, 0, 14, -10}), "multi-rank bundle execution failed");
    std::filesystem::remove_all(local);
    std::filesystem::remove_all(rank0);
    std::filesystem::remove_all(rank1);
}

void test_boot_paths() {
    for (const auto &name : {std::string("v004_host_boot.json"), std::string("v006_native_boot_device.json")}) {
        const auto loaded = RuntimePlanJsonReader::read_file(plan_path("valid", name).string());
        const auto spec = load_spec(loaded.plan);
        const auto input = make_cipher({1, 2}, "ctx-main", 32768, 2, 40);
        const auto result = run_mock_cluster(loaded.plan, spec, {{1, input}}, {}, {}, DiffMode::FinalOnly);
        const auto output = result.artifacts[0].values.at(loaded.plan.final_outputs[0]).value.materialize();
        require(output.metadata.level == 12 && output.metadata.scale_log2 == 40, name + " did not execute");
    }
}

void test_preflight_and_digest_debug_mode() {
    const auto plan = RuntimePlanJsonReader::read_file(plan_path("valid", "v002_device_mul_relin_rescale_rotate.json").string());
    const auto spec = load_spec(plan.plan);
    const auto input = make_cipher({1, 2, 3, 4}, "ctx-main", 32768, 5, 40);
    MockClusterConfig missing;
    missing.missing_keys.insert(KeyRequirement{KeyKind::Galois, {PlaceKind::Device, 0, 0}, 16383});
    expect_throw([&] { run_mock_cluster(plan.plan, spec, {{1, input}}, missing, {}, DiffMode::FinalOnly); }, "rotation step 16383");
    missing = {};
    missing.missing_capabilities.insert(RequiredCapability::Transfer);
    expect_throw([&] { run_mock_cluster(plan.plan, spec, {{1, input}}, missing, {}, DiffMode::FinalOnly); }, "required capability: transfer");

    const auto bad_spec_plan = RuntimePlanJsonReader::read_file(plan_path("invalid", "i011_operator_spec_digest_mismatch.json").string());
    const auto bad_plan_spec = load_spec(bad_spec_plan.plan);
    expect_throw([&] { run_mock_cluster(bad_spec_plan.plan, bad_plan_spec, {{1, make_cipher({1, 2}, "ctx-main", 32768, 5, 40)}}, {}, {}, DiffMode::FinalOnly); }, "source SHA-256 mismatch");
    run_mock_cluster(bad_spec_plan.plan, bad_plan_spec, {{1, make_cipher({1, 2, 3, 4}, "ctx-main", 32768, 5, 40)}}, {}, {}, DiffMode::FinalOnly, true);
    auto wrong_id_spec = bad_plan_spec;
    wrong_id_spec.spec.id = "wrong-spec";
    expect_throw([&] { run_mock_cluster(bad_spec_plan.plan, wrong_id_spec, {{1, make_cipher({1, 2}, "ctx-main", 32768, 5, 40)}}, {}, {}, DiffMode::FinalOnly, true); }, "id/version mismatch");

    auto check_cluster_preflight = [](bool skip0, bool skip1, std::string digest1, const std::string &needle) {
        MockClusterConfig config;
        config.world_size = 2;
        auto cluster = std::make_shared<MockCluster>(config);
        std::exception_ptr errors[2];
        std::thread a([&] { try { cluster->preflight(0, "sha256:" + std::string(64, '1'), skip0); } catch (...) { errors[0] = std::current_exception(); } });
        std::thread b([&] { try { cluster->preflight(1, std::move(digest1), skip1); } catch (...) { errors[1] = std::current_exception(); } });
        a.join(); b.join();
        require(errors[0] || errors[1], "preflight mismatch did not fail");
        bool found = false;
        for (auto error : errors) if (error) try { std::rethrow_exception(error); }
            catch (const std::exception &e) { found = found || std::string(e.what()).find(needle) != std::string::npos; }
        require(found, "preflight error did not mention " + needle);
    };
    check_cluster_preflight(false, false, "sha256:" + std::string(64, '2'), "SHA-256 mismatch");
    check_cluster_preflight(false, true, "sha256:" + std::string(64, '1'), "skip_artifact_digest_checks mismatch");

    MockClusterConfig debug_config;
    debug_config.world_size = 2;
    auto debug_cluster = std::make_shared<MockCluster>(debug_config);
    std::exception_ptr debug_errors[2];
    std::thread debug_a([&] { try { debug_cluster->preflight(0, "sha256:" + std::string(64, '1'), true); } catch (...) { debug_errors[0] = std::current_exception(); } });
    std::thread debug_b([&] { try { debug_cluster->preflight(1, "sha256:" + std::string(64, '2'), true); } catch (...) { debug_errors[1] = std::current_exception(); } });
    debug_a.join(); debug_b.join();
    require(!debug_errors[0] && !debug_errors[1], "debug mode did not skip cross-rank plan digest mismatch");
}

void test_mock_sync_async_matrix() {
    for (const auto mode : {VecExecMode::Sync, VecExecMode::Async}) {
        const auto built = make_fanout_plan({2, 1});
        const auto cipher = make_cipher({1, 2, 3, 4}, "ctx", 8192, 3, 1);
        const auto plain = make_plain({2, 3, 4, 5}, "ctx", 8192, 3, 1);
        const auto result = run_mock_cluster(built.plan, built.operator_spec, {{0, cipher}, {1, plain}}, {},
                                             VecExecConfig{mode, 17, 2}, DiffMode::FinalOnly);
        compare_values(result.artifacts.back().values.at(built.plan.final_outputs[0]).value,
                       run_fanout_reference(cipher, plain).at(built.reference_output));
    }
}

} // namespace

int main() {
    try {
        run_test("requirements and Vec operations", test_requirements_and_vec_operations);
        run_test("all compute metadata rules", test_all_compute_metadata_rules);
        run_test("inline Encode and Host compute", test_inline_encode_and_host_compute);
        run_test("Device compute and full value validation", test_device_compute_and_value_validation);
        run_test("bundle reuse and rank-local files", test_bundle_reuse_and_rank_local_files);
        run_test("Host and Device Boot paths", test_boot_paths);
        run_test("preflight and digest debug mode", test_preflight_and_digest_debug_mode);
        run_test("Mock sync/async multi-rank matrix", test_mock_sync_async_matrix);
        std::cout << "ALL " << tests_run << " TEST GROUPS PASSED\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
