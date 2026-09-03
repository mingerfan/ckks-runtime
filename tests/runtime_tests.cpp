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

std::vector<double> padded(std::vector<double> values, std::size_t size) {
    values.resize(size, 0.0);
    return values;
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
    require(std::find(requirements.keys.begin(), requirements.keys.end(), KeyRequirement{KeyKind::Relin, {PlaceKind::Device, 0, 0}, std::nullopt, 5}) != requirements.keys.end(), "level-specific relin key was not derived");
    require(std::find(requirements.keys.begin(), requirements.keys.end(), KeyRequirement{KeyKind::Galois, {PlaceKind::Device, 0, 0}, 16383, 4}) != requirements.keys.end(), "level-specific normalized Galois key was not derived");

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
    const auto input = make_cipher(padded({10, 20, 30, 40}, spec.spec.poly_degree / 2),
                                   "ctx-main", 32768, 5, 40);
    const auto result = run_mock_cluster(loaded.plan, spec, {{1, input}}, {}, {}, DiffMode::AllValuesAfterRun);
    const auto encoded = result.artifacts[0].values.at(2).value.materialize();
    const std::vector<double> encoded_prefix{1, 0, 3.5, -2};
    require(encoded.slots.size() == spec.spec.poly_degree / 2,
            "inline Encode did not fill the CKKS slot capacity");
    require(std::equal(encoded_prefix.begin(), encoded_prefix.end(), encoded.slots.begin()),
            "inline Encode prefix is wrong");
    require(std::all_of(encoded.slots.begin() + 4, encoded.slots.end(),
                        [](double value) { return value == 0.0; }),
            "inline Encode padding is not zero");
    require(!std::signbit(encoded.slots[1]), "Encode did not normalize negative zero");
    const auto computed = result.artifacts[0].values.at(3).value.materialize();
    const std::vector<double> computed_prefix{11, 20, 33.5, 38};
    require(std::equal(computed_prefix.begin(), computed_prefix.end(), computed.slots.begin()),
            "Host AddCP result prefix is wrong");
    require(std::all_of(computed.slots.begin() + 4, computed.slots.end(),
                        [](double value) { return value == 0.0; }),
            "Host AddCP padding is not zero");
    require(result.stats[0].compute_calls == 1, "Host compute was not executed");
    const auto &timing = result.artifacts[0].timing;
    require(timing.compute_calls == 1 && timing.boot_calls == 0,
            "Host compute timing counts are wrong");
    require(timing.boot_nanoseconds == 0 &&
                timing.compute_excluding_boot_nanoseconds() ==
                    timing.compute_including_boot_nanoseconds,
            "Host compute timing split is wrong");
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
    const std::vector<double> bundle_prefix{1, 0, 3.5, -2};
    require(first.slots.size() == reuse_spec.spec.poly_degree / 2 && first.slots == second.slots &&
            std::equal(bundle_prefix.begin(), bundle_prefix.end(), first.slots.begin()) &&
            std::all_of(first.slots.begin() + 4, first.slots.end(),
                        [](double value) { return value == 0.0; }) &&
            first.metadata.level == 5 && second.metadata.level == 4 &&
            first.metadata.scale_log2 == 40 && second.metadata.scale_log2 == 20,
            "one bundle content was not encoded with two ValueDesc settings");

    const auto multi = RuntimePlanJsonReader::read_file(plan_path("valid", "v003_bundle_multi_rank.json").string());
    const auto multi_spec = load_spec(multi.plan);
    const auto rank0 = make_rank_bundle("ckks-runtime-rank0", {});
    const auto rank1 = make_rank_bundle("ckks-runtime-rank1", {content});
    const auto input = make_cipher(padded({2, 3, 4, 5}, multi_spec.spec.poly_degree / 2),
                                   "ctx-main", 32768, 5, 40);
    const auto result = run_mock_cluster(multi.plan, multi_spec, {{1, input}}, {}, {}, DiffMode::FinalOnly,
                                         false, {rank0, rank1});
    const auto multi_output = result.artifacts[1].values.at(7).value.materialize();
    const std::vector<double> multi_prefix{2, 0, 14, -10};
    require(std::equal(multi_prefix.begin(), multi_prefix.end(), multi_output.slots.begin()) &&
            std::all_of(multi_output.slots.begin() + 4, multi_output.slots.end(),
                        [](double value) { return value == 0.0; }),
            "multi-rank bundle execution failed");
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
        const auto &timing = result.artifacts[0].timing;
        require(timing.compute_calls == 1 && timing.boot_calls == 1,
                name + " timing counts are wrong");
        require(timing.compute_including_boot_nanoseconds == timing.boot_nanoseconds &&
                    timing.compute_excluding_boot_nanoseconds() == 0,
                name + " Boot timing split is wrong");
    }
}

void test_preflight_and_digest_debug_mode() {
    const auto plan = RuntimePlanJsonReader::read_file(plan_path("valid", "v002_device_mul_relin_rescale_rotate.json").string());
    const auto spec = load_spec(plan.plan);
    const auto input = make_cipher({1, 2, 3, 4}, "ctx-main", 32768, 5, 40);
    MockClusterConfig missing;
    missing.missing_keys.insert(KeyRequirement{KeyKind::Galois, {PlaceKind::Device, 0, 0}, 16383, 4});
    expect_throw([&] { run_mock_cluster(plan.plan, spec, {{1, input}}, missing, {}, DiffMode::FinalOnly); }, "rotation step 16383");
    missing = {};
    missing.missing_capabilities.insert(RequiredCapability::Transfer);
    expect_throw([&] { run_mock_cluster(plan.plan, spec, {{1, input}}, missing, {}, DiffMode::FinalOnly); }, "required capability: transfer");

    const auto bad_spec_plan = RuntimePlanJsonReader::read_file(plan_path("invalid", "i011_operator_spec_digest_mismatch.json").string());
    const auto bad_plan_spec = load_spec(bad_spec_plan.plan);
    expect_throw([&] {
        run_mock_cluster(bad_spec_plan.plan, bad_plan_spec,
            {{1, make_cipher(padded({1, 2}, bad_plan_spec.spec.poly_degree / 2),
                             "ctx-main", 32768, 5, 40)}}, {}, {}, DiffMode::FinalOnly);
    }, "source SHA-256 mismatch");
    run_mock_cluster(bad_spec_plan.plan, bad_plan_spec,
        {{1, make_cipher(padded({1, 2, 3, 4}, bad_plan_spec.spec.poly_degree / 2),
                         "ctx-main", 32768, 5, 40)}}, {}, {}, DiffMode::FinalOnly, true);
    auto wrong_id_spec = bad_plan_spec;
    wrong_id_spec.spec.id = "wrong-spec";
    expect_throw([&] {
        run_mock_cluster(bad_spec_plan.plan, wrong_id_spec,
            {{1, make_cipher(padded({1, 2}, wrong_id_spec.spec.poly_degree / 2),
                             "ctx-main", 32768, 5, 40)}}, {}, {}, DiffMode::FinalOnly, true);
    }, "id/version mismatch");

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

void test_multi_rank_device_workers() {
    const auto built = make_fanout_plan({2, 1});
    const auto cipher = make_cipher({1, 2, 3, 4}, "ctx", 8192, 3, 1);
    const auto plain = make_plain({2, 3, 4, 5}, "ctx", 8192, 3, 1);
    const auto result = run_mock_cluster(
        built.plan, built.operator_spec, {{0, cipher}, {1, plain}}, {}, {},
        DiffMode::FinalOnly, false, {}, DeviceExecutionMode::PerDeviceWorkers);
    compare_values(
        result.artifacts.back().values.at(built.plan.final_outputs[0]).value,
        run_fanout_reference(cipher, plain).at(built.reference_output));
    const auto &cross_rank =
        std::get<CommAction>(built.plan.execution.at(1).body);
    const int source_rank = cross_rank.sources.front().rank;
    const int source_device = cross_rank.sources.front().index;
    for (std::size_t rank = 0; rank < result.stats.size(); ++rank) {
        const auto &stats = result.stats[rank];
        require(stats.coordinator_compute_calls == 0,
                "device compute ran on the coordinator thread");
        require(stats.worker_compute_calls == stats.compute_calls,
                "device compute did not run on a worker thread");
        const auto issuer = stats.communication_threads.find(cross_rank.id);
        require(issuer != stats.communication_threads.end(),
                "cross-rank communication issuer was not recorded");
        const auto source_worker =
            stats.device_compute_threads.find(source_device);
        const auto is_device_worker = [&](std::thread::id thread) {
            return std::any_of(
                stats.device_compute_threads.begin(),
                stats.device_compute_threads.end(),
                [&](const auto &entry) { return entry.second == thread; });
        };
        const auto communication_issuer_count = static_cast<std::size_t>(
            std::count_if(
                issuer->second.begin(), issuer->second.end(),
                [&](std::thread::id thread) {
                    return thread != stats.runtime_thread &&
                           !is_device_worker(thread);
                }));
        require(communication_issuer_count == 1,
                "cross-rank communication did not use exactly one dedicated issuer");
        if (static_cast<int>(rank) == source_rank) {
            require(source_worker != stats.device_compute_threads.end(),
                    "source device worker was not recorded");
            require(std::count(
                        issuer->second.begin(), issuer->second.end(),
                        source_worker->second) == 1,
                    "same-rank copy in a mixed action was not submitted by the source worker");
        }
        require(stats.completed_handles == stats.communicate_calls,
                "parallel communication handle was not completed");
    }
    require(result.stats.back().worker_communicate_calls > 0,
            "local Device-to-Host copy did not run on a worker thread");
}

void test_multi_rank_bidirectional_dependency_chain() {
    auto built = make_fanout_plan({1, 1});
    auto &plan = built.plan;
    const auto old_final =
        std::get<CommAction>(plan.finalization.front().body);
    const ValueId rank1_branch = old_final.inputs.front();
    const ValueId old_final_output = old_final.outputs.front();
    const auto branch_desc = *std::find_if(
        plan.values.begin(), plan.values.end(),
        [&](const ValueDesc &value) { return value.id == rank1_branch; });
    plan.values.erase(
        std::remove_if(
            plan.values.begin(), plan.values.end(),
            [&](const ValueDesc &value) {
                return value.id == old_final_output;
            }),
        plan.values.end());
    ValueId next_value = std::max_element(
        plan.values.begin(), plan.values.end(),
        [](const ValueDesc &a, const ValueDesc &b) { return a.id < b.id; })->id + 1;

    const ValueId returned = next_value++;
    auto returned_desc = branch_desc;
    returned_desc.id = returned;
    returned_desc.place = {PlaceKind::Device, 0, 0};
    plan.values.push_back(returned_desc);

    const TransferId return_transfer = old_final.id;
    const std::size_t return_ordinal = plan.finalization.front().ordinal;
    plan.execution.push_back({
        return_ordinal,
        CommAction{return_transfer, CommKind::Transfer,
                   CommHint::PointToPoint, {rank1_branch}, {returned},
                   {branch_desc.place}, {returned_desc.place},
                   {ValueKind::Ciphertext}}});

    const ValueId rank0_result = next_value++;
    auto result_desc = returned_desc;
    result_desc.id = rank0_result;
    plan.values.push_back(result_desc);
    plan.execution.push_back({
        return_ordinal + 1,
        ComputeOp{ComputeKind::Negate, {returned}, rank0_result,
                  returned_desc.place, {}}});

    const ValueId final = next_value++;
    auto final_desc = result_desc;
    final_desc.id = final;
    final_desc.place = {PlaceKind::Host, 0, 0};
    plan.values.push_back(final_desc);
    plan.finalization = {{
        return_ordinal + 2,
        CommAction{return_transfer + 1, CommKind::Transfer, CommHint::Auto,
                   {rank0_result}, {final}, {result_desc.place},
                   {final_desc.place}, {ValueKind::Ciphertext}}}};
    plan.final_outputs = {final};

    const auto cipher = make_cipher({1, 2, 3, 4}, "ctx", 8192, 3, 1);
    const auto plain = make_plain({2, 3, 4, 5}, "ctx", 8192, 3, 1);
    MockClusterConfig config;
    // Delay the forward publish so the reverse send is posted first. This
    // catches communication implementations that wait for a source while
    // holding the receive-side message lock.
    config.delay_seed = 64;
    config.max_delay_ms = 7;
    const auto result = run_mock_cluster(
        plan, built.operator_spec, {{0, cipher}, {1, plain}}, config,
        VecExecConfig{VecExecMode::Async, 31, 3}, DiffMode::FinalOnly,
        false, {}, DeviceExecutionMode::PerDeviceWorkers);
    compare_values(
        result.artifacts.front().values.at(final).value,
        run_fanout_reference(cipher, plain).at(2));

    const auto &forward = std::get<CommAction>(plan.execution.at(1).body);
    for (const auto &stats : result.stats) {
        const auto forward_threads =
            stats.communication_threads.find(forward.id);
        const auto return_threads =
            stats.communication_threads.find(return_transfer);
        require(forward_threads != stats.communication_threads.end() &&
                    forward_threads->second.size() == 1 &&
                    return_threads != stats.communication_threads.end() &&
                    return_threads->second.size() == 1 &&
                    forward_threads->second.front() ==
                        return_threads->second.front(),
                "bidirectional communication did not use one ordered issuer");
        require(forward_threads->second.front() != stats.runtime_thread,
                "bidirectional communication ran on the Runtime main thread");
        require(std::none_of(
                    stats.device_compute_threads.begin(),
                    stats.device_compute_threads.end(),
                    [&](const auto &entry) {
                        return entry.second == forward_threads->second.front();
                    }),
                "bidirectional communication ran on a device worker");
        for (TransferId id : {forward.id, return_transfer}) {
            const auto waits = stats.wait_compute_calls.find(id);
            require(waits != stats.wait_compute_calls.end() &&
                        std::all_of(
                            waits->second.begin(), waits->second.end(),
                            [](std::size_t compute_calls) {
                                return compute_calls > 0;
                            }),
                    "communication completion was waited before dependent compute was submitted");
        }
    }
}

void test_single_rank_device_worker_replicate() {
    const auto built = make_fanout_plan({3});
    const auto cipher = make_cipher({1, 2, 3, 4}, "ctx", 8192, 3, 1);
    const auto plain = make_plain({2, 3, 4, 5}, "ctx", 8192, 3, 1);
    const auto result = run_mock_cluster(
        built.plan, built.operator_spec, {{0, cipher}, {1, plain}}, {}, {},
        DiffMode::FinalOnly, false, {}, DeviceExecutionMode::PerDeviceWorkers);
    compare_values(
        result.artifacts.front().values.at(built.plan.final_outputs[0]).value,
        run_fanout_reference(cipher, plain).at(built.reference_output));
    const auto &stats = result.stats.front();
    require(stats.worker_compute_calls == stats.compute_calls,
            "single-rank compute did not run on workers");
    require(stats.worker_communicate_calls >= 3,
            "local Replicate was not split across device workers");
    require(stats.completed_handles == stats.communicate_calls,
            "local Replicate left an incomplete communication handle");

    const auto &replicate =
        std::get<CommAction>(built.plan.execution.at(1).body);
    const int source_device = replicate.sources.front().index;
    const auto compute_thread = stats.device_compute_threads.find(source_device);
    const auto communication_threads =
        stats.communication_threads.find(replicate.id);
    require(compute_thread != stats.device_compute_threads.end(),
            "source device worker thread was not recorded");
    require(communication_threads != stats.communication_threads.end() &&
                communication_threads->second.size() == replicate.outputs.size(),
            "local Replicate communication threads were not recorded");
    require(std::all_of(
                communication_threads->second.begin(),
                communication_threads->second.end(),
                [&](std::thread::id thread) {
                    return thread == compute_thread->second;
                }),
            "local Device-to-Device copy was not submitted by its source worker");
}

void test_multi_rank_device_worker_failures() {
    const auto built = make_fanout_plan({2, 1});
    const auto cipher = make_cipher({1, 2, 3, 4}, "ctx", 8192, 3, 1);
    const auto plain = make_plain({2, 3, 4, 5}, "ctx", 8192, 3, 1);
    const auto run = [&](MockClusterConfig config) {
        return run_mock_cluster(
            built.plan, built.operator_spec, {{0, cipher}, {1, plain}},
            std::move(config), {}, DiffMode::FinalOnly, false, {},
            DeviceExecutionMode::PerDeviceWorkers);
    };
    MockClusterConfig compute_failure;
    compute_failure.fail_compute = ComputeKind::Negate;
    expect_throw([&] { run(compute_failure); }, "injected compute failure");

    MockClusterConfig communication_failure;
    communication_failure.fail_communicate.insert(102);
    expect_throw([&] { run(communication_failure); },
                 "injected communicate_async failure");
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
        run_test("multi-rank per-device workers", test_multi_rank_device_workers);
        run_test("multi-rank bidirectional dependency chain",
                 test_multi_rank_bidirectional_dependency_chain);
        run_test("single-rank worker Replicate", test_single_rank_device_worker_replicate);
        run_test("multi-rank device-worker failures", test_multi_rank_device_worker_failures);
        std::cout << "ALL " << tests_run << " TEST GROUPS PASSED\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
