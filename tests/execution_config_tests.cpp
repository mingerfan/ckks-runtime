// ExecutionConfig V1: what the runtime accepts, and what it refuses.
//
// The refusals are the point.  A sidecar that is read leniently is worse than
// no sidecar: it lets a run be configured by a file generated for a different
// program, and the failure surfaces as a wrong number rather than as an error.

#include "runtime/json_execution_config_reader.hpp"
#include "runtime/json_plan_reader.hpp"
#include "runtime/runtime.hpp"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <iostream>
#include <set>
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
            throw std::runtime_error("exception did not contain '" + needle + "': " +
                                     error.what());
        return;
    }
    throw std::runtime_error("expected exception was not thrown");
}

void run_test(const char *name, const std::function<void()> &test) {
    test();
    ++tests_run;
    std::cout << "[PASS] " << name << '\n';
}

LoadedRuntimePlan single_device_plan() {
    return RuntimePlanJsonReader::read_file(
        (source_dir / "docs/runtime-plan/v1/testdata/valid"
                      "/v002_device_mul_relin_rescale_rotate.json").string());
}

const std::string kFakeDigest =
    "sha256:0000000000000000000000000000000000000000000000000000000000000000";

// A configuration that matches `plan` exactly.  Tests mutate one piece of it,
// so every rejection is attributable to the thing that was changed.
std::string config_text(const LoadedRuntimePlan &plan,
                        const std::string &plan_digest = {},
                        const std::string &parent = "null",
                        const std::string &runtime_section = {},
                        const std::string &devices_section = {},
                        const std::string &target_section = {}) {
    const std::string plan_sha =
        plan_digest.empty() ? plan.source_sha256 : plan_digest;
    const std::string target =
        target_section.empty()
            ? R"({"world_size": 1, "device_counts": [1], "rank": 0})"
            : target_section;
    const std::string runtime_body =
        runtime_section.empty()
            ? R"({"device_execution_mode": "sequential", "device_worker_count": 1})"
            : runtime_section;
    const std::string devices =
        devices_section.empty()
            ? R"([{"index": 0, "cuda_device_id": 3, "mempool_bytes": 1024,)"
              R"( "pinned_host_bytes": 0, "mig_instances": 1}])"
            : devices_section;

    return std::string("{\n") +
        R"("format_version": 1,)" "\n"
        R"("config_id": "t",)" "\n"
        R"("binding": {"plan_sha256": ")" + plan_sha + R"(",)"
        R"("parent_plan_sha256": )" + parent + R"(,)"
        R"("operator_spec_sha256": ")" + plan.plan.target.operator_spec.source_sha256 +
        R"(","decision_record_sha256": ")" + kFakeDigest + R"("},)" "\n"
        R"("target": )" + target + ",\n"
        R"("runtime": )" + runtime_body + ",\n"
        R"("devices": )" + devices + ",\n"
        R"("launch": {"grid_dim": 132, "l2_persist_bytes": 0, "l2_persist_hit_ratio": 1.0,)"
        R"( "num_streams": 1, "stream_priorities": [], "launch_mode": "cuda_graph",)"
        R"( "alloc_granularity": "polynomial_set"},)" "\n"
        R"("communication": {"p2p_mode": "disabled", "chunk_bytes": 0,)"
        R"( "num_comm_streams": 0, "collective_proto": "auto", "use_nvls": false,)"
        R"( "use_gpudirect_rdma": false, "use_sharp": false, "net_chunk_bytes": 0},)" "\n"
        R"("keys": {"strategy": "replicate", "rotation_key_policy": "plan_required"})" "\n"
        "}";
}

std::string replace_once(std::string text, const std::string &from, const std::string &to) {
    const auto at = text.find(from);
    if (at == std::string::npos) throw std::runtime_error("fixture lacks '" + from + "'");
    return text.replace(at, from.size(), to);
}

// -- the reader -------------------------------------------------------------

void test_a_matching_configuration_loads() {
    const auto plan = single_device_plan();
    const auto config = ExecutionConfigJsonReader::read_text(config_text(plan));

    require(config.config.config_id == "t", "config_id");
    require(config.config.devices.size() == 1, "one device");
    require(config.config.devices[0].cuda_device_id == 3, "the binding survives");
    require(config.config.launch.launch_mode == ConfigLaunchMode::CudaGraph, "launch mode");
    require(config.source_sha256.rfind("sha256:", 0) == 0, "the config hashes itself");
}

void test_a_bom_is_refused() {
    const auto plan = single_device_plan();
    expect_throw([&] {
        ExecutionConfigJsonReader::read_text("\xEF\xBB\xBF" + config_text(plan));
    }, "BOM");
}

void test_a_duplicate_key_is_refused() {
    expect_throw([] {
        ExecutionConfigJsonReader::read_text(R"({"config_id": "a", "config_id": "b"})");
    }, "duplicate object key");
}

void test_an_unknown_field_is_refused() {
    const auto plan = single_device_plan();
    expect_throw([&] {
        ExecutionConfigJsonReader::read_text(
            replace_once(config_text(plan), R"("config_id": "t",)",
                         R"("config_id": "t", "comment": "hi",)"));
    }, "unknown field 'comment'");
}

void test_a_missing_section_is_refused() {
    const auto plan = single_device_plan();
    expect_throw([&] {
        ExecutionConfigJsonReader::read_text(
            replace_once(config_text(plan),
                         R"("keys": {"strategy": "replicate", "rotation_key_policy": "plan_required"})",
                         R"("config_id_2": 1)"));
    }, "missing required field 'keys'");
}

void test_a_future_format_version_is_refused() {
    const auto plan = single_device_plan();
    expect_throw([&] {
        ExecutionConfigJsonReader::read_text(
            replace_once(config_text(plan), R"("format_version": 1)",
                         R"("format_version": 2)"));
    }, "unsupported format version");
}

void test_an_unknown_enum_value_is_refused() {
    const auto plan = single_device_plan();
    expect_throw([&] {
        ExecutionConfigJsonReader::read_text(
            replace_once(config_text(plan), R"("launch_mode": "cuda_graph")",
                         R"("launch_mode": "warp_specialised")"));
    }, "unknown launch mode");

    expect_throw([&] {
        ExecutionConfigJsonReader::read_text(config_text(
            plan, {}, "null",
            R"({"device_execution_mode": "magic", "device_worker_count": 1})"));
    }, "unknown device execution mode");
}

void test_a_malformed_digest_is_refused() {
    const auto plan = single_device_plan();
    expect_throw([&] {
        ExecutionConfigJsonReader::read_text(config_text(plan, "sha256:abc"));
    }, "expected sha256");
}

void test_devices_must_follow_place_index_order() {
    // devices[] is read positionally, so an entry whose own index disagrees
    // with its position would resolve instructions to the wrong card.
    const auto plan = single_device_plan();
    expect_throw([&] {
        ExecutionConfigJsonReader::read_text(config_text(
            plan, {}, "null", {},
            R"([{"index": 1, "cuda_device_id": 0, "mempool_bytes": 0,)"
            R"( "pinned_host_bytes": 0, "mig_instances": 1}])"));
    }, "Place.index order");
}

void test_one_card_cannot_be_bound_twice() {
    const auto plan = single_device_plan();
    expect_throw([&] {
        ExecutionConfigJsonReader::read_text(config_text(
            plan, {}, "null", {},
            R"([{"index": 0, "cuda_device_id": 2, "mempool_bytes": 0,)"
            R"( "pinned_host_bytes": 0, "mig_instances": 1},)"
            R"({"index": 1, "cuda_device_id": 2, "mempool_bytes": 0,)"
            R"( "pinned_host_bytes": 0, "mig_instances": 1}])"));
    }, "is bound twice");
}

void test_ratios_and_lists_are_range_checked() {
    const auto plan = single_device_plan();
    expect_throw([&] {
        ExecutionConfigJsonReader::read_text(
            replace_once(config_text(plan), R"("l2_persist_hit_ratio": 1.0)",
                         R"("l2_persist_hit_ratio": 1.5)"));
    }, "[0, 1]");

    expect_throw([&] {
        ExecutionConfigJsonReader::read_text(
            replace_once(config_text(plan), R"("stream_priorities": [])",
                         R"("stream_priorities": [0, -1, -2])"));
    }, "one priority per stream");
}

void test_the_target_must_be_self_consistent() {
    const auto plan = single_device_plan();
    expect_throw([&] {
        ExecutionConfigJsonReader::read_text(config_text(
            plan, {}, "null", {}, {},
            R"({"world_size": 2, "device_counts": [1], "rank": 0})"));
    }, "one entry per rank");

    expect_throw([&] {
        ExecutionConfigJsonReader::read_text(config_text(
            plan, {}, "null", {}, {},
            R"({"world_size": 1, "device_counts": [1], "rank": 4})"));
    }, "outside the world");
}

// -- the binding ------------------------------------------------------------

void test_a_configuration_for_another_plan_is_refused() {
    // The whole reason the binding exists.
    const auto plan = single_device_plan();
    const auto config = ExecutionConfigJsonReader::read_text(
        config_text(plan, kFakeDigest));

    expect_throw([&] { verify_execution_config(config, plan, 0, 1, 1); },
                 "it was generated for plan");
}

void test_a_matching_configuration_passes_the_binding() {
    const auto plan = single_device_plan();
    const auto config = ExecutionConfigJsonReader::read_text(config_text(plan));
    verify_execution_config(config, plan, 0, 1, 1);
}

void test_a_rewritten_plan_is_refused_while_no_stage_rewrites() {
    const auto plan = single_device_plan();
    const auto config = ExecutionConfigJsonReader::read_text(
        config_text(plan, {}, "\"" + kFakeDigest + "\""));

    expect_throw([&] { verify_execution_config(config, plan, 0, 1, 1); },
                 "plan-rewriting stage this runtime does not implement");
}

void test_the_binding_checks_rank_world_and_devices() {
    const auto plan = single_device_plan();
    const auto config = ExecutionConfigJsonReader::read_text(config_text(plan));

    expect_throw([&] { verify_execution_config(config, plan, 1, 2, 1); },
                 "world_size");
    expect_throw([&] { verify_execution_config(config, plan, 0, 1, 4); },
                 "device(s) but this run has");
}

void test_the_binding_checks_the_operator_spec() {
    const auto plan = single_device_plan();
    auto text = config_text(plan);
    text = replace_once(text, plan.plan.target.operator_spec.source_sha256, kFakeDigest);
    const auto config = ExecutionConfigJsonReader::read_text(text);

    expect_throw([&] { verify_execution_config(config, plan, 0, 1, 1); },
                 "operator spec digest");
}

// -- adapters and reach -----------------------------------------------------

void test_the_binding_reaches_the_api_constructor() {
    const auto plan = single_device_plan();
    const auto config = ExecutionConfigJsonReader::read_text(config_text(plan));

    const auto ids = cuda_device_ids_of(config.config);
    require(ids.size() == 1 && ids[0] == 3,
            "cuda_device_ids_of feeds PoseidonGpuApi in Place.index order");
    require(device_worker_count_of(config.config) == 1, "worker count");
}

void test_every_execution_mode_maps_onto_the_scheduler() {
    const auto plan = single_device_plan();
    const std::vector<std::pair<std::string, DeviceExecutionMode>> cases = {
        {"sequential", DeviceExecutionMode::Sequential},
        {"per_device", DeviceExecutionMode::PerDeviceWorkers},
        {"ready_backfill", DeviceExecutionMode::PerDeviceReadyWorkers},
        {"dependency_workers", DeviceExecutionMode::PerDeviceDependencyWorkers},
    };
    for (const auto &[name, expected] : cases) {
        const auto config = ExecutionConfigJsonReader::read_text(config_text(
            plan, {}, "null",
            R"({"device_execution_mode": ")" + name +
                R"(", "device_worker_count": 2})"));
        require(device_execution_mode_of(config.config) == expected,
                "mode " + name + " must reach the scheduler");
    }
}

void test_the_runtime_states_which_parameters_it_honours() {
    // Seventeen of the twenty are parsed and not acted on.  Saying so is what
    // keeps a report of "configured by mgfhe" from overclaiming.
    const auto applied = applied_fields();
    const auto ignored = ignored_fields();

    require(applied.size() == 3, "three parameters reach an execution path");
    require(ignored.size() == 20, "the rest are parsed and not acted on");

    std::set<std::string> both(applied.begin(), applied.end());
    for (const auto &name : ignored)
        require(both.insert(name).second,
                "a field cannot be both applied and ignored: " + name);
}

} // namespace

int main() {
    try {
        run_test("a matching configuration loads", test_a_matching_configuration_loads);
        run_test("a BOM is refused", test_a_bom_is_refused);
        run_test("a duplicate key is refused", test_a_duplicate_key_is_refused);
        run_test("an unknown field is refused", test_an_unknown_field_is_refused);
        run_test("a missing section is refused", test_a_missing_section_is_refused);
        run_test("a future format version is refused", test_a_future_format_version_is_refused);
        run_test("an unknown enum value is refused", test_an_unknown_enum_value_is_refused);
        run_test("a malformed digest is refused", test_a_malformed_digest_is_refused);
        run_test("devices must follow Place.index order",
                 test_devices_must_follow_place_index_order);
        run_test("one card cannot be bound twice", test_one_card_cannot_be_bound_twice);
        run_test("ratios and lists are range checked", test_ratios_and_lists_are_range_checked);
        run_test("the target must be self consistent", test_the_target_must_be_self_consistent);
        run_test("a configuration for another plan is refused",
                 test_a_configuration_for_another_plan_is_refused);
        run_test("a matching configuration passes the binding",
                 test_a_matching_configuration_passes_the_binding);
        run_test("a rewritten plan is refused while no stage rewrites",
                 test_a_rewritten_plan_is_refused_while_no_stage_rewrites);
        run_test("the binding checks rank, world and devices",
                 test_the_binding_checks_rank_world_and_devices);
        run_test("the binding checks the operator spec",
                 test_the_binding_checks_the_operator_spec);
        run_test("the binding reaches the api constructor",
                 test_the_binding_reaches_the_api_constructor);
        run_test("every execution mode maps onto the scheduler",
                 test_every_execution_mode_maps_onto_the_scheduler);
        run_test("the runtime states which parameters it honours",
                 test_the_runtime_states_which_parameters_it_honours);
    } catch (const std::exception &error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
    std::cout << tests_run << " test(s) passed\n";
    return 0;
}
