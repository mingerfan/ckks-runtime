#include "api/mock_api.hpp"
#include "runtime/json_plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/runtime.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace fhegpu;

namespace {

using Json = nlohmann::json;
const std::filesystem::path source_dir = CKKS_RUNTIME_SOURCE_DIR;
int tests_run = 0;

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Function>
void expect_throw(Function function, const std::string &needle) {
    try { function(); }
    catch (const std::exception &error) {
        require(std::string(error.what()).find(needle) != std::string::npos,
                "expected '" + needle + "' in: " + error.what());
        return;
    }
    throw std::runtime_error("expected exception: " + needle);
}

void run_test(const char *name, const std::function<void()> &test) {
    test();
    ++tests_run;
    std::cout << "[PASS] " << name << '\n';
}

Json fixture(const char *name = "cpu_reuse_chain.json") {
    std::ifstream input(source_dir / "docs/runtime-plan/v2/testdata/valid" / name);
    require(static_cast<bool>(input), "cannot open V2 fixture");
    return Json::parse(input);
}

LoadedRuntimePlan read(const Json &document) {
    return RuntimePlanJsonReader::read_text(document.dump());
}

LoadedOperatorSpec load_spec(const RuntimePlan &plan) {
    const char *name = plan.target.operator_spec.id == "poseidon-ckks-cpu-v1"
        ? "poseidon-ckks-cpu.v1.json" : "poseidon-ckks-gpu.v1.json";
    return OperatorSpecReader::read_file((source_dir / "docs/operator-spec/v1/profiles" / name).string());
}

void renumber(Json &plan) {
    std::size_t ordinal = 0;
    for (const char *phase : {"initialization", "execution", "finalization"})
        for (auto &instruction : plan[phase]) instruction["ordinal"] = ordinal++;
}

void remove_release(Json &plan, const char *id) {
    for (const char *phase : {"initialization", "execution", "finalization"}) {
        auto &instructions = plan[phase];
        instructions.erase(std::remove_if(instructions.begin(), instructions.end(),
            [&](const Json &instruction) {
                return instruction["kind"] == "release" && instruction["value"] == id;
            }), instructions.end());
    }
}

void expect_invalid(Json plan, const std::string &needle) {
    renumber(plan);
    const auto loaded = read(plan);
    const auto spec = load_spec(loaded.plan);
    expect_throw([&] { PlanVerifier::verify(loaded.plan, spec); }, needle);
}

void test_v2_chains_and_printing() {
    for (const char *name : {"cpu_reuse_chain.json", "gpu_reuse_chain.json"}) {
        const auto loaded = read(fixture(name));
        const auto requirements = PlanVerifier::verify(loaded.plan, load_spec(loaded.plan));
        require(!requirements.capabilities.empty(), "V2 requirements are empty");
        require(loaded.plan.format_version == 2, "V2 version was not read");
        std::size_t reused = 0;
        for (const auto *phase : {&loaded.plan.initialization, &loaded.plan.execution, &loaded.plan.finalization})
            for (const auto &instruction : *phase)
                if (const auto *op = std::get_if<ComputeOp>(&instruction.body))
                    if (op->reuse_input) {
                        require(*op->reuse_input == 0, "reuse index was not read");
                        ++reused;
                    }
        require(reused == 3, "three-operation reuse chain was not read");
        std::ostringstream printed;
        loaded.plan.print(printed);
        require(printed.str().find("Release(%1)") != std::string::npos, "Release was not printed");
        require(printed.str().find("reuse_input=0") != std::string::npos, "reuse was not printed");
    }
    // Explicit Release is optional; ownership rules still apply to reuse.
    auto plan = fixture();
    remove_release(plan, "1");
    remove_release(plan, "2");
    renumber(plan);
    const auto loaded = read(plan);
    PlanVerifier::verify(loaded.plan, load_spec(loaded.plan));
}

void test_versions_and_strict_fields() {
    auto plan = fixture();
    plan["format_version"] = 1;
    expect_throw([&] { read(plan); }, "Release requires format version 2");
    remove_release(plan, "1");
    remove_release(plan, "2");
    expect_throw([&] { read(plan); }, "reuse_input requires format version 2");
    auto v3 = fixture();
    v3["format_version"] = 3;
    v3["execution"].push_back({{"kind", "fence"}, {"ordinal", 0}});
    renumber(v3);
    auto valid = read(v3);
    PlanVerifier::verify(valid.plan, load_spec(valid.plan));
    v3["execution"].back()["output"] = "99";
    expect_throw([&] { read(v3); }, "unknown field");
    v3["execution"].back().erase("output");
    v3["format_version"] = 2;
    expect_throw([&] { read(v3); }, "Fence requires format version 3");
    plan["format_version"] = 4;
    expect_throw([&] { read(plan); }, "unsupported format version");
    for (const Json &index : {Json(-1), Json(0.0), Json("0"), Json(true), Json(nullptr), Json(2147483648ULL)}) {
        auto invalid = fixture();
        invalid["execution"][0]["reuse_input"] = index;
        expect_throw([&] { read(invalid); }, ".reuse_input");
    }
    plan = fixture();
    plan["initialization"][2]["wait"] = true;
    expect_throw([&] { read(plan); }, "unknown field 'wait'");
    plan["initialization"][2].erase("wait");
    plan["initialization"][2].erase("value");
    expect_throw([&] { read(plan); }, "missing required field 'value'");
    plan = fixture();
    plan["initialization"][2]["value"] = 1;
    expect_throw([&] { read(plan); }, "expected string");
    plan = fixture();
    plan["execution"][1]["attrs"]["reuse_input"] = 0;
    expect_throw([&] { read(plan); }, "unknown field 'reuse_input'");

    // Programmatically built plans must obey the version rules too.
    auto parsed = read(fixture()).plan;
    const auto spec = load_spec(parsed);
    parsed.format_version = 1;
    expect_throw([&] { PlanVerifier::verify(parsed, spec); }, "requires format version 2");
    parsed.initialization.resize(2);
    std::size_t ordinal = 2;
    for (auto *phase : {&parsed.execution, &parsed.finalization})
        for (auto &instruction : *phase) instruction.ordinal = ordinal++;
    expect_throw([&] { PlanVerifier::verify(parsed, spec); }, "reuse_input requires format version 2");
}

void test_release_errors() {
    auto plan = fixture();
    plan["initialization"][3]["value"] = "1";
    expect_invalid(plan, "already released at instruction #2");
    plan = fixture();
    plan["initialization"][2]["value"] = "4";
    expect_invalid(plan, "use-before-definition");
    plan["initialization"][2]["value"] = "99";
    expect_invalid(plan, "Release ValueId 99 has no ValueDesc");
    plan = fixture();
    plan["initialization"][2]["value"] = "3";
    expect_invalid(plan, "ValueId 3 was already released");
    plan = fixture("gpu_reuse_chain.json");
    plan["initialization"].insert(plan["initialization"].begin() + 1,
                                    Json{{"kind", "release"}, {"value", "1"}});
    expect_invalid(plan, "ValueId 1 was already released");
    plan = fixture();
    plan["finalization"].push_back({{"kind", "release"}, {"value", "6"}});
    expect_invalid(plan, "final output and cannot be released");
    plan = fixture();
    plan["execution"].push_back({{"kind", "release"}, {"value", "3"}});
    expect_invalid(plan, "ValueId 3 was already overwritten by reuse at instruction #4");
}

void test_reuse_ownership_and_support() {
    auto plan = fixture();
    remove_release(plan, "1");
    plan["execution"][0]["inputs"] = {"1"};
    expect_invalid(plan, "input must be produced by a computation");
    plan = fixture("gpu_reuse_chain.json");
    remove_release(plan, "7");
    plan["execution"][0]["inputs"][0] = "7";
    expect_invalid(plan, "input must be produced by a computation");
    plan = fixture();
    plan["final_outputs"].push_back("3");
    expect_invalid(plan, "final output cannot be overwritten");
    plan = fixture();
    plan["execution"][0]["reuse_input"] = 1;
    expect_invalid(plan, "only input 0 can be reused");
    plan["execution"][0]["reuse_input"] = 99;
    expect_invalid(plan, "only input 0 can be reused");
    plan["execution"][0]["inputs"] = Json::array();
    expect_invalid(plan, "reuse_input requires input 0");
    plan = fixture();
    remove_release(plan, "2");
    plan["execution"][0]["op"] = "add_cp";
    plan["execution"][0]["inputs"] = {"3", "2"};
    expect_invalid(plan, "AddCP does not support reuse on Host");
    plan = fixture("gpu_reuse_chain.json");
    plan["execution"][0]["op"] = "negate";
    plan["execution"][0]["inputs"] = {"3"};
    expect_invalid(plan, "Negate does not support reuse on Device");
    plan = fixture();
    plan["values"][3]["scale_log2"] = 41;
    expect_invalid(plan, "instruction #4 reuse_input for ValueId 3: input and output metadata must match");
    plan = fixture();
    plan["values"][3]["place"] = {{"kind", "device"}, {"rank", 0}, {"index", 0}};
    expect_invalid(plan, "input and output metadata must match");
}

void test_uses_across_phases_and_communication() {
    auto plan = fixture();
    auto extra_value = plan["values"][0];
    extra_value["id"] = "9";
    plan["values"].push_back(extra_value);
    plan["finalization"].push_back({{"kind", "compute"}, {"op", "negate"},
        {"place", extra_value["place"]}, {"inputs", {"3"}}, {"output", "9"}});
    expect_invalid(plan, "including communication (2 uses)");
    plan["finalization"].erase(plan["finalization"].end() - 1);
    plan["initialization"].push_back({{"kind", "compute"}, {"op", "add_cc"},
        {"place", extra_value["place"]}, {"inputs", {"3", "3"}}, {"output", "9"}});
    expect_invalid(plan, "including communication (3 uses)");

    plan = fixture("gpu_reuse_chain.json");
    auto transfer = plan["initialization"][1];
    transfer["transfer_id"] = "3";
    transfer["inputs"] = {"3"};
    transfer["outputs"] = {"9"};
    transfer["sources"] = transfer["destinations"];
    transfer["destinations"] = {plan["values"][0]["place"]};
    extra_value = plan["values"][0];
    extra_value["id"] = "9";
    plan["values"].push_back(extra_value);
    plan["finalization"].push_back(transfer);
    expect_invalid(plan, "including communication (2 uses)");
}

void test_v1_v2_without_memory_instructions() {
    const auto path = source_dir / "docs/runtime-plan/v1/testdata/valid/v001_inline_encode_host_compute.json";
    auto v1 = RuntimePlanJsonReader::read_file(path.string());
    const auto spec = load_spec(v1.plan);
    const auto old_requirements = PlanVerifier::verify(v1.plan, spec);
    v1.plan.format_version = 2;
    const auto new_requirements = PlanVerifier::verify(v1.plan, spec);
    require(old_requirements.capabilities == new_requirements.capabilities &&
            old_requirements.keys == new_requirements.keys, "V2 changed ordinary plan requirements");
    auto cluster = std::make_shared<MockCluster>(MockClusterConfig{});
    MockVecApi api(0, cluster);
    SequentialRuntime<MockVecApi> runtime(0, 1, 1, api);
    const auto input = make_cipher(std::vector<double>(spec.spec.poly_degree / 2, 1.0),
                                  "ctx-main", spec.spec.poly_degree, 5, 40);
    const auto result = runtime.run(v1, RuntimeResources{spec, {}, false}, {{1, input}});
    require(result.values.count(3) == 1 && api.stats().compute_calls == 1, "ordinary V2 plan did not execute");
}

void test_memory_execution_rejected_before_api_work() {
    for (const auto mode : {DeviceExecutionMode::Sequential, DeviceExecutionMode::PerDeviceWorkers}) {
        for (bool reuse_only : {false, true}) {
            auto document = fixture();
            if (reuse_only) {
                remove_release(document, "1");
                remove_release(document, "2");
                renumber(document);
            }
            const auto loaded = read(document);
            const auto spec = load_spec(loaded.plan);
            for (const auto diff_mode : {DiffMode::AllValuesAfterRun}) {
                auto cluster = std::make_shared<MockCluster>(MockClusterConfig{});
                MockVecApi api(0, cluster);
                SequentialRuntime<MockVecApi> runtime(0, 1, 1, api, mode);
                const std::string reason = "AllValuesAfterRun is incompatible";
                // Empty bindings prove the whole-plan check precedes input/API work.
                expect_throw([&] { runtime.run(loaded, RuntimeResources{spec, {}, false}, {}, diff_mode); }, reason);
                const auto stats = api.stats();
                require(stats.compute_calls == 0 && stats.communicate_calls == 0, "rejected plan submitted API work");
            }
        }
    }
}

void test_reuse_execution() {
    class ObservedApi : public MockVecApi {
    public:
        using MockVecApi::MockVecApi;
        std::size_t reuses = 0;
        const void *allocation = nullptr;
        Value compute_reuse(const ComputeOp &op, Value input, const std::vector<Value> &others) {
            if (!allocation) allocation = input.identity();
            require(input.identity() == allocation, "reuse chain changed allocation");
            auto output = MockVecApi::compute_reuse(op, std::move(input), others);
            require(output.identity() == allocation, "reuse produced a new allocation");
            ++reuses;
            return output;
        }
    };
    for (const char *name : {"cpu_reuse_chain.json", "gpu_reuse_chain.json"}) {
        const auto loaded = read(fixture(name));
        const auto spec = load_spec(loaded.plan);
        const auto input = make_cipher(std::vector<double>(spec.spec.poly_degree / 2, 1.0),
                                      "ctx-main", spec.spec.poly_degree, 5, 40);
        auto baseline = fixture(name);
        for (const char *phase : {"initialization", "execution", "finalization"})
            for (auto &instruction : baseline[phase]) instruction.erase("reuse_input");
        MockVecApi ordinary(0, std::make_shared<MockCluster>(MockClusterConfig{}));
        SequentialRuntime<MockVecApi> base_runtime(0, 1, 1, ordinary);
        const auto expected = base_runtime.run(read(baseline), RuntimeResources{spec, {}, false}, {{1, input}});
        for (auto mode : {DeviceExecutionMode::Sequential, DeviceExecutionMode::PerDeviceWorkers}) {
            if (mode == DeviceExecutionMode::PerDeviceWorkers && name == std::string("cpu_reuse_chain.json")) continue;
            VecExecConfig config; config.mode = VecExecMode::Async; config.max_delay_ms = 3;
            for (int iteration = 0; iteration < 3; ++iteration) {
                ObservedApi api(0, std::make_shared<MockCluster>(MockClusterConfig{}), config);
                SequentialRuntime<ObservedApi> runtime(0, 1, 1, api, mode);
                const auto actual = runtime.run(loaded, RuntimeResources{spec, {}, false}, {{1, input}});
                require(actual.values.at(6).value.materialize().slots == expected.values.at(6).value.materialize().slots,
                        "reuse changed numerical results");
                require(input.materialize().slots.front() == 1.0, "reuse changed caller input");
                require(api.reuses == 3, "reuse calls were not dispatched");
            }
        }
    }
    class UnsupportedApi : public MockVecApi {
    public:
        using MockVecApi::MockVecApi;
        bool supports_reuse(const ComputeOp &) const { return false; }
    };
    const auto loaded = read(fixture());
    const auto spec = load_spec(loaded.plan);
    UnsupportedApi api(0, std::make_shared<MockCluster>(MockClusterConfig{}));
    SequentialRuntime<UnsupportedApi> runtime(0, 1, 1, api);
    expect_throw([&] { runtime.run(loaded, RuntimeResources{spec, {}, false}, {}); }, "Api does not support reuse_input");
    require(api.stats().compute_calls == 0, "unsupported reuse submitted work");
}

void test_release_only_v2_execution() {
    for (const char *name : {"cpu_reuse_chain.json", "gpu_reuse_chain.json"}) {
        auto document = fixture(name);
        for (const char *phase : {"initialization", "execution", "finalization"})
            for (auto &instruction : document[phase]) instruction.erase("reuse_input");
        const auto loaded = read(document);
        const auto spec = load_spec(loaded.plan);
        const auto input = make_cipher(std::vector<double>(spec.spec.poly_degree / 2, 1.0),
                                      "ctx-main", spec.spec.poly_degree, 5, 40);
        auto baseline_document = document;
        for (const char *phase : {"initialization", "execution", "finalization"}) {
            auto &instructions = baseline_document[phase];
            instructions.erase(std::remove_if(instructions.begin(), instructions.end(),
                [](const Json &instruction) { return instruction["kind"] == "release"; }), instructions.end());
        }
        renumber(baseline_document);
        const auto run = [&](const LoadedRuntimePlan &plan, DeviceExecutionMode mode) {
            MockVecApi api(0, std::make_shared<MockCluster>(MockClusterConfig{}));
            SequentialRuntime<MockVecApi> runtime(0, 1, 1, api, mode);
            return runtime.run(plan, RuntimeResources{spec, {}, false}, {{1, input}});
        };
        const auto baseline = run(read(baseline_document), DeviceExecutionMode::Sequential);
        for (auto mode : {DeviceExecutionMode::Sequential, DeviceExecutionMode::PerDeviceWorkers}) {
            if (mode == DeviceExecutionMode::PerDeviceWorkers && name == std::string("cpu_reuse_chain.json")) continue;
            const auto artifact = run(loaded, mode);
            require(artifact.values.at(6).value.materialize().slots == baseline.values.at(6).value.materialize().slots,
                    "Release-only V2 fixture changed the numerical result");
        }
    }
}

} // namespace

int main() {
    try {
        run_test("V2 CPU/GPU reuse chains and printing", test_v2_chains_and_printing);
        run_test("version gates and strict memory fields", test_versions_and_strict_fields);
        run_test("Release lifetime errors", test_release_errors);
        run_test("reuse ownership, operation and metadata rules", test_reuse_ownership_and_support);
        run_test("operand uses across phases and communication", test_uses_across_phases_and_communication);
        run_test("ordinary V1/V2 compatibility", test_v1_v2_without_memory_instructions);
        run_test("memory execution rejected before API work", test_memory_execution_rejected_before_api_work);
        run_test("Release-only CPU/GPU V2 plans execute", test_release_only_v2_execution);
        run_test("reuse chains preserve allocation and caller input", test_reuse_execution);
        std::cout << "ALL " << tests_run << " MEMORY TEST GROUPS PASSED\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
