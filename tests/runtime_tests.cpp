#include "testing/testing.hpp"

#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace fhegpu;

namespace {

int tests_run = 0;

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Function>
void expect_throw(Function function, const std::string &needle = {}) {
    bool threw = false;
    try { function(); }
    catch (const std::exception &error) {
        threw = true;
        if (!needle.empty() && std::string(error.what()).find(needle) == std::string::npos)
            throw std::runtime_error("exception did not contain '" + needle + "': " + error.what());
    }
    if (!threw) throw std::runtime_error("expected exception was not thrown");
}

void run_test(const char *name, const std::function<void()> &test) {
    test();
    ++tests_run;
    std::cout << "[PASS] " << name << '\n';
}

ComputeOp *first_compute(RuntimePlan &plan) {
    for (auto &instruction : plan.execution)
        if (auto *op = std::get_if<ComputeOp>(&instruction.body)) return op;
    throw std::runtime_error("test plan has no compute");
}

CommAction *first_comm(RuntimePlan &plan) {
    for (auto *list : {&plan.initialization, &plan.execution, &plan.finalization})
        for (auto &instruction : *list)
            if (auto *action = std::get_if<CommAction>(&instruction.body)) return action;
    throw std::runtime_error("test plan has no communication");
}

void test_plan_and_fingerprint() {
    auto built = make_fanout_plan({2, 1});
    PlanVerifier::verify(built.plan);
    const auto fingerprint = built.plan.fingerprint();
    require(fingerprint == built.plan.fingerprint(), "fingerprint is unstable");
    std::ostringstream printed;
    built.plan.print(printed);
    require(printed.str().find("Initialization") != std::string::npos, "plan printer omitted phases");
    require(printed.str().find("Replicate") != std::string::npos, "plan printer omitted communication");
    std::cout << printed.str();
}

void test_verifier_rejections() {
    const RuntimePlan base = make_fanout_plan({1, 1}).plan;
    {
        auto plan = base; plan.values.push_back(plan.values.front());
        expect_throw([&] { PlanVerifier::verify(plan); }, "duplicate ValueId");
    }
    {
        auto plan = base; first_compute(plan)->inputs[0] = plan.final_outputs[0];
        expect_throw([&] { PlanVerifier::verify(plan); }, "use-before-definition");
    }
    {
        auto plan = base; first_compute(plan)->place = {PlaceKind::Device, 1, 0};
        expect_throw([&] { PlanVerifier::verify(plan); }, "cross-Place");
    }
    {
        auto plan = base;
        auto &last = std::get<CommAction>(plan.finalization.front().body);
        last.id = std::get<CommAction>(plan.initialization.front().body).id;
        expect_throw([&] { PlanVerifier::verify(plan); }, "duplicate TransferId");
    }
    {
        auto plan = base; auto *comm = first_comm(plan); comm->destinations[0] = {PlaceKind::Device, 1, 0};
        expect_throw([&] { PlanVerifier::verify(plan); }, "mapping mismatch");
    }
    {
        auto plan = base; first_comm(plan)->kind = static_cast<CommKind>(99);
        expect_throw([&] { PlanVerifier::verify(plan); }, "not implemented");
    }
    {
        auto plan = base; first_comm(plan)->hint = CommHint::Ring;
        expect_throw([&] { PlanVerifier::verify(plan); }, "hint is not implemented");
    }
    expect_throw([&] { PlanVerifier::verify_runtime_target(base, 0, 3, 1); }, "world size");
}

void test_vec_operations() {
    const Place p{PlaceKind::Device, 0, 0};
    VecExecutor executor;
    const VecValue ct = make_cipher({1, 2, 3, 4}, "ctx", 8192, 4, 3, true, 2);
    const VecValue ct2 = make_cipher({4, 3, 2, 1}, "ctx", 8192, 4, 3, true, 2);
    const VecValue pt = make_plain({2, 2, 2, 2}, "ctx", 8192, 4, 3, true);
    auto op = [&](ComputeKind kind, std::vector<VecValue> inputs, ComputeAttrs attrs = {}) {
        return executor.compute(ComputeOp{kind, {}, 9, p, std::move(attrs)}, inputs);
    };
    require(op(ComputeKind::AddCC, {ct, ct2}).materialize().slots[0] == 5, "AddCC failed");
    require(op(ComputeKind::SubCP, {ct, pt}).materialize().slots[0] == -1, "SubCP failed");
    auto mul = op(ComputeKind::MulCC, {ct, ct2}).materialize();
    require(mul.metadata.scale_log2 == 6 && mul.metadata.components == 3, "MulCC metadata failed");
    auto relin = op(ComputeKind::Relinearize, {VecValue::ready(mul)}).materialize();
    require(relin.metadata.components == 2, "Relinearize failed");
    auto rescale = op(ComputeKind::Rescale, {ct}, RescaleAttrs{3, 1}).materialize();
    require(rescale.metadata.level == 3 && rescale.metadata.scale_log2 == 1, "Rescale metadata failed");
    require(op(ComputeKind::ModSwitch, {ct}, ModSwitchAttrs{2}).materialize().metadata.level == 2, "ModSwitch failed");
    auto boot = op(ComputeKind::Boot, {ct},
                   BootAttrs{7, 4, 2, "test-native-boot", BootImplementation::Native}).materialize();
    require(boot.metadata.level == 7 && boot.metadata.scale_log2 == 4, "Boot failed");
    require(op(ComputeKind::Rotate, {ct}, RotateAttrs{-1}).materialize().slots[0] == 4, "negative rotate failed");
    require(op(ComputeKind::Rotate, {ct}, RotateAttrs{5}).materialize().slots[0] == 2, "normalized rotate failed");
    expect_throw([&] { op(ComputeKind::AddCC, {ct, pt}).materialize(); }, "type mismatch");
    expect_throw([&] { op(ComputeKind::Rescale, {ct}, RescaleAttrs{4, 2}).materialize(); }, "lower level");

    VecExecutor async({VecExecMode::Async, 42, 2});
    VecValue pending = async.compute(ComputeOp{ComputeKind::AddCP, {}, 10, p, {}}, {ct, pt});
    require(pending.materialize().slots[3] == 6, "async Vec compute failed");
}

void validate_layout(const std::vector<int> &layout, VecExecMode mode) {
    const auto built = make_fanout_plan(layout);
    const VecValue cipher = make_cipher({1, 2, 3, 4}, "ctx", 8192, 3, 1);
    const VecValue plain = make_plain({2, 3, 4, 5}, "ctx", 8192, 3, 1);
    const auto reference = run_fanout_reference(cipher, plain);
    MockClusterConfig cluster_config;
    cluster_config.world_size = static_cast<int>(layout.size());
    cluster_config.delay_seed = 91;
    cluster_config.max_delay_ms = 3;
    auto result = run_mock_cluster(built.plan, {{0, cipher.deep_copy()}, {1, plain.deep_copy()}},
                                   cluster_config,
                                   VecExecConfig{mode, 17, 2}, DiffMode::AllValuesAfterRun);
    std::map<ValueId, ArtifactValue<VecValue>> merged;
    for (const auto &artifact : result.artifacts)
        for (const auto &item : artifact.values)
            require(merged.emplace(item.first, item.second).second, "ValueId materialized on multiple runtimes");
    const ValueId final = built.plan.final_outputs.front();
    compare_values(merged.at(final).value, reference.at(built.reference_output));
    require(merged.at(final).place == built.plan.values.back().place, "final output Place mismatch");
    for (const auto &point : built.diff_map) compare_values(merged.at(point.distributed_value).value, reference.at(point.reference_value));

    std::vector<const void *> product_copies;
    for (const auto &point : built.diff_map)
        if (point.reference_value == 2) product_copies.push_back(merged.at(point.distributed_value).value.identity());
    for (std::size_t i = 0; i < product_copies.size(); ++i)
        for (std::size_t j = i + 1; j < product_copies.size(); ++j)
            require(product_copies[i] != product_copies[j], "communication shared a VecValue object");
    for (const auto &stats : result.stats) {
        require(stats.communicate_calls >= 3, "not every rank interpreted every communication");
        require(stats.wait_calls == stats.completed_handles, "communication handle was not completed");
    }
}

void test_mock_matrix() {
    for (const auto &layout : std::vector<std::vector<int>>{{1}, {2}, {2, 2}, {3, 3}, {2, 2, 2, 2}}) {
        validate_layout(layout, VecExecMode::Sync);
        validate_layout(layout, VecExecMode::Async);
    }
}

void test_mock_fail_fast() {
    const auto built = make_fanout_plan({1, 1});
    const VecValue cipher = make_cipher({1, 2}, "ctx", 8192, 3, 1);
    const VecValue plain = make_plain({2, 3}, "ctx", 8192, 3, 1);
    const TransferId failed = std::get<CommAction>(built.plan.execution.at(1).body).id;
    const TransferId init_transfer = std::get<CommAction>(built.plan.initialization.front().body).id;
    MockClusterConfig config;
    config.fail_compute = ComputeKind::MulCP;
    expect_throw([&] { run_mock_cluster(built.plan, {{0, cipher}, {1, plain}}, config, {}, DiffMode::FinalOnly); });
    config = {};
    config.fail_communicate.insert(init_transfer);
    expect_throw([&] { run_mock_cluster(built.plan, {{0, cipher}, {1, plain}}, config, {}, DiffMode::FinalOnly); });
    config = {};
    config.fail_publish.insert(init_transfer);
    expect_throw([&] { run_mock_cluster(built.plan, {{0, cipher}, {1, plain}}, config, {}, DiffMode::FinalOnly); });
    config = {};
    config.fail_wait.insert(failed);
    expect_throw([&] { run_mock_cluster(built.plan, {{0, cipher}, {1, plain}}, config, {}, DiffMode::FinalOnly); });
    config = {};
    config.corrupt_output_count.insert(failed);
    expect_throw([&] { run_mock_cluster(built.plan, {{0, cipher}, {1, plain}}, config, {}, DiffMode::FinalOnly); });
    config = {};
    config.corrupt_output_type.insert(failed);
    expect_throw([&] { run_mock_cluster(built.plan, {{0, cipher}, {1, plain}}, config, {}, DiffMode::FinalOnly); });

    RuntimePlan async_error_plan;
    async_error_plan.plan_id = 77;
    async_error_plan.target.target_id = "async-final-error";
    async_error_plan.target.world_size = 1;
    async_error_plan.target.device_counts = {1};
    async_error_plan.target.capability_version = 1;
    async_error_plan.target.operator_spec = {"test-operator-spec", 1,
        "sha256:0000000000000000000000000000000000000000000000000000000000000000"};
    async_error_plan.target.required_capabilities = {RequiredCapability::Transfer};
    const Place host{PlaceKind::Host, 0, 0};
    const Place device{PlaceKind::Device, 0, 0};
    async_error_plan.values = {{0, ValueKind::Ciphertext, host, "ctx", 3, 1, true, 2},
                               {1, ValueKind::Ciphertext, device, "ctx", 3, 1, true, 2},
                               {2, ValueKind::Ciphertext, device, "ctx", 3, 0, true, 2}};
    async_error_plan.external_inputs = {0};
    async_error_plan.initialization = {{0, CommAction{1, CommKind::Transfer, CommHint::Auto,
        {0}, {1}, {host}, {device}, {ValueKind::Ciphertext}}}};
    async_error_plan.execution = {{1, ComputeOp{ComputeKind::Rescale, {1}, 2, device,
                                                RescaleAttrs{3, 0}}}};
    async_error_plan.final_outputs = {2};
    expect_throw([&] {
        run_mock_cluster(async_error_plan, {{0, cipher}}, {},
                         VecExecConfig{VecExecMode::Async, 5, 1}, DiffMode::FinalOnly);
    }, "rescale must lower level");
}

} // namespace

int main() {
    try {
        run_test("plan printer and stable fingerprint", test_plan_and_fingerprint);
        run_test("plan verifier rejection matrix", test_verifier_rejections);
        run_test("Vec operations and metadata", test_vec_operations);
        run_test("Mock multi-runtime sync/async matrix", test_mock_matrix);
        run_test("Mock fail-fast", test_mock_fail_fast);
        std::cout << "ALL " << tests_run << " TEST GROUPS PASSED\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
