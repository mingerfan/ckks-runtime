#include "testing/testing.hpp"

#include <chrono>
#include <functional>
#include <iostream>

using namespace fhegpu;

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

// Observe Runtime ownership independently of the Mock executor's async payloads.
// The wrapped API retains its own VecValue copies when accepting submissions.
class TrackingApi {
public:
    using Value = std::shared_ptr<VecValue>;
    struct CommHandle {
        MockVecApi::CommHandle inner;
        std::vector<std::optional<Value>> outputs;
    };
    std::unique_ptr<MockVecApi> inner;
    std::function<void(const ComputeOp &)> before_compute;
    std::function<void(const CommAction &)> before_communication;
    std::function<void()> on_collect, on_drain;
    bool defer_outputs = false;
    std::function<void(ValueId)> before_encode;

    std::string name() const { return "TrackingApi"; }
    Value observe(ValueId id, VecValue value) {
        auto result = std::make_shared<VecValue>(std::move(value));
        std::lock_guard<std::mutex> lock(mutex_);
        observed_[id] = result;
        return result;
    }
    bool expired(ValueId id) {
        std::lock_guard<std::mutex> lock(mutex_);
        return observed_.at(id).expired();
    }
    Value encode_plaintext(const ValueDesc &desc, const std::vector<double> &slots) {
        if (before_encode) before_encode(desc.id);
        return observe(desc.id, inner->encode_plaintext(desc, slots));
    }
    Value compute(const ComputeOp &op, const std::vector<Value> &inputs) {
        if (before_compute) before_compute(op);
        std::vector<VecValue> copied;
        for (const auto &input : inputs) copied.push_back(*input);
        return observe(op.output, inner->compute(op, copied));
    }
    CommHandle communicate_async(const CommAction &action, const std::vector<Value> &inputs,
                                 const std::vector<ValueDesc> &descs) {
        if (before_communication) before_communication(action);
        std::vector<VecValue> copied;
        for (const auto &input : inputs) copied.push_back(*input);
        CommHandle handle;
        handle.inner = inner->communicate_async(action, copied, descs);
        auto outputs = inner->posted_outputs(handle.inner);
        handle.outputs.resize(outputs.size());
        for (std::size_t slot = 0; slot < outputs.size(); ++slot)
            if (outputs[slot]) handle.outputs[slot] = observe(action.outputs[slot], std::move(*outputs[slot]));
        return handle;
    }
    std::vector<std::optional<Value>> posted_outputs(CommHandle &handle) {
        if (defer_outputs) return std::vector<std::optional<Value>>(handle.outputs.size());
        return take_outputs(handle);
    }
    std::vector<std::optional<Value>> wait(CommHandle &handle) {
        inner->wait(handle.inner);
        return take_outputs(handle);
    }
    void collect_completed() { if (on_collect) on_collect(); }
    void drain() { inner->drain(); if (on_drain) on_drain(); }
    void synchronize(Value &value) { inner->synchronize(*value); }
    void validate_value(const Value &value, const ValueDesc &desc) const { inner->validate_value(*value, desc); }
    void preflight(std::string_view sha, bool skip, const TargetConfig &target,
                   const OperatorSpec &spec, const PlanRequirements &requirements) {
        inner = std::make_unique<MockVecApi>(0, std::make_shared<MockCluster>(MockClusterConfig{}));
        inner->preflight(sha, skip, target, spec, requirements);
    }
    [[noreturn]] void abort_all(int code, const std::string &reason) { inner->abort_all(code, reason); }
private:
    std::vector<std::optional<Value>> take_outputs(CommHandle &handle) {
        std::vector<std::optional<Value>> result(handle.outputs.size());
        result.swap(handle.outputs);
        return result;
    }
    std::mutex mutex_;
    std::unordered_map<ValueId, std::weak_ptr<VecValue>> observed_;
};

struct Gate {
    std::mutex mutex;
    std::condition_variable condition;
    bool released = false, first_sent = false;
    void wait(bool first = false) {
        std::unique_lock<std::mutex> lock(mutex);
        require(condition.wait_for(lock, std::chrono::seconds(5), [&] {
            return released && (!first || first_sent);
        }), "timed out waiting for the other Runtime worker");
    }
    void signal(bool first = false) {
        std::lock_guard<std::mutex> lock(mutex);
        if (first) first_sent = true; else released = true;
        condition.notify_all();
    }
};

Place host() { return {PlaceKind::Host, 0, 0}; }
Place device(int index) { return {PlaceKind::Device, 0, index}; }
ValueDesc value(ValueId id, ValueKind kind, Place place) {
    return {id, kind, place, "ctx", 3, 1, true, kind == ValueKind::Plaintext ? 1 : 2};
}
CommAction transfer(TransferId id, ValueId input, ValueId output, Place src, Place dst) {
    return {id, CommKind::Transfer, CommHint::PointToPoint, {input}, {output}, {src}, {dst}, {ValueKind::Ciphertext}};
}
void renumber(RuntimePlan &plan) {
    std::size_t ordinal = 0;
    for (auto *phase : {&plan.initialization, &plan.execution, &plan.finalization})
        for (auto &instruction : *phase) instruction.ordinal = ordinal++;
}

BuiltPlan split_plan() {
    auto built = make_fanout_plan({3});
    auto &plan = built.plan;
    plan.format_version = 2;
    plan.values = {value(0, ValueKind::Plaintext, host()), value(1, ValueKind::Ciphertext, host()),
        value(2, ValueKind::Ciphertext, device(1)), value(3, ValueKind::Ciphertext, device(2)),
        value(4, ValueKind::Ciphertext, device(2)), value(5, ValueKind::Plaintext, device(1)),
        value(6, ValueKind::Plaintext, device(2)), value(7, ValueKind::Ciphertext, device(1)),
        value(8, ValueKind::Ciphertext, device(2)), value(9, ValueKind::Ciphertext, host())};
    plan.external_inputs = {1};
    plan.initialization = {
        {0, EncodeOp{InlineEncodePayload{{2, 3, 4, 5}}, 0}},
        {0, ComputeOp{ComputeKind::AddCP, {1, 0}, 9, host(), {}}},
        {0, transfer(100, 1, 2, host(), device(1))},
        {0, transfer(101, 1, 3, host(), device(2))}};
    plan.execution = {
        {0, ComputeOp{ComputeKind::Negate, {3}, 4, device(2), {}}},
        {0, CommAction{102, CommKind::Replicate, CommHint::Broadcast, {0}, {5, 6},
            {host()}, {device(1), device(2)}, {ValueKind::Plaintext, ValueKind::Plaintext}}},
        {0, ReleaseOp{0}},
        {0, ComputeOp{ComputeKind::AddCP, {2, 5}, 7, device(1), {}}},
        {0, ComputeOp{ComputeKind::AddCP, {4, 6}, 8, device(2), {}}}};
    plan.finalization.clear();
    plan.final_outputs = {7, 8};
    renumber(plan);
    return built;
}

std::vector<double> slots(std::initializer_list<double> prefix) {
    std::vector<double> result(prefix);
    result.resize(4096);
    return result;
}

void test_split_release(bool fail_second_send) {
    const auto built = split_plan();
    TrackingApi api;
    Gate gate;
    api.on_collect = [&] { gate.signal(); };
    api.before_compute = [&](const ComputeOp &op) {
        if (op.output == 4) gate.wait(true); // Hold back the second physical send.
        if (op.output == 7) {
            gate.wait();
            require(!api.expired(0), "Release dropped source before its second send");
            gate.signal(true);
        }
        if (op.output == 8) require(api.expired(0), "Runtime retained source after all sends submitted");
    };
    api.before_communication = [&](const CommAction &action) {
        if (action.outputs == std::vector<ValueId>{6} && fail_second_send) {
            require(!api.expired(0), "failed submission lost the source");
            throw std::runtime_error("injected second-send failure");
        }
    };
    auto input = api.observe(1, make_cipher(slots({1, 2, 3, 4}), "ctx", 8192, 3, 1));
    SequentialRuntime<TrackingApi> runtime(0, 1, 3, api, DeviceExecutionMode::PerDeviceWorkers);
    if (fail_second_send) {
        try { runtime.run({built.plan, "sha256:test"}, {built.operator_spec, {}, false}, {{1, input}}); }
        catch (const ClusterPanic &error) {
            require(std::string(error.what()).find("injected second-send failure") != std::string::npos,
                    "wrong submission failure");
            require(!api.expired(0), "failed send consumed its remaining use");
            return;
        }
        throw std::runtime_error("expected a submission failure");
    }
    const auto artifact = runtime.run({built.plan, "sha256:test"}, {built.operator_spec, {}, false}, {{1, input}});
    compare_values(*artifact.values.at(7).value, make_cipher(slots({3, 5, 7, 9}), "ctx", 8192, 3, 1));
    compare_values(*artifact.values.at(8).value, make_cipher(slots({1, 1, 1, 1}), "ctx", 8192, 3, 1));
    require(!api.expired(1), "Runtime destroyed caller's external input");
}

void test_abandoned_output(bool parallel, bool deferred) {
    auto built = split_plan();
    auto &plan = built.plan;
    plan.values = {value(1, ValueKind::Ciphertext, host()), value(3, ValueKind::Ciphertext, device(2)),
                   value(9, ValueKind::Ciphertext, host())};
    plan.initialization = {{0, transfer(100, 1, 3, host(), device(2))}};
    plan.execution = {{0, transfer(101, 3, 9, device(2), host())}, {0, ReleaseOp{9}}};
    plan.final_outputs = {1};
    renumber(plan);
    TrackingApi api;
    Gate gate;
    api.defer_outputs = deferred;
    api.on_collect = [&] { gate.signal(); };
    if (parallel) api.before_communication = [&](const CommAction &action) {
        if (action.id == 101) gate.wait(); // Release precedes output publication.
    };
    api.on_drain = [&] { require(api.expired(9), "late output was retained after Release"); };
    auto input = api.observe(1, make_cipher(slots({1, 2, 3, 4}), "ctx", 8192, 3, 1));
    SequentialRuntime<TrackingApi> runtime(0, 1, 3, api,
        parallel ? DeviceExecutionMode::PerDeviceWorkers : DeviceExecutionMode::Sequential);
    const auto artifact = runtime.run({plan, "sha256:test"}, {built.operator_spec, {}, false}, {{1, input}});
    require(artifact.values.size() == 1 && artifact.values.count(1), "wrong final output");
}

void test_cross_phase_and_repeated_operands() {
    auto built = split_plan();
    auto &plan = built.plan;
    // A Host input used in every phase; repeated AddCC inputs each count once.
    plan.values = {value(1, ValueKind::Ciphertext, host()), value(2, ValueKind::Ciphertext, host()),
                   value(3, ValueKind::Ciphertext, host()), value(4, ValueKind::Ciphertext, host())};
    plan.initialization = {{0, ComputeOp{ComputeKind::Negate, {1}, 2, host(), {}}}};
    plan.execution = {{0, ComputeOp{ComputeKind::AddCC, {1, 1}, 3, host(), {}}}};
    plan.finalization = {{0, ComputeOp{ComputeKind::SubCC, {3, 1}, 4, host(), {}}},
                         {0, ReleaseOp{1}}, {0, ReleaseOp{2}}, {0, ReleaseOp{3}}};
    plan.final_outputs = {4};
    renumber(plan);
    TrackingApi api;
    auto input = api.observe(1, make_cipher(slots({1, 2, 3, 4}), "ctx", 8192, 3, 1));
    api.on_drain = [&] { require(api.expired(2) && api.expired(3), "Release retained intermediates"); };
    SequentialRuntime<TrackingApi> runtime(0, 1, 3, api);
    for (int iteration = 0; iteration < 3; ++iteration) {
        const auto artifact = runtime.run({plan, "sha256:test"}, {built.operator_spec, {}, false}, {{1, input}});
        compare_values(*artifact.values.at(4).value, *input);
    }
    try { runtime.run({plan, "sha256:test"}, {built.operator_spec, {}, false}, {}, DiffMode::AllValuesAfterRun); }
    catch (const ClusterPanic &error) {
        require(std::string(error.what()).find("AllValuesAfterRun is incompatible") != std::string::npos,
                "Release historical diff mode was not rejected before input binding");
        return;
    }
    throw std::runtime_error("Release must reject AllValuesAfterRun");
}

void test_mock_multirank_release() {
    const auto cipher = make_cipher(slots({1, 2, 3, 4}), "ctx", 8192, 3, 1);
    const auto plain = make_plain(slots({2, 3, 4, 5}), "ctx", 8192, 3, 1);
    const auto reference = run_fanout_reference(cipher, plain);
    for (const auto &counts : {std::vector<int>{3}, std::vector<int>{2, 2}, std::vector<int>{1, 1, 1, 1}}) {
        const auto built = make_fanout_plan(counts);
        const auto plan = with_releases(built.plan);
        for (auto mode : {DeviceExecutionMode::Sequential, DeviceExecutionMode::PerDeviceWorkers})
            for (int seed = 0; seed < 3; ++seed) {
                MockClusterConfig config;
                config.delay_seed = seed;
                config.max_delay_ms = 3;
                const auto result = run_mock_cluster(plan, built.operator_spec, {{0, cipher}, {1, plain}},
                    config, {}, DiffMode::FinalOnly, false, {}, mode);
                const auto &artifact = result.artifacts.back();
                compare_values(artifact.values.at(plan.final_outputs.front()).value, reference.at(built.reference_output));
            }
    }
}

void test_streaming(bool parallel, bool deferred, bool fail_encode = false) {
    auto built = split_plan();
    auto &plan = built.plan;
    plan.format_version = 3;
    plan.values = {value(1, ValueKind::Ciphertext, host()),
                   value(2, ValueKind::Ciphertext, device(1))};
    plan.initialization = {{0, transfer(100, 1, 2, host(), device(1))}};
    plan.execution.clear();
    ValueId input = 2;
    for (int batch = 0; batch < 3; ++batch) {
        ValueId weight = 10 + batch * 3, uploaded = weight + 1, output = weight + 2;
        plan.values.push_back(value(weight, ValueKind::Plaintext, host()));
        plan.values.push_back(value(uploaded, ValueKind::Plaintext, device(1)));
        plan.values.push_back(value(output, ValueKind::Ciphertext, device(1)));
        plan.execution.push_back({0, EncodeOp{InlineEncodePayload{{2, 3, 4, 5}}, weight}});
        auto upload = transfer(200 + batch, weight, uploaded, host(), device(1));
        upload.output_types = {ValueKind::Plaintext};
        plan.execution.push_back({0, upload});
        plan.execution.push_back({0, ReleaseOp{weight}});
        plan.execution.push_back({0, ComputeOp{ComputeKind::AddCP, {input, uploaded}, output, device(1), {}}});
        plan.execution.push_back({0, ReleaseOp{uploaded}});
        plan.execution.push_back({0, ReleaseOp{input}});
        plan.execution.push_back({0, FenceOp{}});
        input = output;
    }
    plan.final_outputs = {input};
    renumber(plan);
    TrackingApi api;
    api.defer_outputs = deferred;
    int drains = 0;
    api.on_drain = [&] {
        if (drains < 3) {
            require(api.expired(10 + drains * 3), "Fence retained host encoding");
            require(api.expired(11 + drains * 3), "Fence retained uploaded weight");
        }
        ++drains;
    };
    api.before_encode = [&](ValueId id) {
        if (id == 13 && fail_encode) throw std::runtime_error("injected Encode failure");
        require(drains == int((id - 10) / 3), "Encode crossed unfinished Fence");
    };
    auto external = api.observe(1, make_cipher(slots({1, 2, 3, 4}), "ctx", 8192, 3, 1));
    SequentialRuntime<TrackingApi> runtime(0, 1, 3, api,
        parallel ? DeviceExecutionMode::PerDeviceWorkers : DeviceExecutionMode::Sequential);
    try {
        auto result = runtime.run({plan, "sha256:test"}, {built.operator_spec, {}, false}, {{1, external}});
        require(!fail_encode, "expected Encode failure");
        compare_values(*result.values.at(input).value, make_cipher(slots({7, 11, 15, 19}), "ctx", 8192, 3, 1));
        require(drains == 4, "missing Fence or final drain");
    } catch (const ClusterPanic &error) {
        require(fail_encode && std::string(error.what()).find("injected Encode failure") != std::string::npos,
                "wrong streaming failure");
    }
    for (int version : {1, 2}) {
        plan.format_version = version;
        try { PlanVerifier::verify(plan, built.operator_spec); }
        catch (const std::exception &) { continue; }
        throw std::runtime_error("old version accepted online Encode");
    }
}

void test_streaming_scale(bool parallel) {
    auto built = split_plan();
    auto &plan = built.plan;
    built.operator_spec.spec.poly_degree = 16;
    plan.format_version = 3;
    plan.values = {value(1, ValueKind::Ciphertext, host())};
    plan.initialization.clear(); plan.execution.clear();
    constexpr ValueId count = 20000, batch_size = 32;
    for (ValueId step = 0; step < count; ++step) {
        Place source = step % 2 ? device(1) : host();
        Place destination = step % 2 ? host() : device(1);
        plan.values.push_back(value(step + 2, ValueKind::Ciphertext, destination));
        plan.execution.push_back({0, transfer(step, step + 1, step + 2, source, destination)});
        plan.execution.push_back({0, ReleaseOp{step + 1}});
        if ((step + 1) % batch_size == 0) plan.execution.push_back({0, FenceOp{}});
    }
    plan.final_outputs = {count + 1};
    renumber(plan);
    TrackingApi api;
    api.defer_outputs = true;
    ValueId drains = 0;
    api.on_drain = [&] {
        if (drains < count / batch_size) {
            const auto first = drains * batch_size + 2;
            for (ValueId id = first; id < first + batch_size - 1; ++id)
                require(api.expired(id), "scale test retained a completed batch output");
        }
        ++drains;
    };
    api.before_communication = [&](const CommAction &action) {
        require(drains == action.id / batch_size, "scale test crossed Fence early");
    };
    auto input = api.observe(1, make_cipher(std::vector<double>(8, 2.0), "ctx", 16, 3, 1));
    SequentialRuntime<TrackingApi> runtime(0, 1, 3, api,
        parallel ? DeviceExecutionMode::PerDeviceWorkers : DeviceExecutionMode::Sequential);
    auto result = runtime.run({plan, "sha256:test"}, {built.operator_spec, {}, false}, {{1, input}});
    compare_values(*result.values.at(count + 1).value, *input);
    require(drains == count / batch_size + 1, "scale test missed a Fence");
}

} // namespace

int main() {
    try {
        for (bool parallel : {false, true})
            for (bool deferred : {false, true}) {
                test_streaming(parallel, deferred);
                test_streaming(parallel, deferred, true);
            }
        std::cout << "[PASS] V3 streaming, Fence reclamation, deferred outputs and CPU failure\n";
        test_streaming_scale(false);
        test_streaming_scale(true);
        std::cout << "[PASS] 20000 transfers and 625 Fence boundaries in each execution mode\n";
        test_cross_phase_and_repeated_operands();
        std::cout << "[PASS] cross-phase counts, repeated operands and repeated runs\n";
        test_split_release(false);
        std::cout << "[PASS] Release before split Replicate submissions\n";
        test_split_release(true);
        std::cout << "[PASS] failed submission preserves source\n";
        for (bool parallel : {false, true})
            for (bool deferred : {false, true}) test_abandoned_output(parallel, deferred);
        std::cout << "[PASS] abandoned posted/pending outputs in both execution modes\n";
        test_mock_multirank_release();
        std::cout << "[PASS] Mock multi-device and multi-rank Release fanout\n";
        std::cout << "ALL 7 RELEASE TEST GROUPS PASSED\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
