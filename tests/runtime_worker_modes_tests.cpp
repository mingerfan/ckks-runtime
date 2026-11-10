#include "testing/testing.hpp"

#include <iostream>
#include <random>

using namespace fhegpu;

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr DeviceExecutionMode kModes[] = {
    DeviceExecutionMode::Sequential,
    DeviceExecutionMode::PerDeviceWorkers,
    DeviceExecutionMode::PerDeviceReadyWorkers,
    DeviceExecutionMode::PerDeviceDependencyWorkers,
};

const char *mode_name(DeviceExecutionMode mode) {
    switch (mode) {
        case DeviceExecutionMode::Sequential: return "sequential";
        case DeviceExecutionMode::PerDeviceWorkers: return "per_device";
        case DeviceExecutionMode::PerDeviceReadyWorkers: return "ready_backfill";
        case DeviceExecutionMode::PerDeviceDependencyWorkers: return "dependency_workers";
    }
    return "unknown";
}

// A posted copy whose wait may move to the end of the phase, as the GPU Api
// reports for peer copies.  Only the dependency workers act on it.
class DeferringApi : public MockVecApi {
public:
    using MockVecApi::MockVecApi;
    bool wait_can_be_deferred(const CommHandle &) const { return true; }
};

using Outputs = std::map<ValueId, VecValue>;

template <class Api>
Outputs run_mode(const RuntimePlan &plan, const LoadedOperatorSpec &spec,
                 const std::unordered_map<ValueId, VecValue> &inputs,
                 DeviceExecutionMode mode, VecExecConfig exec, int devices) {
    auto cluster = std::make_shared<MockCluster>(MockClusterConfig{});
    Api api(0, cluster, exec);
    SequentialRuntime<Api> runtime(0, 1, devices, api, mode);
    Outputs result;
    auto artifact = runtime.run({plan, "sha256:" + std::string(64, '0')},
                                {spec, std::nullopt, false}, inputs);
    for (auto &[id, value] : artifact.values) result.emplace(id, value.value);
    return result;
}

void compare_outputs(const Outputs &actual, const Outputs &expected, const std::string &what) {
    require(actual.size() == expected.size(), what + ": wrong output count");
    for (const auto &[id, value] : expected) {
        const auto found = actual.find(id);
        require(found != actual.end(), what + ": missing output " + std::to_string(id));
        try {
            compare_values(found->second, value);
        } catch (const std::exception &error) {
            throw std::runtime_error(what + ": output " + std::to_string(id) + ": " + error.what());
        }
    }
}

template <class Api>
void check_all_modes(const std::string &name, const RuntimePlan &plan,
                     const LoadedOperatorSpec &spec,
                     const std::unordered_map<ValueId, VecValue> &inputs,
                     const Outputs &expected, int devices) {
    const VecExecConfig execs[] = {{VecExecMode::Sync, 0, 0}, {VecExecMode::Async, 17, 2},
                                   {VecExecMode::Async, 91, 1}};
    for (const auto mode : kModes)
        for (const auto &exec : execs) {
            const std::string what = name + "/" + mode_name(mode) +
                (exec.mode == VecExecMode::Sync ? "/sync" : "/async" + std::to_string(exec.delay_seed));
            compare_outputs(run_mode<Api>(plan, spec, inputs, mode, exec, devices), expected, what);
        }
}

void test_fanout_all_modes() {
    const auto built = make_fanout_plan({3});
    const auto cipher = make_cipher({1, 2, 3, 4}, "ctx", 8192, 3, 1);
    const auto plain = make_plain({2, 3, 4, 5}, "ctx", 8192, 3, 1);
    const std::unordered_map<ValueId, VecValue> inputs{{0, cipher}, {1, plain}};
    const Outputs expected{{built.plan.final_outputs.front(),
                            run_fanout_reference(cipher, plain).at(built.reference_output)}};
    for (const auto &plan : {built.plan, with_releases(built.plan)}) {
        check_all_modes<MockVecApi>("fanout", plan, built.operator_spec, inputs, expected, 3);
        check_all_modes<DeferringApi>("fanout-deferred", plan, built.operator_spec, inputs, expected, 3);
    }
}

// A random single-rank graph over three devices: each step computes on one
// device and copies in whatever input lives elsewhere, so the plan carries
// many cross-device dependencies; every value is released after its last use.
RuntimePlan random_plan(const BuiltPlan &base, std::uint64_t seed, std::size_t steps) {
    std::mt19937_64 random(seed);
    auto plan = base.plan;
    const auto host = Place{PlaceKind::Host, 0, 0};
    const auto device = [](int index) { return Place{PlaceKind::Device, 0, index}; };
    const auto describe = [](ValueId id, ValueKind kind, Place place) {
        return ValueDesc{id, kind, place, "ctx", 3, 1, true, kind == ValueKind::Plaintext ? 1 : 2};
    };
    plan.values.clear();
    plan.initialization.clear();
    plan.execution.clear();
    plan.finalization.clear();
    ValueId next = 0;
    TransferId transfer = 100;
    const ValueId cipher_host = next++, plain_host = next++;
    plan.values.push_back(describe(cipher_host, ValueKind::Ciphertext, host));
    plan.values.push_back(describe(plain_host, ValueKind::Plaintext, host));
    plan.external_inputs = {cipher_host, plain_host};

    struct Live { ValueId id; int device; };
    std::vector<Live> ciphers;
    std::vector<ValueId> plains(3);
    const ValueId first = next++;
    plan.values.push_back(describe(first, ValueKind::Ciphertext, device(0)));
    plan.initialization.push_back({0, CommAction{transfer++, CommKind::Transfer, CommHint::PointToPoint,
        {cipher_host}, {first}, {host}, {device(0)}, {ValueKind::Ciphertext}}});
    ciphers.push_back({first, 0});
    std::vector<Place> places;
    for (int d = 0; d < 3; ++d) {
        plains[d] = next++;
        places.push_back(device(d));
        plan.values.push_back(describe(plains[d], ValueKind::Plaintext, device(d)));
    }
    plan.initialization.push_back({0, CommAction{transfer++, CommKind::Replicate, CommHint::Broadcast,
        {plain_host}, plains, {host}, places,
        {ValueKind::Plaintext, ValueKind::Plaintext, ValueKind::Plaintext}}});

    // Copies are made where they are used; a value copied to a device once
    // is reused there, which also leaves values with several consumers.
    std::map<std::pair<ValueId, int>, ValueId> copies;
    const auto local = [&](const Live &value, int target) {
        if (value.device == target) return value.id;
        const auto key = std::make_pair(value.id, target);
        if (const auto found = copies.find(key); found != copies.end()) return found->second;
        const ValueId copy = next++;
        plan.values.push_back(describe(copy, ValueKind::Ciphertext, device(target)));
        plan.execution.push_back({0, CommAction{transfer++, CommKind::Transfer, CommHint::PointToPoint,
            {value.id}, {copy}, {device(value.device)}, {device(target)}, {ValueKind::Ciphertext}}});
        copies.emplace(key, copy);
        return copy;
    };
    for (std::size_t step = 0; step < steps; ++step) {
        const int target = static_cast<int>(random() % 3);
        const auto pick = [&] { return ciphers[random() % ciphers.size()]; };
        const ValueId output = next++;
        plan.values.push_back(describe(output, ValueKind::Ciphertext, device(target)));
        switch (random() % 3) {
            case 0:
                plan.execution.push_back({0, ComputeOp{ComputeKind::Negate,
                    {local(pick(), target)}, output, device(target), {}}});
                break;
            case 1: {
                const ValueId left = local(pick(), target);
                const ValueId right = local(pick(), target);
                plan.execution.push_back({0, ComputeOp{ComputeKind::AddCC,
                    {left, right}, output, device(target), {}}});
                break;
            }
            default:
                plan.execution.push_back({0, ComputeOp{ComputeKind::AddCP,
                    {local(pick(), target), plains[target]}, output, device(target), {}}});
        }
        ciphers.push_back({output, target});
    }
    plan.final_outputs = {ciphers[ciphers.size() - 1].id, ciphers[ciphers.size() - 2].id};
    return with_releases(plan);
}

void test_random_graphs_all_modes() {
    const auto base = make_fanout_plan({3});
    const auto cipher = make_cipher({1, 2, 3, 4}, "ctx", 8192, 3, 1);
    const auto plain = make_plain({2, 3, 4, 5}, "ctx", 8192, 3, 1);
    const std::unordered_map<ValueId, VecValue> inputs{{0, cipher}, {1, plain}};
    for (std::uint64_t seed = 1; seed <= 12; ++seed) {
        const auto plan = random_plan(base, seed, 40 + seed * 5);
        const auto expected = run_mode<MockVecApi>(plan, base.operator_spec, inputs,
            DeviceExecutionMode::Sequential, {VecExecMode::Sync, 0, 0}, 3);
        const std::string name = "random" + std::to_string(seed);
        check_all_modes<MockVecApi>(name, plan, base.operator_spec, inputs, expected, 3);
        check_all_modes<DeferringApi>(name + "-deferred", plan, base.operator_spec, inputs, expected, 3);
    }
}

void test_worker_modes_refuse_shared_dispatcher_work() {
    // An online Encode needs the PerDeviceWorkers dispatcher; the other two
    // executors must say so instead of waiting for a value nobody produces.
    auto built = make_fanout_plan({2});
    auto &plan = built.plan;
    plan.format_version = 3;
    const ValueId encoded = 1000;
    plan.values.push_back(ValueDesc{encoded, ValueKind::Plaintext, {PlaceKind::Host, 0, 0},
                                    "ctx", 3, 1, true, 1});
    plan.execution.insert(plan.execution.begin(),
                          {0, EncodeOp{InlineEncodePayload{{1, 2, 3, 4}}, encoded}});
    std::size_t ordinal = 0;
    for (auto *phase : {&plan.initialization, &plan.execution, &plan.finalization})
        for (auto &instruction : *phase) instruction.ordinal = ordinal++;
    const auto cipher = make_cipher({1, 2, 3, 4}, "ctx", 8192, 3, 1);
    const auto plain = make_plain({2, 3, 4, 5}, "ctx", 8192, 3, 1);
    for (const auto mode : {DeviceExecutionMode::PerDeviceReadyWorkers,
                            DeviceExecutionMode::PerDeviceDependencyWorkers}) {
        try {
            run_mode<MockVecApi>(plan, built.operator_spec, {{0, cipher}, {1, plain}}, mode,
                                 {VecExecMode::Sync, 0, 0}, 2);
        } catch (const std::exception &error) {
            require(std::string(error.what()).find("does not support online Encode") != std::string::npos,
                    std::string(mode_name(mode)) + ": wrong refusal: " + error.what());
            continue;
        }
        throw std::runtime_error(std::string(mode_name(mode)) + " accepted an online Encode");
    }
}

} // namespace

int main() {
    const std::pair<const char *, void (*)()> tests[] = {
        {"fanout in every mode", test_fanout_all_modes},
        {"random multi-device graphs with Release in every mode", test_random_graphs_all_modes},
        {"worker modes refuse dispatcher-only work", test_worker_modes_refuse_shared_dispatcher_work},
    };
    int failures = 0;
    for (const auto &[name, test] : tests) {
        try {
            test();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception &error) {
            ++failures;
            std::cout << "[FAIL] " << name << ": " << error.what() << '\n';
        }
    }
    return failures == 0 ? 0 : 1;
}
