#include "testing/testing.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>

using namespace fhegpu;

namespace {
void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

// Synchronous evaluation and deferred receive publication, as in Poseidon CPU.
// Mock transport is only the in-process wire; every API entry is thread checked.
class HostApi {
public:
    using Value = VecValue;
    using CommHandle = MockVecApi::CommHandle;
    HostApi(int rank, std::shared_ptr<MockCluster> cluster)
        : inner_(rank, std::move(cluster)), main_(std::this_thread::get_id()) {}
    std::function<void(const ComputeOp &)> on_compute;
    std::function<void(TransferId)> on_post, on_wait;
    std::vector<ValueId> computed;
    std::string name() const { return "HostApi"; }
    Value compute(const ComputeOp &op, const std::vector<Value> &inputs) {
        require(std::this_thread::get_id() != main_, "Host compute ran on main");
        if (on_compute) on_compute(op);
        computed.push_back(op.output);
        return inner_.compute(op, inputs);
    }
    Value encode_plaintext(const ValueDesc &desc, const std::vector<double> &slots) {
        require(std::this_thread::get_id() != main_, "Host Encode ran on main");
        return inner_.encode_plaintext(desc, slots);
    }
    CommHandle communicate_async(const CommAction &action, const std::vector<Value> &inputs,
                                 const std::vector<ValueDesc> &descs) {
        require(std::this_thread::get_id() == main_, "Host communication left main");
        if (on_post) on_post(action.id);
        return inner_.communicate_async(action, inputs, descs);
    }
    std::vector<std::optional<Value>> wait(CommHandle &handle) {
        require(std::this_thread::get_id() == main_, "Host communication wait left main");
        if (on_wait) on_wait(handle.id);
        return inner_.wait(handle);
    }
    void preflight(std::string_view sha, bool skip, const TargetConfig &target,
                   const OperatorSpec &spec, const PlanRequirements &requirements) {
        inner_.preflight(sha, skip, target, spec, requirements);
    }
    void validate_value(const Value &v, const ValueDesc &d) const { inner_.validate_value(v, d); }
    void synchronize(Value &v) { inner_.synchronize(v); }
    void drain() { inner_.drain(); }
    void collect_completed() {}
    [[noreturn]] void abort_all(int code, const std::string &reason) {
        require(std::this_thread::get_id() == main_, "Host abort left main");
        inner_.abort_all(code, reason);
    }
private:
    MockVecApi inner_;
    const std::thread::id main_;
};

BuiltPlan host_plan() {
    auto built = make_fanout_plan({1, 1});
    auto &p = built.plan;
    p.format_version = 2;
    p.target.device_counts = {0, 0};
    const Place a{PlaceKind::Host, 0, 0}, b{PlaceKind::Host, 1, 0};
    p.values.clear();
    for (ValueId id = 0; id < 8; ++id)
        p.values.push_back({id, id == 1 ? ValueKind::Plaintext : ValueKind::Ciphertext,
            id >= 3 && id <= 5 ? b : a, "ctx", 3, 1, true, id == 1 ? 1 : 2});
    const auto send = [](TransferId id, ValueId input, ValueId output, Place src, Place dst) {
        return CommAction{id, CommKind::Transfer, CommHint::PointToPoint,
            {input}, {output}, {src}, {dst}, {ValueKind::Ciphertext}};
    };
    p.external_inputs = {0};
    p.initialization = {{0, EncodeOp{InlineEncodePayload{{2, 3, 4, 5}}, 1}}};
    p.execution = {
        {1, ComputeOp{ComputeKind::AddCP, {0, 1}, 2, a, {}}},
        {2, send(100, 0, 3, a, b)},
        {3, send(101, 2, 4, a, b)},
        {4, ComputeOp{ComputeKind::AddCC, {3, 4}, 5, b, {}}},
        {5, send(102, 5, 6, b, a)},
        {6, ComputeOp{ComputeKind::AddCC, {2, 6}, 7, a, {}}}};
    p.finalization.clear();
    p.final_outputs = {7};
    return built;
}

struct Rendezvous {
    std::mutex mutex;
    std::condition_variable condition;
    bool computing = false, communicated = false;
    void during_compute() {
        std::unique_lock<std::mutex> lock(mutex);
        computing = true;
        condition.notify_all();
        require(condition.wait_for(lock, std::chrono::seconds(5), [&] { return communicated; }),
                "main could not process communication during synchronous evaluation");
    }
    void during_wait() {
        std::unique_lock<std::mutex> lock(mutex);
        require(condition.wait_for(lock, std::chrono::seconds(5), [&] { return computing; }),
                "communication did not rendezvous with Host compute");
        communicated = true;
        condition.notify_all();
    }
};

void run_cluster(bool release, int failure) {
    auto built = host_plan();
    if (release) built.plan = with_releases(std::move(built.plan));
    MockClusterConfig config;
    config.world_size = 2;
    auto cluster = std::make_shared<MockCluster>(config);
    std::exception_ptr errors[2];
    RunArtifact<VecValue> artifacts[2];
    Rendezvous gate;
    std::vector<std::thread> ranks;
    for (int rank = 0; rank < 2; ++rank) ranks.emplace_back([&, rank] {
        try {
            HostApi api(rank, cluster);
            if (rank == 0 && failure == 0) {
                api.on_compute = [&](const ComputeOp &op) {
                    if (op.output == 2) gate.during_compute();
                };
                api.on_wait = [&](TransferId id) { if (id == 100) gate.during_wait(); };
            }
            if (rank == 0 && failure == 1)
                api.on_compute = [](const ComputeOp &) { throw std::runtime_error("injected Host compute failure"); };
            if (rank == 1 && failure == 2)
                api.on_wait = [](TransferId) { throw std::runtime_error("injected Host receive failure"); };
            SequentialRuntime<HostApi> runtime(rank, 2, 0, api, DeviceExecutionMode::PerDeviceWorkers);
            std::unordered_map<ValueId, VecValue> inputs;
            std::vector<double> slots{1, 2, 3, 4};
            slots.resize(4096);
            if (rank == 0) inputs.emplace(0, make_cipher(slots, "ctx", 8192, 3, 1));
            artifacts[rank] = runtime.run({built.plan, "sha256:" + std::string(64, '1')},
                {built.operator_spec, {}, false}, inputs);
            require(api.computed == (rank == 0 ? std::vector<ValueId>{2, 7} : std::vector<ValueId>{5}),
                    "Host task queue order changed");
        } catch (...) { errors[rank] = std::current_exception(); }
    });
    for (auto &rank : ranks) rank.join();
    if (failure) {
        require(errors[0] && errors[1] && cluster->aborted(), "failure did not cancel all waiting ranks");
        bool original = false;
        for (const auto &error : errors) try { std::rethrow_exception(error); }
        catch (const std::exception &e) { original |= std::string(e.what()).find("injected Host") != std::string::npos; }
        require(original, "worker exception was lost");
    } else {
        for (const auto &error : errors) if (error) std::rethrow_exception(error);
        require(gate.communicated, "communication/compute rendezvous was skipped");
        std::vector<double> expected{7, 12, 17, 22};
        expected.resize(4096);
        compare_values(artifacts[0].values.at(7).value,
            make_cipher(expected, "ctx", 8192, 3, 1));
    }
}

void reject_reuse_of_send_source() {
    auto built = host_plan();
    auto output = built.plan.values.at(2);
    output.id = 8;
    built.plan.values.push_back(output);
    ComputeOp overwrite{ComputeKind::Negate, {2}, 8, output.place, {}};
    overwrite.reuse_input = 0;
    built.plan.execution.push_back({7, overwrite});
    built.plan.final_outputs.push_back(8);
    try { PlanVerifier::verify(built.plan, built.operator_spec); }
    catch (const std::exception &e) {
        require(std::string(e.what()).find("exactly one use") != std::string::npos,
                "send/reuse was rejected for the wrong reason");
        return;
    }
    throw std::runtime_error("verifier allowed reuse of a send source");
}
} // namespace

int main() {
    try {
        for (bool release : {false, true}) run_cluster(release, 0);
        for (int failure : {1, 2}) run_cluster(true, failure);
        reject_reuse_of_send_source();
        std::cout << "[PASS] Host FIFO, synchronous compute/main communication rendezvous, Release and cancellation\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
