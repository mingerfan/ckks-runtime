#include "testing/testing.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

namespace fhegpu {
namespace {

Instruction instruction(std::size_t ordinal, InstructionBody body) {
    return Instruction{ordinal, std::move(body)};
}

ValueDesc value_desc(ValueId id, ValueKind kind, Place place, int scale_log2 = 1) {
    return ValueDesc{id, kind, place, "ctx", 3, scale_log2, true,
                     kind == ValueKind::Plaintext ? 1 : 2};
}

LoadedOperatorSpec make_test_operator_spec() {
    OperatorSpec spec;
    spec.id = "vec-operator-spec";
    spec.version = 1;
    spec.status = "placeholder";
    spec.target_id = "vec-fixed-target";
    spec.context_id = "ctx";
    spec.poly_degree = 8192;
    spec.rns_moduli_log2 = {60, 40, 40, 40, 60};
    spec.max_modulus_log2 = 60;
    spec.default_scale_log2 = 40;
    spec.level_lower_bound = 0;
    spec.level_upper_bound = 4;
    for (ComputeKind kind : {ComputeKind::AddCC, ComputeKind::AddCP, ComputeKind::SubCC,
             ComputeKind::SubCP, ComputeKind::MulCC, ComputeKind::MulCP, ComputeKind::Negate,
             ComputeKind::Rotate, ComputeKind::Rescale, ComputeKind::ModSwitch,
             ComputeKind::Relinearize, ComputeKind::Boot}) {
        OperatorSupport support;
        support.supported = true;
        if (kind == ComputeKind::Rescale) support.max_levels_per_op = 2;
        spec.operators.emplace(kind, std::move(support));
    }
    BootProfile boot;
    boot.profile_id = "test-boot";
    boot.implementation = BootImplementation::DecryptReencrypt;
    boot.input_level_min = 0;
    boot.input_level_max = 4;
    boot.input_components = 2;
    boot.output_level = 4;
    boot.output_scale_log2 = 1;
    boot.output_components = 2;
    boot.latency_us = 1;
    boot.needs_secret_key = true;
    boot.needs_host_compute = true;
    spec.boot_profiles.push_back(std::move(boot));
    return {std::move(spec), "sha256:0000000000000000000000000000000000000000000000000000000000000000"};
}

} // namespace

BuiltPlan make_fanout_plan(const std::vector<int> &device_counts) {
    if (device_counts.empty()) throw std::runtime_error("fanout plan requires at least one rank");
    std::vector<Place> devices;
    for (std::size_t rank = 0; rank < device_counts.size(); ++rank) {
        if (device_counts[rank] < 0) throw std::runtime_error("negative device count");
        for (int index = 0; index < device_counts[rank]; ++index)
            devices.push_back(Place{PlaceKind::Device, static_cast<int>(rank), index});
    }
    if (devices.empty()) throw std::runtime_error("fanout plan requires at least one device");

    BuiltPlan built;
    built.operator_spec = make_test_operator_spec();
    auto &plan = built.plan;
    plan.plan_id = 0x46504750554c4cULL;
    plan.target.target_id = "vec-fixed-target";
    plan.target.world_size = static_cast<int>(device_counts.size());
    plan.target.device_counts = device_counts;
    plan.target.capability_version = 1;
    plan.target.operator_spec = {"vec-operator-spec", 1,
        "sha256:0000000000000000000000000000000000000000000000000000000000000000"};
    ValueId next_value = 0;
    TransferId next_transfer = 100;
    std::size_t ordinal = 0;
    const ValueId cipher_host = next_value++;
    const ValueId plain_host = next_value++;
    plan.values.push_back(value_desc(cipher_host, ValueKind::Ciphertext, {PlaceKind::Host, 0, 0}));
    plan.values.push_back(value_desc(plain_host, ValueKind::Plaintext, {PlaceKind::Host, 0, 0}));
    plan.external_inputs = {cipher_host, plain_host};

    const ValueId cipher_device = next_value++;
    plan.values.push_back(value_desc(cipher_device, ValueKind::Ciphertext, devices.front()));
    plan.initialization.push_back(instruction(ordinal++, CommAction{next_transfer++, CommKind::Transfer, CommHint::PointToPoint,
        {cipher_host}, {cipher_device}, {{PlaceKind::Host, 0, 0}}, {devices.front()}, {ValueKind::Ciphertext}}));
    built.diff_map.push_back({0, cipher_device, cipher_host});

    std::vector<ValueId> plains;
    std::vector<ValueKind> plain_types;
    for (const Place &place : devices) {
        const ValueId id = next_value++;
        plains.push_back(id);
        plain_types.push_back(ValueKind::Plaintext);
        plan.values.push_back(value_desc(id, ValueKind::Plaintext, place));
    }
    const bool replicate_plain = devices.size() > 1;
    plan.initialization.push_back(instruction(ordinal++, CommAction{
        next_transfer++, replicate_plain ? CommKind::Replicate : CommKind::Transfer,
        replicate_plain ? CommHint::Broadcast : CommHint::PointToPoint,
        {plain_host}, plains, {{PlaceKind::Host, 0, 0}}, devices, plain_types}));
    for (ValueId id : plains) built.diff_map.push_back({1, id, plain_host});

    const ValueId product = next_value++;
    plan.values.push_back(value_desc(product, ValueKind::Ciphertext, devices.front(), 2));
    plan.execution.push_back(instruction(ordinal++, ComputeOp{ComputeKind::MulCP, {cipher_device, plains.front()}, product, devices.front(), {}}));

    std::vector<ValueId> products(devices.size());
    products[0] = product;
    if (devices.size() > 1) {
        std::vector<Place> destinations(devices.begin() + 1, devices.end());
        std::vector<ValueId> outputs;
        std::vector<ValueKind> types;
        for (const Place &place : destinations) {
            const ValueId id = next_value++;
            outputs.push_back(id); types.push_back(ValueKind::Ciphertext);
            products[outputs.size()] = id;
            plan.values.push_back(value_desc(id, ValueKind::Ciphertext, place, 2));
        }
        const CommKind product_comm_kind = outputs.size() == 1 ? CommKind::Transfer : CommKind::Replicate;
        plan.execution.push_back(instruction(ordinal++, CommAction{next_transfer++, product_comm_kind, CommHint::Broadcast,
            {product}, outputs, {devices.front()}, destinations, types}));
    }

    std::vector<ValueId> branches;
    for (std::size_t i = 0; i < devices.size(); ++i) {
        const ValueId id = next_value++;
        branches.push_back(id);
        plan.values.push_back(value_desc(id, ValueKind::Ciphertext, devices[i], 2));
        plan.execution.push_back(instruction(ordinal++, ComputeOp{ComputeKind::Negate, {products[i]}, id, devices[i], {}}));
    }

    const Place output_host{PlaceKind::Host, devices.back().rank, 0};
    const ValueId final = next_value++;
    plan.values.push_back(value_desc(final, ValueKind::Ciphertext, output_host, 2));
    plan.finalization.push_back(instruction(ordinal++, CommAction{next_transfer++, CommKind::Transfer, CommHint::Auto,
        {branches.back()}, {final}, {devices.back()}, {output_host}, {ValueKind::Ciphertext}}));
    plan.final_outputs = {final};

    built.reference_output = 3;
    built.diff_map.push_back({2, product, 2});
    for (std::size_t i = 1; i < products.size(); ++i) built.diff_map.push_back({3, products[i], 2});
    for (ValueId id : branches) built.diff_map.push_back({4, id, 3});
    built.diff_map.push_back({ordinal - 1, final, 3});
    return built;
}

std::unordered_map<ValueId, VecValue> SequentialReferenceExecutor::run(
    const std::vector<ComputeOp> &ops, const std::unordered_map<ValueId, VecValue> &inputs) {
    std::unordered_map<ValueId, VecValue> values = inputs;
    VecExecutor executor;
    for (const auto &op : ops) {
        std::vector<VecValue> operands;
        for (ValueId id : op.inputs) operands.push_back(values.at(id));
        if (!values.emplace(op.output, executor.compute(op, operands)).second)
            throw std::runtime_error("reference executor duplicate ValueId");
    }
    return values;
}

std::unordered_map<ValueId, VecValue> run_fanout_reference(const VecValue &cipher, const VecValue &plain) {
    const Place place{PlaceKind::Device, 0, 0};
    const std::vector<ComputeOp> ops = {
        {ComputeKind::MulCP, {0, 1}, 2, place, {}},
        {ComputeKind::Negate, {2}, 3, place, {}}
    };
    return SequentialReferenceExecutor{}.run(ops, {{0, cipher.deep_copy()}, {1, plain.deep_copy()}});
}

void compare_values(const VecValue &actual_value, const VecValue &expected_value,
                    double absolute_tolerance, double relative_tolerance) {
    const VecPayload actual = actual_value.materialize();
    const VecPayload expected = expected_value.materialize();
    if (actual.kind != expected.kind) throw std::runtime_error("value kind mismatch");
    if (!(actual.metadata == expected.metadata)) throw std::runtime_error("value metadata mismatch");
    if (actual.slots.size() != expected.slots.size()) throw std::runtime_error("slot count mismatch");
    for (std::size_t i = 0; i < actual.slots.size(); ++i) {
        const double a = actual.slots[i], e = expected.slots[i];
        if (std::isnan(a) || std::isnan(e)) { if (!(std::isnan(a) && std::isnan(e))) throw std::runtime_error("NaN classification mismatch"); continue; }
        if (std::isinf(a) || std::isinf(e)) { if (a != e) throw std::runtime_error("Inf classification mismatch"); continue; }
        const double error = std::abs(a - e);
        const double limit = absolute_tolerance + relative_tolerance * std::abs(e);
        if (error > limit) throw std::runtime_error("slot value mismatch at index " + std::to_string(i));
    }
}

HarnessResult run_mock_cluster(const RuntimePlan &plan,
                               const LoadedOperatorSpec &operator_spec,
                               const std::unordered_map<ValueId, VecValue> &rank0_inputs,
                               MockClusterConfig cluster_config,
                               VecExecConfig exec_config,
                               DiffMode diff_mode,
                               bool skip_artifact_digest_checks,
                               const std::vector<std::optional<std::filesystem::path>> &bundle_dirs) {
    cluster_config.world_size = plan.target.world_size;
    auto cluster = std::make_shared<MockCluster>(cluster_config);
    std::vector<std::unique_ptr<MockVecApi>> apis;
    std::vector<std::unique_ptr<SequentialRuntime<MockVecApi>>> runtimes;
    std::vector<RunArtifact<VecValue>> artifacts(static_cast<std::size_t>(plan.target.world_size));
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(plan.target.world_size));
    for (int rank = 0; rank < plan.target.world_size; ++rank) {
        apis.push_back(std::make_unique<MockVecApi>(rank, cluster, exec_config,
                                                    cluster_config.fail_compute,
                                                    cluster_config.fail_communicate));
        runtimes.push_back(std::make_unique<SequentialRuntime<MockVecApi>>(rank, plan.target.world_size,
            plan.target.device_counts.at(static_cast<std::size_t>(rank)), *apis.back()));
    }
    std::vector<std::thread> threads;
    for (int rank = 0; rank < plan.target.world_size; ++rank) {
        threads.emplace_back([&, rank] {
            try {
                const std::unordered_map<ValueId, VecValue> empty;
                const LoadedRuntimePlan loaded{plan,
                    "sha256:1111111111111111111111111111111111111111111111111111111111111111"};
                const auto bundle_dir = bundle_dirs.empty() ? std::optional<std::filesystem::path>{}
                    : bundle_dirs.at(static_cast<std::size_t>(rank));
                const RuntimeResources resources{operator_spec, bundle_dir, skip_artifact_digest_checks};
                artifacts[static_cast<std::size_t>(rank)] = runtimes[static_cast<std::size_t>(rank)]->run(
                    loaded, resources, rank == 0 ? rank0_inputs : empty, diff_mode);
            } catch (...) { errors[static_cast<std::size_t>(rank)] = std::current_exception(); }
        });
    }
    for (auto &thread : threads) thread.join();
    for (const auto &error : errors) if (error) std::rethrow_exception(error);
    HarnessResult result;
    result.artifacts = std::move(artifacts);
    for (const auto &api : apis) result.stats.push_back(api->stats());
    return result;
}

} // namespace fhegpu
