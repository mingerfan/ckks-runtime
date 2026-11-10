#include "runtime/execution_config.hpp"

#include <stdexcept>
#include <string>

namespace fhegpu {

std::vector<std::string> applied_fields() {
    return {
        "runtime.device_execution_mode",
        "runtime.device_worker_count",
        "devices[].cuda_device_id",
    };
}

std::vector<std::string> ignored_fields() {
    return {
        "devices[].mempool_bytes",
        "devices[].pinned_host_bytes",
        "devices[].mig_instances",
        "launch.grid_dim",
        "launch.l2_persist_bytes",
        "launch.l2_persist_hit_ratio",
        "launch.num_streams",
        "launch.stream_priorities",
        "launch.launch_mode",
        "launch.alloc_granularity",
        "communication.p2p_mode",
        "communication.chunk_bytes",
        "communication.num_comm_streams",
        "communication.collective_proto",
        "communication.use_nvls",
        "communication.use_gpudirect_rdma",
        "communication.use_sharp",
        "communication.net_chunk_bytes",
        "keys.strategy",
        "keys.rotation_key_policy",
    };
}

namespace {

[[noreturn]] void mismatch(const std::string &what) {
    throw std::runtime_error("execution config does not match this run: " + what);
}

} // namespace

void verify_execution_config(const LoadedExecutionConfig &loaded,
                             const LoadedRuntimePlan &plan, int rank,
                             int world_size, int local_devices) {
    const ExecutionConfig &config = loaded.config;

    // The digest is the whole safety belt.  It covers the plan's raw bytes,
    // which is the same thing the ranks compare in preflight, so a
    // configuration generated for a different program cannot be mistaken for
    // one generated for this one.
    if (config.binding.plan_sha256 != plan.source_sha256)
        mismatch("it was generated for plan " + config.binding.plan_sha256 +
                 " but this run loaded " + plan.source_sha256);

    // A parent digest means some stage rewrote the plan.  Nothing does yet, so
    // its presence means the configuration came through a pipeline this build
    // does not have; refusing is safer than assuming the rewrite was a no-op.
    if (config.binding.parent_plan_sha256.has_value())
        mismatch("it carries parent_plan_sha256, so it was produced by a "
                 "plan-rewriting stage this runtime does not implement");

    if (config.binding.operator_spec_sha256 != plan.plan.target.operator_spec.source_sha256)
        mismatch("its operator spec digest " + config.binding.operator_spec_sha256 +
                 " differs from the plan's " +
                 plan.plan.target.operator_spec.source_sha256);

    if (config.target.world_size != world_size)
        mismatch("it describes world_size " + std::to_string(config.target.world_size) +
                 " but this run has " + std::to_string(world_size));

    if (config.target.rank != rank)
        mismatch("it describes rank " + std::to_string(config.target.rank) +
                 " but this is rank " + std::to_string(rank));

    if (static_cast<int>(config.target.device_counts.size()) != world_size)
        mismatch("device_counts has " +
                 std::to_string(config.target.device_counts.size()) +
                 " entries for world_size " + std::to_string(world_size));

    if (config.target.device_counts[static_cast<std::size_t>(rank)] != local_devices)
        mismatch("it gives rank " + std::to_string(rank) + " " +
                 std::to_string(config.target.device_counts[static_cast<std::size_t>(rank)]) +
                 " device(s) but this run has " + std::to_string(local_devices));

    // devices[] is indexed by Place.index, so a short list would leave some
    // instruction with no card to resolve to.
    if (static_cast<int>(config.devices.size()) != local_devices)
        mismatch("devices[] describes " + std::to_string(config.devices.size()) +
                 " device(s) but rank " + std::to_string(rank) + " has " +
                 std::to_string(local_devices));
}

std::vector<int> cuda_device_ids_of(const ExecutionConfig &config) {
    std::vector<int> ids;
    ids.reserve(config.devices.size());
    for (const auto &device : config.devices) ids.push_back(device.cuda_device_id);
    return ids;
}

std::size_t device_worker_count_of(const ExecutionConfig &config) {
    return config.runtime.device_worker_count;
}

std::string to_string(ConfigDeviceExecutionMode mode) {
    switch (mode) {
        case ConfigDeviceExecutionMode::Sequential: return "sequential";
        case ConfigDeviceExecutionMode::PerDevice: return "per_device";
        case ConfigDeviceExecutionMode::ReadyBackfill: return "ready_backfill";
        case ConfigDeviceExecutionMode::DependencyWorkers: return "dependency_workers";
    }
    return "unknown";
}

std::string to_string(ConfigLaunchMode mode) {
    switch (mode) {
        case ConfigLaunchMode::PerOpLaunch: return "per_op_launch";
        case ConfigLaunchMode::CudaGraph: return "cuda_graph";
        case ConfigLaunchMode::PersistentKernel: return "persistent_kernel";
    }
    return "unknown";
}

std::string to_string(ConfigAllocGranularity granularity) {
    switch (granularity) {
        case ConfigAllocGranularity::Polynomial: return "polynomial";
        case ConfigAllocGranularity::PolynomialSet: return "polynomial_set";
        case ConfigAllocGranularity::Arena: return "arena";
    }
    return "unknown";
}

std::string to_string(ConfigP2PMode mode) {
    switch (mode) {
        case ConfigP2PMode::Disabled: return "disabled";
        case ConfigP2PMode::DmaCopy: return "dma_copy";
        case ConfigP2PMode::InlineLoadStore: return "inline_load_store";
    }
    return "unknown";
}

std::string to_string(ConfigCollectiveProto proto) {
    switch (proto) {
        case ConfigCollectiveProto::Auto: return "auto";
        case ConfigCollectiveProto::LL: return "ll";
        case ConfigCollectiveProto::LL128: return "ll128";
        case ConfigCollectiveProto::Simple: return "simple";
    }
    return "unknown";
}

std::string to_string(ConfigKeyStrategy strategy) {
    switch (strategy) {
        case ConfigKeyStrategy::Replicate: return "replicate";
        case ConfigKeyStrategy::ReplicateOnline: return "replicate-online";
        case ConfigKeyStrategy::Split: return "split";
        case ConfigKeyStrategy::TierToHost: return "tier-to-host";
    }
    return "unknown";
}

} // namespace fhegpu
