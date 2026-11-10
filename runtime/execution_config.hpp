#pragma once

// ExecutionConfig V1 -- the sidecar that says *how* to execute a RuntimePlan.
//
// The plan says what is computed, where each instruction runs and when the
// ranks communicate; none of that is negotiable here.  This file carries the
// resource posture the same instruction stream is executed with: which physical
// card a Place resolves to, how instructions are issued across the local
// devices, how big the pools are.  The plan is not rewritten and not reread --
// the two files are bound by digest, and a mismatch is refused rather than
// reconciled.
//
// Two properties are deliberate:
//
// *Everything is parsed, even what is not yet honoured.*  Seventeen of the
// twenty parameters have no execution path in this runtime today.  Accepting
// them silently would let a configuration claim an effect it does not have, so
// they are read, range-checked, and then listed by `ignored_fields()`.  A run
// can therefore state which knobs it actually applied instead of leaving the
// reader to assume all of them.
//
// *The binding is checked before anything runs.*  `binding.plan_sha256` covers
// the raw plan bytes, which is the same digest the ranks compare in preflight,
// so a configuration generated for a different program is rejected at load
// rather than producing a run that silently used the wrong resource posture.

#include "runtime/plan.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace fhegpu {

// Mirrors DeviceExecutionMode in runtime.hpp.  Kept as a separate enum so the
// file format does not become whatever the scheduler enum happens to be today;
// `device_execution_mode_of` is the one place the two are tied together.
enum class ConfigDeviceExecutionMode {
    Sequential,
    PerDevice,
    ReadyBackfill,
    DependencyWorkers
};

enum class ConfigLaunchMode { PerOpLaunch, CudaGraph, PersistentKernel };

enum class ConfigAllocGranularity { Polynomial, PolynomialSet, Arena };

enum class ConfigP2PMode { Disabled, DmaCopy, InlineLoadStore };

enum class ConfigCollectiveProto { Auto, LL, LL128, Simple };

enum class ConfigKeyStrategy { Replicate, ReplicateOnline, Split, TierToHost };

struct ExecutionConfigBinding {
    std::string plan_sha256;
    // set only once a plan-rewriting stage exists; null until then
    std::optional<std::string> parent_plan_sha256;
    std::string operator_spec_sha256;
    std::string decision_record_sha256;
};

struct ExecutionConfigTarget {
    int world_size = 1;
    std::vector<int> device_counts;
    int rank = 0;
};

struct ExecutionConfigRuntime {
    ConfigDeviceExecutionMode device_execution_mode = ConfigDeviceExecutionMode::Sequential;
    std::size_t device_worker_count = 0;
};

// One entry per local device of `target.rank`, in Place.index order: entry i is
// the physical card that Place{device, rank, i} resolves to.
struct ExecutionConfigDevice {
    int index = 0;
    int cuda_device_id = 0;
    std::uint64_t mempool_bytes = 0;
    std::uint64_t pinned_host_bytes = 0;
    int mig_instances = 1;
};

struct ExecutionConfigLaunch {
    int grid_dim = 0;
    std::uint64_t l2_persist_bytes = 0;
    double l2_persist_hit_ratio = 1.0;
    int num_streams = 1;
    std::vector<int> stream_priorities;
    ConfigLaunchMode launch_mode = ConfigLaunchMode::PerOpLaunch;
    ConfigAllocGranularity alloc_granularity = ConfigAllocGranularity::PolynomialSet;
};

struct ExecutionConfigCommunication {
    ConfigP2PMode p2p_mode = ConfigP2PMode::Disabled;
    std::uint64_t chunk_bytes = 0;
    int num_comm_streams = 0;
    ConfigCollectiveProto collective_proto = ConfigCollectiveProto::Auto;
    bool use_nvls = false;
    bool use_gpudirect_rdma = false;
    bool use_sharp = false;
    std::uint64_t net_chunk_bytes = 0;
};

struct ExecutionConfigKeys {
    ConfigKeyStrategy strategy = ConfigKeyStrategy::Replicate;
    std::string rotation_key_policy;
};

struct ExecutionConfig {
    int format_version = 1;
    std::string config_id;
    ExecutionConfigBinding binding;
    ExecutionConfigTarget target;
    ExecutionConfigRuntime runtime;
    std::vector<ExecutionConfigDevice> devices;
    ExecutionConfigLaunch launch;
    ExecutionConfigCommunication communication;
    ExecutionConfigKeys keys;
};

struct LoadedExecutionConfig {
    ExecutionConfig config;
    std::string source_sha256;
};

// -- what this runtime actually does with it -------------------------------

// The three parameters that reach an execution path today.  Named here rather
// than described in a comment so a caller can print them.
std::vector<std::string> applied_fields();

// Everything else the format carries.  Parsed and validated, but nothing in
// this runtime consults it yet; a caller that reports these has told the truth
// about what its numbers mean.
std::vector<std::string> ignored_fields();

// Refuse a configuration that was not generated for this plan, this rank, or
// this device count.  Throws std::runtime_error with the mismatch named.
void verify_execution_config(const LoadedExecutionConfig &config,
                             const LoadedRuntimePlan &plan, int rank,
                             int world_size, int local_devices);

// Adapters onto the existing constructor arguments.
std::vector<int> cuda_device_ids_of(const ExecutionConfig &config);
std::size_t device_worker_count_of(const ExecutionConfig &config);

std::string to_string(ConfigDeviceExecutionMode mode);
std::string to_string(ConfigLaunchMode mode);
std::string to_string(ConfigAllocGranularity granularity);
std::string to_string(ConfigP2PMode mode);
std::string to_string(ConfigCollectiveProto proto);
std::string to_string(ConfigKeyStrategy strategy);

} // namespace fhegpu
