#include "runtime/json_execution_config_reader.hpp"
#include "runtime/json_utils.hpp"

#include <istream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>

namespace fhegpu {
namespace {

using namespace json_utils;
constexpr const char *doc = "execution config";
constexpr int kSupportedFormatVersion = 1;

ConfigDeviceExecutionMode read_device_execution_mode(const Json &value,
                                                    const std::string &path) {
    const std::string mode = read_string(value, doc, path);
    if (mode == "sequential") return ConfigDeviceExecutionMode::Sequential;
    if (mode == "per_device") return ConfigDeviceExecutionMode::PerDevice;
    if (mode == "ready_backfill") return ConfigDeviceExecutionMode::ReadyBackfill;
    if (mode == "dependency_workers") return ConfigDeviceExecutionMode::DependencyWorkers;
    fail(doc, path, "unknown device execution mode '" + mode + "'");
}

ConfigLaunchMode read_launch_mode(const Json &value, const std::string &path) {
    const std::string mode = read_string(value, doc, path);
    if (mode == "per_op_launch") return ConfigLaunchMode::PerOpLaunch;
    if (mode == "cuda_graph") return ConfigLaunchMode::CudaGraph;
    if (mode == "persistent_kernel") return ConfigLaunchMode::PersistentKernel;
    fail(doc, path, "unknown launch mode '" + mode + "'");
}

ConfigAllocGranularity read_alloc_granularity(const Json &value, const std::string &path) {
    const std::string value_name = read_string(value, doc, path);
    if (value_name == "polynomial") return ConfigAllocGranularity::Polynomial;
    if (value_name == "polynomial_set") return ConfigAllocGranularity::PolynomialSet;
    if (value_name == "arena") return ConfigAllocGranularity::Arena;
    fail(doc, path, "unknown allocation granularity '" + value_name + "'");
}

ConfigP2PMode read_p2p_mode(const Json &value, const std::string &path) {
    const std::string mode = read_string(value, doc, path);
    if (mode == "disabled") return ConfigP2PMode::Disabled;
    if (mode == "dma_copy") return ConfigP2PMode::DmaCopy;
    if (mode == "inline_load_store") return ConfigP2PMode::InlineLoadStore;
    fail(doc, path, "unknown p2p mode '" + mode + "'");
}

ConfigCollectiveProto read_collective_proto(const Json &value, const std::string &path) {
    const std::string proto = read_string(value, doc, path);
    if (proto == "auto") return ConfigCollectiveProto::Auto;
    if (proto == "ll") return ConfigCollectiveProto::LL;
    if (proto == "ll128") return ConfigCollectiveProto::LL128;
    if (proto == "simple") return ConfigCollectiveProto::Simple;
    fail(doc, path, "unknown collective protocol '" + proto + "'");
}

ConfigKeyStrategy read_key_strategy(const Json &value, const std::string &path) {
    const std::string strategy = read_string(value, doc, path);
    if (strategy == "replicate") return ConfigKeyStrategy::Replicate;
    if (strategy == "replicate-online") return ConfigKeyStrategy::ReplicateOnline;
    if (strategy == "split") return ConfigKeyStrategy::Split;
    if (strategy == "tier-to-host") return ConfigKeyStrategy::TierToHost;
    fail(doc, path, "unknown key strategy '" + strategy + "'");
}

ExecutionConfigBinding read_binding(const Json &value, const std::string &path) {
    // parent_plan_sha256 is required to be *present* and allowed to be null:
    // "no plan rewrite happened" is a statement the file makes, not one the
    // reader should have to infer from an absent key.
    require_members(value, doc, path,
                    {"plan_sha256", "parent_plan_sha256", "operator_spec_sha256",
                     "decision_record_sha256"});
    ExecutionConfigBinding binding;
    binding.plan_sha256 = read_sha256(value.at("plan_sha256"), doc, path + ".plan_sha256");
    const Json &parent = value.at("parent_plan_sha256");
    if (!parent.is_null())
        binding.parent_plan_sha256 =
            read_sha256(parent, doc, path + ".parent_plan_sha256");
    binding.operator_spec_sha256 =
        read_sha256(value.at("operator_spec_sha256"), doc, path + ".operator_spec_sha256");
    binding.decision_record_sha256 =
        read_sha256(value.at("decision_record_sha256"), doc, path + ".decision_record_sha256");
    return binding;
}

ExecutionConfigTarget read_target(const Json &value, const std::string &path) {
    require_members(value, doc, path, {"world_size", "device_counts", "rank"});
    ExecutionConfigTarget target;
    target.world_size = read_positive_int(value.at("world_size"), doc, path + ".world_size");
    const Json &counts = value.at("device_counts");
    if (!counts.is_array()) fail(doc, path + ".device_counts", "expected array");
    for (std::size_t i = 0; i < counts.size(); ++i)
        target.device_counts.push_back(read_nonnegative_int(
            counts[i], doc, item_path(path + ".device_counts", i)));
    target.rank = read_nonnegative_int(value.at("rank"), doc, path + ".rank");
    if (target.rank >= target.world_size)
        fail(doc, path + ".rank", "rank is outside the world");
    if (static_cast<int>(target.device_counts.size()) != target.world_size)
        fail(doc, path + ".device_counts", "one entry per rank is required");
    return target;
}

ExecutionConfigRuntime read_runtime(const Json &value, const std::string &path) {
    require_members(value, doc, path, {"device_execution_mode", "device_worker_count"});
    ExecutionConfigRuntime runtime;
    runtime.device_execution_mode = read_device_execution_mode(
        value.at("device_execution_mode"), path + ".device_execution_mode");
    runtime.device_worker_count = static_cast<std::size_t>(read_safe_uint(
        value.at("device_worker_count"), doc, path + ".device_worker_count", 0, 4096));
    return runtime;
}

ExecutionConfigDevice read_device(const Json &value, const std::string &path,
                                  std::size_t expected_index) {
    require_members(value, doc, path,
                    {"index", "cuda_device_id", "mempool_bytes", "pinned_host_bytes",
                     "mig_instances"});
    ExecutionConfigDevice device;
    device.index = read_nonnegative_int(value.at("index"), doc, path + ".index");
    // devices[] is read positionally by Place.index, so a list whose entries
    // disagree with their own position would resolve instructions to the wrong
    // card while looking well formed.
    if (static_cast<std::size_t>(device.index) != expected_index)
        fail(doc, path + ".index",
             "device entries must be in Place.index order starting at 0");
    device.cuda_device_id =
        read_nonnegative_int(value.at("cuda_device_id"), doc, path + ".cuda_device_id");
    device.mempool_bytes = read_safe_uint(value.at("mempool_bytes"), doc,
                                          path + ".mempool_bytes", 0,
                                          std::numeric_limits<std::uint64_t>::max());
    device.pinned_host_bytes = read_safe_uint(value.at("pinned_host_bytes"), doc,
                                              path + ".pinned_host_bytes", 0,
                                              std::numeric_limits<std::uint64_t>::max());
    device.mig_instances =
        read_positive_int(value.at("mig_instances"), doc, path + ".mig_instances");
    return device;
}

ExecutionConfigLaunch read_launch(const Json &value, const std::string &path) {
    require_members(value, doc, path,
                    {"grid_dim", "l2_persist_bytes", "l2_persist_hit_ratio", "num_streams",
                     "stream_priorities", "launch_mode", "alloc_granularity"});
    ExecutionConfigLaunch launch;
    launch.grid_dim = read_nonnegative_int(value.at("grid_dim"), doc, path + ".grid_dim");
    launch.l2_persist_bytes = read_safe_uint(value.at("l2_persist_bytes"), doc,
                                             path + ".l2_persist_bytes", 0,
                                             std::numeric_limits<std::uint64_t>::max());
    launch.l2_persist_hit_ratio =
        read_finite_double(value.at("l2_persist_hit_ratio"), doc, path + ".l2_persist_hit_ratio");
    if (launch.l2_persist_hit_ratio < 0.0 || launch.l2_persist_hit_ratio > 1.0)
        fail(doc, path + ".l2_persist_hit_ratio", "ratio must lie in [0, 1]");
    launch.num_streams = read_positive_int(value.at("num_streams"), doc, path + ".num_streams");
    const Json &priorities = value.at("stream_priorities");
    if (!priorities.is_array()) fail(doc, path + ".stream_priorities", "expected array");
    for (std::size_t i = 0; i < priorities.size(); ++i)
        launch.stream_priorities.push_back(read_int(
            priorities[i], doc, item_path(path + ".stream_priorities", i), -128, 128));
    if (!launch.stream_priorities.empty() &&
        static_cast<int>(launch.stream_priorities.size()) != launch.num_streams)
        fail(doc, path + ".stream_priorities",
             "give one priority per stream, or an empty list for the default");
    launch.launch_mode = read_launch_mode(value.at("launch_mode"), path + ".launch_mode");
    launch.alloc_granularity =
        read_alloc_granularity(value.at("alloc_granularity"), path + ".alloc_granularity");
    return launch;
}

ExecutionConfigCommunication read_communication(const Json &value, const std::string &path) {
    require_members(value, doc, path,
                    {"p2p_mode", "chunk_bytes", "num_comm_streams", "collective_proto",
                     "use_nvls", "use_gpudirect_rdma", "use_sharp", "net_chunk_bytes"});
    ExecutionConfigCommunication comm;
    comm.p2p_mode = read_p2p_mode(value.at("p2p_mode"), path + ".p2p_mode");
    comm.chunk_bytes = read_safe_uint(value.at("chunk_bytes"), doc, path + ".chunk_bytes", 0,
                                      std::numeric_limits<std::uint64_t>::max());
    comm.num_comm_streams =
        read_nonnegative_int(value.at("num_comm_streams"), doc, path + ".num_comm_streams");
    comm.collective_proto =
        read_collective_proto(value.at("collective_proto"), path + ".collective_proto");
    comm.use_nvls = read_bool(value.at("use_nvls"), doc, path + ".use_nvls");
    comm.use_gpudirect_rdma =
        read_bool(value.at("use_gpudirect_rdma"), doc, path + ".use_gpudirect_rdma");
    comm.use_sharp = read_bool(value.at("use_sharp"), doc, path + ".use_sharp");
    comm.net_chunk_bytes = read_safe_uint(value.at("net_chunk_bytes"), doc,
                                          path + ".net_chunk_bytes", 0,
                                          std::numeric_limits<std::uint64_t>::max());
    return comm;
}

ExecutionConfigKeys read_keys(const Json &value, const std::string &path) {
    require_members(value, doc, path, {"strategy", "rotation_key_policy"});
    ExecutionConfigKeys keys;
    keys.strategy = read_key_strategy(value.at("strategy"), path + ".strategy");
    keys.rotation_key_policy =
        read_string(value.at("rotation_key_policy"), doc, path + ".rotation_key_policy");
    return keys;
}

ExecutionConfig read_document(const Json &root) {
    require_members(root, doc, "$",
                    {"format_version", "config_id", "binding", "target", "runtime",
                     "devices", "launch", "communication", "keys"});

    ExecutionConfig config;
    config.format_version =
        read_positive_int(root.at("format_version"), doc, "$.format_version");
    if (config.format_version != kSupportedFormatVersion)
        fail(doc, "$.format_version",
             "unsupported format version " + std::to_string(config.format_version) +
                 "; this build reads version " + std::to_string(kSupportedFormatVersion));

    config.config_id = read_string(root.at("config_id"), doc, "$.config_id");
    config.binding = read_binding(root.at("binding"), "$.binding");
    config.target = read_target(root.at("target"), "$.target");
    config.runtime = read_runtime(root.at("runtime"), "$.runtime");

    const Json &devices = root.at("devices");
    if (!devices.is_array()) fail(doc, "$.devices", "expected array");
    for (std::size_t i = 0; i < devices.size(); ++i)
        config.devices.push_back(read_device(devices[i], item_path("$.devices", i), i));

    // Two ranks assigned the same physical card would silently halve the
    // memory each thinks it has, and the topology-aware binding this file
    // exists to carry would be meaningless.
    std::set<int> seen;
    for (const auto &device : config.devices)
        if (!seen.insert(device.cuda_device_id).second)
            fail(doc, "$.devices",
                 "cuda_device_id " + std::to_string(device.cuda_device_id) +
                     " is bound twice");

    config.launch = read_launch(root.at("launch"), "$.launch");
    config.communication = read_communication(root.at("communication"), "$.communication");
    config.keys = read_keys(root.at("keys"), "$.keys");
    return config;
}

} // namespace

LoadedExecutionConfig ExecutionConfigJsonReader::read(std::istream &input) {
    const std::string text{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
    return read_text(text);
}

LoadedExecutionConfig ExecutionConfigJsonReader::read_text(std::string_view text) {
    return {read_document(json_utils::parse(text, doc)), json_utils::source_sha256(text)};
}

LoadedExecutionConfig ExecutionConfigJsonReader::read_file(const std::string &path) {
    return read_text(json_utils::read_file_bytes(path));
}

} // namespace fhegpu
