#include "api/mpi_api.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace fhegpu {

void MpiVecApi::check_mpi(int code, const char *operation) {
    if (code == MPI_SUCCESS) return;
    char text[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(code, text, &length);
    throw std::runtime_error(std::string(operation) + " failed: " + std::string(text, static_cast<std::size_t>(length)));
}

MpiVecApi::MpiVecApi(MPI_Comm communicator) : communicator_(communicator), executor_({VecExecMode::Sync, 0, 0}) {
    check_mpi(MPI_Comm_rank(communicator_, &rank_), "MPI_Comm_rank");
    check_mpi(MPI_Comm_size(communicator_, &world_size_), "MPI_Comm_size");
    int *tag_limit = nullptr;
    int present = 0;
    check_mpi(MPI_Comm_get_attr(communicator_, MPI_TAG_UB, &tag_limit, &present), "MPI_Comm_get_attr(MPI_TAG_UB)");
    if (!present || !tag_limit) throw std::runtime_error("MPI_TAG_UB is unavailable");
    tag_upper_bound_ = *tag_limit;
}

int MpiVecApi::tag(TransferId id, std::size_t slot, int part) const {
    if (slot >= 32 || part < 0 || part > 1) throw std::runtime_error("MPI communication output slot exceeds tag layout");
    const std::uint64_t suffix = slot * 2ULL + static_cast<std::uint64_t>(part);
    const std::uint64_t upper_bound = static_cast<std::uint64_t>(tag_upper_bound_);
    if (suffix > upper_bound || id > (upper_bound - suffix) / 64ULL)
        throw std::runtime_error("TransferId exceeds MPI_TAG_UB");
    const std::uint64_t value = id * 64ULL + suffix;
    return static_cast<int>(value);
}

MpiVecApi::WireHeader MpiVecApi::encode_header(const VecPayload &payload) {
    if (payload.metadata.context.size() >= 64) throw std::runtime_error("Vec context is too long for MPI wire header");
    WireHeader header;
    header.slot_count = payload.slots.size();
    header.degree = payload.metadata.degree;
    header.kind = static_cast<int>(payload.kind);
    header.level = payload.metadata.level;
    header.scale_log2 = payload.metadata.scale_log2;
    header.ntt = payload.metadata.ntt ? 1 : 0;
    header.components = payload.metadata.components;
    std::copy(payload.metadata.context.begin(), payload.metadata.context.end(), header.context.begin());
    return header;
}

VecValue MpiVecApi::decode_value(const WireHeader &header, std::vector<double> slots) {
    if (header.kind != static_cast<int>(ValueKind::Plaintext) && header.kind != static_cast<int>(ValueKind::Ciphertext))
        throw std::runtime_error("invalid value kind in MPI wire header");
    const auto end = std::find(header.context.begin(), header.context.end(), '\0');
    const std::string context(header.context.begin(), end);
    VecPayload payload{static_cast<ValueKind>(header.kind), std::move(slots),
                       VecMetadata{context, header.degree, header.level, header.scale_log2,
                                   header.ntt != 0, header.components}};
    return VecValue::ready(std::move(payload));
}

MpiVecApi::Value MpiVecApi::compute(const ComputeOp &op, const std::vector<Value> &inputs) {
    return executor_.compute(op, inputs);
}

MpiVecApi::Value MpiVecApi::encode_plaintext(const ValueDesc &output_desc,
                                             const std::vector<double> &slots) {
    const std::size_t slot_capacity = static_cast<std::size_t>(poly_degree_ / 2);
    if (slots.size() > slot_capacity)
        throw std::runtime_error("plaintext payload exceeds CKKS slot capacity");
    std::vector<double> padded = slots;
    padded.resize(slot_capacity, 0.0);
    return make_plain(std::move(padded), output_desc.context, poly_degree_, output_desc.level,
                      output_desc.scale_log2, output_desc.ntt);
}

MpiVecApi::CommHandle MpiVecApi::communicate_async(const CommAction &action,
                                                    const std::vector<Value> &local_inputs) {
    CommHandle handle;
    handle.id = action.id;
    for (std::size_t i = 0; i < action.destinations.size(); ++i)
        if (action.destinations[i].rank == rank_) handle.local_slots.push_back(i);

    const bool source_local = action.sources.at(0).rank == rank_;
    if (source_local && local_inputs.size() != 1) throw std::runtime_error("MPI sender requires one local input");
    if (!source_local && !local_inputs.empty()) throw std::runtime_error("MPI non-source supplied a local input");

    VecPayload source;
    if (source_local) source = local_inputs[0].materialize();
    std::size_t remote_send_count = 0;
    for (const auto &destination : action.destinations)
        if (source_local && destination.rank != rank_) ++remote_send_count;
    handle.sends.reserve(remote_send_count);
    handle.receives.reserve(handle.local_slots.size());
    handle.locals.reserve(handle.local_slots.size());

    for (std::size_t slot = 0; slot < action.destinations.size(); ++slot) {
        const int destination = action.destinations[slot].rank;
        if (source_local && destination == rank_) {
            handle.locals.push_back(LocalState{slot, VecValue::ready(source)});
        } else if (source_local) {
            handle.sends.push_back(SendState{encode_header(source), source.slots});
        } else if (destination == rank_) {
            handle.receives.push_back(RecvState{slot, action.sources[0].rank, {}, {}, MPI_REQUEST_NULL});
        }
    }

    std::size_t send_index = 0;
    for (std::size_t slot = 0; slot < action.destinations.size(); ++slot) {
        const int destination = action.destinations[slot].rank;
        if (!source_local || destination == rank_) continue;
        auto &send = handle.sends.at(send_index++);
        if (send.slots.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) throw std::runtime_error("MPI slot count exceeds int");
        check_mpi(MPI_Isend(&send.header, static_cast<int>(sizeof(WireHeader)), MPI_BYTE, destination,
                            tag(action.id, slot, 0), communicator_, &send.requests[0]), "MPI_Isend(header)");
        check_mpi(MPI_Isend(send.slots.data(), static_cast<int>(send.slots.size()), MPI_DOUBLE, destination,
                            tag(action.id, slot, 1), communicator_, &send.requests[1]), "MPI_Isend(slots)");
    }
    for (auto &recv : handle.receives) {
        check_mpi(MPI_Irecv(&recv.header, static_cast<int>(sizeof(WireHeader)), MPI_BYTE, recv.source_rank,
                            tag(action.id, recv.slot, 0), communicator_, &recv.header_request), "MPI_Irecv(header)");
    }
    return handle;
}

std::vector<MpiVecApi::Value> MpiVecApi::wait(CommHandle &handle) {
    if (handle.waited) throw std::runtime_error("MPI communication handle waited twice");
    handle.waited = true;
    for (auto &send : handle.sends)
        check_mpi(MPI_Waitall(2, send.requests.data(), MPI_STATUSES_IGNORE), "MPI_Waitall(send)");

    std::vector<LocalState> completed = std::move(handle.locals);
    for (auto &recv : handle.receives) {
        check_mpi(MPI_Wait(&recv.header_request, MPI_STATUS_IGNORE), "MPI_Wait(header)");
        if (recv.header.slot_count == 0 || recv.header.slot_count > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("invalid MPI slot count");
        recv.slots.resize(static_cast<std::size_t>(recv.header.slot_count));
        MPI_Request slots_request = MPI_REQUEST_NULL;
        check_mpi(MPI_Irecv(recv.slots.data(), static_cast<int>(recv.slots.size()), MPI_DOUBLE, recv.source_rank,
                            tag(handle.id, recv.slot, 1), communicator_, &slots_request), "MPI_Irecv(slots)");
        check_mpi(MPI_Wait(&slots_request, MPI_STATUS_IGNORE), "MPI_Wait(slots)");
        completed.push_back(LocalState{recv.slot, decode_value(recv.header, std::move(recv.slots))});
    }
    std::sort(completed.begin(), completed.end(), [](const LocalState &a, const LocalState &b) { return a.slot < b.slot; });
    std::vector<Value> outputs;
    outputs.reserve(completed.size());
    for (auto &item : completed) outputs.push_back(std::move(item.value));
    return outputs;
}

void MpiVecApi::synchronize(Value &value) { static_cast<void>(value.materialize()); }

void MpiVecApi::preflight(std::string_view plan_source_sha256,
                          bool skip_artifact_digest_checks,
                          const TargetConfig &target,
                          const OperatorSpec &operator_spec,
                          const PlanRequirements &) {
    poly_degree_ = operator_spec.poly_degree;
    if (target.capability_version != 1) throw std::runtime_error("MpiVecApi does not support target capability_version");
    const int local_skip = skip_artifact_digest_checks ? 1 : 0;
    std::vector<int> skip_values(static_cast<std::size_t>(world_size_));
    check_mpi(MPI_Allgather(&local_skip, 1, MPI_INT, skip_values.data(), 1, MPI_INT,
                            communicator_), "MPI_Allgather(skip_artifact_digest_checks)");
    for (int value : skip_values) if (value != local_skip)
        throw std::runtime_error("skip_artifact_digest_checks mismatch across MPI ranks");
    if (skip_artifact_digest_checks) return;
    if (plan_source_sha256.size() != 71) throw std::runtime_error("invalid plan source SHA-256");
    std::array<char, 72> local{};
    std::copy(plan_source_sha256.begin(), plan_source_sha256.end(), local.begin());
    std::vector<char> all(static_cast<std::size_t>(world_size_) * local.size());
    check_mpi(MPI_Allgather(local.data(), static_cast<int>(local.size()), MPI_CHAR,
                            all.data(), static_cast<int>(local.size()), MPI_CHAR,
                            communicator_), "MPI_Allgather(plan_source_sha256)");
    for (int rank = 0; rank < world_size_; ++rank) {
        const char *digest = all.data() + static_cast<std::size_t>(rank) * local.size();
        if (!std::equal(local.begin(), local.end(), digest))
            throw std::runtime_error("plan source SHA-256 mismatch across MPI ranks");
    }
}

void MpiVecApi::validate_value(const Value &value, const ValueDesc &expected) const {
    const VecMetadata metadata = value.metadata();
    if (value.kind() != expected.kind || metadata.context != expected.context ||
        metadata.degree != poly_degree_ || metadata.level != expected.level ||
        metadata.scale_log2 != expected.scale_log2 || metadata.ntt != expected.ntt ||
        metadata.components != expected.components)
        throw std::runtime_error("Api value metadata does not match ValueDesc " + std::to_string(expected.id));
}

[[noreturn]] void MpiVecApi::abort_all(int exit_code, const std::string &reason) {
    std::fprintf(stderr, "[rank %d] MPI abort: %s\n", rank_, reason.c_str());
    std::fflush(stderr);
    MPI_Abort(communicator_, exit_code);
    std::abort();
}

} // namespace fhegpu
