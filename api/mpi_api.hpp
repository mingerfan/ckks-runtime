#pragma once

#include "api/vec_api.hpp"
#include "runtime/operator_spec.hpp"

#include <mpi.h>

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace fhegpu {

class MpiVecApi {
public:
    using Value = VecValue;

    struct WireHeader {
        std::uint64_t slot_count = 0;
        std::uint64_t degree = 0;
        int kind = 0;
        int level = 0;
        int scale_log2 = 0;
        int ntt = 0;
        int components = 1;
        std::array<char, 64> context{};
    };
    struct SendState {
        WireHeader header;
        std::vector<double> slots;
        std::array<MPI_Request, 2> requests{MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    };
    struct RecvState {
        std::size_t slot = 0;
        int source_rank = 0;
        WireHeader header;
        std::vector<double> slots;
        MPI_Request header_request = MPI_REQUEST_NULL;
    };
    struct LocalState { std::size_t slot = 0; Value value; };
    struct CommHandle {
        TransferId id = 0;
        std::vector<std::size_t> local_slots;
        std::vector<SendState> sends;
        std::vector<RecvState> receives;
        std::vector<LocalState> locals;
        bool waited = false;
    };

    explicit MpiVecApi(MPI_Comm communicator = MPI_COMM_WORLD);
    std::string name() const { return "MpiVecApi"; }
    Value encode_plaintext(const ValueDesc &output_desc, const std::vector<double> &slots);
    Value compute(const ComputeOp &op, const std::vector<Value> &inputs);
    CommHandle communicate_async(const CommAction &action, const std::vector<Value> &local_inputs);
    std::vector<Value> wait(CommHandle &handle);
    void synchronize(Value &value);
    void preflight(std::string_view plan_source_sha256,
                   bool skip_artifact_digest_checks,
                   const TargetConfig &target,
                   const OperatorSpec &operator_spec,
                   const PlanRequirements &requirements);
    [[noreturn]] void abort_all(int exit_code, const std::string &reason);
    void validate_value(const Value &value, const ValueDesc &expected) const;

private:
    int tag(TransferId id, std::size_t slot, int part) const;
    static WireHeader encode_header(const VecPayload &payload);
    static VecValue decode_value(const WireHeader &header, std::vector<double> slots);
    static void check_mpi(int code, const char *operation);

    MPI_Comm communicator_;
    int rank_ = 0;
    int world_size_ = 0;
    int tag_upper_bound_ = 0;
    VecExecutor executor_;
    std::uint64_t poly_degree_ = 0;
};

} // namespace fhegpu
