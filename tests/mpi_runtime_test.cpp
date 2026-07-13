#include "api/mpi_api.hpp"
#include "runtime/runtime.hpp"
#include "testing/testing.hpp"

#include <mpi.h>

#include <cstdio>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

using namespace fhegpu;

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int rank = 0;
    int world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (provided < MPI_THREAD_FUNNELED || (world != 2 && world != 4)) {
        if (rank == 0) std::fprintf(stderr, "mpi_runtime_test requires MPI thread funneled and -n 2 or -n 4\n");
        MPI_Finalize();
        return 2;
    }

    const bool inject_error = argc == 2 && std::strcmp(argv[1], "--inject-error") == 0;
    const bool inject_large_transfer_id = argc == 2 && std::strcmp(argv[1], "--inject-large-transfer-id") == 0;
    auto built = make_fanout_plan(std::vector<int>(static_cast<std::size_t>(world), 1));
    if (inject_large_transfer_id)
        std::get<CommAction>(built.plan.initialization.at(1).body).id = std::numeric_limits<TransferId>::max();
    const VecValue cipher = make_cipher({1, 2, 3, 4}, "mpi-ctx", 16384, 3, 2.0);
    const VecValue plain = make_plain({2, 3, 4, 5}, "mpi-ctx", 16384, 3, 2.0);
    const auto reference = run_fanout_reference(cipher, plain);

    MpiVecApi api;
    SequentialRuntime<MpiVecApi> runtime(rank, world, 1, api);
    std::unordered_map<ValueId, VecValue> inputs;
    if (rank == 0) {
        inputs.emplace(0, inject_error ? make_plain({1, 2, 3, 4}, "mpi-ctx", 16384, 3, 2.0) : cipher.deep_copy());
        inputs.emplace(1, plain.deep_copy());
    }
    const auto artifact = runtime.run(built.plan, inputs, DiffMode::FinalOnly);

    int local_ok = 1;
    if (rank == world - 1) {
        try {
            compare_values(artifact.values.at(built.plan.final_outputs.front()).value,
                           reference.at(built.reference_output));
        } catch (const std::exception &error) {
            std::fprintf(stderr, "[rank %d] result mismatch: %s\n", rank, error.what());
            local_ok = 0;
        }
    } else if (!artifact.values.empty()) {
        local_ok = 0;
    }
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) std::printf("MPI runtime end-to-end (%d ranks): %s\n", world, global_ok ? "PASS" : "FAIL");
    MPI_Finalize();
    return global_ok ? 0 : 1;
}
