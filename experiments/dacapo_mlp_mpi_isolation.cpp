#include "api/mpi_api.hpp"
#include "experiments/dacapo_mlp_experiment.hpp"

#include <mpi.h>

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

using namespace fhegpu;

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int rank = 0;
    int world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (provided < MPI_THREAD_FUNNELED || argc != 5) {
        if (rank == 0)
            std::fprintf(stderr, "usage: dacapo_mlp_mpi_isolation_experiment PLAN OPERATOR_SPEC BUNDLE_DIR REPORT_DIR\n");
        MPI_Finalize();
        return 2;
    }

    int local_ok = 1;
    std::array<char, 65> local_hash{};
    try {
        const auto context = experiment::load_context(argv[1], argv[2], argv[3]);
        MPI_Comm isolated = MPI_COMM_NULL;
        if (MPI_Comm_split(MPI_COMM_WORLD, rank, 0, &isolated) != MPI_SUCCESS)
            throw std::runtime_error("MPI_Comm_split failed");
        MpiVecApi api(isolated);
        SequentialRuntime<MpiVecApi> runtime(0, 1, 0, api);
        const RuntimeResources resources{
            context.operator_spec, context.bundle_dir, false};
        const auto artifact = runtime.run(
            context.loaded_plan, resources, {{context.input_id, context.input}},
            DiffMode::AllValuesAfterRun);

        const std::filesystem::path report_dir = argv[4];
        std::filesystem::create_directories(report_dir);
        const auto report_path =
            report_dir / ("mpi.rank" + std::to_string(rank) + ".diff.txt");
        std::ofstream report(report_path);
        if (!report) throw std::runtime_error("cannot open MPI diff report");
        const auto summary =
            experiment::diff_against_reference(context, artifact, report);
        experiment::print_summary(std::cout, summary);
        std::printf("rank=%d diff_report=%s\n", rank, report_path.c_str());
        if (summary.mismatch_lines != 0) local_ok = 0;
        std::copy(summary.final_sha256.begin(), summary.final_sha256.end(),
                  local_hash.begin());
        MPI_Comm_free(&isolated);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "rank=%d FAIL %s\n", rank, error.what());
        local_ok = 0;
    }

    std::vector<char> hashes(static_cast<std::size_t>(world) * local_hash.size());
    MPI_Allgather(local_hash.data(), static_cast<int>(local_hash.size()), MPI_CHAR,
                  hashes.data(), static_cast<int>(local_hash.size()), MPI_CHAR,
                  MPI_COMM_WORLD);
    for (int other = 1; other < world; ++other)
        if (!std::equal(hashes.begin(), hashes.begin() + local_hash.size(),
                        hashes.begin() + static_cast<std::size_t>(other) * local_hash.size()))
            local_ok = 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0)
        std::printf("MPI isolated MLP (%d ranks): %s final_sha256=%s\n",
                    world, global_ok ? "PASS" : "FAIL", local_hash.data());
    MPI_Finalize();
    return global_ok ? 0 : 1;
}
