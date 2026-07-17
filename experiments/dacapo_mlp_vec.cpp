#include "api/mock_api.hpp"
#include "experiments/dacapo_mlp_experiment.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>

using namespace fhegpu;

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "usage: dacapo_mlp_vec_experiment PLAN OPERATOR_SPEC BUNDLE_DIR DIFF_REPORT\n";
        return 2;
    }

    try {
        const auto context = experiment::load_context(argv[1], argv[2], argv[3]);

        MockClusterConfig config;
        config.world_size = 1;
        auto cluster = std::make_shared<MockCluster>(config);
        MockVecApi api(0, cluster);
        SequentialRuntime<MockVecApi> runtime(0, 1, 0, api);
        const RuntimeResources resources{
            context.operator_spec, context.bundle_dir, false};
        const auto artifact = runtime.run(
            context.loaded_plan, resources, {{context.input_id, context.input}},
            DiffMode::AllValuesAfterRun);

        const std::filesystem::path report_path = argv[4];
        std::filesystem::create_directories(report_path.parent_path());
        std::ofstream report(report_path);
        if (!report) throw std::runtime_error("cannot open diff report");
        const auto summary =
            experiment::diff_against_reference(context, artifact, report);
        std::cout << (summary.mismatch_lines == 0 ? "PASS " : "FAIL ");
        experiment::print_summary(std::cout, summary);
        std::cout << "diff_report=" << report_path << '\n';
        return summary.mismatch_lines == 0 ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
