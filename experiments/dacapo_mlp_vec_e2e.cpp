#include "api/mock_api.hpp"
#include "experiments/dacapo_mlp_experiment.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace fhegpu;

namespace {

using Json = nlohmann::json;

Json read_json(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open JSON file: " + path.string());
    Json value;
    input >> value;
    return value;
}

std::vector<double> read_numbers(const Json &root, const char *member,
                                 std::size_t expected_size) {
    const auto &array = root.at(member);
    if (!array.is_array() || array.size() != expected_size)
        throw std::runtime_error(std::string(member) + " has the wrong length");
    std::vector<double> result;
    result.reserve(array.size());
    for (const auto &value : array) {
        if (!value.is_number())
            throw std::runtime_error(std::string(member) + " contains a non-number");
        const double number = value.get<double>();
        if (!std::isfinite(number))
            throw std::runtime_error(std::string(member) + " contains a non-finite number");
        result.push_back(number);
    }
    return result;
}

double max_abs_difference(const std::vector<double> &expected,
                          const std::vector<double> &actual) {
    if (expected.size() != actual.size())
        throw std::runtime_error("cannot compare outputs with different lengths");
    double result = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i)
        result = std::max(result, std::abs(expected[i] - actual[i]));
    return result;
}

void write_json(const std::filesystem::path &path, const Json &value) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write JSON file: " + path.string());
    output << value.dump(2) << '\n';
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 7) {
        std::cerr << "usage: dacapo_mlp_vec_e2e PLAN OPERATOR_SPEC BUNDLE_DIR "
                     "FIXTURE RESULT_JSON DIFF_REPORT\n";
        return 2;
    }

    try {
        const Json fixture = read_json(argv[4]);
        if (fixture.at("format_version") != 1)
            throw std::runtime_error("unsupported model fixture format_version");
        const std::string model = fixture.value("model", "mlp");
        const std::size_t input_size = model == "resnet20" ? 3 * 32 * 32 : 784;
        if (model != "mlp" && model != "resnet20")
            throw std::runtime_error("unsupported model fixture");
        const std::vector<double> input =
            read_numbers(fixture, "input", input_size);
        const std::vector<double> python_output =
            read_numbers(fixture, "python_output", 10);

        auto context = experiment::load_context(argv[1], argv[2], argv[3]);
        experiment::set_input(
            context, model == "resnet20"
                         ? experiment::pack_resnet20_input(input)
                         : experiment::pack_mlp_input(input));

        MockClusterConfig config;
        config.world_size = 1;
        auto cluster = std::make_shared<MockCluster>(config);
        MockVecApi api(0, cluster);
        SequentialRuntime<MockVecApi> runtime(
            0, 1, context.loaded_plan.plan.target.device_counts.at(0), api);
        const RuntimeResources resources{
            context.operator_spec, context.bundle_dir, false};
        const DiffMode diff_mode = model == "resnet20"
            ? DiffMode::FinalOnly : DiffMode::AllValuesAfterRun;
        const auto artifact = runtime.run(
            context.loaded_plan, resources, {{context.input_id, context.input}},
            diff_mode);

        const std::filesystem::path report_path = argv[6];
        if (!report_path.parent_path().empty())
            std::filesystem::create_directories(report_path.parent_path());
        std::ofstream report(report_path);
        if (!report) throw std::runtime_error("cannot open diff report");
        experiment::DiffSummary summary;
        if (model == "resnet20") {
            const ValueId output_id =
                context.loaded_plan.plan.final_outputs.at(0);
            const VecPayload output =
                artifact.values.at(output_id).value.materialize();
            summary.compared_lines = 1;
            summary.final_sha256 = experiment::slots_sha256(output.slots);
            summary.final_prefix.assign(output.slots.begin(),
                                        output.slots.begin() + 10);
            report << "source=runtime_final output=" << output_id << '\n';
        } else {
            summary = experiment::diff_against_reference(
                context, artifact, report);
        }
        const double python_max_abs = max_abs_difference(
            python_output, summary.final_prefix);
        report << "source=python_final max_abs_diff=" << python_max_abs << '\n';

        const double python_tolerance = model == "resnet20" ? 0.1 : 1e-9;
        const bool python_within_tolerance =
            python_max_abs <= python_tolerance;
        const bool output_finite = std::all_of(
            summary.final_prefix.begin(), summary.final_prefix.end(),
            [](double value) { return std::isfinite(value); });
        const bool passed = output_finite && summary.mismatch_lines == 0 &&
                            (model == "resnet20" || python_within_tolerance);
        Json result{
            {"format_version", 1},
            {"passed", passed},
            {"model", model},
            {"seed", fixture.at("seed")},
            {"model_sha256", fixture.at("model_sha256")},
            {"plan_sha256", context.loaded_plan.source_sha256},
            {"compared_lines", summary.compared_lines},
            {"instruction_mismatches", summary.mismatch_lines},
            {"python_max_abs_diff", python_max_abs},
            {"python_tolerance", python_tolerance},
            {"python_within_tolerance", python_within_tolerance},
            {"output_finite", output_finite},
            {"output", summary.final_prefix},
            {"final_slots_sha256", summary.final_sha256},
        };
        write_json(argv[5], result);

        std::cout << (passed ? "PASS" : "FAIL")
                  << " compared_lines=" << summary.compared_lines
                  << " instruction_mismatches=" << summary.mismatch_lines
                  << " python_max_abs_diff=" << python_max_abs << '\n'
                  << "result_json=" << argv[5] << '\n'
                  << "diff_report=" << report_path << '\n';
        return passed ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
