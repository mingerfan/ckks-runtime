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
#include <type_traits>
#include <unordered_map>
#include <vector>

using namespace fhegpu;

namespace {

using Json = nlohmann::json;

// A waterline plan (V4) runs only under the per-device executor and needs an Api
// that prepares plaintexts.  Here a Prepare is the Encode it replaced and costs
// nothing, so the ledger admits every group and only the arithmetic is checked
// -- the same arithmetic as the eager plan the schedule was made from.
class PrepareMockVecApi : public MockVecApi {
public:
    PrepareMockVecApi(std::shared_ptr<MockCluster> cluster, std::size_t devices)
        : MockVecApi(0, std::move(cluster)), devices_(devices) {}
    ResourceUsage value_resources(const ValueDesc &) const { return ResourceUsage(devices_); }
    PrepareResources prepare_resources(const ValueDesc &, PrepareBackend) const {
        return {ResourceUsage(devices_), ResourceUsage(devices_), true};
    }
    ResourceUsage instruction_workspace(const Instruction &) const { return ResourceUsage(devices_); }
    Value prepare_plaintext_async(const ValueDesc &desc, const std::vector<double> &slots,
                                  PrepareBackend) {
        return encode_plaintext(desc, slots);
    }
    bool is_ready(Value &value) const { return value.is_ready(); }

private:
    std::size_t devices_;
};

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

std::vector<double> numbers_of(const Json &array, const std::string &what) {
    if (!array.is_array()) throw std::runtime_error(what + " is not an array");
    std::vector<double> result;
    result.reserve(array.size());
    for (const auto &value : array) {
        if (!value.is_number()) throw std::runtime_error(what + " contains a non-number");
        const double number = value.get<double>();
        if (!std::isfinite(number)) throw std::runtime_error(what + " contains a non-finite number");
        result.push_back(number);
    }
    return result;
}

// A fixture of whole slot vectors (model "qwen"): inputs[i] holds the packed
// slots plan.external_inputs[i] encrypts, expected[i] the valid prefix of
// plan.final_outputs[i].  Every slot past that prefix must stay zero.  The
// reference is the model's exact function, so the bound here covers the
// polynomial approximations only (MockVec has no CKKS noise and an ideal Boot).
int run_slot_fixture(const Json &fixture, char **argv) {
    experiment::Context context;
    context.loaded_plan = RuntimePlanReader::read_file(argv[1]);
    context.operator_spec = OperatorSpecReader::read_file(argv[2]);
    context.bundle_dir = argv[3];
    context.slot_capacity =
        static_cast<std::size_t>(context.operator_spec.spec.poly_degree / 2);
    const auto &plan = context.loaded_plan.plan;
    const auto &input_rows = fixture.at("inputs");
    const auto &expected_rows = fixture.at("expected");
    if (plan.external_inputs.size() != input_rows.size() ||
        plan.final_outputs.size() != expected_rows.size())
        throw std::runtime_error("fixture inputs/expected do not match the plan's inputs/outputs");
    const double tolerance = fixture.at("tolerance").at("mock_absolute").get<double>();

    std::unordered_map<ValueId, VecValue> inputs;
    for (std::size_t i = 0; i < plan.external_inputs.size(); ++i) {
        auto slots = numbers_of(input_rows.at(i), "inputs[" + std::to_string(i) + "]");
        if (slots.size() != context.slot_capacity)
            throw std::runtime_error("inputs[" + std::to_string(i) + "] is not one full slot vector");
        const ValueId id = plan.external_inputs[i];
        const auto &desc = experiment::find_value(plan, id);
        inputs.emplace(id, make_cipher(std::move(slots), desc.context,
                                       context.operator_spec.spec.poly_degree, desc.level,
                                       desc.scale_log2, desc.ntt, desc.components));
    }

    MockClusterConfig config;
    config.world_size = 1;
    auto cluster = std::make_shared<MockCluster>(config);
    const int devices = plan.target.device_counts.at(0);
    const RuntimeResources resources{context.operator_spec, context.bundle_dir, false};
    const auto run = [&](auto &api, DeviceExecutionMode mode) {
        SequentialRuntime<std::decay_t<decltype(api)>> runtime(0, 1, devices, api, mode);
        return runtime.run(context.loaded_plan, resources, inputs, DiffMode::FinalOnly);
    };
    const auto artifact = [&] {
        if (plan.waterline) {
            PrepareMockVecApi api(cluster, static_cast<std::size_t>(devices));
            return run(api, DeviceExecutionMode::PerDeviceWorkers);
        }
        MockVecApi api(0, cluster);
        return run(api, DeviceExecutionMode::Sequential);
    }();

    std::ofstream report(argv[6]);
    if (!report) throw std::runtime_error("cannot open diff report");
    Json outputs = Json::array(), rows = Json::array();
    double max_error = 0.0, max_tail = 0.0;
    bool finite = true;
    for (std::size_t i = 0; i < plan.final_outputs.size(); ++i) {
        const ValueId id = plan.final_outputs[i];
        const auto expected = numbers_of(expected_rows.at(i), "expected[" + std::to_string(i) + "]");
        const auto slots = artifact.values.at(id).value.materialize().slots;
        if (expected.empty() || expected.size() > slots.size())
            throw std::runtime_error("expected[" + std::to_string(i) + "] does not fit its output");
        double error = 0.0, tail = 0.0;
        for (std::size_t s = 0; s < slots.size(); ++s) {
            finite = finite && std::isfinite(slots[s]);
            if (s < expected.size()) error = std::max(error, std::abs(slots[s] - expected[s]));
            else tail = std::max(tail, std::abs(slots[s]));
        }
        max_error = std::max(max_error, error);
        max_tail = std::max(max_tail, tail);
        const std::string name = fixture.contains("output_names")
            ? fixture.at("output_names").at(i).get<std::string>() : std::to_string(i);
        rows.push_back({{"name", name}, {"value_id", id}, {"valid", expected.size()},
                        {"max_abs_error", error}, {"max_abs_tail", tail}});
        outputs.push_back(slots);
        report << "output=" << id << " name=" << name << " max_abs_error=" << error
               << " max_abs_tail=" << tail << '\n';
    }
    const bool passed = finite && max_error <= tolerance && max_tail <= tolerance;
    write_json(argv[5], Json{
        {"format_version", 1},
        {"passed", passed},
        {"model", fixture.at("model")},
        {"input_encoding", fixture.value("input_encoding", "frontend-packed-slots")},
        {"require_python_match", true},
        {"seed", fixture.at("seed")},
        {"model_sha256", fixture.at("model_sha256")},
        {"plan_sha256", context.loaded_plan.source_sha256},
        {"python_max_abs_diff", max_error},
        {"python_max_abs_tail", max_tail},
        {"python_tolerance", tolerance},
        {"python_within_tolerance", max_error <= tolerance && max_tail <= tolerance},
        {"output_finite", finite},
        {"per_output", rows},
        {"outputs", outputs},
    });
    std::cout << (passed ? "PASS" : "FAIL") << " outputs=" << outputs.size()
              << " python_max_abs_diff=" << max_error << " python_max_abs_tail=" << max_tail
              << '\n' << "result_json=" << argv[5] << '\n';
    return passed ? 0 : 1;
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
        if (model == "qwen") return run_slot_fixture(fixture, argv);
        const std::size_t input_size = model == "resnet20" ? 3 * 32 * 32 : 784;
        if (model != "mlp" && model != "resnet20")
            throw std::runtime_error("unsupported model fixture");
        const std::vector<double> input =
            read_numbers(fixture, "input", input_size);
        const std::vector<double> python_output =
            read_numbers(fixture, "python_output", 10);

        auto context = experiment::load_context(argv[1], argv[2], argv[3]);
        const bool frontend_packed =
            fixture.value("input_encoding", "legacy") == "frontend-packed-slots";
        experiment::set_input(
            context, frontend_packed
                         ? read_numbers(fixture, "packed_input", context.slot_capacity)
                         : model == "resnet20"
                         ? experiment::pack_resnet20_input(input)
                         : experiment::pack_mlp_input(input));

        MockClusterConfig config;
        config.world_size = 1;
        auto cluster = std::make_shared<MockCluster>(config);
        const int devices = context.loaded_plan.plan.target.device_counts.at(0);
        const RuntimeResources resources{
            context.operator_spec, context.bundle_dir, false};
        const DiffMode diff_mode = model == "resnet20"
            ? DiffMode::FinalOnly : DiffMode::AllValuesAfterRun;
        const auto run = [&](auto &api, DeviceExecutionMode mode) {
            SequentialRuntime<std::decay_t<decltype(api)>> runtime(0, 1, devices, api, mode);
            return runtime.run(
                context.loaded_plan, resources, {{context.input_id, context.input}},
                diff_mode);
        };
        const auto artifact = [&] {
            if (context.loaded_plan.plan.waterline) {
                PrepareMockVecApi api(cluster, static_cast<std::size_t>(devices));
                return run(api, DeviceExecutionMode::PerDeviceWorkers);
            }
            MockVecApi api(0, cluster);
            return run(api, DeviceExecutionMode::Sequential);
        }();

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

        const double python_tolerance = model == "resnet20" && !frontend_packed ? 0.1 : 1e-9;
        const bool python_within_tolerance =
            python_max_abs <= python_tolerance;
        const bool output_finite = std::all_of(
            summary.final_prefix.begin(), summary.final_prefix.end(),
            [](double value) { return std::isfinite(value); });
        const bool passed = output_finite && summary.mismatch_lines == 0 &&
                            ((model == "resnet20" && !frontend_packed) || python_within_tolerance);
        Json result{
            {"format_version", 1},
            {"passed", passed},
            {"model", model},
            {"input_encoding", fixture.value("input_encoding", "legacy")},
            {"require_python_match", frontend_packed || model != "resnet20"},
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
