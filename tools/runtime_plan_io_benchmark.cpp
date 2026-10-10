#include "runtime/json_plan_reader.hpp"
#include "runtime/json_utils.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/verifier.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sys/resource.h>

using namespace fhegpu;
using Clock = std::chrono::steady_clock;

static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

static long peak_rss_bytes() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0)
        throw std::runtime_error("getrusage failed");
    return usage.ru_maxrss * 1024L;
}

int main(int argc, char **argv) {
    try {
        if (argc < 3 || argc > 4 ||
            (std::string(argv[1]) != "--dom" && std::string(argv[1]) != "--plan"))
            throw std::runtime_error("usage: runtime_plan_io_benchmark --dom FILE | --plan FILE [OPERATOR_SPEC]");
        json_utils::Json report{{"file", argv[2]}, {"bytes", std::filesystem::file_size(argv[2])},
                               {"instruction_size", sizeof(Instruction)}, {"value_desc_size", sizeof(ValueDesc)}};
        auto start = Clock::now();
        if (std::string(argv[1]) == "--dom") {
            auto bytes = json_utils::read_file_bytes(argv[2]);
            report["read_seconds"] = seconds(start);
            start = Clock::now();
            auto root = json_utils::parse(bytes, argv[2]);
            report["parse_seconds"] = seconds(start);
            if (root.contains("execution")) report["execution_records"] = root.at("execution").size();
            start = Clock::now();
            report["source_sha256"] = json_utils::source_sha256(bytes);
            report["hash_seconds"] = seconds(start);
            report["peak_rss_bytes"] = peak_rss_bytes();
            start = Clock::now();
            root = nullptr;
            bytes = {};
            report["destroy_seconds"] = seconds(start);
        } else {
            auto loaded = RuntimePlanJsonReader::read_file(argv[2]);
            report["load_seconds"] = seconds(start);
            report["source_sha256"] = loaded.source_sha256;
            report["values"] = loaded.plan.values.size();
            report["initialization"] = loaded.plan.initialization.size();
            report["execution"] = loaded.plan.execution.size();
            report["finalization"] = loaded.plan.finalization.size();
            report["metadata_peak_rss_bytes"] = peak_rss_bytes();
            if (argc == 4) {
                auto spec = OperatorSpecReader::read_file(argv[3]);
                start = Clock::now();
                auto requirements = PlanVerifier::verify(loaded.plan, spec);
                report["verify_seconds"] = seconds(start);
                report["capabilities"] = requirements.capabilities.size();
            }
            report["peak_rss_bytes"] = peak_rss_bytes();
            start = Clock::now();
            loaded = {};
            report["destroy_seconds"] = seconds(start);
        }
        std::cout << report.dump() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
