#include "runtime/json_plan_reader.hpp"
#include "runtime/json_utils.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/plaintext_bundle.hpp"
#include "runtime/verifier.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
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

// Compare complete records without keeping either large document in memory.
// Bundle references have a new source digest after compact serialization;
// callers must separately compare the manifests and verify their source hashes.
static json_utils::Json record_summary(const std::string &path, bool plan) {
    using namespace json_utils;
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open file: " + path);
    std::map<std::string, Sha256> hashes;
    std::map<std::string, std::size_t> counts;
    StrictJsonSax sax(path, {"values", "external_inputs", "initialization", "execution", "finalization", "final_outputs", "blobs"},
        [&](const std::string &field, std::size_t, Json &&record) {
            hashes[field].update(record.dump());
            hashes[field].update("\n");
            ++counts[field];
        });
    HashingInputBuffer buffer(input, path);
    std::istream stream(&buffer);
    Json::sax_parse(stream, &sax);
    auto result = sax.take_result();
    for (const char *field : {"values", "external_inputs", "initialization", "execution", "finalization", "final_outputs", "blobs"})
        if (result.contains(field) && result.at(field).is_array())
            result[field] = {{"records", counts[field]}, {"records_sha256", "sha256:" + hashes[field].hex_digest()}};
    if (plan && result.contains("plaintext_bundle")) result["plaintext_bundle"].erase("manifest_sha256");
    return result;
}

int main(int argc, char **argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (argc == 4 && (mode == "--compare-plans" || mode == "--compare-manifests")) {
            const auto start = Clock::now();
            const auto left = record_summary(argv[2], mode == "--compare-plans");
            const auto right = record_summary(argv[3], mode == "--compare-plans");
            if (left != right) throw std::runtime_error("JSON record summaries differ");
            std::cout << json_utils::Json{{"equal", true}, {"mode", mode}, {"seconds", seconds(start)},
                                         {"peak_rss_bytes", peak_rss_bytes()}, {"summary", left}}.dump() << '\n';
            return 0;
        }
        if (argc < 3 || argc > 5 || (mode != "--dom" && mode != "--plan"))
            throw std::runtime_error("usage: runtime_plan_io_benchmark --dom FILE | --plan FILE [OPERATOR_SPEC [BUNDLE_DIR]] | --compare-plans OLD NEW | --compare-manifests OLD NEW");
        json_utils::Json report{{"file", argv[2]}, {"bytes", std::filesystem::file_size(argv[2])},
                               {"instruction_size", sizeof(Instruction)}, {"value_desc_size", sizeof(ValueDesc)}};
        auto start = Clock::now();
        if (mode == "--dom") {
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
            JsonReadStats stats;
            auto loaded = RuntimePlanJsonReader::read_file(argv[2], &stats);
            report["load_seconds"] = seconds(start);
            report["read_seconds"] = stats.read_seconds;
            report["hash_seconds"] = stats.hash_seconds;
            report["parse_build_seconds"] = stats.parse_build_seconds;
            report["source_bytes"] = stats.source_bytes;
            report["source_sha256"] = loaded.source_sha256;
            report["values"] = loaded.plan.values.size();
            report["initialization"] = loaded.plan.initialization.size();
            report["execution"] = loaded.plan.execution.size();
            report["finalization"] = loaded.plan.finalization.size();
            report["typed_array_bytes"] = loaded.plan.values.size() * sizeof(ValueDesc) +
                (loaded.plan.initialization.size() + loaded.plan.execution.size() + loaded.plan.finalization.size()) * sizeof(Instruction);
            report["typed_array_capacity_bytes"] = loaded.plan.values.capacity() * sizeof(ValueDesc) +
                (loaded.plan.initialization.capacity() + loaded.plan.execution.capacity() + loaded.plan.finalization.capacity()) * sizeof(Instruction);
            report["metadata_peak_rss_bytes"] = peak_rss_bytes();
            if (argc >= 4) {
                auto spec = OperatorSpecReader::read_file(argv[3]);
                start = Clock::now();
                auto requirements = PlanVerifier::verify(loaded.plan, spec);
                report["verify_seconds"] = seconds(start);
                report["capabilities"] = requirements.capabilities.size();
                report["verify_peak_rss_bytes"] = peak_rss_bytes();
                if (argc == 5) {
                    if (!loaded.plan.plaintext_bundle) throw std::runtime_error("plan has no plaintext bundle reference");
                    start = Clock::now();
                    auto bundle = PlaintextBundleLoader::open(argv[4], *loaded.plan.plaintext_bundle, {}, spec.spec.poly_degree / 2, false);
                    report["bundle_index_seconds"] = seconds(start);
                    report["bundle_index_peak_rss_bytes"] = peak_rss_bytes();
                }
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
