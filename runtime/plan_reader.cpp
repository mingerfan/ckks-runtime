#include "runtime/plan_reader.hpp"
#include "runtime/binary_plan_parallel.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>

namespace fhegpu {
LoadedRuntimePlan RuntimePlanReader::read_file(const std::string &path,
                                              PlanReadOptions options, PlanReadStats *stats) {
    PlanReadStats measured;
    const auto extension = std::filesystem::path(path).extension();
    if (extension == ".json") {
        auto result = RuntimePlanJsonReader::read_file(path, &measured);
        if (stats) *stats = measured;
        return result;
    }
    if (extension != ".bin") throw std::runtime_error("plan filename must end in .json or .bin: " + path);
    unsigned threads = options.threads;
    if (!threads) {
        const char *env = std::getenv("CKKS_RUNTIME_PLAN_READ_THREADS");
        threads = 4;
        if (env) {
            const std::string value(env);
            if (value.empty() || value.size() > 2 || value.find_first_not_of("0123456789") != std::string::npos)
                throw std::runtime_error("CKKS_RUNTIME_PLAN_READ_THREADS must be between 1 and 64");
            threads = static_cast<unsigned>(std::stoul(value));
        }
    }
    if (!threads || threads > 64) throw std::runtime_error("plan read threads must be between 1 and 64");
    const auto start = std::chrono::steady_clock::now();
    const auto size = std::filesystem::file_size(path);
    if (size > (8ULL << 30) || size > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("binary plan exceeds 8 GiB limit: " + path);
    std::unique_ptr<char[]> bytes(new char[static_cast<std::size_t>(size)]);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open binary plan: " + path);
    constexpr std::uint64_t block = 8 * 1024 * 1024;
    for (std::uint64_t position = 0; position < size;) {
        const auto count = std::min(block, size - position);
        input.read(bytes.get() + position, static_cast<std::streamsize>(count));
        if (!input) throw std::runtime_error("truncated binary plan: " + path);
        position += count;
    }
    if (input.peek() != std::char_traits<char>::eof() || input.bad())
        throw std::runtime_error("binary plan size changed: " + path);
    const auto decode_start = std::chrono::steady_clock::now();
    measured.binary = true;
    measured.threads = threads;
    measured.source_bytes = size;
    measured.read_seconds = std::chrono::duration<double>(decode_start - start).count();
    LoadedRuntimePlan loaded;
    loaded.plan = binary_io::read_plan_parallel({bytes.get(), static_cast<std::size_t>(size)}, threads,
                                              &measured.scan_allocate_seconds, &measured.decode_seconds);
    // Only the small canonical metadata is digested for the existing API's
    // cross-rank identity check. Neither instructions nor blob bytes are hashed.
    loaded.preflight_identity = json_utils::source_sha256(binary_io::metadata(loaded.plan).dump());
    measured.parse_build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
    if (stats) *stats = measured;
    return loaded;
}
} // namespace fhegpu
