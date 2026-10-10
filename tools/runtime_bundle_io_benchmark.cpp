#include "runtime/plaintext_bundle.hpp"
#include "runtime/json_utils.hpp"
#include <chrono>
#include <iostream>
#include <sys/resource.h>

int main(int argc, char **argv) {
    try {
        if (argc != 6) throw std::runtime_error("usage: runtime_bundle_io_benchmark BUNDLE ID VERSION MANIFEST_SHA256 RESIDENT_BYTE_LIMIT");
        fhegpu::PlaintextBundleRef ref{argv[2], std::stoi(argv[3]), argv[4]};
        const auto start = std::chrono::steady_clock::now();
        auto bundle = fhegpu::PlaintextBundleLoader::open(argv[1], ref, {}, 0, false, {std::stoull(argv[5])});
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        std::cout << fhegpu::json_utils::Json{
            {"open_seconds", seconds}, {"resident_bytes", bundle.resident_bytes()},
            {"resident_load_seconds", bundle.resident_load_seconds()},
            {"metadata_seconds", seconds - bundle.resident_load_seconds()},
            {"peak_rss_bytes", usage.ru_maxrss * 1024L},
            {"blob_hashes_verified", 0}, {"blob_payloads_decoded", 0}}
            .dump() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
