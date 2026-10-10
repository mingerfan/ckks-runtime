#pragma once

#include "runtime/plan.hpp"

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace fhegpu {

struct LoadedPlaintextBundle {
    std::map<std::string, std::vector<double>> slots_by_content;
    std::string manifest_source_sha256;
};

struct BundleReadOptions {
    // Zero keeps a single pack file open for offset reads. A positive budget
    // loads its raw bytes once; an oversized pack is an error.
    std::uint64_t resident_byte_limit = 0;
};

class PlaintextBundleLoader {
public:
    static PlaintextBundleLoader open(
        const std::filesystem::path &directory, const PlaintextBundleRef &reference,
        const std::vector<std::string> &required_contents, std::size_t slot_capacity,
        bool skip_artifact_digest_checks, BundleReadOptions options = {});
    std::vector<double> read(const std::string &content) const;
    std::uint64_t resident_bytes() const;
    double resident_load_seconds() const;
    static LoadedPlaintextBundle load(
        const std::filesystem::path &directory,
        const PlaintextBundleRef &reference,
        const std::vector<std::string> &required_contents,
        std::size_t slot_capacity,
        bool skip_artifact_digest_checks, BundleReadOptions options = {});
private:
    struct Entry { std::uint64_t length, offset; };
    struct PackedData;
    std::filesystem::path directory_;
    std::unordered_map<std::string, Entry> entries_;
    std::shared_ptr<PackedData> packed_;
    std::size_t slot_capacity_ = 0;
    std::string manifest_digest_;
};

} // namespace fhegpu
