#pragma once

#include "runtime/plan.hpp"

#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>

namespace fhegpu {

struct LoadedPlaintextBundle {
    std::map<std::string, std::vector<double>> slots_by_content;
    std::string manifest_source_sha256;
};

class PlaintextBundleLoader {
public:
    static PlaintextBundleLoader open(
        const std::filesystem::path &directory, const PlaintextBundleRef &reference,
        const std::vector<std::string> &required_contents, std::size_t slot_capacity,
        bool skip_artifact_digest_checks);
    std::vector<double> read(const std::string &content) const;
    static LoadedPlaintextBundle load(
        const std::filesystem::path &directory,
        const PlaintextBundleRef &reference,
        const std::vector<std::string> &required_contents,
        std::size_t slot_capacity,
        bool skip_artifact_digest_checks);
private:
    std::filesystem::path directory_;
    std::unordered_map<std::string, std::uint64_t> lengths_;
    std::size_t slot_capacity_ = 0;
    std::string manifest_digest_;
};

} // namespace fhegpu
