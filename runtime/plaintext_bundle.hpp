#pragma once

#include "runtime/plan.hpp"

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace fhegpu {

struct LoadedPlaintextBundle {
    std::map<std::string, std::vector<double>> slots_by_content;
    std::string manifest_source_sha256;
};

class PlaintextBundleLoader {
public:
    static LoadedPlaintextBundle load(
        const std::filesystem::path &directory,
        const PlaintextBundleRef &reference,
        const std::vector<std::string> &required_contents,
        std::size_t slot_capacity,
        bool skip_artifact_digest_checks);
};

} // namespace fhegpu
