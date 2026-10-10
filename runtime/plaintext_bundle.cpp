#include "runtime/plaintext_bundle.hpp"
#include "runtime/json_utils.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <unordered_map>

namespace fhegpu {
namespace {

using namespace json_utils;
constexpr const char *doc = "plaintext bundle manifest";

struct BlobEntry {
    std::string content;
    std::uint64_t byte_length = 0;
};

std::vector<BlobEntry> read_manifest(const Json &root, const PlaintextBundleRef &reference) {
    require_members(root, doc, "$", {"bundle_format_version", "bundle_id", "version", "blobs"});
    if (read_nonnegative_int(root.at("bundle_format_version"), doc, "$.bundle_format_version") != 1)
        fail(doc, "$.bundle_format_version", "unsupported format version");
    if (read_string(root.at("bundle_id"), doc, "$.bundle_id") != reference.id)
        fail(doc, "$.bundle_id", "does not match RuntimePlan reference");
    if (read_positive_int(root.at("version"), doc, "$.version") != reference.version)
        fail(doc, "$.version", "does not match RuntimePlan reference");
    const auto &blobs = root.at("blobs");
    if (!blobs.is_array()) fail(doc, "$.blobs", "expected array");
    std::set<std::string> seen;
    std::vector<BlobEntry> result;
    for (std::size_t i = 0; i < blobs.size(); ++i) {
        const std::string path = item_path("$.blobs", i);
        require_members(blobs[i], doc, path, {"content", "byte_length"});
        BlobEntry entry;
        entry.content = read_sha256(blobs[i].at("content"), doc, path + ".content");
        entry.byte_length = read_safe_uint(blobs[i].at("byte_length"), doc, path + ".byte_length",
                                           8, (1ULL << 53) - 1);
        if (entry.byte_length % 8 != 0) fail(doc, path + ".byte_length", "must be a multiple of 8");
        if (!seen.insert(entry.content).second) fail(doc, path + ".content", "duplicate content");
        result.push_back(std::move(entry));
    }
    return result;
}

std::vector<double> decode_slots(const std::string &bytes, const std::string &content) {
    std::vector<double> result(bytes.size() / 8);
    for (std::size_t i = 0; i < result.size(); ++i) {
        std::uint64_t bits = 0;
        for (int byte = 0; byte < 8; ++byte)
            bits |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i * 8 + static_cast<std::size_t>(byte)]))
                    << (byte * 8);
        double value = 0.0;
        static_assert(sizeof(value) == sizeof(bits), "float64 is required");
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value)) throw std::runtime_error("bundle blob contains non-finite float64: " + content);
        result[i] = value == 0.0 ? 0.0 : value;
    }
    return result;
}

} // namespace

PlaintextBundleLoader PlaintextBundleLoader::open(
    const std::filesystem::path &directory, const PlaintextBundleRef &reference,
    const std::vector<std::string> &required_contents, std::size_t slot_capacity,
    bool skip_artifact_digest_checks) {
    const auto bytes = json_utils::read_file_bytes((directory / "manifest.json").string());
    PlaintextBundleLoader result;
    result.directory_ = directory;
    result.slot_capacity_ = slot_capacity;
    result.manifest_digest_ = json_utils::source_sha256(bytes);
    if (!skip_artifact_digest_checks && result.manifest_digest_ != reference.manifest_sha256)
        throw std::runtime_error("plaintext bundle manifest SHA-256 mismatch");
    for (const auto &entry : read_manifest(json_utils::parse(bytes, doc), reference))
        result.lengths_.emplace(entry.content, entry.byte_length);
    for (const auto &content : required_contents) {
        auto found = result.lengths_.find(content);
        if (found == result.lengths_.end())
            throw std::runtime_error("bundle content is absent from manifest: " + content);
        if (found->second / 8 > slot_capacity)
            throw std::runtime_error("bundle blob exceeds CKKS slot capacity: " + content);
    }
    return result;
}

std::vector<double> PlaintextBundleLoader::read(const std::string &content) const {
    auto found = lengths_.find(content);
    if (found == lengths_.end())
        throw std::runtime_error("bundle content is absent from manifest: " + content);
    if (found->second / 8 > slot_capacity_)
        throw std::runtime_error("bundle blob exceeds CKKS slot capacity: " + content);
    const auto path = directory_ / "data" / (content.substr(7) + ".bin");
    // Check before allocating: malformed files must not bypass the raw slot bound.
    std::error_code error;
    const auto length = std::filesystem::file_size(path, error);
    if (error) throw std::runtime_error("cannot open file: " + path.string() + ": " + error.message());
    if (length != found->second)
        throw std::runtime_error("bundle blob byte length mismatch: " + content);
    const auto bytes = json_utils::read_file_bytes(path.string());
    if (bytes.size() != found->second)
        throw std::runtime_error("bundle blob byte length mismatch: " + content);
    if (json_utils::source_sha256(bytes) != content)
        throw std::runtime_error("bundle blob content SHA-256 mismatch: " + content);
    return decode_slots(bytes, content);
}

LoadedPlaintextBundle PlaintextBundleLoader::load(
    const std::filesystem::path &directory, const PlaintextBundleRef &reference,
    const std::vector<std::string> &required_contents, std::size_t slot_capacity,
    bool skip_artifact_digest_checks) {
    auto index = open(directory, reference, required_contents, slot_capacity,
                      skip_artifact_digest_checks);
    LoadedPlaintextBundle result;
    result.manifest_source_sha256 = index.manifest_digest_;
    for (const auto &content : required_contents)
        if (!result.slots_by_content.count(content))
            result.slots_by_content.emplace(content, index.read(content));
    return result;
}

} // namespace fhegpu
