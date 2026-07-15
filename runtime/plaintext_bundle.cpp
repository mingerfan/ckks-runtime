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

LoadedPlaintextBundle PlaintextBundleLoader::load(
    const std::filesystem::path &directory,
    const PlaintextBundleRef &reference,
    const std::vector<std::string> &required_contents,
    std::size_t slot_capacity,
    bool skip_artifact_digest_checks) {
    const std::string manifest_bytes = json_utils::read_file_bytes((directory / "manifest.json").string());
    const std::string manifest_digest = json_utils::source_sha256(manifest_bytes);
    if (!skip_artifact_digest_checks && manifest_digest != reference.manifest_sha256)
        throw std::runtime_error("plaintext bundle manifest SHA-256 mismatch");
    const auto entries = read_manifest(json_utils::parse(manifest_bytes, doc), reference);
    std::unordered_map<std::string, std::uint64_t> lengths;
    for (const auto &entry : entries) lengths.emplace(entry.content, entry.byte_length);

    std::set<std::string> unique_required(required_contents.begin(), required_contents.end());
    LoadedPlaintextBundle result;
    result.manifest_source_sha256 = manifest_digest;
    for (const std::string &content : unique_required) {
        const auto length_it = lengths.find(content);
        if (length_it == lengths.end()) throw std::runtime_error("bundle content is absent from manifest: " + content);
        const std::string filename = content.substr(7) + ".bin";
        const std::string bytes = json_utils::read_file_bytes((directory / "data" / filename).string());
        if (bytes.size() != length_it->second) throw std::runtime_error("bundle blob byte length mismatch: " + content);
        if (json_utils::source_sha256(bytes) != content) throw std::runtime_error("bundle blob content SHA-256 mismatch: " + content);
        auto slots = decode_slots(bytes, content);
        if (slots.size() > slot_capacity) throw std::runtime_error("bundle blob exceeds CKKS slot capacity: " + content);
        result.slots_by_content.emplace(content, std::move(slots));
    }
    return result;
}

} // namespace fhegpu
