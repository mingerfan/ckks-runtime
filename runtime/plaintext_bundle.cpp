#include "runtime/plaintext_bundle.hpp"
#include "runtime/json_utils.hpp"
#include "runtime/binary_manifest_codec.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <mutex>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <unordered_map>

namespace fhegpu {
namespace {

using namespace json_utils;
constexpr const char *doc = "plaintext bundle manifest";

int read_manifest_metadata(const Json &root, const PlaintextBundleRef &reference) {
    require_members(root, doc, "$", {"bundle_format_version", "bundle_id", "version", "blobs"}, {"pack_byte_length"});
    const int format = read_nonnegative_int(root.at("bundle_format_version"), doc, "$.bundle_format_version");
    if (format != 1 && format != 2) fail(doc, "$.bundle_format_version", "unsupported format version");
    if (format == 1 && root.contains("pack_byte_length")) fail(doc, "$.pack_byte_length", "requires bundle format 2");
    if (format == 2 && !root.contains("pack_byte_length")) fail(doc, "$", "missing required field 'pack_byte_length'");
    if (read_string(root.at("bundle_id"), doc, "$.bundle_id") != reference.id)
        fail(doc, "$.bundle_id", "does not match RuntimePlan reference");
    if (read_positive_int(root.at("version"), doc, "$.version") != reference.version)
        fail(doc, "$.version", "does not match RuntimePlan reference");
    if (!root.at("blobs").is_array()) fail(doc, "$.blobs", "expected array");
    return format;
}

std::vector<double> decode_slots(std::string_view bytes, const std::string &content) {
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

struct PlaintextBundleLoader::PackedData {
    std::ifstream input;
    std::mutex mutex;
    std::unique_ptr<char[]> resident;
    std::uint64_t length = 0;
    double resident_load_seconds = 0;
};

PlaintextBundleLoader PlaintextBundleLoader::open(
    const std::filesystem::path &directory, const PlaintextBundleRef &reference,
    const std::vector<std::string> &required_contents, std::size_t slot_capacity,
    bool skip_artifact_digest_checks, BundleReadOptions options) {
    const bool binary = reference.manifest_format == "binary";
    if (!binary && reference.manifest_format != "json") throw std::runtime_error("unknown manifest format");
    const auto path = (directory / (binary ? "manifest.bin" : "manifest.json")).string();
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open file: " + path);
    PlaintextBundleLoader result;
    result.directory_ = directory;
    result.slot_capacity_ = slot_capacity;
    constexpr auto no_offset = std::numeric_limits<std::uint64_t>::max();
    Json root;
    int format;
    if (binary) {
        const auto size = std::filesystem::file_size(path);
        if (size > (8ULL << 30)) throw std::runtime_error("binary manifest exceeds 8 GiB limit");
        binary_io::Archive<true> archive(input, size);
        binary_io::Manifest manifest;
        binary_io::manifest_records(archive, manifest);
        root = std::move(manifest.metadata);
        result.entries_.reserve(manifest.entries.size());
        for (auto &[content, entry] : manifest.entries)
            result.entries_.emplace(std::move(content), Entry{entry.length, entry.offset});
        format = read_manifest_metadata(root, reference);
        result.verify_blob_digest_ = false;
    } else {
        StrictJsonSax sax(path, {"blobs"}, [&](const std::string &, std::size_t index, Json &&entry) {
            try {
                require_members(entry, doc, "$", {"content", "byte_length"}, {"offset"});
                auto content = read_sha256(entry.at("content"), doc, "$.content");
                const auto length = read_safe_uint(entry.at("byte_length"), doc, "$.byte_length", 8, (1ULL << 53) - 1);
                if (length % 8 != 0) fail(doc, "$.byte_length", "must be a multiple of 8");
                const auto offset = entry.contains("offset")
                    ? read_safe_uint(entry.at("offset"), doc, "$.offset", 0, (1ULL << 53) - 1) : no_offset;
                if (!result.entries_.emplace(std::move(content), Entry{length, offset}).second)
                    fail(doc, "$.content", "duplicate content");
            } catch (const std::runtime_error &error) {
                throw std::runtime_error(path + " at " + item_path("$.blobs", index) + ": " + error.what());
            }
        });
        HashingInputBuffer buffer(input, path);
        std::istream parser_input(&buffer);
        Json::sax_parse(parser_input, &sax);
        root = sax.take_result();
        try { format = read_manifest_metadata(root, reference); }
        catch (const std::runtime_error &error) { throw std::runtime_error(path + ": " + error.what()); }
        result.manifest_digest_ = buffer.source_sha256();
        if (!skip_artifact_digest_checks && result.manifest_digest_ != reference.manifest_sha256)
            throw std::runtime_error("plaintext bundle manifest SHA-256 mismatch: " + path);
    }
    for (const auto &content : required_contents) {
        auto found = result.entries_.find(content);
        if (found == result.entries_.end())
            throw std::runtime_error("bundle content is absent from manifest: " + content);
        if (found->second.length / 8 > slot_capacity)
            throw std::runtime_error("bundle blob exceeds CKKS slot capacity: " + content);
    }
    if (format == 1) {
        if (options.resident_byte_limit) throw std::runtime_error("resident mode requires packed bundle format 2");
        for (const auto &[content, entry] : result.entries_)
            if (entry.offset != no_offset) fail(path, "$.blobs", "offset requires bundle format 2");
        return result;
    }
    const auto length = read_safe_uint(root.at("pack_byte_length"), path, "$.pack_byte_length", 0, (1ULL << 53) - 1);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    ranges.reserve(result.entries_.size());
    for (const auto &[content, entry] : result.entries_) {
        if (entry.offset == no_offset) fail(path, "$.blobs", "missing required field 'offset'");
        if (entry.offset % 8 || entry.offset > length || entry.length > length - entry.offset)
            fail(path, "$.blobs", "pack range is unaligned or out of bounds");
        ranges.emplace_back(entry.offset, entry.length);
    }
    std::sort(ranges.begin(), ranges.end());
    std::uint64_t end = 0;
    for (const auto &[offset, size] : ranges) {
        if (offset != end) fail(path, "$.blobs", "pack ranges overlap or contain gaps");
        end += size;
    }
    if (end != length) fail(path, "$.pack_byte_length", "pack ranges do not cover declared length");
    const auto data_path = directory / "data.bin";
    std::error_code error;
    const auto actual = std::filesystem::file_size(data_path, error);
    if (error) throw std::runtime_error("cannot open file: " + data_path.string() + ": " + error.message());
    if (actual != length) throw std::runtime_error("bundle pack byte length mismatch: " + data_path.string());
    auto data = std::make_shared<PackedData>();
    data->length = length;
    data->input.open(data_path, std::ios::binary);
    if (!data->input) throw std::runtime_error("cannot open file: " + data_path.string());
    if (options.resident_byte_limit) {
        if (length > options.resident_byte_limit || length > std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("bundle pack exceeds resident byte budget");
        const auto start = std::chrono::steady_clock::now();
        data->resident.reset(new char[static_cast<std::size_t>(length)]);
        constexpr std::uint64_t block = 8 * 1024 * 1024;
        for (std::uint64_t position = 0; position < length;) {
            const auto count = std::min(block, length - position);
            data->input.read(data->resident.get() + position, static_cast<std::streamsize>(count));
            if (!data->input) throw std::runtime_error("failed to read bundle pack: " + data_path.string());
            position += count;
        }
        if (data->input.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("bundle pack byte length changed: " + data_path.string());
        data->input.close();
        data->resident_load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    result.packed_ = std::move(data);
    return result;
}

std::uint64_t PlaintextBundleLoader::resident_bytes() const {
    return packed_ && packed_->resident ? packed_->length : 0;
}

double PlaintextBundleLoader::resident_load_seconds() const {
    return packed_ ? packed_->resident_load_seconds : 0;
}

std::vector<double> PlaintextBundleLoader::read(const std::string &content) const {
    auto found = entries_.find(content);
    if (found == entries_.end()) throw std::runtime_error("bundle content is absent from manifest: " + content);
    const auto &entry = found->second;
    if (entry.length / 8 > slot_capacity_) throw std::runtime_error("bundle blob exceeds CKKS slot capacity: " + content);
    if (entry.length > std::numeric_limits<std::size_t>::max() ||
        entry.length > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::runtime_error("bundle blob is too large for this platform");
    std::string storage;
    std::string_view bytes;
    if (packed_ && packed_->resident) {
        bytes = std::string_view(packed_->resident.get() + entry.offset, static_cast<std::size_t>(entry.length));
    } else {
        if (!packed_) {
            const auto path = directory_ / "data" / (content.substr(7) + ".bin");
            std::error_code error;
            const auto length = std::filesystem::file_size(path, error);
            if (error) throw std::runtime_error("cannot open file: " + path.string() + ": " + error.message());
            if (length != entry.length) throw std::runtime_error("bundle blob byte length mismatch: " + content);
        }
        storage.resize(static_cast<std::size_t>(entry.length));
        if (packed_) {
            // Copies of the loader share one open file. Seeking is serialized;
            // resident reads access immutable ranges without this lock.
            std::lock_guard<std::mutex> lock(packed_->mutex);
            packed_->input.clear();
            packed_->input.seekg(static_cast<std::streamoff>(entry.offset));
            packed_->input.read(storage.data(), static_cast<std::streamsize>(storage.size()));
            if (!packed_->input) throw std::runtime_error("failed to read bundle pack range: " + content);
        } else {
            const auto path = directory_ / "data" / (content.substr(7) + ".bin");
            std::ifstream input(path, std::ios::binary);
            if (!input) throw std::runtime_error("cannot open file: " + path.string());
            input.read(storage.data(), static_cast<std::streamsize>(storage.size()));
            if (!input || input.peek() != std::char_traits<char>::eof())
                throw std::runtime_error("bundle blob byte length mismatch: " + content);
        }
        bytes = storage;
    }
    if (verify_blob_digest_ && json_utils::source_sha256(bytes) != content)
        throw std::runtime_error("bundle blob content SHA-256 mismatch: " + content);
    return decode_slots(bytes, content);
}

LoadedPlaintextBundle PlaintextBundleLoader::load(
    const std::filesystem::path &directory, const PlaintextBundleRef &reference,
    const std::vector<std::string> &required_contents, std::size_t slot_capacity,
    bool skip_artifact_digest_checks, BundleReadOptions options) {
    auto index = open(directory, reference, required_contents, slot_capacity, skip_artifact_digest_checks, options);
    LoadedPlaintextBundle result;
    result.manifest_source_sha256 = index.manifest_digest_;
    for (const auto &content : required_contents)
        if (!result.slots_by_content.count(content)) result.slots_by_content.emplace(content, index.read(content));
    return result;
}

} // namespace fhegpu
