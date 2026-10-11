#include "experiments/binary_io_codec.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/verifier.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/resource.h>

using namespace fhegpu;
namespace expio = fhegpu::binary_io_experiment;
using Json = json_utils::Json;
using Clock = std::chrono::steady_clock;

static double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
static long rss() {
    rusage r{};
    getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss * 1024L;
}

// Independent JSON projection used only by tiny correctness tests, never timed.
static Json place_json(const Place &p) {
    Json r{{"kind", p.kind == PlaceKind::Host ? "host" : "device"}, {"rank", p.rank}};
    if (p.kind == PlaceKind::Device)
        r["index"] = p.index;
    return r;
}
static Json ids_json(const std::vector<ValueId> &ids) {
    Json r = Json::array();
    for (auto id : ids)
        r.push_back(std::to_string(id));
    return r;
}
static Json projection(const RuntimePlan &p) {
    const char *compute_names[] = {"add_cc", "add_cp", "sub_cc",  "sub_cp",     "mul_cc",      "mul_cp",
                                   "negate", "rotate", "rescale", "mod_switch", "relinearize", "boot"};
    const char *hint_names[] = {"auto", "point_to_point", "broadcast", "tree", "ring", "host_staged"};
    Json root = expio::metadata(p);
    for (const auto &v : p.values)
        root["values"].push_back({{"id", std::to_string(v.id)},
                                  {"kind", v.kind == ValueKind::Plaintext ? "plaintext" : "ciphertext"},
                                  {"place", place_json(v.place)},
                                  {"context", v.context},
                                  {"level", v.level},
                                  {"scale_log2", v.scale_log2},
                                  {"ntt", v.ntt},
                                  {"components", v.components}});
    root["external_inputs"] = ids_json(p.external_inputs);
    root["final_outputs"] = ids_json(p.final_outputs);
    const char *names[] = {"initialization", "execution", "finalization"};
    const std::vector<Instruction> *phases[] = {&p.initialization, &p.execution, &p.finalization};
    for (int n = 0; n < 3; ++n)
        for (const auto &i : *phases[n]) {
            Json r{{"ordinal", i.ordinal}};
            if (const auto *v = std::get_if<EncodeOp>(&i.body)) {
                r["kind"] = "encode";
                r["output"] = std::to_string(v->output);
                if (const auto *payload = std::get_if<InlineEncodePayload>(&v->payload))
                    r["payload"] = {{"kind", "inline"}, {"values", payload->values}};
                else
                    r["payload"] = {{"kind", "bundle"}, {"content", std::get<BundleEncodePayload>(v->payload).content}};
            } else if (const auto *v = std::get_if<ComputeOp>(&i.body)) {
                r.update({{"kind", "compute"},
                          {"op", compute_names[static_cast<std::size_t>(v->kind)]},
                          {"inputs", ids_json(v->inputs)},
                          {"output", std::to_string(v->output)},
                          {"place", place_json(v->place)}});
                if (v->reuse_input)
                    r["reuse_input"] = *v->reuse_input;
                if (const auto *a = std::get_if<RotateAttrs>(&v->attrs))
                    r["attrs"] = {{"steps", a->steps}};
                if (const auto *a = std::get_if<RescaleAttrs>(&v->attrs))
                    r["attrs"] = {{"target_level", a->target_level}, {"target_scale_log2", a->target_scale_log2}};
                if (const auto *a = std::get_if<ModSwitchAttrs>(&v->attrs))
                    r["attrs"] = {{"target_level", a->target_level}};
                if (const auto *a = std::get_if<BootAttrs>(&v->attrs))
                    r["attrs"] = {{"target_level", a->target_level},
                                  {"target_scale_log2", a->target_scale_log2},
                                  {"target_components", a->target_components},
                                  {"operator_profile", a->operator_profile},
                                  {"implementation", to_string(a->implementation)}};
            } else if (const auto *v = std::get_if<CommAction>(&i.body)) {
                Json sources = Json::array(), destinations = Json::array(), types = Json::array();
                for (const auto &p : v->sources)
                    sources.push_back(place_json(p));
                for (const auto &p : v->destinations)
                    destinations.push_back(place_json(p));
                for (auto kind : v->output_types)
                    types.push_back(kind == ValueKind::Plaintext ? "plaintext" : "ciphertext");
                r.update({{"kind", v->kind == CommKind::Transfer ? "transfer" : "replicate"},
                          {"transfer_id", std::to_string(v->id)},
                          {"hint", hint_names[static_cast<std::size_t>(v->hint)]},
                          {"inputs", ids_json(v->inputs)},
                          {"outputs", ids_json(v->outputs)},
                          {"sources", sources},
                          {"destinations", destinations},
                          {"output_kinds", types}});
            } else if (const auto *v = std::get_if<ReleaseOp>(&i.body))
                r.update({{"kind", "release"}, {"value", std::to_string(v->value)}});
            else
                r["kind"] = "fence";
            root[names[n]].push_back(std::move(r));
        }
    return root;
}

static RuntimePlan load_binary_plan(const std::string &path, JsonReadStats &stats) {
    const auto start = Clock::now();
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open binary plan");
    json_utils::HashingInputBuffer buffer(in, path, &stats);
    std::istream parser(&buffer);
    auto result = expio::read_plan(parser, std::filesystem::file_size(path));
    buffer.source_sha256();
    stats.parse_build_seconds = seconds(start) - stats.read_seconds - stats.hash_seconds;
    return result;
}
static expio::Manifest load_manifest(const std::string &path, bool binary, JsonReadStats &stats) {
    const auto start = Clock::now();
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open manifest");
    json_utils::HashingInputBuffer buffer(in, path, &stats);
    std::istream parser(&buffer);
    expio::Manifest m;
    if (binary) {
        expio::Archive<true> a(parser, std::filesystem::file_size(path));
        expio::manifest_records(a, m);
    } else
        m = expio::read_json_manifest(parser);
    buffer.source_sha256();
    stats.parse_build_seconds = seconds(start) - stats.read_seconds - stats.hash_seconds;
    return m;
}
static Json metrics(const std::string &path, const JsonReadStats &s, double elapsed) {
    return {{"file", path},
            {"bytes", std::filesystem::file_size(path)},
            {"load_seconds", elapsed},
            {"read_seconds", s.read_seconds},
            {"hash_seconds", s.hash_seconds},
            {"parse_build_seconds", s.parse_build_seconds},
            {"source_bytes", s.source_bytes},
            {"peak_rss_bytes", rss()},
            {"blob_payloads_read", 0}};
}

static void self_test(const std::filesystem::path &source, const std::filesystem::path &extra = {}) {
    int cases = 0;
    std::vector<std::filesystem::path> files;
    for (const auto &file : std::filesystem::directory_iterator(source / "docs/runtime-plan/v1/testdata/valid"))
        if (file.path().extension() == ".json")
            files.push_back(file.path());
    if (!extra.empty())
        files.push_back(extra);
    for (const auto &file : files) {
        auto loaded = RuntimePlanJsonReader::read_file(file.string());
        // Test independent projection against the source before using it to compare.
        const auto original = Json::parse(json_utils::read_file_bytes(file.string()));
        if (projection(loaded.plan) != original)
            throw std::runtime_error("projection differs from fixture: " + file.string());
        std::ostringstream out(std::ios::binary);
        expio::write_plan(out, loaded.plan);
        const auto bytes = out.str();
        std::istringstream in(bytes, std::ios::binary);
        const auto decoded = expio::read_plan(in, bytes.size());
        if (projection(decoded) != original)
            throw std::runtime_error("binary roundtrip differs");
        for (const auto &bad :
             {bytes.substr(0, bytes.size() - 1), bytes + "extra", std::string("badmagic") + bytes.substr(8)}) {
            bool rejected = false;
            try {
                std::istringstream stream(bad, std::ios::binary);
                expio::read_plan(stream, bad.size());
            } catch (const std::exception &) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("damaged binary plan accepted");
        }
        ++cases;
    }
    std::cout << Json{{"roundtrip_fixtures", cases}, {"truncation_trailing_magic_rejected", true}}.dump() << '\n';
}

int main(int argc, char **argv) {
    try {
        if (argc < 3)
            throw std::runtime_error(
                "usage: runtime_binary_io_experiment self-test SOURCE | convert-plan JSON BINARY | convert-manifest "
                "JSON BINARY | plan-json|plan-binary FILE [SPEC] | manifest-json|manifest-binary FILE");
        const std::string mode = argv[1], path = argv[2];
        if (mode == "self-test" && (argc == 3 || argc == 4)) {
            self_test(path, argc == 4 ? std::filesystem::path(argv[3]) : std::filesystem::path{});
            return 0;
        }
        JsonReadStats stats;
        const auto start = Clock::now();
        if (mode == "convert-plan" && argc == 4) {
            auto loaded = RuntimePlanJsonReader::read_file(path, &stats);
            const auto load_seconds = seconds(start);
            const auto emit_start = Clock::now();
            std::ofstream out(argv[3], std::ios::binary);
            if (!out)
                throw std::runtime_error("cannot open output");
            expio::write_plan(out, loaded.plan);
            out.close();
            if (!out)
                throw std::runtime_error("failed binary close");
            std::cout << Json{{"json_bytes", std::filesystem::file_size(path)},
                              {"binary_bytes", std::filesystem::file_size(argv[3])},
                              {"json_load_seconds", load_seconds},
                              {"binary_emit_seconds", seconds(emit_start)},
                              {"peak_rss_bytes", rss()}}
                             .dump()
                      << '\n';
        } else if (mode == "convert-manifest" && argc == 4) {
            auto m = load_manifest(path, false, stats);
            const auto load_seconds = seconds(start);
            const auto emit_start = Clock::now();
            std::ofstream out(argv[3], std::ios::binary);
            if (!out)
                throw std::runtime_error("cannot open output");
            expio::Archive<false> a(out);
            expio::manifest_records(a, m);
            out.close();
            if (!out)
                throw std::runtime_error("failed binary close");
            JsonReadStats check_stats;
            auto decoded = load_manifest(argv[3], true, check_stats);
            if (decoded.metadata != m.metadata || decoded.entries.size() != m.entries.size())
                throw std::runtime_error("manifest roundtrip mismatch");
            for (const auto &[content, e] : m.entries) {
                const auto &other = decoded.entries.at(content);
                if (other.offset != e.offset || other.length != e.length)
                    throw std::runtime_error("manifest entry mismatch");
            }
            std::cout << Json{{"json_bytes", std::filesystem::file_size(path)},
                              {"binary_bytes", std::filesystem::file_size(argv[3])},
                              {"json_load_seconds", load_seconds},
                              {"binary_emit_and_check_seconds", seconds(emit_start)},
                              {"entries_equal", true},
                              {"blob_payloads_read", 0}}
                             .dump()
                      << '\n';
        } else if ((mode == "plan-json" || mode == "plan-binary") && (argc == 3 || argc == 4)) {
            auto p = mode == "plan-json" ? RuntimePlanJsonReader::read_file(path, &stats).plan
                                         : load_binary_plan(path, stats);
            auto r = metrics(path, stats, seconds(start));
            r.update({{"mode", mode},
                      {"values", p.values.size()},
                      {"instructions", p.initialization.size() + p.execution.size() + p.finalization.size()},
                      {"typed_array_capacity_bytes",
                       p.values.capacity() * sizeof(ValueDesc) +
                           (p.initialization.capacity() + p.execution.capacity() + p.finalization.capacity()) *
                               sizeof(Instruction)}});
            if (argc == 4) {
                auto spec = OperatorSpecReader::read_file(argv[3]);
                const auto verify_start = Clock::now();
                const auto requirements = PlanVerifier::verify(p, spec);
                r.update({{"verify_seconds", seconds(verify_start)},
                          {"capabilities", requirements.capabilities.size()},
                          {"keys", requirements.keys.size()},
                          {"verify_peak_rss_bytes", rss()}});
            }
            std::cout << r.dump() << '\n';
        } else if ((mode == "manifest-json" || mode == "manifest-binary") && argc == 3) {
            auto m = load_manifest(path, mode == "manifest-binary", stats);
            auto r = metrics(path, stats, seconds(start));
            r.update({{"mode", mode}, {"entries", m.entries.size()}});
            std::cout << r.dump() << '\n';
        } else
            throw std::runtime_error("invalid experiment arguments");
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
