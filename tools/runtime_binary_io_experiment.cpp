#include "experiments/binary_io_codec.hpp"
#include "experiments/binary_io_parallel.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/plan_reader.hpp"
#include "runtime/verifier.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <sys/resource.h>
#include <tuple>

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

// Compare every typed field without constructing another multi-GB JSON DOM
// or computing a content hash. Small fixtures also check this against projection.
static bool attrs_equal(const ComputeAttrs &a, const ComputeAttrs &b) {
    if (a.index() != b.index())
        return false;
    switch (a.index()) {
    case 0: return true;
    case 1: return std::get<RotateAttrs>(a).steps == std::get<RotateAttrs>(b).steps;
    case 2: {
        const auto &x = std::get<RescaleAttrs>(a), &y = std::get<RescaleAttrs>(b);
        return std::tie(x.target_level, x.target_scale_log2) == std::tie(y.target_level, y.target_scale_log2);
    }
    case 3: return std::get<ModSwitchAttrs>(a).target_level == std::get<ModSwitchAttrs>(b).target_level;
    case 4: {
        const auto &x = std::get<BootAttrs>(a), &y = std::get<BootAttrs>(b);
        return std::tie(x.target_level, x.target_scale_log2, x.target_components, x.operator_profile, x.implementation) ==
               std::tie(y.target_level, y.target_scale_log2, y.target_components, y.operator_profile, y.implementation);
    }
    default: throw std::runtime_error("unknown comparison attrs");
    }
}
static bool instruction_equal(const Instruction &a, const Instruction &b) {
    if (a.ordinal != b.ordinal || a.body.index() != b.body.index())
        return false;
    switch (a.body.index()) {
    case 0: {
        const auto &x = std::get<EncodeOp>(a.body), &y = std::get<EncodeOp>(b.body);
        if (x.output != y.output || x.payload.index() != y.payload.index())
            return false;
        if (const auto *v = std::get_if<InlineEncodePayload>(&x.payload)) {
            const auto &other = std::get<InlineEncodePayload>(y.payload).values;
            return v->values.size() == other.size() &&
                   (other.empty() || std::memcmp(v->values.data(), other.data(), other.size() * sizeof(double)) == 0);
        }
        return std::get<BundleEncodePayload>(x.payload).content == std::get<BundleEncodePayload>(y.payload).content;
    }
    case 1: {
        const auto &x = std::get<ComputeOp>(a.body), &y = std::get<ComputeOp>(b.body);
        return std::tie(x.kind, x.inputs, x.output, x.place, x.reuse_input) ==
                   std::tie(y.kind, y.inputs, y.output, y.place, y.reuse_input) && attrs_equal(x.attrs, y.attrs);
    }
    case 2: {
        const auto &x = std::get<CommAction>(a.body), &y = std::get<CommAction>(b.body);
        return std::tie(x.id, x.kind, x.hint, x.inputs, x.outputs, x.sources, x.destinations, x.output_types) ==
               std::tie(y.id, y.kind, y.hint, y.inputs, y.outputs, y.sources, y.destinations, y.output_types);
    }
    case 3: return std::get<ReleaseOp>(a.body).value == std::get<ReleaseOp>(b.body).value;
    case 4: return true;
    default: throw std::runtime_error("unknown comparison instruction");
    }
}
static void compare_plans(const RuntimePlan &a, const RuntimePlan &b) {
    if (expio::metadata(a) != expio::metadata(b) || a.external_inputs != b.external_inputs ||
        a.final_outputs != b.final_outputs || a.values.size() != b.values.size())
        throw std::runtime_error("roundtrip metadata/inputs/outputs/value count mismatch");
    for (std::size_t i = 0; i < a.values.size(); ++i) {
        const auto &x = a.values[i], &y = b.values[i];
        if (std::tie(x.id, x.kind, x.place, x.context, x.level, x.scale_log2, x.ntt, x.components) !=
            std::tie(y.id, y.kind, y.place, y.context, y.level, y.scale_log2, y.ntt, y.components))
            throw std::runtime_error("roundtrip value mismatch at " + std::to_string(i));
    }
    const std::vector<Instruction> *left[] = {&a.initialization, &a.execution, &a.finalization};
    const std::vector<Instruction> *right[] = {&b.initialization, &b.execution, &b.finalization};
    for (int phase = 0; phase < 3; ++phase) {
        if (left[phase]->size() != right[phase]->size())
            throw std::runtime_error("roundtrip phase size mismatch");
        for (std::size_t i = 0; i < left[phase]->size(); ++i)
            if (!instruction_equal((*left[phase])[i], (*right[phase])[i]))
                throw std::runtime_error("roundtrip instruction mismatch in phase " + std::to_string(phase) +
                                         " at " + std::to_string(i));
    }
}

static RuntimePlan load_binary_plan(const std::string &path, JsonReadStats &stats) {
    const auto start = Clock::now();
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open binary plan");
    json_utils::HashingInputBuffer buffer(in, path, &stats, false);
    std::istream parser(&buffer);
    auto result = expio::read_plan(parser, std::filesystem::file_size(path));
    stats.parse_build_seconds = seconds(start) - stats.read_seconds - stats.hash_seconds;
    return result;
}
static RuntimePlan load_parallel_plan(const std::string &path, JsonReadStats &stats, unsigned threads,
                                     double &scan_seconds, double &decode_seconds) {
    const auto start = Clock::now();
    const auto size = std::filesystem::file_size(path);
    if (size > (8ULL << 30)) throw std::runtime_error("binary file exceeds 8 GiB experiment limit");
    std::unique_ptr<char[]> bytes(new char[static_cast<std::size_t>(size)]);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open binary plan");
    for (std::size_t offset = 0; offset < size;) {
        const auto count = std::min<std::size_t>(8 << 20, size - offset);
        in.read(bytes.get() + offset, static_cast<std::streamsize>(count));
        if (!in) throw std::runtime_error("truncated binary file");
        offset += count;
    }
    if (in.peek() != std::char_traits<char>::eof() || in.bad()) throw std::runtime_error("binary file size changed");
    stats.read_seconds = seconds(start);
    stats.source_bytes = size;
    auto p = expio::read_plan_parallel({bytes.get(), static_cast<std::size_t>(size)}, threads, &scan_seconds, &decode_seconds);
    stats.parse_build_seconds = seconds(start) - stats.read_seconds;
    return p;
}
static unsigned read_threads(const char *text) {
    unsigned count = 0;
    const auto *end = text + std::strlen(text);
    const auto parsed = std::from_chars(text, end, count);
    if (parsed.ec != std::errc{} || parsed.ptr != end || count == 0 || count > 64)
        throw std::runtime_error("threads must be between 1 and 64");
    return count;
}
static expio::Manifest load_manifest(const std::string &path, bool binary, JsonReadStats &stats) {
    const auto start = Clock::now();
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open manifest");
    json_utils::HashingInputBuffer buffer(in, path, &stats, false);
    std::istream parser(&buffer);
    expio::Manifest m;
    if (binary) {
        expio::Archive<true> a(parser, std::filesystem::file_size(path));
        expio::manifest_records(a, m);
    } else
        m = expio::read_json_manifest(parser);
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
        compare_plans(loaded.plan, decoded);
        for (unsigned threads : {1U, 4U})
            compare_plans(loaded.plan, expio::read_plan_parallel(bytes, threads));
        auto production_bytes = bytes;
        production_bytes.replace(0, 8, "CKKSPL01");
        compare_plans(loaded.plan, binary_io::read_plan_parallel(production_bytes, 4));
        if (projection(decoded) != original)
            throw std::runtime_error("binary roundtrip differs");
        auto changed = decoded;
        changed.plan_id ^= 1;
        bool mismatch_rejected = false;
        try { compare_plans(loaded.plan, changed); }
        catch (const std::exception &) { mismatch_rejected = true; }
        if (!mismatch_rejected)
            throw std::runtime_error("plan comparison accepted changed metadata");
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
            rejected = false;
            try { expio::read_plan_parallel(bad, 4); }
            catch (const std::exception &) { rejected = true; }
            if (!rejected) throw std::runtime_error("damaged parallel binary plan accepted");
        }
        // Corrupt a descriptor enum and the first instruction tag so errors
        // exercise both worker validation and the boundary scanner.
        expio::Archive<true> cursor(bytes);
        cursor.magic("PLNEXP01");
        std::string root;
        cursor.text(root);
        cursor.sequence(cursor.strings.values, 4, [&](std::string &s) { cursor.text(s); });
        std::uint64_t count = 0;
        cursor.number(count);
        const auto value_offset = static_cast<std::size_t>(cursor.cursor() - bytes.data());
        cursor.skip(count * 35);
        expio::skip_sequence(cursor, 8);
        cursor.number(count);
        if (!count) cursor.number(count); // empty initialization -> execution
        const auto instruction_offset = static_cast<std::size_t>(cursor.cursor() - bytes.data());
        for (auto offset : {value_offset + 8, instruction_offset + 8}) {
            auto bad = bytes;
            bad.at(offset) = static_cast<char>(255);
            bool rejected = false;
            try { expio::read_plan_parallel(bad, 4); }
            catch (const std::exception &) { rejected = true; }
            if (!rejected) throw std::runtime_error("invalid parallel enum/tag accepted");
        }
        ++cases;
    }
    std::cout << Json{{"roundtrip_fixtures", cases}, {"truncation_trailing_magic_rejected", true}}.dump() << '\n';
}

int main(int argc, char **argv) {
    try {
        if (argc < 3)
            throw std::runtime_error(
                "usage: runtime_binary_io_experiment self-test SOURCE | convert-plan JSON BINARY | "
                "convert-plan-check JSON BINARY [SPEC] | convert-manifest "
                "JSON BINARY | plan-json|plan-binary FILE [SPEC] | manifest-json|manifest-binary FILE");
        const std::string mode = argv[1], path = argv[2];
        if (mode == "self-test" && (argc == 3 || argc == 4)) {
            self_test(path, argc == 4 ? std::filesystem::path(argv[3]) : std::filesystem::path{});
            return 0;
        }
        JsonReadStats stats;
        const auto start = Clock::now();
        if ((mode == "promote-plan" || mode == "convert-production-plan") && argc == 4) {
            if (std::filesystem::exists(argv[3])) throw std::runtime_error("output already exists");
            auto p = mode == "promote-plan" ? load_binary_plan(path, stats) : RuntimePlanReader::read_file(path).plan;
            if (p.plaintext_bundle) { p.plaintext_bundle->manifest_format = "binary"; p.plaintext_bundle->manifest_sha256.clear(); }
            std::ofstream out(argv[3], std::ios::binary);
            binary_io::write_plan(out, p); out.close();
            if (!out) throw std::runtime_error("failed to close binary output");
            std::cout << Json{{"seconds", seconds(start)}, {"bytes", std::filesystem::file_size(argv[3])},
                             {"hash_seconds", 0}, {"blob_payloads_read", 0}}.dump() << '\n';
        } else if (mode == "promote-manifest" && argc == 4) {
            if (std::filesystem::exists(argv[3])) throw std::runtime_error("output already exists");
            auto manifest = load_manifest(path, false, stats);
            std::ofstream out(argv[3], std::ios::binary);
            binary_io::Archive<false> archive(out);
            binary_io::manifest_records(archive, manifest); out.close();
            if (!out) throw std::runtime_error("failed to close binary manifest");
            std::cout << Json{{"seconds", seconds(start)}, {"bytes", std::filesystem::file_size(argv[3])},
                             {"entries", manifest.entries.size()}, {"hash_seconds", 0}, {"blob_payloads_read", 0}}.dump() << '\n';
        } else if ((mode == "compare-production-plan" || mode == "compare-production-json") && argc == 4) {
            auto baseline = mode == "compare-production-plan" ? load_binary_plan(path, stats) : RuntimePlanReader::read_file(path).plan;
            auto production = RuntimePlanReader::read_file(argv[3], {4});
            if (baseline.plaintext_bundle) {
                baseline.plaintext_bundle->manifest_format = "binary";
                baseline.plaintext_bundle->manifest_sha256.clear();
            }
            compare_plans(baseline, production.plan);
            std::cout << Json{{"all_fields_equal", true}, {"bundle_reference_format_changed", true}, {"source_hashing", mode == "compare-production-json"},
                             {"values", baseline.values.size()},
                             {"instructions", baseline.initialization.size() + baseline.execution.size() + baseline.finalization.size()},
                             {"seconds", seconds(start)}, {"hash_seconds", 0}, {"blob_payloads_read", 0}, {"peak_rss_bytes", rss()}}.dump() << '\n';
        } else if (mode == "compare-binary-readers" && argc == 4) {
            auto baseline = load_binary_plan(path, stats);
            JsonReadStats parallel_stats;
            double scan = 0, decode = 0;
            auto parallel = load_parallel_plan(path, parallel_stats, read_threads(argv[3]), scan, decode);
            const auto compare_start = Clock::now();
            compare_plans(baseline, parallel);
            std::cout << Json{{"all_fields_equal", true}, {"threads", read_threads(argv[3])},
                             {"values", baseline.values.size()},
                             {"instructions", baseline.initialization.size() + baseline.execution.size() + baseline.finalization.size()},
                             {"compare_seconds", seconds(compare_start)}, {"hash_seconds", 0},
                             {"blob_payloads_read", 0}, {"peak_rss_bytes", rss()}}.dump() << '\n';
        } else if (mode == "convert-plan-check" && (argc == 4 || argc == 5)) {
            std::cerr << "loading source JSON\n";
            auto loaded = RuntimePlanJsonReader::read_file(path, &stats);
            auto baseline = metrics(path, stats, seconds(start));
            baseline.update({{"mode", "plan-json"}, {"source_sha256", loaded.source_sha256},
                             {"values", loaded.plan.values.size()},
                             {"instructions", loaded.plan.initialization.size() + loaded.plan.execution.size() +
                                                  loaded.plan.finalization.size()}});
            std::cerr << baseline.dump() << '\n';
            if (argc == 5) {
                auto spec = OperatorSpecReader::read_file(argv[4]);
                const auto verify_start = Clock::now();
                const auto requirements = PlanVerifier::verify(loaded.plan, spec);
                baseline.update({{"verify_seconds", seconds(verify_start)},
                                 {"capabilities", requirements.capabilities.size()},
                                 {"keys", requirements.keys.size()}, {"verify_peak_rss_bytes", rss()}});
                std::cerr << "source PlanVerifier passed\n";
            }
            const auto emit_start = Clock::now();
            std::ofstream out(argv[3], std::ios::binary);
            if (!out)
                throw std::runtime_error("cannot open output");
            expio::write_plan(out, loaded.plan);
            out.close();
            if (!out)
                throw std::runtime_error("failed binary close");
            const auto emit_seconds = seconds(emit_start);
            std::cerr << "binary written; decoding for field comparison\n";
            JsonReadStats binary_stats;
            const auto decode_start = Clock::now();
            auto decoded = load_binary_plan(argv[3], binary_stats);
            auto readback = metrics(argv[3], binary_stats, seconds(decode_start));
            readback["rss_includes_retained_json_plan"] = true;
            const auto compare_start = Clock::now();
            compare_plans(loaded.plan, decoded);
            std::cout << Json{{"json_baseline", baseline}, {"binary_readback", readback},
                             {"binary_emit_seconds", emit_seconds}, {"field_compare_seconds", seconds(compare_start)},
                             {"all_fields_equal", true}, {"blob_payloads_read", 0}}.dump() << '\n';
        } else if (mode == "convert-plan" && argc == 4) {
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
        } else if (((mode == "plan-json" || mode == "plan-binary") && (argc == 3 || argc == 4)) ||
                   (mode == "plan-binary-parallel" && argc == 5)) {
            const unsigned threads = mode == "plan-binary-parallel" ? read_threads(argv[4]) : 1;
            double scan = 0, decode = 0;
            auto p = mode == "plan-json" ? RuntimePlanJsonReader::read_file(path, &stats).plan
                     : mode == "plan-binary" ? load_binary_plan(path, stats)
                     : load_parallel_plan(path, stats, threads, scan, decode);
            auto r = metrics(path, stats, seconds(start));
            r.update({{"mode", mode},
                      {"values", p.values.size()},
                      {"instructions", p.initialization.size() + p.execution.size() + p.finalization.size()},
                      {"typed_array_capacity_bytes",
                       p.values.capacity() * sizeof(ValueDesc) +
                           (p.initialization.capacity() + p.execution.capacity() + p.finalization.capacity()) *
                               sizeof(Instruction)}});
            if (mode != "plan-json") r["source_hashing"] = false;
            r["threads"] = threads;
            if (mode == "plan-binary-parallel") r.update({{"scan_allocate_seconds", scan}, {"decode_seconds", decode}});
            if (argc >= 4) {
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
