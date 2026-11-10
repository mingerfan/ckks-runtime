#include "runtime/binary_plan_parallel.hpp"
#include "runtime/plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/runtime.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>

using namespace fhegpu;

namespace {
void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
template <class Function>
void expect_error(Function function, const std::string &expected) {
    try { function(); }
    catch (const std::exception &error) {
        require(std::string(error.what()).find(expected) != std::string::npos,
                "wrong error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected error: " + expected);
}
Place device() { return {PlaceKind::Device, 0, 0}; }
LoadedOperatorSpec operator_spec() {
    LoadedOperatorSpec loaded;
    auto &spec = loaded.spec;
    spec.id = "prepared-api";
    spec.version = 1;
    spec.target_id = "prepared-device";
    spec.context_id = "prepared-context";
    spec.poly_degree = 8;
    spec.rns_moduli_log2 = {40, 40, 40, 40};
    spec.level_lower_bound = 0;
    spec.level_upper_bound = 3;
    loaded.source_sha256 = "sha256:" + std::string(64, '0');
    // No arithmetic support entries or ordinary Boot profiles are needed.
    return loaded;
}
RuntimePlan fixture() {
    RuntimePlan plan;
    plan.format_version = 7;
    plan.plan_id = 7;
    plan.target = {"prepared-device", 1, {1}, 1,
                   {"prepared-api", 1, operator_spec().source_sha256}};
    plan.values = {
        {0, ValueKind::Ciphertext, device(), "prepared-context", 1, 20.25, true, 2},
        {1, ValueKind::Ciphertext, device(), "prepared-context", 3, 22.5, true, 2},
        {2, ValueKind::Ciphertext, device(), "prepared-context", 2, 13.75, true, 2},
        {3, ValueKind::Ciphertext, device(), "prepared-context", 0, 8.25, true, 2}};
    plan.execution = {
        {0, ComputeOp{ComputeKind::ApiCall, {}, 0, device(), ApiCallAttrs{"stem", 0}}},
        {1, ComputeOp{ComputeKind::ApiCall, {0}, 1, device(), ApiCallAttrs{"bootstrap", 1}}},
        {2, ComputeOp{ComputeKind::ApiCall, {0, 1}, 2, device(), ApiCallAttrs{"residual_add", 2}}},
        {3, ReleaseOp{0}},
        {4, ReleaseOp{1}},
        {5, ComputeOp{ComputeKind::ApiCall, {2}, 3, device(), ApiCallAttrs{"head", 3}}},
        {6, ReleaseOp{2}}};
    plan.final_outputs = {3};
    return plan;
}
json_utils::Json json_fixture(const RuntimePlan &plan) {
    using Json = json_utils::Json;
    auto root = binary_io::metadata(plan);
    const Json place{{"kind", "device"}, {"rank", 0}, {"index", 0}};
    for (const auto &value : plan.values)
        root["values"].push_back({{"id", std::to_string(value.id)}, {"kind", "ciphertext"},
            {"place", place}, {"context", value.context}, {"level", value.level},
            {"scale_log2", value.scale_log2}, {"ntt", value.ntt}, {"components", value.components}});
    for (const auto &instruction : plan.execution) {
        if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) {
            const auto &attrs = std::get<ApiCallAttrs>(op->attrs);
            Json inputs = Json::array();
            for (auto id : op->inputs) inputs.push_back(std::to_string(id));
            root["execution"].push_back({{"ordinal", instruction.ordinal}, {"kind", "compute"},
                {"op", "ApiCall"}, {"inputs", inputs}, {"output", std::to_string(op->output)},
                {"place", place}, {"attrs", {{"callee", attrs.callee}, {"call_id", std::to_string(attrs.call_id)}}}});
        } else {
            root["execution"].push_back({{"ordinal", instruction.ordinal}, {"kind", "release"},
                {"value", std::to_string(std::get<ReleaseOp>(instruction.body).value)}});
        }
    }
    for (auto id : plan.final_outputs) root["final_outputs"].push_back(std::to_string(id));
    return root;
}
std::string binary(RuntimePlan plan) {
    std::ostringstream output(std::ios::binary);
    binary_io::write_plan(output, plan);
    return output.str();
}
void renumber(RuntimePlan &plan) {
    std::size_t ordinal = 0;
    for (auto &instruction : plan.execution) instruction.ordinal = ordinal++;
}
void test_roundtrip() {
    auto plan = fixture();
    // JSON identifiers retain their full uint64 range, including zero.
    std::get<ApiCallAttrs>(std::get<ComputeOp>(plan.execution[5].body).attrs).call_id =
        std::numeric_limits<std::uint64_t>::max();
    const auto json = json_fixture(plan);
    const auto from_json = RuntimePlanJsonReader::read_text(json.dump()).plan;
    require(json_fixture(from_json) == json, "V7 JSON fields changed");
    require(binary(from_json) == binary(plan), "JSON decoding changed binary records");
    const auto bytes = binary(plan);
    std::istringstream input(bytes, std::ios::binary);
    require(binary(binary_io::read_plan(input, bytes.size())) == bytes, "V7 stream binary roundtrip changed");
    for (unsigned threads : {1U, 4U})
        require(binary(binary_io::read_plan_parallel(bytes, threads)) == bytes, "V7 parallel binary roundtrip changed");

    const auto path = std::filesystem::temp_directory_path() /
        ("ckks-prepared-api-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code error; std::filesystem::remove(path, error); }
    } cleanup{path};
    { std::ofstream output(path, std::ios::binary); output.write(bytes.data(), bytes.size());
      require(bool(output), "cannot write test plan"); }
    for (unsigned threads : {1U, 4U})
        require(binary(RuntimePlanReader::read_file(path.string(), {threads}).plan) == bytes,
                "V7 production file reader changed records");
    std::ostringstream printed;
    plan.print(printed);
    require(printed.str().find("ApiCall()") != std::string::npos &&
            printed.str().find("callee=stem call_id=0") != std::string::npos,
            "plan print omitted prepared call identity");
}
void test_verifier() {
    const auto spec = operator_spec();
    const auto plan = fixture();
    const auto requirements = PlanVerifier::verify(plan, spec);
    require(requirements.capabilities.empty() && requirements.keys.empty(),
            "opaque bootstrap acquired arithmetic/Boot requirements");
    auto bad = plan;
    std::get<ApiCallAttrs>(std::get<ComputeOp>(bad.execution[0].body).attrs).callee.clear();
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "non-empty callee");
    bad = plan;
    std::get<ComputeOp>(bad.execution[1].body).reuse_input = 0;
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "ApiCall does not support reuse_input");
    bad = plan;
    std::swap(bad.execution[2], bad.execution[3]);
    renumber(bad);
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "already released");
    bad = plan;
    bad.execution.push_back({7, ReleaseOp{3}});
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "final output and cannot be released");
    bad = plan;
    bad.values[0].place = {PlaceKind::Host, 0, 0};
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "compute output Place mismatch");
    bad = plan;
    bad.values[0].kind = ValueKind::Plaintext;
    bad.values[0].components = 1;
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "compute output must be ciphertext");
    bad = plan;
    bad.format_version = 5;
    expect_error([&] { PlanVerifier::verify(bad, spec); }, "ApiCall requires format version 7");
    expect_error([&] { binary(bad); }, "ApiCall requires plan V7");
}
void test_json_errors() {
    auto json = json_fixture(fixture());
    json["execution"][0]["attrs"]["callee"] = "";
    expect_error([&] { RuntimePlanJsonReader::read_text(json.dump()); }, "string must not be empty");
    for (const auto &call_id : {"-1", "01", "18446744073709551616"}) {
        json = json_fixture(fixture());
        json["execution"][0]["attrs"]["call_id"] = call_id;
        expect_error([&] { RuntimePlanJsonReader::read_text(json.dump()); }, "identifier");
    }
    json = json_fixture(fixture());
    json["execution"][0]["attrs"]["call_id"] = 0;
    expect_error([&] { RuntimePlanJsonReader::read_text(json.dump()); }, "expected string");
    json = json_fixture(fixture());
    json["format_version"] = 5;
    expect_error([&] { RuntimePlanJsonReader::read_text(json.dump()); }, "ApiCall requires format version 7");
    json["format_version"] = 6;
    expect_error([&] { RuntimePlanJsonReader::read_text(json.dump()); }, "unsupported format version");
}

// Only this explicit test backend implements the prepared calls. No Vec/Mock
// arithmetic backend is used to stand in for opaque GPU library operations.
class PreparedApi {
public:
    struct Payload { ValueDesc desc; int number; };
    using Value = std::shared_ptr<Payload>;
    struct CommHandle {};
    explicit PreparedApi(const RuntimePlan &plan) : plan_(plan) {}
    std::vector<std::uint64_t> calls;
    std::map<ValueId, std::weak_ptr<Payload>> observed;
    std::string name() const { return "TestPreparedApi"; }
    void preflight(std::string_view, bool, const TargetConfig &, const OperatorSpec &,
                   const PlanRequirements &requirements) {
        require(requirements.capabilities.empty() && requirements.keys.empty(), "unexpected requirements");
    }
    Value compute(const ComputeOp &op, const std::vector<Value> &inputs) {
        require(op.kind == ComputeKind::ApiCall, "test backend only supports ApiCall");
        const auto &attrs = std::get<ApiCallAttrs>(op.attrs);
        const auto &callback = callbacks_.at(attrs.call_id);
        require(attrs.callee == callback.first, "prepared callee/call_id mismatch");
        if (attrs.call_id == 3)
            require(observed.at(0).expired() && observed.at(1).expired(), "Release retained consumed inputs");
        calls.push_back(attrs.call_id);
        auto result = std::make_shared<Payload>(Payload{plan_.values.at(op.output), callback.second(inputs)});
        observed[op.output] = result;
        return result;
    }
    void validate_value(const Value &value, const ValueDesc &desc) const {
        const auto &actual = value->desc;
        require(actual.id == desc.id && actual.kind == desc.kind && actual.place == desc.place &&
                actual.context == desc.context && actual.level == desc.level && actual.scale_log2 == desc.scale_log2 &&
                actual.ntt == desc.ntt && actual.components == desc.components, "prepared output descriptor mismatch");
    }
    Value encode_plaintext(const ValueDesc &, const std::vector<double> &) {
        throw std::runtime_error("TestPreparedApi does not support Encode");
    }
    CommHandle communicate_async(const CommAction &, const std::vector<Value> &, const std::vector<ValueDesc> &) {
        throw std::runtime_error("TestPreparedApi does not support communication");
    }
    std::vector<std::optional<Value>> wait(CommHandle &) {
        throw std::runtime_error("TestPreparedApi does not support communication");
    }
    void collect_completed() {}
    void drain() {}
    void synchronize(Value &) {}
    [[noreturn]] void abort_all(int, const std::string &reason) { throw std::runtime_error(reason); }
private:
    const RuntimePlan &plan_;
    using Callback = std::function<int(const std::vector<Value> &)>;
    const std::map<std::uint64_t, std::pair<std::string, Callback>> callbacks_{
        {0, {"stem", [](const auto &inputs) { require(inputs.empty(), "stem has operands"); return 7; }}},
        {1, {"bootstrap", [](const auto &inputs) { return inputs.at(0)->number * 2; }}},
        {2, {"residual_add", [](const auto &inputs) { return inputs.at(0)->number + inputs.at(1)->number; }}},
        {3, {"head", [](const auto &inputs) { return inputs.at(0)->number + 1; }}}};
};
void test_dispatch() {
    const auto plan = fixture();
    const auto spec = operator_spec();
    PreparedApi api(plan);
    SequentialRuntime<PreparedApi> runtime(0, 1, 1, api);
    const auto loaded = RuntimePlanJsonReader::read_text(json_fixture(plan).dump());
    const auto artifact = runtime.run(loaded, {spec, {}}, {});
    require(api.calls == std::vector<std::uint64_t>({0, 1, 2, 3}), "prepared calls did not dispatch in order");
    require(artifact.values.size() == 1 && artifact.values.at(3).value->number == 22, "wrong prepared result");
    require(api.observed.at(0).expired() && api.observed.at(1).expired() && api.observed.at(2).expired(),
            "runtime retained released values");
    require(artifact.timing.compute_calls == 4 && artifact.timing.boot_calls == 0,
            "opaque bootstrap was counted as generic Boot");
}
void test_compiler_output(const char *json_path, const char *binary_path, const char *spec_path) {
    const auto expected = RuntimePlanReader::read_file(json_path).plan;
    require(expected.format_version == 7, "compiler ApiCall plan must be V7");
    std::size_t calls = 0, releases = 0;
    for (const auto &instruction : expected.execution) {
        if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) {
            require(op->kind == ComputeKind::ApiCall && !op->reuse_input, "unexpected compiler computation");
            ++calls;
        } else if (std::holds_alternative<ReleaseOp>(instruction.body)) ++releases;
    }
    require(calls && releases, "compiler fixture must contain calls and releases");
    for (unsigned threads : {1U, 4U})
        require(binary(RuntimePlanReader::read_file(binary_path, {threads}).plan) == binary(expected),
                "compiler JSON and binary records differ");
    if (spec_path) PlanVerifier::verify(expected, OperatorSpecReader::read_file(spec_path));
    std::cout << "PASS compiler V7 JSON/binary: calls=" << calls << " releases=" << releases
              << " readers=1,4 verified=" << bool(spec_path) << '\n';
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc == 1) {
            test_roundtrip();
            test_verifier();
            test_json_errors();
            test_dispatch();
            std::cout << "PASS prepared ApiCall V7: JSON/binary, 0/1/2 operands, verification, Release, dispatch\n";
        } else if (argc == 3 || argc == 4) {
            test_compiler_output(argv[1], argv[2], argc == 4 ? argv[3] : nullptr);
        } else throw std::runtime_error("usage: runtime_prepared_api_tests [COMPILER_JSON COMPILER_BINARY [OPERATOR_SPEC]]");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
