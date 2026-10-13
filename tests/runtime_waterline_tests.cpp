#include "api/mock_api.hpp"
#include "runtime/runtime.hpp"
#include "runtime/plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/binary_plan_codec.hpp"
#include <iostream>
#include <atomic>
#include <algorithm>
#include <limits>
#include <sstream>
using namespace fhegpu;
void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
template <class F> void expect_error(F &&run, const char *message) {
    try { run(); }
    catch (const std::exception &e) {
        if (std::string(e.what()).find(message) != std::string::npos) return;
        throw std::runtime_error(std::string("unexpected error, expected '") + message + "': " + e.what());
    }
    throw std::runtime_error(std::string("expected error containing '") + message + "'");
}
class PrepareApi : public MockVecApi {
public:
    explicit PrepareApi(std::shared_ptr<MockCluster> cluster):MockVecApi(0,std::move(cluster)) {}
    std::atomic<int> cpu{0},gpu{0};
    std::atomic<int> host_encodes{0}, gpu_resource_queries{0};
    int delay_ms=5;bool unsupported=false;
    struct Event { const char *kind; ValueId value; std::thread::id thread; };
    std::vector<Event> events;
    std::vector<std::future<void>> preparations;
    std::mutex mutex;
    void event(const char *kind, ValueId id) {
        std::lock_guard<std::mutex> lock(mutex);
        events.push_back({kind, id, std::this_thread::get_id()});
    }
    ResourceUsage value_resources(const ValueDesc &v) {
        ResourceUsage r(1);if(v.place.kind==PlaceKind::Host)r.host=v.components*100;
        else if(v.kind==ValueKind::Plaintext)r.plain[0]=100;else r.front[0]=200;return r;
    }
    PrepareResources prepare_resources(const ValueDesc &v,PrepareBackend backend) {
        if (backend == PrepareBackend::Gpu) ++gpu_resource_queries;
        auto resident=value_resources(v),peak=resident;peak.host=100;peak.pinned=50;peak.raw=20;
        if(backend==PrepareBackend::Gpu)peak.plain[0]+=50;
        return {peak,resident,!unsupported};
    }
    Value encode_plaintext(const ValueDesc &desc, const std::vector<double> &slots) {
        ++host_encodes;
        return MockVecApi::encode_plaintext(desc, slots);
    }
    ResourceUsage instruction_workspace(const Instruction &i) {
        ResourceUsage r(1);if(std::holds_alternative<ComputeOp>(i.body))r.front[0]=100;return r;
    }
    Value prepare_plaintext_async(const ValueDesc &desc,const std::vector<double> &slots,PrepareBackend backend) {
        event("prepare", desc.id);
        if(backend==PrepareBackend::Cpu)++cpu;else ++gpu;
        auto host=desc;host.place={PlaceKind::Host,0,0};auto source=encode_plaintext(host,slots);
        auto result=Value::pending(source.kind(),source.metadata());
        std::lock_guard<std::mutex> lock(mutex);
        preparations.push_back(std::async(std::launch::async,[=,this] {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));result.fulfill(source.materialize());
        }));return result;
    }
    Value compute(const ComputeOp &op, const std::vector<Value> &inputs) {
        event("compute_start", op.output);
        auto result = MockVecApi::compute(op, inputs);
        event("compute_complete", op.output);
        return result;
    }
    bool is_ready(Value &v) {return v.is_ready();}
    void drain() {for(auto &future:preparations)future.get();preparations.clear();MockVecApi::drain();}
};

std::size_t event_index(const PrepareApi &api, const char *kind, ValueId id) {
    for (std::size_t i = 0; i < api.events.size(); ++i)
        if (std::string(api.events[i].kind) == kind && api.events[i].value == id) return i;
    throw std::runtime_error(std::string("missing event '") + kind + "' for ValueId " + std::to_string(id));
}

void number_instructions(RuntimePlan &plan) {
    std::size_t ordinal = 0;
    for (auto *phase : {&plan.initialization, &plan.execution, &plan.finalization})
        for (auto &instruction : *phase) instruction.ordinal = ordinal++;
}

RuntimePlan binary_round_trip(RuntimePlan plan) {
    std::stringstream wire(std::ios::in | std::ios::out | std::ios::binary);
    binary_io::write_plan(wire, plan);
    const auto size = wire.str().size();
    wire.seekg(0);
    return binary_io::read_plan(wire, size);
}

void test_json_and_negative_config(const RuntimePlan &template_plan, const LoadedOperatorSpec &spec) {
    using Json = binary_io::Json;
    auto root = binary_io::metadata(template_plan);
    root["plan_id"] = "18446744073709551615";
    const Json device{{"kind", "device"}, {"rank", 0}, {"index", 0}};
    const Json host{{"kind", "host"}, {"rank", 0}};
    for (unsigned id = 0; id < 4; ++id)
        root["values"].push_back({{"id", std::to_string(id)},
            {"kind", id == 1 ? "plaintext" : "ciphertext"}, {"place", id == 0 ? host : device},
            {"context", spec.spec.context_id}, {"level", 3}, {"scale_log2", 30},
            {"ntt", true}, {"components", id == 1 ? 1 : 2}});
    root["external_inputs"] = {"0"};
    root["initialization"] = {{{"ordinal", 0}, {"kind", "transfer"}, {"transfer_id", "0"},
        {"hint", "point_to_point"}, {"inputs", {"0"}}, {"outputs", {"3"}},
        {"sources", {host}}, {"destinations", {device}}, {"output_kinds", {"ciphertext"}}}};
    root["execution"] = {
        {{"ordinal", 1}, {"kind", "prepare_plaintext"}, {"output", "1"},
            {"payload", {{"kind", "inline"}, {"values", {.5, -1.0}}}}},
        {{"ordinal", 2}, {"kind", "compute"}, {"op", "add_cp"}, {"place", device},
            {"inputs", {"3", "1"}}, {"output", "2"}},
        {{"ordinal", 3}, {"kind", "release"}, {"value", "1"}},
        {{"ordinal", 4}, {"kind", "release"}, {"value", "3"}}};
    root["final_outputs"] = {"2"};

    for (const char *backend : {"cpu", "gpu", "auto"}) {
        root["waterline"]["backend"] = backend;
        auto loaded = RuntimePlanJsonReader::read_text(root.dump());
        PlanVerifier::verify(loaded.plan, spec, false);
        auto parsed = binary_round_trip(loaded.plan);
        PlanVerifier::verify(parsed, spec, false);
        require(binary_io::metadata(parsed) == binary_io::metadata(loaded.plan),
                "V4 JSON/binary round trip changed Waterline metadata");
        require(parsed.plan_id == std::numeric_limits<std::uint64_t>::max() &&
                parsed.external_inputs == loaded.plan.external_inputs &&
                parsed.final_outputs == loaded.plan.final_outputs && parsed.values.size() == 4,
                "V4 round trip changed identifiers or values");
        const auto &prepare = std::get<PreparePlaintextOp>(parsed.execution.at(0).body);
        require(prepare.output == 1 && std::get<InlineEncodePayload>(prepare.payload).values ==
                std::vector<double>({.5, -1.0}), "V4 round trip changed inline Prepare payload");
        const auto &compute = std::get<ComputeOp>(parsed.execution.at(1).body);
        require(compute.kind == ComputeKind::AddCP && compute.inputs == std::vector<ValueId>({3, 1}) &&
                compute.output == 2 && compute.place == Place{PlaceKind::Device, 0, 0} &&
                std::get<ReleaseOp>(parsed.execution.at(2).body).value == 1,
                "V4 round trip changed Compute/Release records");
    }
    // Bundle Prepare uses the binary string table as well as the new instruction tag.
    const auto digest = "sha256:" + std::string(64, 'a');
    auto bundled = root;
    bundled["execution"][0]["payload"] = {{"kind", "bundle"}, {"content", digest}};
    bundled["plaintext_bundle"] = {{"id", "waterline-test"}, {"version", 1}, {"manifest_sha256", digest}};
    auto bundle_plan = RuntimePlanJsonReader::read_text(bundled.dump()).plan;
    PlanVerifier::verify(bundle_plan, spec, false);
    auto bundle_parsed = binary_round_trip(bundle_plan);
    PlanVerifier::verify(bundle_parsed, spec, false);
    require(std::get<BundleEncodePayload>(std::get<PreparePlaintextOp>(bundle_parsed.execution[0].body).payload).content == digest &&
            bundle_parsed.plaintext_bundle->manifest_sha256 == digest,
            "V4 round trip changed bundle Prepare/string table");

    const auto reject_json = [&](const Json &bad, const char *message) {
        expect_error([&] { RuntimePlanJsonReader::read_text(bad.dump()); }, message);
    };
    auto bad = root;bad["waterline"]["backend"] = "fft";reject_json(bad, "expected cpu, gpu or auto");
    bad = root;bad["waterline"]["device_capacity_bytes"] = 1000;reject_json(bad, "device budgets must be arrays");
    bad = root;bad["waterline"]["device_fractions"] = .25;reject_json(bad, "expected array");
    bad = root;bad["waterline"]["device_fractions"] = {true};reject_json(bad, "expected number");
    bad = root;bad["waterline"]["host_budget_bytes"] = -1;reject_json(bad, "outside the supported range");
    bad = root;bad["waterline"]["host_budget_bytes"] = .5;reject_json(bad, "expected integer");
    bad = root;bad["waterline"]["host_budget_bytes"] = (1ULL << 53);reject_json(bad, "outside the supported range");
    bad = root;bad["waterline"]["lookahead_groups"] = 0;reject_json(bad, "outside the supported range");
    bad = root;bad["waterline"]["obsolete_batch_size"] = 4;reject_json(bad, "unknown field");
    bad = root;bad["format_version"] = 3;reject_json(bad, "waterline requires V4");
    bad.erase("waterline");reject_json(bad, "PreparePlaintext requires format version 4");

    const auto valid = RuntimePlanJsonReader::read_text(root.dump()).plan;
    auto legacy = valid;legacy.format_version = 3;legacy.waterline.reset();
    expect_error([&] { binary_round_trip(legacy); }, "binary PreparePlaintext requires plan V4");
    const auto reject_plan = [&](const RuntimePlan &invalid, const char *message) {
        expect_error([&] { PlanVerifier::verify(invalid, spec, false); }, message);
    };
    for (int fault = 0; fault < 3; ++fault) {
        auto invalid = valid;
        if (fault == 0) invalid.waterline->device_capacity_bytes.clear();
        if (fault == 1) invalid.waterline->device_reserve_bytes.clear();
        if (fault == 2) invalid.waterline->device_fractions.clear();
        reject_plan(invalid, "device budget table mismatch");
    }
    for (double fraction : {0.0, 1.0, -.1, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
        auto invalid = valid;invalid.waterline->device_fractions = {fraction};
        reject_plan(invalid, "device fraction");
    }
    for (int fault = 0; fault < 6; ++fault) {
        auto invalid = valid;
        if (fault == 0) invalid.waterline->device_capacity_bytes = {0};
        if (fault == 1) invalid.waterline->device_reserve_bytes = invalid.waterline->device_capacity_bytes;
        if (fault == 2) invalid.waterline->host_budget_bytes = 0;
        if (fault == 3) invalid.waterline->pinned_budget_bytes = 0;
        if (fault == 4) invalid.waterline->raw_budget_bytes = 0;
        if (fault == 5) invalid.waterline->lookahead_groups = 0;
        reject_plan(invalid, fault < 2 ? "capacity/reserve" : "positive budgets");
    }
    auto invalid = valid;invalid.waterline.reset();reject_plan(invalid, "PreparePlaintext requires V4");
    for (int fault = 0; fault < 4; ++fault) {
        invalid = valid;
        if (fault == 0) invalid.values[1].place.kind = PlaceKind::Host;
        if (fault == 1) {invalid.values[1].kind = ValueKind::Ciphertext;invalid.values[1].components = 2;}
        if (fault == 2) invalid.values[1].ntt = false;
        if (fault == 3) invalid.values[1].components = 2;
        reject_plan(invalid, fault == 3 ? "plaintext components" : "Device NTT plaintext");
    }
    invalid = valid;
    std::get<InlineEncodePayload>(std::get<PreparePlaintextOp>(invalid.execution[0].body).payload).values =
        {std::numeric_limits<double>::quiet_NaN()};
    reject_plan(invalid, "non-finite PreparePlaintext payload");
    invalid = valid;invalid.initialization.push_back(invalid.execution.front());
    invalid.execution.erase(invalid.execution.begin());number_instructions(invalid);
    reject_plan(invalid, "PreparePlaintext requires V4 execution");
    std::cout << "PASS V4 JSON/binary inline and bundle round trips, strict config and Prepare rejection\n";
}

void test_host_waiter(const RuntimePlan &rolling, const LoadedOperatorSpec &spec,
                      const std::vector<double> &input) {
    auto plan = rolling;
    plan.waterline->backend = PrepareBackend::Cpu;
    plan.execution.resize(4); // First Prepare/AddCP/Release pair from the rolling graph.
    auto received = plan.values.at(0);received.id = 50;
    auto output = received;output.id = 51;
    plan.values.erase(std::remove_if(plan.values.begin(), plan.values.end(), [](const ValueDesc &value) {
        return value.id != 0 && value.id != 41 && value.id != 1 && value.id != 2;
    }), plan.values.end());
    plan.values.push_back(received);plan.values.push_back(output);
    const Place device{PlaceKind::Device, 0, 0}, host{PlaceKind::Host, 0, 0};
    plan.execution.push_back({0, CommAction{2, CommKind::Transfer, CommHint::PointToPoint,
        {2}, {50}, {device}, {host}, {ValueKind::Ciphertext}}});
    plan.execution.push_back({0, ComputeOp{ComputeKind::Negate, {50}, 51, host, {}}});
    plan.execution.push_back({0, ReleaseOp{50}});
    plan.execution.push_back({0, ReleaseOp{2}});
    plan.finalization.clear();plan.final_outputs = {51};number_instructions(plan);
    // The CPU queue begins with Negate, whose producer needs this same CPU
    // worker to Prepare. Dispatching the undefined Host consumer would deadlock.
    PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));api.delay_ms = 20;
    SequentialRuntime<PrepareApi> runtime(0, 1, 1, api, DeviceExecutionMode::PerDeviceWorkers);
    auto artifact = runtime.run({plan, "sha256:" + std::string(64, '1')}, {spec, std::nullopt, false},
        {{0, make_cipher(input, spec.spec.context_id, spec.spec.poly_degree, 3, 30)}});
    const auto result = artifact.values.at(51).value.materialize();
    require(std::abs(result.slots[0] + 1.5) < 1e-9 && std::abs(result.slots[1] - 3) < 1e-9,
            "Host consumer failed to read prepared result");
    const auto prepare = event_index(api, "prepare", 1), host_compute = event_index(api, "compute_start", 51);
    require(prepare < host_compute && api.events[prepare].thread == api.events[host_compute].thread &&
            api.cpu == 1 && api.gpu == 0 && artifact.timing.fence_calls == 0,
            "Host waiter blocked or bypassed the single CPU preparation worker");
    std::cout << "PASS Host waiter leaves the single CPU worker available for required Prepare\n";
}

void test_small_waterline(const RuntimePlan &rolling, const LoadedOperatorSpec &spec,
                          const std::vector<double> &input) {
    for (const auto backend : {PrepareBackend::Cpu, PrepareBackend::Auto}) {
        auto plan = rolling;plan.waterline->backend = backend;
        plan.waterline->device_fractions = {.1}; // CPU peak fits one plaintext; GPU peak does not.
        PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));api.delay_ms = 10;
        SequentialRuntime<PrepareApi> runtime(0, 1, 1, api, DeviceExecutionMode::PerDeviceWorkers);
        auto artifact = runtime.run({plan, "sha256:" + std::string(64, '1')}, {spec, std::nullopt, false},
            {{0, make_cipher(input, spec.spec.context_id, spec.spec.poly_degree, 3, 30)}});
        for (unsigned i = 1; i < 20; ++i)
            require(event_index(api, "compute_complete", 2 * i) < event_index(api, "prepare", 1 + 2 * i),
                    "small waterline submitted the next group before the current consumer completed");
        auto result = artifact.values.at(42).value.materialize();
        require(std::abs(result.slots[0] - 11) < 1e-9 && std::abs(result.slots[1] + 22) < 1e-9 &&
                api.cpu == 20 && api.gpu == 0 && artifact.timing.fence_calls == 0,
                "small waterline failed to resume all groups after reclaim");
        require(backend != PrepareBackend::Auto || api.gpu_resource_queries > 0,
                "Auto small-waterline test never evaluated an oversized GPU candidate");
        std::cout << "PASS one-plaintext waterline backend=" << int(backend)
                  << " resumes 20 consumers, oversized GPU candidate keeps legal CPU path\n";
    }
}

void test_initialization_preflight(const RuntimePlan &rolling, const LoadedOperatorSpec &spec,
                                   const std::vector<double> &input) {
    auto plan = rolling;plan.waterline->backend = PrepareBackend::Cpu;
    plan.waterline->device_capacity_bytes = {450}; // Later AddCP needs input + output + workspace = 500.
    auto plaintext = plan.values.at(0);plaintext.id = 60;plaintext.kind = ValueKind::Plaintext;plaintext.components = 1;
    plan.values.push_back(plaintext);
    plan.initialization.insert(plan.initialization.begin(), {0, ReleaseOp{60}});
    plan.initialization.insert(plan.initialization.begin(), {0, EncodeOp{InlineEncodePayload{{.5, -1}}, 60}});
    number_instructions(plan);
    PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));
    SequentialRuntime<PrepareApi> runtime(0, 1, 1, api, DeviceExecutionMode::PerDeviceWorkers);
    expect_error([&] {
        runtime.run({plan, "sha256:" + std::string(64, '1')}, {spec, std::nullopt, false},
            {{0, make_cipher(input, spec.spec.context_id, spec.spec.poly_degree, 3, 30)}});
    }, "single-group");
    require(api.host_encodes == 0 && api.cpu == 0 && api.gpu == 0 && api.stats().communicate_calls == 0,
            "infeasible execution submitted initialization Encode/Transfer before full-plan preflight");
    std::cout << "PASS later infeasible group rejects before initialization Encode/Transfer submission\n";
}

void test_prepare_across_fence_and_unused(const RuntimePlan &rolling, const LoadedOperatorSpec &spec,
                                         const std::vector<double> &input) {
    auto plan = rolling;plan.waterline->backend = PrepareBackend::Cpu;
    plan.execution.insert(plan.execution.begin() + 1, {0, FenceOp{}});
    auto unused = *std::find_if(plan.values.begin(), plan.values.end(), [](const ValueDesc &value) {
        return value.id == 1;
    });
    unused.id = 61;plan.values.push_back(unused);
    plan.execution.push_back({0, PreparePlaintextOp{InlineEncodePayload{{7.0, 2.0}}, 61}});
    number_instructions(plan);
    PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));
    SequentialRuntime<PrepareApi> runtime(0, 1, 1, api, DeviceExecutionMode::PerDeviceWorkers);
    auto artifact = runtime.run({plan, "sha256:" + std::string(64, '1')}, {spec, std::nullopt, false},
        {{0, make_cipher(input, spec.spec.context_id, spec.spec.poly_degree, 3, 30)}});
    auto result = artifact.values.at(42).value.materialize();
    require(std::abs(result.slots[0] - 11) < 1e-9 && std::abs(result.slots[1] + 22) < 1e-9 &&
            api.cpu == 20 && api.gpu == 0 && artifact.timing.fence_calls == 1,
            "Prepare registered before Fence was lost or unused Prepare blocked drain");
    require(std::none_of(api.events.begin(), api.events.end(), [](const PrepareApi::Event &event) {
        return std::string(event.kind) == "prepare" && event.value == 61;
    }), "unused Prepare submitted work without a consumer group");
    std::cout << "PASS Prepare crosses explicit Fence and unused Prepare finishes without submission\n";
}
int main() {
 try {
    WaterlineConfig c; c.device_capacity_bytes={1000};c.device_reserve_bytes={0};c.device_fractions={.4};
    c.host_budget_bytes=600;c.pinned_budget_bytes=300;c.raw_budget_bytes=100;c.lookahead_groups=8;
    WaterlineLedger ledger(c);ResourceUsage first(1);first.front[0]=400;first.plain[0]=300;ledger.reserve(first);
    ResourceUsage two(1);two.plain[0]=200;two.front[0]=200;two.host=100;
    ResourceUsage one(1);one.plain[0]=100;
    require(ledger.fits(one) && !ledger.fits(two),"two-plaintext complete group was partially admitted");
    expect_error([&] { ledger.reserve(two); }, "complete group admission");
    require(ledger.used.front[0]==400 && ledger.used.plain[0]==300 && ledger.used.host==0,
            "failed complete-group reservation mutated the ledger");
    ledger.release(first);require(ledger.fits(two), "group did not become admissible after completion reclaim");
    ledger.reserve(two);ledger.release(two);
    require(ledger.used.front[0]==0 && ledger.used.plain[0]==0 && ledger.used.host==0,
            "complete group failed to return its resource credits");
    auto spec=OperatorSpecReader::read_file(std::string(CKKS_RUNTIME_SOURCE_DIR)+"/docs/operator-spec/v1/profiles/poseidon-ckks-gpu.v1.json");
    RuntimePlan plan;plan.format_version=4;plan.plan_id=404;plan.target={spec.spec.target_id,1,{1},1,{spec.spec.id,spec.spec.version,spec.source_sha256}};
    const Place d{PlaceKind::Device,0,0};const auto v=[&](ValueId id,ValueKind kind) {return ValueDesc{id,kind,d,spec.spec.context_id,3,30,true,kind==ValueKind::Plaintext?1:2};};
    plan.values.push_back(v(0,ValueKind::Ciphertext));plan.external_inputs={0};
    const Place host{PlaceKind::Host,0,0};
    plan.values.front().place=host;
    auto initial=v(41,ValueKind::Ciphertext);plan.values.push_back(initial);
    plan.initialization={{0,CommAction{0,CommKind::Transfer,CommHint::PointToPoint,{0},{41},{host},{d},{ValueKind::Ciphertext}}}};
    ValueId previous=41;
    for(unsigned i=0;i<20;++i) {
        const auto p=ValueId(1+i*2),out=p+1;plan.values.push_back(v(p,ValueKind::Plaintext));plan.values.push_back(v(out,ValueKind::Ciphertext));
        plan.execution.push_back({0,PreparePlaintextOp{InlineEncodePayload{{.5,-1}},p}});
        plan.execution.push_back({0,ComputeOp{ComputeKind::AddCP,{previous,p},out,d,{}}});
        plan.execution.push_back({0,ReleaseOp{p}});if(previous)plan.execution.push_back({0,ReleaseOp{previous}});previous=out;
    }
    auto result_desc=v(42,ValueKind::Ciphertext);result_desc.place=host;plan.values.push_back(result_desc);
    plan.finalization={{0,CommAction{1,CommKind::Transfer,CommHint::PointToPoint,{previous},{42},{d},{host},{ValueKind::Ciphertext}}},{0,ReleaseOp{previous}}};
    previous=42;plan.final_outputs={previous};
    std::size_t ordinal=0;for(auto *phase:{&plan.initialization,&plan.execution,&plan.finalization})for(auto &i:*phase)i.ordinal=ordinal++;
    c.device_fractions={.25};plan.waterline=c;
    std::stringstream wire(std::ios::in|std::ios::out|std::ios::binary);binary_io::write_plan(wire,plan);const auto size=wire.str().size();wire.seekg(0);
    auto parsed=binary_io::read_plan(wire,size);require(parsed.format_version==4 && parsed.waterline && parsed.execution.size()==plan.execution.size(),"V4 binary round trip changed plan");
    require(std::get<PreparePlaintextOp>(parsed.execution[0].body).output==1,"binary Prepare tag/output corrupted");
    PlanVerifier::verify(parsed,spec,false);
    std::vector<double> input(spec.spec.poly_degree/2);input[0]=1;input[1]=-2;
    test_host_waiter(parsed, spec, input);
    test_small_waterline(parsed, spec, input);
    test_json_and_negative_config(parsed, spec);
    test_initialization_preflight(parsed, spec, input);
    test_prepare_across_fence_and_unused(parsed, spec, input);
    for(auto backend:{PrepareBackend::Cpu,PrepareBackend::Gpu,PrepareBackend::Auto}) {
        parsed.waterline->backend=backend;
        // Auto needs room for two groups including GPU's extra 50-byte workspace.
        parsed.waterline->device_capacity_bytes={backend==PrepareBackend::Auto?1200u:1000u};
        PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));
        SequentialRuntime<PrepareApi> runtime(0,1,1,api,DeviceExecutionMode::PerDeviceWorkers);
        auto artifact=runtime.run({parsed,"sha256:"+std::string(64,'1')},{spec,std::nullopt,false},
            {{0,make_cipher(input,spec.spec.context_id,spec.spec.poly_degree,3,30)}});
        auto result=artifact.values.at(previous).value.materialize();require(std::abs(result.slots[0]-11)<1e-9 && std::abs(result.slots[1]+22)<1e-9,"rolling Prepare changed arithmetic");
        require(api.cpu+api.gpu==20 && artifact.timing.fence_calls==0,"Prepare repeated or inserted Fence");
        require(backend!=PrepareBackend::Cpu || (api.cpu==20 && api.gpu==0), "CPU-only selected a GPU Prepare");
        require(backend!=PrepareBackend::Gpu || (api.gpu==20 && api.cpu==0), "GPU-only selected a CPU Prepare");
        require(backend!=PrepareBackend::Auto || (api.cpu>0 && api.gpu>0), "Auto rolling test did not exercise both backends");
        std::cout<<"PASS rolling backend="<<int(backend)<<" cpu="<<api.cpu<<" gpu="<<api.gpu<<"\n";
    }
    parsed.waterline->device_capacity_bytes={1000};
    for(int fault=0;fault<6;++fault) {
        auto bad=parsed;
        if(fault==0)bad.waterline->device_fractions={.01};
        if(fault==2)bad.waterline->device_capacity_bytes={450}; // Input + output + workspace need 500.
        if(fault==3)bad.waterline->host_budget_bytes=250; // External Host input + Prepare need 300.
        if(fault==4)bad.waterline->pinned_budget_bytes=25;
        if(fault==5)bad.waterline->raw_budget_bytes=10;
        PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));api.unsupported=fault==1;
        SequentialRuntime<PrepareApi> runtime(0,1,1,api,DeviceExecutionMode::PerDeviceWorkers);
        bool rejected=false;
        try {runtime.run({bad,"sha256:"+std::string(64,'1')},{spec,std::nullopt,false},{{0,make_cipher(input,spec.spec.context_id,spec.spec.poly_degree,3,30)}});}
        catch(const ClusterPanic &e) {rejected=std::string(e.what()).find(fault==1?"unsupported complete":"single-group")!=std::string::npos;}
        require(rejected && api.cpu+api.gpu==0,"infeasible/unsupported baseline submitted partial work");
    }
    std::cout<<"PASS complete atomic group, delayed events, continuous reclaim, no partial start on failure, V4 binary\n";
 }catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
