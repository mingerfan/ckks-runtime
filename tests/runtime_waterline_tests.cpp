#include "api/mock_api.hpp"
#include "runtime/runtime.hpp"
#include "runtime/plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/binary_plan_codec.hpp"
#include <iostream>
#include <atomic>
#include <sstream>
using namespace fhegpu;
void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
class PrepareApi : public MockVecApi {
public:
    explicit PrepareApi(std::shared_ptr<MockCluster> cluster):MockVecApi(0,std::move(cluster)) {}
    std::atomic<int> cpu{0},gpu{0};
    int delay_ms=5;bool unsupported=false;
    std::vector<std::future<void>> preparations;
    std::mutex mutex;
    ResourceUsage value_resources(const ValueDesc &v) {
        ResourceUsage r(1);if(v.place.kind==PlaceKind::Host)r.host=v.components*100;
        else if(v.kind==ValueKind::Plaintext)r.plain[0]=100;else r.front[0]=200;return r;
    }
    PrepareResources prepare_resources(const ValueDesc &v,PrepareBackend backend) {
        auto resident=value_resources(v),peak=resident;peak.host=100;peak.pinned=50;peak.raw=20;
        if(backend==PrepareBackend::Gpu)peak.plain[0]+=50;
        return {peak,resident,!unsupported};
    }
    ResourceUsage instruction_workspace(const Instruction &i) {
        ResourceUsage r(1);if(std::holds_alternative<ComputeOp>(i.body))r.front[0]=100;return r;
    }
    Value prepare_plaintext_async(const ValueDesc &desc,const std::vector<double> &slots,PrepareBackend backend) {
        if(backend==PrepareBackend::Cpu)++cpu;else ++gpu;
        auto host=desc;host.place={PlaceKind::Host,0,0};auto source=encode_plaintext(host,slots);
        auto result=Value::pending(source.kind(),source.metadata());
        std::lock_guard<std::mutex> lock(mutex);
        preparations.push_back(std::async(std::launch::async,[=,this] {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));result.fulfill(source.materialize());
        }));return result;
    }
    bool is_ready(Value &v) {return v.is_ready();}
    void drain() {for(auto &future:preparations)future.get();preparations.clear();MockVecApi::drain();}
};
int main() {
 try {
    WaterlineConfig c; c.device_capacity_bytes={1000};c.device_reserve_bytes={0};c.device_fractions={.4};
    c.host_budget_bytes=600;c.pinned_budget_bytes=300;c.raw_budget_bytes=100;c.lookahead_groups=8;
    WaterlineLedger ledger(c);ResourceUsage first(1);first.front[0]=400;first.plain[0]=300;ledger.reserve(first);
    ResourceUsage two(1);two.plain[0]=200;two.front[0]=200;two.host=100;
    require(!ledger.fits(two),"partial dependency group admitted");require(ledger.used.plain[0]==300 && ledger.used.host==0,"failed admission mutated ledger");
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
    for(auto backend:{PrepareBackend::Cpu,PrepareBackend::Gpu,PrepareBackend::Auto}) {
        parsed.waterline->backend=backend;
        PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));
        SequentialRuntime<PrepareApi> runtime(0,1,1,api,DeviceExecutionMode::PerDeviceWorkers);
        auto artifact=runtime.run({parsed,"sha256:"+std::string(64,'1')},{spec,std::nullopt,false},
            {{0,make_cipher(input,spec.spec.context_id,spec.spec.poly_degree,3,30)}});
        auto result=artifact.values.at(previous).value.materialize();require(std::abs(result.slots[0]-11)<1e-9 && std::abs(result.slots[1]+22)<1e-9,"rolling Prepare changed arithmetic");
        require(api.cpu+api.gpu==20 && artifact.timing.fence_calls==0,"Prepare repeated or inserted Fence");
        std::cout<<"PASS rolling backend="<<int(backend)<<" cpu="<<api.cpu<<" gpu="<<api.gpu<<"\n";
    }
    for(int fault=0;fault<2;++fault) {
        auto bad=parsed;if(fault==0)bad.waterline->device_fractions={.01};
        PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));api.unsupported=fault==1;
        SequentialRuntime<PrepareApi> runtime(0,1,1,api,DeviceExecutionMode::PerDeviceWorkers);
        bool rejected=false;
        try {runtime.run({bad,"sha256:"+std::string(64,'1')},{spec,std::nullopt,false},{{0,make_cipher(input,spec.spec.context_id,spec.spec.poly_degree,3,30)}});}
        catch(const ClusterPanic &e) {rejected=std::string(e.what()).find(fault==0?"single-group":"unsupported complete")!=std::string::npos;}
        require(rejected && api.cpu+api.gpu==0,"infeasible/unsupported baseline submitted partial work");
    }
    std::cout<<"PASS complete atomic group, delayed events, continuous reclaim, no partial start on failure, V4 binary\n";
 }catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
