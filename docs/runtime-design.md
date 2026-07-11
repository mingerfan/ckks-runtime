# Runtime 设计

## 1. 定位

Runtime 是 executable plan 的纯执行器：

- 不做 device/Place placement；
- 不读取完整 TopologyModel；
- 不选择通信路由；
- 不选择 MPI、NCCL 或 CUDA；
- 不决定 fallback；
- 不修复非法计划；
- 不实现 retry、cancel 或恢复。

Runtime 负责验证计划、管理 Ready/Pending Value、按顺序调用 Api，并在任何错误发生时打印上下文后终止整个 execution group。

## 2. 依赖关系

~~~text
Executable Plan
    |
    v
Runtime<Api>
├── PlanVerifier
├── ValueStore
└── SequentialExecutor
        |
        v
Api
├── Value
├── compute functions
├── communicate_async
├── wait
└── abort_all
~~~

Runtime 通过模板使用 Api：

~~~cpp
template <class Api>
class Runtime {
public:
    using Value = typename Api::Value;
    using CommHandle = typename Api::CommHandle;
};
~~~

Runtime 不理解 Value 的内部结构。

## 3. Runtime 输入

Executable plan 已经包含：

- 固定 plan/version ID；
- 每个 op 的稳定 ordinal；
- PureCompute op；
- CommAction；
- source/destination 物理 Place；
- output ValueType；
- 可选 CommHint；
- Initialization、Execution、Finalization 阶段；
- 编译目标的 world size 和本地设备要求。

Runtime 只需要运行时身份：

~~~cpp
struct LocalIdentity {
    int rank;
};
~~~

实际 rank 数量或本地设备不符合编译目标时直接 panic。

## 4. ValueStore

首期只维护两种状态：

~~~cpp
enum class ValueState {
    Ready,
    Pending
};
~~~

概念结构：

~~~cpp
struct ReadyEntry {
    Place place;
    Api::Value value;
};

using PendingGroupId = std::uint64_t;

struct PendingEntry {
    Place place;
    PendingGroupId group_id;
    std::size_t output_slot;
};

struct PendingGroup {
    std::vector<ValueId> local_output_ids;
    Api::CommHandle handle;
};
~~~

ValueStore 严格按 ValueId 寻址：

~~~text
ValueId -> ReadyEntry or PendingEntry
~~~

每个 ReadyEntry/PendingEntry 只包含一个 Place。一个多输出 CommHandle 可以由多个不同 ValueId 的 PendingEntry 共享，但每个 PendingEntry 通过 output_slot 映射到唯一输出；同一个 ValueId 不得表示多个 Place。等待该 group 时，Runtime 一次安装它在本 rank 上的全部结果。

不存在的 Value 属于 fatal plan/runtime error。异步失败由 Api.wait 抛出并立即终止，不需要长期 Failed 状态。

## 5. 加载与验证

执行前检查：

- schema/version；
- ValueId 唯一定义；
- 每个 ValueId 恰好绑定一个 Place；
- use-before-def；
- op arity 和 ValueType；
- compute op 必须是 PureCompute；
- compute op 只有一个执行 Place；
- compute input 已在该 Place 物化；
- CommKind 的 input/output 关系合法；
- Replicate 的每个 destination 对应一个不同 output ValueId；
- source/destination Place 属于编译目标；
- TransferId 唯一；
- world size 和本地设备符合目标；
- 外部 input 已绑定；
- 所有 rank 加载同一个 plan。

Verifier 只拒绝非法计划，不插入传输或重新 placement。

## 6. Compute 执行

伪代码：

~~~cpp
execute_compute(op):
    assert op is PureCompute
    assert op.place belongs to local rank

    inputs = []
    for input in op.inputs:
        inputs.push(ensure_ready(input, op.place))

    output = api.compute(op, inputs)
    value_store.define_ready(op.output, op.place, output)
~~~

Api 首期契约：

> compute 返回后，输出已经可以安全交给 communicate_async。

如果 Poseidon 内部异步启动 GPU kernel，由 Poseidon Api 自己处理同步或 stream/event 连接。

Runtime 不接触 CUDA event。

## 7. CommAction 执行

所有 rank 以相同顺序解释同一个 CommAction：

~~~cpp
execute_comm(action):
    local_inputs = collect_sources_owned_by_local_rank(action)

    group = register_pending_group(
        api.communicate_async(action, local_inputs))
    register_each_local_output_by_value_id(action, group)
    retain_group_until_finalization(group)
~~~

对于 Replicate，Runtime 按位置建立映射：`action.outputs[i]` 属于 `action.destinations[i]`。同一个 handle 可以完成多个输出，但不会让一个 output ValueId 对应多个 destination。

角色由物理 Place 和 local rank 直接确定：

- source 在本 rank：提供 source Value；
- destination 在本 rank：Api 准备本地接收和输出；
- source/destination 都在本 rank：Api 可以直接本地 copy；
- 与本 rank 无关：Api 返回 no-op handle。

Runtime 不知道 communicate_async 最终调用了 send、recv、broadcast、gather、memcpy、MPI 或 NCCL。

## 8. ensure_ready

consumer 使用输入时：

~~~cpp
Value &ensure_ready(ValueId id, Place expected_place):
    entry = value_store.lookup(id)
    assert entry.place == expected_place

    if entry is Ready:
        return entry.value

    if entry is Pending:
        group = pending_groups.lookup(entry.group_id)
        outputs = api.wait(group.handle)
        install_ready_outputs_by_slot(group, outputs)
        return value_store.lookup_ready(id).value

    panic("missing value")
~~~

`expected_place` 只用于检查 consumer 的本地性，不参与 ValueStore 寻址。等待只发生在真正消费数据时；一次 wait 可以同时把同一 action 的多个独立 output ValueId 变为 Ready。独立 op 可以继续执行，直到遇到自己的 Pending input。

首期不暴露 test、progress 或 cancel。Api 可以在 communicate_async/wait 内部使用 progress thread、MPI_Test、CUDA event 或其他机制。

## 9. 三个执行阶段

### Initialization

- 绑定 Host input；
- 绑定 plaintext/ciphertext 常量；
- 执行编译器生成的 HostToDevice Transfer；
- 上传 Api 需要的 context/key；
- 等待初始化所需 Value Ready。

首期采用 eager preload：执行阶段需要的外部值和 plaintext 全部预加载，并保留到 run 结束。

### Execution

- 顺序解释 compute 和 communication action；
- 根据 rank/Place mask；
- consumer 前 ensure_ready。

### Finalization

- 执行 DeviceToHost output action；
- wait 所有 outstanding source CommHandle；
- 返回或打印结果；
- `AllValuesAfterRun` 测试模式额外 wait 本 rank 的所有 diff-point Pending output；
- 测试模式可在全部相关通信完成后、释放前导出 test-only RunArtifact；
- 统一释放 Value。

## 10. 生命周期

为避免过度设计，首期不做 last-use 回收：

- 所有 Ready Value 保留到 Finalization；
- 所有 source CommHandle 保留到 Finalization；
- Finalization 完成全部 wait 后统一释放。

这会增加内存占用，但能直接保证 source 不会在异步发送完成前被释放。

逐指令 difftest 利用这个首期生命周期，在所有 plan action 已发起后完成必要的 finalization wait，复制仍然存活的 Value，并在 Runtime 停止后离线比较。Runtime 正常执行路径不插入逐 op observer 或 barrier；未开启该测试选项时不生成 RunArtifact。RunArtifact 是测试设施，不属于 Runtime/Api 的稳定 ABI，开启它的测试不用于性能计时。

未来 plaintext/显存规模要求更高时，由编译器 memory planning 插入 Prefetch 和 Release/Evict。Runtime 只执行显式 action。

## 11. 同一全局计划

本 demo 采用 SPMD 解释方式：

- 所有 rank 读取同一 plan；
- op ordinal 顺序一致；
- compute 根据 op.place.rank mask；
- CommAction 在所有 rank 上都被解释；
- Api 根据本地角色执行或 no-op；
- collective 顺序由全局 plan 保证。

一个 rank 对应一个 Runtime 实例。真实 MPI 部署通常是一个 process/rank 内一个 Runtime；Mock 多 rank 测试则在同一进程内使用多个 host thread，每个线程运行一个独立 Runtime。每个实例拥有自己的 LocalIdentity、ValueStore 和 Api，只共享 MockCluster 的 mailbox/abort 状态，不共享 Api::Value。

多个 Runtime 之间不设置逐 op barrier。它们可以以不同速度推进，仅通过 CommAction 和最终测试 join 协调，从而覆盖 destination/source 到达顺序、并发 wait 和 execution-group abort。

Runtime 启动时至少检查：

- world size；
- local rank；
- plan ID/version；
- 本地设备数量；
- 编译目标标识。

不设计复杂 capability 协商。

## 12. Fail-fast

Api 或 Runtime 发现任何错误时抛出异常。顶层只捕获一次：

~~~cpp
try {
    runtime.run(plan);
} catch (const std::exception &error) {
    print_error_with_runtime_context(error);
    flush_logs();
    api.abort_all(EXIT_FAILURE);
}
~~~

错误上下文至少包括：

- plan ID；
- op ordinal/kind；
- input/output ValueId；
- TransferId；
- source/destination Place；
- local rank；
- Api 名称；
- 原始错误原因。

MPI Api 使用 MPI_Abort。Vec/Mock Api 抛出 ClusterPanic 或终止模拟 execution group。

不设计：

- Failed value 后继续执行；
- retry；
- rollback；
- cancel；
- timeout recovery；
- rank recovery；
- 部分结果返回。

## 13. Runtime 类建议

首期只需要：

~~~text
RuntimePlan
PlanVerifier
ValueStore<Api>
SequentialRuntime<Api>
VecApi
MockCommunicationApi
~~~

测试侧另外需要：

~~~text
MockCluster
MultiRuntimeHarness
SequentialReferenceExecutor
DiffMap / RunArtifact
~~~

这些类型不进入 Runtime-facing Api。

如果计算和通信由同一 Api bundle 提供，可以进一步简化为：

~~~text
VecApi
PoseidonApi
~~~

不要求在 Runtime ABI 中拆出 ObjectAdapter、BufferTransport 或 Topology。
