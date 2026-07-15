# Runtime 设计

## 1. 定位

Runtime 是可执行计划的纯执行器。它不做这些事：

- 不做设备分配；
- 不读取完整的拓扑模型；
- 不选择通信路由；
- 不选择用 MPI、NCCL 还是 CUDA；
- 不决定实现降级；
- 不修复非法计划；
- 不做重试、取消或恢复。

它做的事只有：验证计划、管理值的 Ready/Pending 状态、按顺序调用 Api，任何错误发生时打印上下文然后终止整个执行组。

## 2. 依赖关系

~~~text
可执行计划
    |
    v
Runtime<Api>
├── PlanVerifier        计划验证器
├── ValueStore          值存储
└── SequentialExecutor  顺序执行器
        |
        v
Api
├── Value               值类型
├── encode_plaintext    把 float64 slots 编码成 Host plaintext
├── 计算函数
├── communicate_async   发起异步通信
├── wait                等待通信完成
└── abort_all           终止整个执行组
~~~

Runtime 通过模板使用 Api，不理解值的内部结构：

~~~cpp
template <class Api>
class Runtime {
public:
    using Value = typename Api::Value;
    using CommHandle = typename Api::CommHandle;
};
~~~

## 3. Runtime 的输入

可执行计划已经包含：固定的计划/版本 ID、目标 capability、OperatorSpec 的 id/版本/指纹、rescale/boot 模式、每条指令的稳定序号、Encode/纯计算/通信指令、可选的明文数据包引用、来源/目标的物理 Place、每个值的 context/level/`scale_log2`/NTT/分量数、可选的通信 hint、初始化/执行/收尾三个阶段的划分、编译目标要求的节点数和本地设备数。

除了计划,每个 Runtime 实例启动时还要收到三样本地输入：本 rank 的身份、由调用方提供的本 rank external inputs、以及可选的明文数据包目录。

~~~cpp
struct LocalIdentity {
    int rank;
};

struct RuntimeInputs {
    LocalIdentity identity;
    std::unordered_map<ValueId, Api::Value> external_inputs;
    std::optional<std::filesystem::path> plaintext_bundle_dir;
};
~~~

`external_inputs` 只放 ValueDesc 位于本 rank Host 的调用方参数,不能拿它传模型权重。`plaintext_bundle_dir` 是部署路径,所以不写进计划 JSON:同一个逻辑数据包在不同节点上的路径可以不同。计划含 bundle Encode 时,每个 rank 都必须拿到一个完整、可读且指纹相符的数据包目录;没有 bundle Encode 时不需要它。

启动时如果实际节点数、本地设备数、外部输入集合或数据包不符合计划，直接报错终止。Runtime 不搜索其他目录,也不在缺文件时下载数据。

## 4. ValueStore（值存储）

首期只维护两种状态：

~~~cpp
enum class ValueState {
    Ready,    // 数据已就绪，可以直接用
    Pending   // 通信还没完成
};
~~~

概念上的数据结构：

~~~cpp
struct ReadyEntry {
    Place place;
    Api::Value value;
};

using PendingGroupId = std::uint64_t;

struct PendingEntry {
    Place place;
    PendingGroupId group_id;   // 属于哪一次通信
    std::size_t output_slot;   // 是那次通信的第几个输出
};

struct PendingGroup {
    std::vector<ValueId> local_output_ids;
    Api::CommHandle handle;
};
~~~

值存储严格按 ValueId 寻址：一个 ValueId 对应一个 ReadyEntry 或 PendingEntry，每个条目只有一个 Place。一次通信可能在本 rank 产生多个输出，这些输出共享同一个通信句柄，但各自通过 `output_slot` 对应到唯一的位置。等待这次通信时，runtime 一次性把它在本 rank 的全部输出安装成 Ready。

查一个不存在的值属于致命错误。异步通信失败由 `Api.wait` 抛异常并立即终止，所以不需要一个长期存在的 Failed 状态。

## 5. 加载与验证

执行前的检查清单：

- 格式/版本、目标 capability 和 OperatorSpec/profile；
- ValueId 唯一定义，且每个 ValueId 恰好绑定一个 Place；
- 不允许"先用后定义"；
- 指令的参数个数和值类型正确；
- Encode 只在 initialization,输出是 Host plaintext,payload 两种形态严格二选一;
- inline Encode 的 float64 数组合法;bundle manifest 中所有 blob 的文件、长度、哈希和有限值都通过 preflight,每个 Encode 引用的 content 和 slot 容量也分别合法;
- 每个值都有完整的 CKKS 元信息,`level` 和 `scale_log2` 都是非负整数；
- 计算指令必须是纯计算，且只有一个执行 Place；该 Place 必须支持这个算子和实现模式；
- 计算指令的输入已在该 Place 上物化；
- Rescale/Boot 的目标 level、`target_scale_log2`、目标分量数与输出 ValueDesc 一致；
- `decrypt_reencrypt` Boot 只能在 Host,并且计划声明了所需的 CPU 能力和密钥；
- 通信动作的输入输出关系符合其类型；
- Transfer/Replicate 前后 CKKS 元信息一致；
- Replicate 的每个目标对应一个不同的输出 ValueId；
- 来源/目标 Place 属于编译目标；
- 传输 ID 唯一；
- 节点数和本地设备数符合目标；
- 调用方的 external_inputs 都已绑定；
- 所有 rank 加载的是同一份计划;使用 bundle 时,各自目录中的 id/version/fingerprint 和完整 blob 集合也一致。

验证器只拒绝非法计划，不会插入传输或重新分配。

## 6. Encode 与计算指令的执行

Encode 没有普通 SSA 输入。Runtime 先从指令内联数组或 bundle `content` 得到 float64 slots,再按输出 ValueDesc 调 Api:

~~~cpp
execute_encode(op):
    desc = value_desc(op.output)
    assert desc.kind == Plaintext
    assert desc.place.kind == Host

    if desc.place.rank != local_identity.rank:
        return

    slots = resolve_inline_or_bundle_payload(op.payload)
    output = api.encode_plaintext(desc, slots)
    value_store.define_ready(op.output, desc.place, output)
~~~

所有 rank 都按同一顺序解释 Encode,但只有输出 Host Place 所属的 rank 真正执行;其他 rank 明确跳过。同一个 `content` 可以被多个 Encode 引用。Runtime 可以缓存已经校验和读取的原始 slots,但每个 Encode 仍按自己的输出 ValueDesc 单独编码和定义 ValueId。

### 普通计算

伪代码：

~~~cpp
execute_compute(op):
    assert op 是纯计算
    assert op.place 属于本 rank

    inputs = []
    for input in op.inputs:
        inputs.push(ensure_ready(input, op.place))

    output = api.compute(op, inputs)
    value_store.define_ready(op.output, op.place, output)
~~~

计算函数的完成契约（详见架构文档第 13 节）：

> 计算函数返回后，输出对同一个 Api 实例上后续发起的调用保证可见（包括交给 `communicate_async`）。指令执行期间阻塞 host 的同步点只有 `wait`；发布最终输出前调用 `synchronize`。

也就是说，Api 内部可以异步地启动 GPU kernel，用 stream/event 保证调用之间的先后顺序，而不必每个算子都同步一次。Runtime 不接触 CUDA event，也感知不到这层异步。

计算不等于“只能在 Device 上算”。目标 capability 可以允许某些算子在 Host 执行。例如 GPU boot 的 CPU 模拟在可执行计划里是 Device→Host Transfer、Host 上的 `Boot(implementation=decrypt_reencrypt)`、Host→Device Transfer。Runtime 只是按顺序执行这三条指令,不会在原生 GPU Boot 失败后临时拼出这条路径。

## 7. 通信动作的执行

所有 rank 按相同顺序解释同一个通信动作：

~~~cpp
execute_comm(action):
    local_inputs = 收集本 rank 拥有的源值

    group = register_pending_group(
        api.communicate_async(action, local_inputs))
    把本 rank 的每个输出 ValueId 登记到这个 group
    group 保留到收尾阶段
~~~

对 Replicate，runtime 按位置建立映射：`action.outputs[i]` 属于 `action.destinations[i]`。同一个句柄可以完成多个输出，但一个输出 ValueId 不会对应多个目标。

每个 rank 在这条指令里扮演什么角色，由物理 Place 和本地 rank 直接算出：

- 源在本 rank：提供源值；
- 目标在本 rank：Api 准备本地接收和输出；
- 源和目标都在本 rank：Api 可以直接本地拷贝；
- 都不在本 rank：Api 返回一个空句柄（什么都不做）。

Runtime 不知道 `communicate_async` 底下调的是 send、recv、broadcast、gather、memcpy、MPI 还是 NCCL。

## 8. ensure_ready（用前等待）

消费一个输入时：

~~~cpp
Value &ensure_ready(ValueId id, Place expected_place):
    entry = value_store.lookup(id)
    assert entry.place == expected_place

    if entry 是 Ready:
        return entry.value

    if entry 是 Pending:
        group = pending_groups.lookup(entry.group_id)
        outputs = api.wait(group.handle)
        按 slot 把这批输出安装成 Ready
        return value_store.lookup_ready(id).value

    panic("值不存在")
~~~

`expected_place` 只用来检查消费方的本地性，不参与寻址。等待只发生在真正要用数据的时候；一次 `wait` 可以同时把同一个通信动作的多个输出变成 Ready。不依赖这个值的指令可以继续执行，直到碰上自己的 Pending 输入。

首期不提供"探测是否完成""推进进度""取消"这类接口。Api 内部想用进度线程、`MPI_Test`、CUDA event 都可以，对 runtime 不可见。

**关于单 rank 多卡的并行性**：runtime 单线程地**发起**指令，但这不等于执行是串行的。异步实现的 Api（GPU 后端）在 `compute` 里提交任务即返回，多张卡的任务同时执行——发起串行、执行并行。真正会损失重叠的只有两种情况：`wait` 阻塞了 host 线程、挡住了后面本可独立发起的指令（靠编译器排序解决：通信早发起、消费晚安排）；以及 Api 本身是同步实现（VecApi 的默认同步模式就是，多卡在它之下纯串行，但不影响正确性验证；VecApi 另提供每设备一个工作线程的异步模式来模拟真实并行时序，接口不变，见测试文档 2.1 节）。详见架构文档第 13 节。

## 9. 三个执行阶段

### 初始化

- 绑定调用方提供的 Host external_inputs；
- 执行 Encode,从 inline 或 bundle 数据生成 Host 明文常量；
- 执行编译器生成的 Host 到 Device 的 Transfer；
- 上传 Api 需要的上下文和密钥；
- 等待初始化涉及的值全部就绪。

首期采用一次性预加载：执行阶段需要的外部值和明文全部提前搬好，并保留到运行结束。

### 执行

- 顺序解释计算和通信指令；
- 按 rank/Place 决定本地要不要动手；
- 消费前调用 `ensure_ready`。

### 收尾

- 执行 Device 到 Host 的输出搬运；
- 等待所有还没收尾的发送句柄；
- 在发布本 rank 的最终输出前调用 Api 的 `synchronize`，暴露尚未完成的异步计算错误；
- 返回或打印结果；
- 若开启了"运行后逐指令对比"模式，额外等待本 rank 所有对比点的 Pending 输出；
- 测试模式可以在通信全部完成后、释放之前，导出仅供测试用的运行产物（RunArtifact）；
- 统一释放所有值。

## 10. 生命周期

为避免过度设计，首期不做"最后一次使用后回收"：

- 所有 Ready 的值保留到收尾阶段；
- 所有发送句柄保留到收尾阶段；
- 收尾完成全部等待后统一释放。

这会增加内存占用，但直接保证了发送源不会在异步发送完成前被释放。

逐指令对比正是利用了这个特性：所有指令发起完之后，做完收尾等待，把还活着的值拷贝出来，等 runtime 停止后离线比较。正常执行路径不插入逐指令的观察点或同步；不开这个测试选项就没有任何额外开销。RunArtifact 是测试设施，不属于 Runtime/Api 的稳定接口，开启它的测试不用于性能计时。

未来数据规模上去后，由编译器的内存规划插入 Prefetch 和 Release/Evict，runtime 依旧只执行显式动作。

## 11. 同一份全局计划

本 demo 采用 SPMD 方式（Single Program Multiple Data，所有节点跑同一个程序，各自处理自己的部分）：

- 所有 rank 读同一份计划，指令顺序完全一致；
- Encode 按输出 Host Place 的 rank 决定由谁执行,其他 rank 跳过；
- 计算指令按 `op.place.rank` 决定本 rank 是否执行；
- 通信动作在所有 rank 上都会被解释，Api 按本地角色执行或跳过；
- 集合通信的顺序由全局计划保证一致。

一个 rank 对应一个 Runtime 实例。真实 MPI 部署是一个进程一个 runtime；模拟多 rank 测试则在同一进程里开多个线程，每个线程跑一个独立 runtime。每个实例有自己的 LocalIdentity、ValueStore 和 Api，只共享 MockCluster 的消息队列和终止状态，不共享任何 `Api::Value`。

多个 runtime 之间不加逐指令同步，它们可以以不同速度推进，只通过通信动作和最终的测试汇合点协调。这样才能测出"接收方先到/发送方先到""并发等待""全组终止"这些真实场景。

Runtime 启动时至少检查：节点总数、本地 rank、计划 ID/版本、本地设备数、编译目标标识、capability 版本、OperatorSpec id/版本/指纹、rescale/boot 模式。不设计复杂的能力协商；不匹配就直接终止。

## 12. Fail-fast

Api 或 runtime 发现任何错误就抛异常，顶层只捕获一次：

~~~cpp
try {
    runtime.run(plan);
} catch (const std::exception &error) {
    print_error_with_runtime_context(error);
    flush_logs();
    api.abort_all(EXIT_FAILURE);
}
~~~

错误上下文至少包括：计划 ID、指令序号和类型、输入输出 ValueId、传输 ID、来源/目标 Place、本地 rank、Api 名称、原始错误原因。

MPI 版 Api 用 `MPI_Abort`；Vec/Mock 版抛出 ClusterPanic 或终止模拟执行组。

不设计：出错后继续执行、重试、回滚、取消、超时恢复、节点恢复、返回部分结果。

## 13. 类结构建议

首期只需要：

~~~text
RuntimePlan                  可执行计划
PlanVerifier                 计划验证器
ValueStore<Api>              值存储
SequentialRuntime<Api>       顺序执行 runtime
VecApi                       明文计算参考实现
MockCommunicationApi         模拟通信实现
~~~

测试侧另外需要（不进入 Runtime 对外接口）：

~~~text
MockCluster                  模拟集群（消息队列 + 终止状态）
MultiRuntimeHarness          多 runtime 测试驱动
SequentialReferenceExecutor  单设备顺序参考执行器
DiffMap / RunArtifact        对比映射 / 运行产物
~~~

如果计算和通信由同一个 Api 打包提供，可以进一步简化为 `VecApi` 和 `PoseidonApi` 两个实现。不要求在 Runtime 接口层拆出对象适配器、传输层或拓扑这类抽象。
