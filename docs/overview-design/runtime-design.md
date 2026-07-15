# Runtime 设计

> 本文描述 Runtime 的**目标形态**。当前代码还没有 Encode/bundle、Host compute 和 OperatorSpec/密钥完整校验，`run()` 也不接收 bundle 目录。准确边界见[实现状态](implementation-status.md)和当前 [`runtime/runtime.hpp`](../../runtime/runtime.hpp)。

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
└── SequentialRuntime   顺序执行器
        |
        v
Api
├── Value               值类型
├── preflight           计划原始字节摘要、target/context/密钥启动检查
├── validate_value      按 ValueDesc 核对实际值
├── encode_plaintext    把 float64 slots 编码成 Host plaintext
├── 计算函数
├── communicate_async   发起异步通信
├── wait                等待通信完成
├── synchronize         发布最终输出前同步
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

可执行计划提供指令序列、每个值的 ValueDesc、目标配置，以及初始化/执行/收尾三个阶段。完整字段和约束以 [RuntimePlan V1](../runtime-plan/v1/specification.md)为准，这里只说明 Runtime 启动时还需要哪些本地信息。

除了计划，每个 Runtime 实例还需要实际运行环境、调用方输入、可选的明文数据包目录和 OperatorSpec 副本。Api 应在创建 Runtime 前完成 context 和密钥配置；Runtime 只检查，不在 initialization 中临时补配置。

~~~cpp
struct LoadedRuntimePlan {
    RuntimePlan plan;
    std::string source_sha256;
};

struct LoadedOperatorSpec {
    OperatorSpec spec;
    std::string source_sha256;
};

struct RuntimeEnvironment {
    int rank;
    int world_size;
    int local_device_count;
    const LoadedOperatorSpec *operator_spec;
};

struct RuntimeResources {
    const LoadedOperatorSpec &operator_spec;
    std::optional<std::filesystem::path> plaintext_bundle_dir;
    bool skip_artifact_digest_checks = false;
};
~~~

`LoadedRuntimePlan` 和 `LoadedOperatorSpec` 都由各自 reader 从文件完整原始字节计算 `source_sha256`,不是调用方随意填写的摘要。RuntimePlan JSON 本身不保存自己的摘要。

- `external_inputs` 放本 rank Host 上由调用方在这次运行中传入的参数。随计划固定发布的权重应由 Encode 产生；如果某份权重本来就是每次调用动态传入的参数，它仍可以是 external input。
- `plaintext_bundle_dir` 是本机部署路径，不写进计划 JSON。存在 bundle Encode 时，每个 rank 都读取 manifest，但只要求本地存在该 rank 实际使用的 blob；否则不需要这个目录。
- Runtime 读取计划、OperatorSpec 和 manifest 时保留各自完整原始字节的 SHA-256。计划自身不保存摘要;OperatorSpec 和 manifest 的摘要由计划引用。
- Runtime 把 `plan_source_sha256`、target、OperatorSpec 和密钥要求交给 Api 做 `preflight`；Api 用自己的实际配置逐项核对。Mock/MPI 参考后端可显式接受 placeholder spec，PoseidonApi 必须拒绝。
- `skip_artifact_digest_checks` 默认是 `false`。调试时显式设为 `true` 只跳过计划跨 rank、OperatorSpec 和 manifest 的原始字节摘要比较,不跳过其他验证。

启动时如果实际节点数、本地设备数、外部输入集合或数据包不符合计划，直接报错终止。Runtime 不搜索其他目录,也不在缺文件时下载数据。

当前代码使用 `SequentialRuntime(rank, world_size, local_devices, api)`，并调用 `run(loaded_plan, resources, local_inputs, diff_mode)`。

## 4. ValueStore（值存储）

首期只维护两种状态：

~~~cpp
enum class ValueState {
    Ready,    // Runtime 已拿到可交给 Api 的值句柄
    Pending   // 通信还没产出本地值句柄
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

值存储严格按 ValueId 寻址：一个 ValueId 只对应一个 ReadyEntry 或 PendingEntry，也只对应一个 Place。

一次通信可以在本 rank 产生多个输出。它们共享同一个通信句柄，但各自用 `output_slot` 对应自己的 ValueId。等待该句柄时，runtime 一次性把这组本地输出全部安装成 Ready。

查一个不存在的值属于致命错误。异步通信失败由 `Api.wait` 抛异常并立即终止，所以不需要一个长期存在的 Failed 状态。

## 5. 加载与验证

执行前的检查清单：

- 格式/版本、目标 capability 和 OperatorSpec/profile；
- ValueId 唯一定义，且每个 ValueId 恰好绑定一个 Place；
- 不允许"先用后定义"；
- 指令的参数个数和值类型正确；
- Encode 只在 initialization，输出是 Host plaintext，payload 两种形态严格二选一；
- inline Encode 的 float64 数组非空、有限且不超过 slot 容量；
- bundle manifest、全部 blob 文件、长度、哈希和 float64 字节都通过 preflight；
- 每个 bundle Encode 引用的 `content` 存在，并按自己的输出描述检查 slot 容量；
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
- 默认模式下所有 rank 加载的计划原始字节 SHA-256 相同;使用 bundle 时,各自目录中的 id/version/manifest_sha256 和完整 blob 集合也一致。

验证器只拒绝非法计划，不会插入传输或重新分配。

绑定 external input 时也要检查实际值，不能只相信调用方：

~~~cpp
for (id, value) in inputs.external_inputs:
    desc = value_desc(id)
    api.validate_value(value, desc)
    value_store.define_ready(id, desc.place, value)
~~~

## 6. Encode 与计算指令的执行

Encode 没有普通 SSA 输入。Runtime 先从指令内联数组或 bundle `content` 得到 float64 slots,再按输出 ValueDesc 调 Api:

~~~cpp
execute_encode(op):
    desc = value_desc(op.output)
    assert desc.kind == Plaintext
    assert desc.place.kind == Host

    if desc.place.rank != local_rank:
        return

    slots = resolve_inline_or_bundle_payload(op.payload)
    output = api.encode_plaintext(desc, slots)
    api.validate_value(output, desc)
    value_store.define_ready(op.output, desc.place, output)
~~~

所有 rank 都按同一顺序解释 Encode，但只有输出 Host Place 所属的 rank 真正执行，其他 rank 明确跳过。

同一个 `content` 可以被多个 Encode 引用。每条 Encode 仍按自己的输出 ValueDesc 单独编码，并定义自己的 ValueId。

### 普通计算

伪代码：

~~~cpp
execute_compute(op):
    assert op 是纯计算

    if op.place.rank != local_rank:
        return

    inputs = []
    for input in op.inputs:
        inputs.push(ensure_ready(input, op.place))

    output = api.compute(op, inputs)
    api.validate_value(output, value_desc(op.output))
    value_store.define_ready(op.output, op.place, output)
~~~

计算函数的完成契约（详见架构文档第 13 节）：

> 计算函数返回后，输出对同一个 Api 实例上后续发起的调用保证可见（包括交给 `communicate_async`）。`wait` 是 Runtime 对异步通信的唯一显式等待点;同步计算、文件读取或 Encode 本身仍可能占用调用线程。发布最终输出前调用 `synchronize`。

也就是说，Api 内部可以异步地启动 GPU kernel，用 stream/event 保证调用之间的先后顺序，而不必每个算子都同步一次。紧接着调用的 `validate_value` 只检查句柄自带的不可变元信息，不能等待 kernel 或物化 slots。Runtime 不接触 CUDA event，也感知不到这层异步。

Host compute 必须由计划明确指定。GPU boot 的 CPU 模拟就是 Device→Host Transfer、Host Boot、Host→Device Transfer 三条显式指令；Runtime 不会在原生 GPU Boot 失败后临时拼出这条路径。

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
        for (output_id, output) in 按 slot 对应的本地输出:
            api.validate_value(output, value_desc(output_id))
            value_store.define_ready(output_id, value_desc(output_id).place, output)
        return value_store.lookup_ready(id).value

    panic("值不存在")
~~~

`expected_place` 只用来检查消费方的本地性，不参与寻址。等待只发生在真正要用数据的时候；一次 `wait` 可以同时把同一个通信动作的多个输出变成 Ready。不依赖这个值的指令可以继续执行，直到碰上自己的 Pending 输入。

首期不提供"探测是否完成""推进进度""取消"这类接口。Api 内部想用进度线程、`MPI_Test`、CUDA event 都可以，对 runtime 不可见。

**关于单 rank 多卡的并行性**：

- Runtime 的单线程只负责按计划发起指令；
- 异步 Api 可以让不同设备上的任务并行执行；
- `wait` 排得太早会挡住后续工作，因此编译器应尽早发起通信、尽量推迟消费；
- 当前 VecExecutor 的同步模式没有重叠，异步模式用每设备一个工作线程模拟真实时序。

完整契约见架构文档第 13 节和测试文档第 2.1 节。

## 9. 三个执行阶段

### 初始化

- 绑定调用方提供的 Host external_inputs；
- 执行 Encode,从 inline 或 bundle 数据生成 Host 明文常量；
- 执行编译器生成的 Host 到 Device 的 Transfer；
- 核对 Api 在启动前配置好的 context、capability 和密钥；
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
- 若开启了"运行后逐指令对比"模式，额外等待本 rank 所有对比点的 Pending 输出；
- 在通信全部完成后复制最终输出和可选的 RunArtifact；
- 释放 Runtime 内部保存的值和句柄；
- 最后把已经独立保存的结果返回给调用方。

## 10. 生命周期

目标实现为避免过度设计，首期不做"最后一次使用后回收"：

- 所有 Ready 的值保留到收尾阶段；
- 所有发送句柄保留到收尾阶段；
- 收尾完成全部等待后统一释放。

这会增加内存占用，但直接保证了发送源不会在异步发送完成前被释放。

> **当前实现：** `SequentialRuntime` 的 `ValueStore` 和通信组是对象成员，`run()` 前后都不会清空。因此一个实例只能执行一次；如果以后支持复用，必须在每次运行前重置全部状态。

逐指令对比正是利用了这个特性：所有指令发起完之后，做完收尾等待，把还活着的值拷贝出来，等 runtime 停止后离线比较。正常执行路径不插入逐指令的观察点或同步；不开这个测试选项就没有任何额外开销。RunArtifact 是测试设施，不属于 Runtime/Api 的稳定接口，开启它的测试不用于性能计时。

未来数据规模上去后，由编译器的内存规划插入 Prefetch 和 Release/Evict，runtime 依旧只执行显式动作。

## 11. 同一份全局计划

本 demo 采用 SPMD 方式（Single Program Multiple Data，所有节点跑同一个程序，各自处理自己的部分）：

- 所有 rank 读同一份计划，指令顺序完全一致；
- Encode 按输出 Host Place 的 rank 决定由谁执行,其他 rank 跳过；
- 计算指令按 `op.place.rank` 决定本 rank 是否执行；
- 通信动作在所有 rank 上都会被解释，Api 按本地角色执行或跳过；
- 集合通信的顺序由全局计划保证一致。

一个 rank 对应一个 Runtime 实例。真实 MPI 部署是一个进程一个 runtime；模拟多 rank 测试则在同一进程里开多个线程，每个线程跑一个独立 runtime。每个实例有自己的 RuntimeEnvironment、ValueStore 和 Api，只共享 MockCluster 的消息队列和终止状态，不共享任何 `Api::Value`。

多个 runtime 之间不加逐指令同步，它们可以以不同速度推进，只通过通信动作和最终的测试汇合点协调。这样才能测出"接收方先到/发送方先到""并发等待""全组终止"这些真实场景。

目标 Runtime 启动时检查节点数、rank、设备数、计划版本、target/capability、OperatorSpec、rescale/boot 模式，以及 Api 的 capability 和密钥配置。不设计能力协商；不匹配就直接终止。

调试选项 `skip_artifact_digest_checks` 由部署方统一设置,不属于 RuntimePlan。开启时 Runtime 必须打印警告,并继续执行全部结构、语义、能力、密钥和 blob 内容检查;各 rank 的选项值不一致仍属于启动错误。

> **当前实现：** 只检查 world size、rank 和本地设备数。

## 12. Fail-fast

Api 或 runtime 发现任何错误就抛异常，顶层只捕获一次：

~~~cpp
try {
    runtime.run(plan, inputs);
} catch (const std::exception &error) {
    print_error_with_runtime_context(error);
    flush_logs();
    api.abort_all(EXIT_FAILURE);
}
~~~

目标错误上下文至少包括：计划 ID、指令序号和类型、输入输出 ValueId、传输 ID、全部来源/目标 Place、本地 rank、Api 名称和原始错误原因。

> **当前实现：** 计算错误不打印全部输入，通信错误只打印第一个来源/目标，加载和 preflight 错误也不一定带指令上下文。

MPI 版 Api 用 `MPI_Abort`；Vec/Mock 版抛出 ClusterPanic 或终止模拟执行组。

不设计：出错后继续执行、重试、回滚、取消、超时恢复、节点恢复、返回部分结果。

## 13. 类结构建议

首期只需要：

~~~text
RuntimePlan                  可执行计划
PlanVerifier                 计划验证器
ValueStore<Api>              值存储
SequentialRuntime<Api>       顺序执行 runtime
VecExecutor                  明文计算内核
MockVecApi + MockCluster     完整的模拟 Api
MpiVecApi                    完整的 MPI Api
~~~

测试侧另外需要（不进入 Runtime 对外接口）：

~~~text
MockCluster                  模拟集群（消息队列 + 终止状态）
run_mock_cluster()           多 runtime 测试驱动
SequentialReferenceExecutor  单设备顺序参考执行器
DiffMap / RunArtifact        对比映射 / 运行产物
~~~

计算和通信由同一个完整 Api 提供:当前是 `MockVecApi`/`MpiVecApi`,未来是 `PoseidonApi`。不要求在 Runtime 接口层拆出对象适配器、传输层或拓扑这类抽象。
