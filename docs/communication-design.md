# 通信设计

## 1. 设计目标

Runtime-facing 通信接口应当尽量小，只表达编译器已经确定的通信语义。具体对象布局、buffer、MPI/NCCL/CUDA 调用、fallback 和格式转换都由 Api 兼容层处理。

本设计面向：

- 编译期固定拓扑；
- 所有 rank 执行同一份物理计划；
- Runtime 只解释计划；
- Api 可以由 Vec、Poseidon 或其他实现提供；
- 错误直接暴露并终止整个集群；
- 不追求 retry、cancel、恢复或动态调优。

## 2. 编译期拓扑与物理 Place

编译器知道完整目标拓扑，并把通信 source/destination 固化到 executable plan。

~~~cpp
enum class PlaceKind {
    Host,
    Device
};

struct Place {
    PlaceKind kind;
    int rank;
    int index;
};
~~~

示例：

~~~text
Host(rank=0)
Device(rank=0, index=0)
Device(rank=1, index=3)
~~~

Host 是一等 Place，但不能使用 Device(rank=0,index=0) 来代替 Host。Api 必须能够区分 HostToDevice、DeviceToHost、DeviceToDevice 和 HostToHost。

Runtime 不需要 TopologyModel，也不选择路由。它只读取物理 Place，并根据 local rank 判断当前通信角色。

## 3. 通信语义

建议 executable plan 使用通用通信 action：

~~~cpp
enum class CommKind {
    Transfer,
    Replicate,
    Gather,
    Scatter,
    AllGather
};
~~~

语义：

~~~text
Transfer:
    一个输入 ValueId -> 一个目标位置上的新 ValueId

Replicate:
    一个输入 ValueId -> N 个数学等价但身份不同的输出 ValueId
    outputs[i] 位于 destinations[i]

Gather:
    多个输入 ValueId -> 一个目标位置上的组合结果 ValueId

Scatter:
    一个输入 ValueId -> N 个位于各自目标位置的切片 ValueId

AllGather:
    多个输入 ValueId -> 每个目标获得一个独立的组合结果 ValueId
~~~

每种 CommKind 必须先定义明确的 input/output 数学关系。Api fallback 只能改变实现方式，不能改变结果。

Executable plan 严格采用单位置物理 SSA：每个 output ValueId 恰好属于一个 Place。数学等价的多个物理 materialization 必须使用不同 ValueId，不能让同一个 ValueId 同时对应多个 destination。

首期只要求实现 Transfer 和 Replicate。Gather 等操作等其 result layout 明确后再加入。

## 4. 编译器 Hint

通信 action 可以携带实现 hint：

~~~cpp
enum class CommHint {
    Auto,
    PointToPoint,
    Collective,
    Broadcast,
    GatherPrimitive,
    Tree,
    Ring,
    HostStaged
};
~~~

例如：

~~~text
CommKind = Replicate
CommHint = Broadcast
~~~

表示编译器建议使用 broadcast，但不强制具体库函数。

Api 可以：

- 使用 NCCL Broadcast；
- 使用 MPI_Ibcast；
- 使用多个 MPI point-to-point；
- 使用多个 CUDA P2P；
- 经 Host staging；
- 使用其他数学等价实现。

如果首选方式不可用，Api 自行 fallback。若不存在等价实现，则抛出 fatal error。

例如语义为 Gather 时，编译器可以给出 GatherPrimitive hint；Api 不支持原生 gather 时，可以退化为多个 point-to-point receive 加本地组装。

为了分析实验结果，Api 可以在 debug trace 中记录实际采用的实现，但 Runtime 不参与选择。

## 5. CommAction

~~~cpp
using TransferId = std::uint64_t;

struct CommAction {
    TransferId id;
    CommKind kind;
    CommHint hint;

    std::vector<ValueId> inputs;
    std::vector<ValueId> outputs;

    std::vector<Place> sources;
    std::vector<Place> destinations;

    std::vector<ValueType> output_types;
};
~~~

CommAction 来自编译器，Runtime 不修改。

output_types 是 SSA 类型信息，用于 Api 在 destination 分配或构造 Value。它不是底层 buffer descriptor。

所有 CommKind 都遵守以下物理结果约束：

- inputs[i] 位于 sources[i]；
- outputs[i] 位于 destinations[i]；
- outputs、destinations 和 output_types 数量一致；
- 每个 output ValueId 唯一，且只能绑定一个 destination。

因此 Replicate 到 N 个 destination 时必须定义 N 个 output ValueId。一个底层 collective request 可以共同完成这些输出，但不能合并它们的 SSA identity。

## 6. 最小 Api

Runtime 模板化在 Api 上：

~~~cpp
template <class Api>
class Runtime {
public:
    using Value = typename Api::Value;
    using CommHandle = typename Api::CommHandle;
};
~~~

通信接口只要求：

~~~cpp
class Api {
public:
    using Value = /* Api-defined */;
    using CommHandle = /* Api-defined */;

    CommHandle communicate_async(
        const CommAction &action,
        std::span<const Value> local_inputs);

    std::vector<Value> wait(CommHandle &handle);

    [[noreturn]] void abort_all(int exit_code);
};
~~~

所有 rank 在解释到同一个 CommAction 时调用 communicate_async：

- source rank 的 local_inputs 包含本地 source Value，顺序与该 rank 在 `action.inputs` 中对应的 input index 顺序一致；
- destination rank 的 local_inputs 可以为空；
- 同一 rank 同时管理 source 和 destination 时，Api 可以直接执行本地 copy；
- 无关 rank 得到一个空/no-op handle；
- Api 根据 action 和本地 rank 发布实际 send、receive、copy 或 collective。

wait 返回当前 rank 上由该 action 产生的本地 output Values，顺序与该 rank 在 `action.outputs` 中对应的 output index 顺序一致。Runtime 使用这些 index 把结果安装到各自独立的 ValueId；返回数量、类型、Place 或顺序不匹配都属于 fatal error。

Source 侧没有 output 时，handle 仍用于保证异步发送在 run 结束前完成。

## 7. 为什么不公开 Object 抽象

Runtime 不需要知道实际对象结构。Api::Value 已经是 backend 自己定义的类型：

~~~text
VecApi:
    variant<VecPlaintext, VecCiphertext>

PoseidonApi:
    variant<GpuPlaintextData, GpuCiphertextData>
~~~

Poseidon Api 可以在内部：

- 展开多个 ciphertext buffer；
- 分配 destination；
- 复制 metadata；
- 转换 CPU/GPU residue；
- 使用 MPI/NCCL/CUDA；
- 管理 pinned host buffer。

这些都不是 Runtime 公共接口的一部分。

因此首期不引入：

- ObjectHandle；
- ObjectDescriptor；
- ObjectAdapter；
- BufferView；
- MemoryKind；
- 通用序列化接口。

如果未来确实需要让一个通用 MPI transport 独立复用多种对象布局，再把 ObjectAdapter 作为 Api 内部模块加入，而不是提前进入 Runtime ABI。

## 8. Host 与 Device 传输

HostToDevice、DeviceToHost 等都使用同一个 CommAction：

~~~text
Transfer:
  source = Host(rank=0)
  destination = Device(rank=0,index=0)
~~~

Runtime 不根据 ValueKind 自动判断是否上传。判断依据始终是编译后的 source/destination Place。

Api 可以在一次 HostToDevice 中完成：

- destination allocation；
- pageable -> pinned staging；
- CPU/GPU 表示转换；
- cudaMemcpy；
- metadata materialization。

只要输出数学值和 ValueType 符合计划，就属于合法实现。

## 9. 初始化、执行和结束

Executable plan 建议划分三个逻辑阶段。

### Initialization

- 绑定 Host 输入和常量；
- 把本次 run 需要的值预加载到 Device；
- 上传 API 所需 context/key；
- 等待初始化通信完成。

首期采用 eager preload：所有需要的 plaintext 和输入对象在执行前搬到目标 Device，并保留到 run 结束。

### Execution

- 执行 compute；
- 执行中间 Transfer/Replicate；
- consumer 使用 Pending output 前调用 wait。

### Finalization

- 下载要求的结果；
- wait 所有 outstanding source handles；
- 打印或返回结果；
- 释放对象。

## 10. 大型 Plaintext 与未来 Memory Planning

大量 plaintext 可能无法全部驻留 GPU。后续由编译器增加 memory planning：

~~~text
Topology-aware placement
  -> communication materialization
  -> memory planning
  -> prefetch/release insertion
~~~

未来 action：

~~~text
Prefetch:
    提前执行 Host -> Device materialization

Release/Evict:
    释放某个 Device ValueId 的物理 materialization；数学等价的 Host 或其他 Place ValueId 仍可保留
~~~

Runtime 不实现动态缓存。它只按计划执行 Prefetch 和 Release。

这使编译器可以利用：

- 显存容量；
- last-use；
- HostToDevice 带宽；
- plaintext 重用次数；
- 多 Device 物理 materialization 成本；
- 通信与计算 overlap。

## 11. Api Fallback

Fallback 完全由 Api 处理：

~~~text
Replicate + Broadcast
    -> NCCL Broadcast
    -> MPI_Ibcast
    -> 多个 point-to-point
    -> Host staging

Gather + Collective
    -> 原生 gather
    -> 多个 receive + 本地组装
~~~

约束：

1. fallback 必须数学等价；
2. output ValueType 必须一致；
3. source/destination Place 不得被悄悄改变；
4. 若无等价 fallback，立即报错；
5. debug 模式可记录实际实现，方便解释性能结果。

Runtime 不需要 capability query，也不参与 fallback。

## 12. 异步与等待

communicate_async 必须在不等待远端完成的情况下返回，具体 request/event 隐藏在 CommHandle 中。

Runtime 保存：

~~~text
ValueId -> Ready Value
ValueId -> Pending (CommHandle, output slot)
~~~

同一 CommHandle 可以被多个不同 output ValueId 引用，但每个 ValueId 都只对应一个 Place 和一个 output slot。

当 compute consumer 需要某个 output 时：

~~~text
Pending
  -> Api.wait
  -> Ready Value
~~~

为保持接口简单，Compute Api 首期采用以下契约：

> 计算函数返回后，输出已经可以安全交给 communicate_async。

Poseidon 兼容层如果内部使用异步 GPU kernel，应自行同步或连接内部 stream/event。

## 13. Source 生命周期

Runtime 首期不做精细 last-use 回收：

- 所有 Value 保留到 run 结束；
- 所有 source CommHandle 保留到 Finalization；
- Finalization 逐个 wait；
- 完成后统一释放。

这会增加内存占用，但能够避免 source 在异步发送完成前被释放。后续 memory planning 再加入精细回收。

## 14. 无死锁约束

正常路径满足：

1. 所有 rank 执行相同 CommAction 顺序；
2. 每个 action 的物理 source/destination 在编译时确定；
3. communicate_async 不等待远端完成；
4. destination 在 action 被解释时立即发布接收；
5. consumer 只等待之前已经启动的 handle；
6. collective 的参与者和顺序一致；
7. compute mask 不跳过通信 action；
8. 任意错误调用 abort_all。

具体 MPI/NCCL 的匹配、group 和 progress 由 Api 兼容层负责。

## 15. Fail-fast

Api 的 compute、communicate_async 或 wait 失败时直接抛出异常。

Runtime 顶层：

~~~cpp
try {
    runtime.run(plan);
} catch (const std::exception &error) {
    print_runtime_context(error);
    flush_logs();
    api.abort_all(EXIT_FAILURE);
}
~~~

诊断至少包含：

- op ordinal 和 kind；
- ValueId；
- TransferId；
- source/destination Place；
- local rank；
- Api 名称；
- 底层错误。

不设计：

- retry；
- cancel；
- timeout recovery；
- rollback；
- rank recovery；
- 部分继续执行。

## 16. 当前 comm_interface.hpp 的调整方向

当前 send_async、broadcast_async、gather_async、receive_async 和 receive_fence 能表达早期想法，但存在：

- T 与 size 的语义不清；
- destination 分配未定义；
- transfer_id 输出引用；
- receive_fence 返回 optional<T>；
- Host 被当成 device 0；
- API 细节会不断泄漏进接口；
- 不能统一表达未来通信 action 和 hint。

建议替换为：

~~~text
communicate_async(CommAction, local_inputs)
wait(CommHandle)
abort_all(exit_code)
~~~

send、receive、broadcast、gather、MPI request、CUDA event 等都留在 Api 实现内部。

## 17. Mock Api

每个模拟 rank 使用独立的 MockCommunicationApi 和 Runtime。Api 实例只共享线程安全的 MockCluster 传输状态，不共享 ValueStore 或 Api::Value。

MockCommunicationApi 实现同一接口：

- Host 和 Device 都是逻辑 Place；
- communicate_async 在线程安全的 mailbox 中发布 send/receive，并创建 pending handle；
- wait 推进模拟事件并返回输出；
- 支持 Transfer 和 Replicate；
- 支持 Broadcast hint fallback 到 point-to-point；
- 支持固定或 seeded delay，以改变各 Runtime 的到达顺序；
- 支持指定 action 失败；
- 传输时复制 payload/metadata，禁止通过共享 C++ 对象完成“零通信”传值；
- abort_all 设置集群终止状态、唤醒所有 wait，并使所有模拟 Runtime 停止。

多 rank 测试需要一个 host thread 对应一个 Runtime，但不要求真实网络、独立 progress thread 或复杂 progress engine。所有 Runtime 只在测试开始和结束处汇合，不增加逐 CommAction barrier。
