# 通信设计

## 1. 设计目标

面向 runtime 的通信接口应当尽量小，只表达编译器已经确定的通信语义。具体的对象布局、buffer、MPI/NCCL/CUDA 调用、等价的传输降级和 CPU/GPU 对象格式转换，都由 Api 兼容层处理。

本设计的前提：

- 拓扑在编译期固定；
- 所有 rank 执行同一份物理计划；
- Runtime 只解释计划；
- Api 可以由 Vec、Poseidon 或其他实现提供；
- 错误直接暴露并终止整个执行组；
- 不追求重试、取消、恢复或动态调优。

## 2. 编译期拓扑与物理 Place

编译器知道完整的目标拓扑，并把通信的来源和目标写死在可执行计划里：

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

Host 是一等 Place，不允许用 `Device(rank=0, index=0)` 冒充。Api 必须能区分 Host→Device、Device→Host、Device→Device、Host→Host 四种搬运。

Runtime 不需要拓扑模型，也不选路由。它只读取计划里的物理 Place，并根据本地 rank 判断自己在这次通信里的角色。

## 3. 通信语义

可执行计划使用通用的通信动作：

~~~cpp
enum class CommKind {
    Transfer,
    Replicate,
    Gather,
    Scatter,
    AllGather
};
~~~

语义定义：

~~~text
Transfer:   一个输入 -> 目标位置上的一个新 ValueId
Replicate:  一个输入 -> N 个数学上相同、身份不同的输出 ValueId，
            outputs[i] 位于 destinations[i]
Gather:     多个输入 -> 一个目标位置上的合并结果
Scatter:    一个输入 -> N 个切片，各自位于自己的目标位置
AllGather:  多个输入 -> 每个目标各得一份独立的合并结果
~~~

每种通信动作必须先定义清楚输入到输出的数学关系。Api 的实现降级只能改变"怎么传"，不能改变"传出来是什么"，也不能改变 context、level、`scale_log2`、NTT 状态或分量数。

计划严格遵守单位置规则：每个输出 ValueId 恰好属于一个 Place。同一份数据出现在多个位置时，必须用多个不同的 ValueId。

首期只要求实现 Transfer 和 Replicate。Gather 等操作等其结果布局定义清楚后再加。

## 4. 编译器 hint

通信动作可以携带实现提示：

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

例如 `Replicate + Broadcast` 表示编译器建议用广播实现，但不强制具体的库函数。Api 可以用 NCCL Broadcast、`MPI_Ibcast`、多个 MPI 点对点、多个 CUDA 点对点拷贝、经 Host 中转，或任何数学等价的方式。首选方式不可用时 Api 自行降级；不存在等价实现时抛出致命错误。

首期只需要 Auto、PointToPoint、Broadcast 三种；Tree/Ring/HostStaged 等枚举值保留但暂不实现。

为了方便分析实验结果，Api 可以在调试日志里记录实际采用的实现方式，但 runtime 不参与选择。

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

CommAction 来自编译器，runtime 不修改。`output_types` 是 SSA 层的明文/密文类型信息，完整 CKKS 元信息来自对应输出的 ValueDesc。Api 用这些信息在目标端分配或构造值，它们不是底层的 buffer 描述。

所有通信动作都满足：

- `inputs[i]` 位于 `sources[i]`；
- `outputs[i]` 位于 `destinations[i]`；
- outputs、destinations、output_types 三者数量一致；
- 每个输出 ValueId 唯一，且只绑定一个目标。

所以 Replicate 到 N 个目标必须定义 N 个输出 ValueId。底层一次集合通信可以同时完成这些输出，但不能合并它们的 SSA 身份。

## 6. 最小 Api

通信接口要求异步通信、等待、最终值同步和全组终止：

~~~cpp
class Api {
public:
    using Value = /* Api 自己定义 */;
    using CommHandle = /* Api 自己定义 */;

    CommHandle communicate_async(
        const CommAction &action,
        const std::vector<Value> &local_inputs);

    std::vector<Value> wait(CommHandle &handle);

    void synchronize(Value &value);

    [[noreturn]] void abort_all(int exit_code);
};
~~~

所有 rank 解释到同一个 CommAction 时都调用 `communicate_async`：

- 源在本 rank：`local_inputs` 里放本地的源值，顺序和该 rank 在 `action.inputs` 中的下标顺序一致；
- 目标在本 rank：`local_inputs` 可以为空，Api 准备本地接收；
- 源和目标都在本 rank：Api 可以直接本地拷贝；
- 都不在本 rank：返回一个空句柄；
- 具体发 send、recv、broadcast、gather、memcpy 还是集合通信，由 Api 根据 action 和本地角色决定。

`wait` 返回本 rank 上由这次通信产生的输出值，顺序和该 rank 在 `action.outputs` 中的下标顺序一致。Runtime 用这些下标把结果安装到各自的 ValueId；数量、类型、位置或顺序不匹配都是致命错误。

`synchronize` 只在运行结束发布最终输出前调用，使 Api 内部异步计算的异常仍由 Runtime 的 fail-fast 顶层捕获。它不用于逐指令同步。

发送侧没有输出时，句柄仍然要保留——用来保证异步发送在运行结束前真正完成。

## 7. 为什么不公开对象抽象

Runtime 不需要知道数据对象的内部结构。`Api::Value` 已经是各后端自己定义的类型：

~~~text
VecApi:         variant<VecPlaintext, VecCiphertext>
PoseidonCpuApi: variant<CpuPlaintextData, CpuCiphertextData>
PoseidonGpuApi: variant<CpuPlaintextData, CpuCiphertextData,
                        GpuPlaintextData, GpuCiphertextData>
~~~

GPU 目标计划可能包含显式的 Host compute，例如 CPU 解密再加密模拟 boot，所以同一个 PoseidonGpuApi 必须同时容纳 Host 和 Device 值。Poseidon 的 Api 可以在内部展开密文的多段 buffer、在目标端分配、复制元信息、转换 CPU/GPU 表示、调 MPI/NCCL/CUDA、管理锁页内存——这些都不进 runtime 的公共接口。poseidon::mgpu 现有的 `GpuObjectCopyMaterializer` 已经实现了"把不透明对象拆成一组分段 buffer 拷贝"这件事，迁移对接时可以直接复用在 Api 内部。

因此首期不引入 ObjectHandle、ObjectDescriptor、ObjectAdapter、BufferView、MemoryKind 或通用序列化接口。如果未来确实要让一个通用的 MPI 传输层独立复用多种对象布局，再把对象适配器作为 Api 的内部模块加进去，而不是提前进入 Runtime 接口。

## 8. Host 与 Device 传输

Host→Device、Device→Host 等都用同一个 CommAction 表达：

~~~text
Transfer:
  source = Host(rank=0)
  destination = Device(rank=0, index=0)
~~~

Runtime 不根据值的类型（明文/密文）自动判断要不要上传，判断依据始终是计划里的来源/目标 Place。

Api 可以在一次 Host→Device 里顺便完成：目标端分配、普通内存到锁页内存的暂存、CPU/GPU 对象表示转换和 cudaMemcpy。只要输出的数学值、类型和全部 CKKS 元信息与计划一致，就是合法实现。

## 9. 初始化、执行和收尾

计划分三个逻辑阶段（与 runtime 文档一致，这里从通信视角复述）：

**初始化**：绑定调用方的 Host external_inputs；执行 Encode 生成 Host 明文常量；把本次运行需要的值预加载到设备；上传上下文和密钥；等初始化通信完成。

**执行**：执行计算；执行中间的 Transfer/Replicate；消费 Pending 输出前调用 `wait`。

**收尾**：下载要求的结果；等待所有未完成的发送句柄；打印或返回结果；释放对象。

## 10. 大型明文与未来的内存规划

大量明文可能装不下显存。后续由编译器增加内存规划：

~~~text
拓扑感知的设备分配
  -> 通信显式化
  -> 内存规划
  -> 插入预取和释放
~~~

未来的动作：

~~~text
Prefetch:       提前执行 Host -> Device 的物化
Release/Evict:  释放某个 Device ValueId 的物理数据；
                数学上相同的 Host 副本（另一个 ValueId）可以保留
~~~

Runtime 不做动态缓存，只按计划执行这些动作。这样编译器可以统筹显存容量、最后使用时间、传输带宽、明文复用次数和通信计算重叠。

## 11. Api 的实现降级

降级完全由 Api 处理：

~~~text
Replicate + Broadcast 提示
    -> NCCL Broadcast
    -> MPI_Ibcast
    -> 多个点对点
    -> 经 Host 中转

Gather + Collective 提示
    -> 原生 gather
    -> 多个接收 + 本地组装
~~~

约束：

1. 降级后的结果必须数学等价；
2. 输出的值类型必须一致；
3. 来源/目标 Place 不得被悄悄改变；
4. 没有等价实现时立即报错；
5. 调试模式可以记录实际用了哪种实现。

Runtime 不需要能力查询，也不参与降级。

这里的“降级”只指通信的等价实现。例如 Broadcast hint 可以改用多个点对点。GPU Boot 改成 CPU 解密再加密会改变计算 Place，还会增加 Device→Host 和 Host→Device，不属于 Api 可以隐藏的降级；它必须由编译器显式写进计划。

## 12. 异步与等待

`communicate_async` 必须在不等待远端完成的情况下返回，具体的 request/event 都藏在 CommHandle 里。

Runtime 维护：

~~~text
ValueId -> Ready 的值
ValueId -> Pending（通信句柄 + 输出下标）
~~~

同一个句柄可以被多个输出 ValueId 引用，但每个 ValueId 只对应一个 Place 和一个输出下标。消费方需要某个输出时：Pending → `Api.wait` → Ready。

计算与通信之间的顺序契约（详见架构文档第 13 节）：

> 计算函数返回后，输出对同一个 Api 实例上后续发起的调用可见（包括 `communicate_async`）。指令执行期间唯一阻塞 host 的同步点是 `wait`；发布最终输出前另有一次 `synchronize`。

Poseidon 兼容层内部使用异步 GPU kernel 时，用 stream/event 保证这个可见性即可，不必每个算子后都做全局同步。

## 13. 发送侧的生命周期

首期不做精细回收：

- 所有值保留到运行结束；
- 所有发送句柄保留到收尾阶段；
- 收尾时逐个等待，全部完成后统一释放。

这会增加内存占用，但避免了"数据还在异步发送、源就被释放了"的问题。后续内存规划再加精细回收。

## 14. 无死锁论证

正常路径下满足以下条件即可无死锁：

1. 所有 rank 按相同顺序执行相同的通信动作；
2. 每个动作的物理来源/目标在编译期确定；
3. `communicate_async` 不等待远端完成；
4. 接收方在解释到这条动作时立即发布接收；
5. 消费方只等待之前已经启动的句柄；
6. 集合通信的参与者和顺序全局一致；
7. 计算的按 rank 跳过不会跳过通信动作；
8. 任意错误调用 `abort_all`。

具体 MPI/NCCL 的消息匹配、分组和进度推进由 Api 兼容层负责。

## 15. Fail-fast

Api 的计算、`communicate_async`、`wait` 或 `synchronize` 失败时直接抛异常，runtime 顶层统一处理（见 runtime 文档第 12 节）。诊断信息至少包含：指令序号和类型、ValueId、传输 ID、来源/目标 Place、本地 rank、Api 名称、底层错误。

不设计重试、取消、超时恢复、回滚、节点恢复或部分继续执行。

## 16. 旧 CommInterface 的替换结果

早期的 `send_async`、`broadcast_async`、`gather_async`、`recive_async` 和 `recive_fence` 已经删除，原因包括：

- 模板参数 T 和 size 的语义不清；
- 接收端怎么分配没有定义；
- 传输 ID 用输出引用传递，容易出错；
- `recive_fence` 返回 `optional<T>`，把"没等到"变成了一个可以被忽略的状态；
- Host 被当成 0 号设备，违反本设计的 Place 原则；
- 底层 API 细节会不断泄漏进接口；
- 无法统一表达未来的通信动作和 hint。

当前接口保留四个函数：

~~~text
communicate_async(CommAction, local_inputs)
wait(CommHandle)
synchronize(Value)
abort_all(exit_code)
~~~

send、receive、broadcast、MPI request 等全部留在 Api 实现内部；首期未实现 Gather。

## 17. Mock Api

每个模拟 rank 使用独立的 MockCommunicationApi 和 runtime。各 Api 实例只共享一个线程安全的 MockCluster（消息队列 + 终止状态），不共享值存储或 `Api::Value`。

MockCommunicationApi 实现同一套接口：

- Host 和 Device 都是逻辑 Place；
- `communicate_async` 在消息队列里登记发送/接收，创建 Pending 句柄；
- `wait` 推进模拟事件并返回输出；
- 支持 Transfer 和 Replicate；
- 支持 Broadcast 提示降级为多个点对点；
- 支持固定或随机（带种子）的延迟，用来改变各 runtime 的到达顺序；
- 支持指定某次通信失败；
- 传输时必须复制数据和元信息，禁止通过共享 C++ 对象"零通信"传值——否则会掩盖真正的通信错误;
- `abort_all` 设置集群终止状态、唤醒所有等待中的 `wait`，让所有模拟 runtime 停下。

多 rank 测试用一个线程跑一个 runtime，只在测试开始和结束处汇合，不加逐指令同步。不需要真实网络、独立进度线程或复杂的进度引擎。
