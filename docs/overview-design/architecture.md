# 总体架构

> 本文描述项目的**目标架构**。当前 C++ 原型只实现 Compute、Transfer、Replicate 及 Vec/Mock/MPI 路径;显式 Encode、bundle 装载、Host compute 和 OperatorSpec 完整校验尚未落地。现状以[实现状态](implementation-status.md)为准。

## 1. 项目定位

本项目是一个面向多设备 CKKS 推理的 runtime 原型。它只实现下面这几件事：

- 用 C++ 表示编译器排好的可执行计划（一串已经定好在哪算、怎么搬的 SSA 指令）；
- 一个只负责执行的 runtime；
- 用普通向量模拟计算的 VecExecutor；
- 组合 VecExecutor 与线程消息队列通信的 MockVecApi/MockCluster；
- 为每个模拟节点各建一个 runtime 的并发测试框架；
- 最终结果对比，以及可选的逐指令对比；
- 合法性检查、异步等待、出错即停的错误处理；
- MPI 环境和基础通信自检。

Poseidon 是未来可以接入的一套 Api 实现，不是 runtime 的固定依赖。GPU、NPU、CPU 或明文实现都应该能套用同一套抽象。

本项目不追求动态集群适配、高可靠、重试、恢复或生产级容错。目标集群在编译时已经确定，运行时一旦发现错误就立刻终止整个执行组，并尽量输出完整的错误位置和原因。

## 2. 核心原则

1. 逻辑层的计算算子保持纯函数语义，不夹带通信。
2. 完整的硬件拓扑在编译时已知。
3. 设备分配和通信显式化都由编译器完成，不留给 runtime。
4. 可执行计划里，每个值的来源、去向和通信语义都已经定死。
5. 可执行计划严格遵守"一个 ValueId 只属于一个 Place"。
6. Runtime 只验证和执行，不重新分配、不选路、不补通信。
7. 面向 runtime 的 Api 尽量简洁，不暴露 MPI、NCCL、CUDA、内存布局这些细节。
8. Api 兼容层负责具体的数据对象、通信协议，以及不改变 Place 和数学语义的等价实现降级。只要会改变计算位置或增加数据搬运,就必须由编译器显式写进计划。
9. Host 是和 Device 平级的一等 Place，但不能拿某张 GPU 冒充 Host。
10. 错误一律 fail-fast，不设计重试、回滚、取消或部分恢复。

## 3. 从上层表示到执行的分层

整条流水线从高层表示逐步下降到实际执行，每一步都由编译器完成，runtime 只接手最后的可执行计划：

~~~text
上层 MLIR / 同态加密表示
        |
        v
逻辑 CKKS SSA
- 只有 CKKS 数学语义和常量编码语义
- 带 CKKS 类型和元信息（level、scale_log2 等）
- 标注对位置的需求，但还没定死
        |
        | 目标相关合法化
        | - CPU eager / GPU lazy rescale
        | - 可选的 GPU boot -> Host 解密再加密模拟
        | - 读取版本化 OperatorSpec
        v
目标可执行的逻辑 CKKS SSA
        |
        | 设备分配 Pass（输入：固定的目标拓扑）
        v
已分配 CKKS SSA
- 每条计算指令有了唯一的执行 Place
- 每个结果标注了"需要出现在哪些 Place"（仅编译期用）
        |
        | 通信显式化 Pass
        v
可执行 CKKS 计划
- Encode 显式定义 Host 明文常量
- 计算指令只产生本地结果
- 每个 ValueId 恰好属于一个 Place
- 一发多收的通信产生多个不同的 ValueId
- V1 用 Transfer / Replicate 表达搬运
- 带物理的来源 / 目标 Place
- 可选的通信实现提示（hint）
        |
        | 验证 + 序列化为 RuntimePlan V1 JSON
        v
Runtime
- 加载时验证
- 管理值的 Ready / Pending 状态
- 按全局统一顺序解释，按本地角色执行
- 需要数据时等待通信完成
- 出错即停
        |
        v
Api
- 定义值类型
- encode_plaintext：把 float64 slots 编码成 Host plaintext
- add / mul / rotate / rescale / boot 等计算
- preflight：用协议 SHA-256 核对各 rank 的计划，并检查 target/context/密钥配置
- validate_value：按 ValueDesc 核对后端实际值
- communicate_async / wait / 最终输出 synchronize
- 适配 MPI / NCCL / CUDA / 向量模拟
- 等价的通信实现降级
- abort_all
~~~

## 4. Place 与编译期拓扑

Host 和 Device 统一用 Place 表示：

~~~cpp
enum class PlaceKind {
    Host,
    Device
};

struct Place {
    PlaceKind kind;
    int rank;    // 节点编号
    int index;   // 该节点内的设备序号
};
~~~

例如：

~~~text
Host(rank=0)
Device(rank=0, index=0)
Device(rank=0, index=7)
Device(rank=1, index=0)
~~~

编译器持有一份完整的拓扑模型，里面可以包含：rank 数量、每个 rank 的 Host 和 Device、NVLink/PCIe/InfiniBand 等连接、带宽和延迟、点对点可达性、内存和显存容量、目标 Api 支持的计算和通信能力。编译器据此完成静态分配、通信选择、内存规划和 hint 生成。

Runtime 不持有也不搜索这份拓扑模型。可执行计划里已经写死了物理 Place;启动环境只需要提供本地 rank、实际 world size 和本地设备数,让 Runtime 对照计划检查。带宽、路由和拓扑代价仍只属于编译器。

## 5. 三种中间表示

### 5.1 逻辑 CKKS

只描述 CKKS 的数学计算：

~~~text
%2 = add %0, %1
~~~

指令没有通信副作用，Place 还没确定。

### 5.2 已分配 CKKS

设备分配之后，给指令加上位置信息：

~~~text
%2 = add %0, %1 {
  compute_place = Device(rank=0, index=0),
  result_places = [
    Device(rank=0, index=0),
    Device(rank=0, index=1),
    Device(rank=1, index=0)
  ]
}
~~~

这里的 `result_places` 是"这个结果最终需要出现在这些位置"的正确性要求，不是说 add 自己会去通信。它只存在于通信显式化之前的中间表示里，也不表示某个 ValueId 同时在多个位置。

### 5.3 可执行计划

通信显式化之后，把远端物化拆成独立的通信动作：

~~~text
%2 = add %0, %1 {
  place = Device(rank=0, index=0)
}

%3, %4 = replicate %2 {
  source = Device(rank=0, index=0),
  destinations = [
    Device(rank=0, index=1),
    Device(rank=1, index=0)
  ],
  hint = Broadcast
}
~~~

`%2`、`%3`、`%4` 在数学上是同一个值，但分别属于三个不同的 Place。Transfer/Replicate 对数据值来说是"原样搬运"，对物理执行来说改变了它所在的位置。Runtime 不会为一个 ValueId 维护多个位置的状态。

原始的逻辑 CKKS 可以保留。Dacapo 与 Runtime 之间的 V1 边界固定为 RuntimePlan JSON;Runtime 读入后再转成自己的 C++ 结构。MLIR 类型和 C++ `plan.hpp` 都不是跨仓库协议。

## 6. 通信动作

通信动作可以逐步扩展,但 RuntimePlan V1 只支持前两种：

~~~text
Transfer   一对一：1 个源 -> 1 个目标
Replicate  一对多：1 个源 -> N 个不同的 ValueId
Gather     多对一：N 个源 -> 1 个合并结果
Scatter    一对多切片：1 个源 -> N 个切片，各去一个目标
AllGather  多对多：N 个源 -> 每个目标各得一份合并结果
~~~

每个通信动作的输出 ValueId 只属于一个目标。以 Replicate 为例，`outputs[i]` 对应 `destinations[i]`，不同目标不能共用同一个输出 ValueId。

通信动作携带两类信息：

**语义**：决定输入如何变成输出，Api 必须遵守，不能改变结果。

**hint（实现提示）**：编译器建议的实现方式，例如：

~~~text
Auto、PointToPoint、Collective、Broadcast、
GatherPrimitive、Tree、Ring、HostStaged
~~~

Api 可以在保持语义不变的前提下用等价的方式实现。比如 Replicate 带 Broadcast 提示，Api 可以真的用 broadcast，也可以退化成多个点对点发送。如果 Api 实现不了相同的语义，就直接报错并触发全局终止。

## 7. Runtime 与 Api 的边界

Runtime 以 Api 为模板参数，不关心 Api 的具体类型：

~~~cpp
template <class Api>
class Runtime {
public:
    using Value = typename Api::Value;
    using CommHandle = typename Api::CommHandle;
};
~~~

Runtime 理解计划声明的 CKKS 元信息,包括 context、level、`scale_log2`、NTT 状态和分量数,因为验证器要靠它们检查算子是否合法。Runtime 不理解的是这些元信息在具体 C++ 对象和 buffer 里怎样布局,也不展开一个密文的各段显存。

Api 负责：定义值类型；在 `preflight` 中核对计划指纹和后端配置；用 `validate_value(value, expected_desc)` 核对不透明值的实际元信息；实现 Encode、计算、通信、等待、最终同步和 `abort_all`。CPU/GPU 对象布局和通信降级都留在 Api 内部。

Api 不负责把一个不支持的 GPU Boot 偷偷改成 CPU 路径。这种替换会改变 Place、需要两次 Transfer,必须由 Dacapo 的目标合法化 Pass 明确生成。

## 8. Runtime 的职责边界

Runtime 负责：

- 加载可执行计划；
- 验证 SSA、Place、通信动作和目标配置；
- 绑定外部输入；
- 校验明文数据包并执行 Encode；
- 所有 rank 按同一个全局指令顺序解释；
- 根据当前 rank 判断自己在每条指令里是计算方、发送方、接收方还是旁观者；
- 保存 Ready 的值和 Pending 的通信句柄；
- 在消费某个值之前，先等它就绪；
- 运行结束前等所有未完成的发送收尾；
- 捕获 Api 抛出的异常，补上指令/值/位置的上下文；
- 打印错误并调用 `abort_all`。

Runtime 不负责：设备分配、动态选路、能力查询与协商、通信降级、重试/取消/恢复、动态缓存策略。

这里的 Ready/Pending 只描述 Runtime 是否已经拿到一个可交给 Api 的值句柄。Ready 不保证 GPU kernel 已经物理完成;异步 Api 可以让句柄内部继续等待。Pending 专指通信尚未产出这个本地值句柄,消费前需要 `wait`。

## 9. Host 与 Device 的数据管理

调用方提供的 external input 和 Encode 产生的明文常量都先位于 Host。要不要搬到设备上，由 Place 决定，而不是由它是明文还是密文决定：

~~~text
当前 Place 和使用它的指令的 Place 不同
    -> 编译器在两者之间插入一个通信动作
~~~

例如：

~~~text
%w_host = encode {
  payload = bundle("sha256:abcd...")
} : HostPlaintext(rank=0)

%w_gpu = transfer %w_host {
  source = Host(rank=0),
  destination = Device(rank=0, index=0)
}
~~~

从 Host 搬到 Device 的这一步里，Api 内部可以顺便完成：锁页/非锁页的暂存、CPU/GPU 对象格式转换、显存分配和 cudaMemcpy。这些细节都不进入 runtime,但传输前后的 context、level、`scale_log2`、NTT 状态和分量数必须保持不变。

### GPU boot 的 CPU 模拟

Poseidon GPU 原生 boot 暂不可用时,可选编译 Pass 生成下面的显式流程:

~~~text
GPU 密文
  -> Transfer(Device -> Host)
  -> Boot(Host, implementation=decrypt_reencrypt,
          target_level=L, target_scale_log2=S)
  -> Transfer(Host -> Device)
  -> GPU 密文
~~~

Host Boot 使用 Poseidon CPU 解密和解码,再按目标 GPU context、level 和 `scale_log2` 重新编码、加密。它会暴露明文和使用 secret key,只用于测试与联调。Runtime 只执行这三条显式指令,不会在 GPU Boot 失败后自动插入这条路径。

### 首期策略：一次性预加载

可执行计划分三个阶段：

~~~text
初始化（Initialization）
执行（Execution）
收尾（Finalization）
~~~

初始化阶段先绑定调用方输入、执行 Encode 生成明文常量,再把本次执行需要的值预加载到目标设备，并一直保留到运行结束。这样实现简单，代价是内存占用偏高。

### 后续的大数据策略

当明文太大、装不下显存时，由编译器增加一个内存规划 Pass：根据显存容量和"最后一次使用"算出每个值该在设备上驻留的时间窗口，插入预取（Prefetch）和释放（Release/Evict），必要时从 Host 重新物化，或选择跨设备/经 Host 的数据路径。Runtime 仍然只执行这些动作，不自己猜缓存策略。

## 10. 错误策略

本项目不做高可靠的错误传播。任何检查、计算、通信或等待中的错误都视为致命：

~~~text
发现错误
  -> 补上指令/值/位置/传输的上下文
  -> 打印到 stderr
  -> 刷新日志
  -> 调用 Api.abort_all
~~~

MPI 版 Api 可以用 `MPI_Abort`；单进程的 MockVecApi 可以抛出 ClusterPanic 或直接终止。

错误信息至少应包含：计划/指令序号、指令类型、输入输出 ValueId、传输 ID、来源/目标 Place、本地 rank、Api 名称、底层错误原因。

不设计重试、回滚、节点恢复、检查点或部分继续执行。

## 11. 可执行计划的不变量

1. 每个 Encode 只在 initialization 中定义一个 Host plaintext,载荷要么内联、要么引用 bundle content。
2. 所有普通计算指令都是纯计算，无副作用。
3. 每条计算指令只有一个执行 Place。
4. 计算指令的输入已经在执行 Place 上就绪。
5. 不存在隐式的跨 Place 操作数。
6. 所有数据搬运都由显式的通信动作表示。
7. 每个值都有完整的 context、level、`scale_log2`、NTT 和分量数,传输不改变这些元信息。
8. 算子的目标元信息与输出 ValueDesc 一致,并符合所引用的 OperatorSpec/profile。
9. Boot 的 `native` 或 `decrypt_reencrypt` 模式在计划中明确,不做运行时隐式回退。
10. 通信动作的数学语义和结果类型都明确。
11. 来源/目标 Place 都来自编译目标。
12. 传输 ID 在同一个计划里稳定且唯一。
13. 所有 rank 使用同一份可执行计划。
14. 使用一个值之前，必须先把它从 Pending 等成 Ready。
15. Api 的通信降级不得改变通信动作的数学语义或 Place。
16. 任意错误都会终止整个执行组。

## 12. 与 poseidon::mgpu 的对接

Poseidon 已有一套单机多卡静态调度代码（`src/poseidon/mgpu`）。本框架是它的演进目标——后续会把其中的通信层和调度产物迁移过来对接，冗余部分删除。为了让迁移顺畅，本设计在这些关键点上和 mgpu 保持一致：

- **单位置 SSA**：mgpu 的 `CopyCipher`/`CopyPlain` 也是显式的拷贝指令，拷贝产生新的 ValueId，和本框架"一个 ValueId 一个位置"的规则一致。
- **不透明对象存储**：mgpu 用 `MgpuObjectStore` 存 `shared_ptr<void>`，用 `ScheduleExecutionBackend` 虚基类隔离计算实现；本框架用 `Runtime<Api>` 模板 + `Api::Value` 达到同样的隔离效果。未来 PoseidonApi 可以直接包住 mgpu 的执行后端和对象拷贝层。
- **对象级拷贝**：mgpu 的 `GpuObjectCopyMaterializer` 已经解决了"密文在显存里不是单段连续 buffer、需要分段拷贝"的问题。这正是本框架把 buffer 细节留在 Api 内部的原因——迁移时这层可以直接复用。

需要在迁移时补齐的差异（mgpu 目前没有，本框架要求）：

- **rank 概念**：mgpu 只有 `int device_id`（单进程单节点），本框架的 `Place{rank, index}` 是它的超集。跨节点传输在 mgpu 里只是一个会明确报错的占位后端，真正的实现要在这里补。
- **完整 CKKS 元信息**：mgpu 的整数属性表不足以表达 V1 的 context、level、`scale_log2`、NTT、分量数和 OperatorSpec/profile。迁移时要按 V1 类型补齐,不能继续依赖无类型的属性名约定。
- **Host compute**：GPU boot 的 CPU 模拟需要显式的 Device→Host、Host Boot、Host→Device。mgpu 当前的 `BootstrapFallback` 只是未实现占位,不能直接当成新方案。
- **异步执行**：mgpu 的执行器逐指令同步执行，拷贝是阻塞的。本框架的 Pending/`wait` 模型更进一步（见下一节的契约说明），迁移时 PoseidonApi 需要用 stream/event 把同步执行包装成异步接口。
- **NCCL/MPI 通信**：mgpu 目前没有集成，这部分从本框架的 MpiApi/NcclApi 开始建。

## 13. 计算与通信的异步契约

为了让 GPU 后端有发挥异步的空间，计算函数的完成契约这样定义：

> 计算函数返回后，它的输出对**同一个 Api 实例上后续发起的调用**保证可见（包括交给 `communicate_async`）。`wait` 是 Runtime 对异步通信的唯一显式等待点;同步 Api 的 `compute`、文件读取或 Encode 本身仍可能占用调用线程。运行结束发布最终输出前还会调用一次 `synchronize`。

这句话的用意是：Api 内部可以用 CUDA stream/event 给算子和通信排序，只要保证"后一个调用能看到前一个的结果"即可，不必在每个算子后面都做一次 `cudaStreamSynchronize`。Runtime 的代码完全不受影响。对 VecExecutor 的同步模式，这个契约天然满足。

**发起串行与执行并行的区分**：首期 runtime 是单线程顺序解释器，指令一条条**发起**。但发起串行不等于执行串行——对异步实现的 Api（如 GPU 后端），`compute` 只是往对应设备的执行流里提交任务就返回，host 线程随即发起下一条指令，多张卡上的任务实际是同时执行的。因此单线程解释器本身不是并行障碍，真正影响重叠程度的是两点：

- **`wait` 会阻塞 host 线程**。如果计划把"等待某个通信"排在"另一张卡上本可独立执行的计算"之前，后者就被无谓推迟。这由编译器的指令排序解决：通信尽早发起，`wait` 对应的消费指令尽量靠后。
- **同步实现的 Api 没有重叠**。VecExecutor 的默认同步模式在 host 线程上当场算完，多卡模拟在它之下是纯串行的——这不影响其验证正确性的用途。异步模式则为每个模拟设备建立工作线程,`compute` 入队即返回,完整 Api 接口与同步模式相同（见测试文档 2.1 节）。

若指令排序仍不够用，还有两条不改变 Runtime↔Api 接口形状的出路：每个设备一条独立解释流，或让计算也返回句柄。

## 14. 当前 Demo 已实现范围

本节只描述当前代码,不是 RuntimePlan V1 草案的最终验收表。更精确的差距见[实现状态](implementation-status.md)。

### 支持

- 单函数、直线 SSA（无分支循环）；
- 固定目标拓扑；
- Host 和 Device Place；
- VecExecutor（同步与异步两种模式）、MockVecApi 和 MockCluster；
- 每个模拟 rank 一个独立 runtime 和值存储；
- 线程安全的 MockCluster；
- Transfer 和 Replicate；
- Auto、PointToPoint、Broadcast 三种 hint；
- 一次性预加载；
- Ready/Pending 状态管理；
- 与单设备顺序流做最终结果对比；
- 可选的运行后逐指令对比；
- fail-fast；
- 1、2、4、6、8 个逻辑设备的模拟；
- MpiVecApi：明文计算 + MPI 通信的多进程集成测试（`mpiexec` 下每进程一个 runtime），验证多进程执行 runtime 的可行性。

### 后续

- 显式 Encode、inline/bundle 双 payload 和数据包 preflight；
- OperatorSpec、能力、密钥和实际值元信息的完整校验；
- Host compute 与 `decrypt_reencrypt` Boot；
- Gather；
- 显存内存规划；
- Prefetch、Release/Evict；
- 真实的 Poseidon/NCCL/CUDA Api（密文级的 MPI 通信在明文 MpiVecApi 验证后接入）；
- 密文分片的 gather/scatter；
- 更复杂的控制流。

### 不做

- 动态拓扑；
- 动态设备分配；
- 重试/取消/恢复；
- 生产级容错；
- 在线通信调优。
