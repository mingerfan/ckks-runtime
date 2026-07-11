# 总体架构

## 1. 项目定位

本项目是一个面向多设备 CKKS 推理的 runtime demo。它只实现：

- 形式化 SSA/executable plan 的 C++ 表示；
- 纯执行 runtime；
- 明文 VecApi；
- MockCommunicationApi；
- 每个模拟 rank 一个 Runtime 的并发测试 harness；
- 最终结果和可选逐指令 difftest；
- 合法性检查、异步等待和 fail-fast 错误暴露；
- MPI 环境和基础通信自检。

Poseidon 是未来可以接入的一套 API 实现，不是 runtime 的固定依赖。其他 GPU、NPU、CPU 或明文实现应能适配同一抽象。

本项目不追求动态集群适配、高可靠、重试、恢复或生产级容错。目标集群在编译时已知，运行时发现任何错误后立即终止整个 execution group，并输出尽可能完整的错误位置和原因。

## 2. 核心原则

1. Logical CKKS SSA 中的计算 op 保持逻辑纯净。
2. 完整硬件拓扑在编译时已知。
3. Device placement 和 communication materialization 由编译器完成。
4. Executable plan 中的 source、destination 和通信语义已经确定。
5. Executable plan 严格采用单位置物理 SSA：每个 ValueId 恰好属于一个 Place。
6. Runtime 只验证和执行，不重新 placement、选路或补通信。
7. Runtime-facing API 尽量简洁，不暴露 MPI、NCCL、CUDA、buffer layout 等细节。
8. API 兼容层负责具体对象、通信协议和等价 fallback。
9. Host 是一种一等 Place，但不能伪装成 GPU device 0。
10. 错误采用 fail-fast；不设计 retry、rollback、cancel 或部分恢复。

## 3. 编译与执行分层

~~~text
上层 MLIR / HE 表示
        |
        v
Logical CKKS SSA
- 纯计算语义
- CKKS type/metadata
- placement requirement
        |
        | Device Placement Pass
        | 输入：固定目标拓扑
        v
Placed CKKS SSA
- 每个 compute op 有唯一执行 Place
- 每个数学结果有 compiler-only required materialization Places
        |
        | Communication Materialization Pass
        v
Executable CKKS Plan
- compute op 只产生本地结果
- 每个 ValueId 恰好属于一个 Place
- 一对多通信产生多个不同 ValueId
- Transfer/Replicate/Gather 等通信 action
- 物理 source/destination Place
- 可选通信实现 hint
        |
        | Verify + Translate
        v
Runtime
- load-time validation
- ready/pending value 管理
- 顺序解释和 role mask
- wait
- fail-fast
        |
        v
Api
- Value 类型
- add/mul/rotate/rescale/boot
- communicate_async/wait
- MPI/NCCL/CUDA/Vec 适配
- fallback
- abort_all
~~~

## 4. Place 与编译期拓扑

Host 和 Device 统一建模为 Place：

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
Device(rank=0, index=7)
Device(rank=1, index=0)
~~~

编译器的 TopologyModel 可以包含：

- rank 数量；
- 每个 rank 的 Host 和 Device；
- NVLink、PCIe、InfiniBand 等连接；
- 带宽和延迟；
- P2P 可达性；
- Host 内存和 Device 显存；
- 目标 API 支持的计算和通信能力。

编译器据此完成静态 placement、通信选择、memory planning 和 hint 生成。

Runtime 不持有或搜索完整 TopologyModel。Executable plan 已经包含物理 Place。Runtime 只需要知道当前 rank，并在启动时检查实际 world size 和本地设备数量是否符合编译目标。

## 5. 三种 IR 状态

### 5.1 Logical CKKS

只描述 CKKS 数学计算：

~~~text
%2 = add %0, %1
~~~

op 没有通信副作用，Place 可以尚未确定。

### 5.2 Placed CKKS

Device placement 后：

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

result_places 是 correctness requirement，而不是 add 自行通信的副作用。它仅存在于通信物化前的 placed IR，不是 RuntimePlan 中一个 ValueId 的多位置属性。

### 5.3 Executable Plan

通信物化后：

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

这里 `%2`、`%3`、`%4` 数学等价，但分别属于三个不同 Place。Transfer/Replicate 在数学值上是 identity，在物理执行上具有 placement effect；Runtime 不会为一个 ValueId 建立多位置状态。

原始 Logical CKKS 可以保留。Executable plan 可以继续使用 MLIR，也可以翻译为 C++ schedule。

## 6. 通信 action

Executable plan 可以逐步支持：

~~~text
Transfer   1 -> 1
Replicate  1 -> N 个不同 ValueId
Gather     N -> 1
Scatter    1 -> N 个不同 ValueId、不同切片
AllGather  N -> N 个不同 ValueId
~~~

每个通信 output ValueId 只属于一个 destination。对于 Replicate，outputs[i] 与 destinations[i] 一一对应；多目标传输不能复用同一个 output ValueId。

通信 action 包含两类信息：

### 语义

决定输入如何形成输出，不能被 API 忽略。

### Hint

表示编译器建议的实现方式，例如：

~~~text
Auto
PointToPoint
Collective
Broadcast
GatherPrimitive
Tree
Ring
HostStaged
~~~

API 可以使用等价 fallback。例如 Replicate + Broadcast hint 可以退化为多个 point-to-point。若无法实现相同语义，API 直接报错并触发全局 panic。

## 7. Runtime 与 Api 边界

Runtime 模板化在 Api 上：

~~~cpp
template <class Api>
class Runtime {
public:
    using Value = typename Api::Value;
    using CommHandle = typename Api::CommHandle;
};
~~~

Runtime 不理解：

- Value 的实际 C++ 类型；
- ciphertext 有多少 component；
- 对象是否连续；
- destination 如何分配；
- 是否使用 MPI、NCCL、CUDA P2P 或 host staging；
- Gather/Broadcast 如何 fallback；
- CPU/GPU 表示是否需要转换。

Api 负责：

- 定义 Value；
- 实现纯计算函数；
- 实现 communicate_async；
- 实现 wait；
- 对 Host/Device 做特殊处理；
- 选择或 fallback 到等价传输方式；
- 抛出详细错误；
- 实现 abort_all。


## 8. Runtime 职责

Runtime 负责：

- 加载 executable plan；
- 验证 SSA、Place、通信 action 和目标配置；
- 绑定外部输入；
- 所有 rank 按同一全局 op 顺序解释；
- 根据当前 rank 执行 compute/source/destination/no-op role；
- 保存 Ready Value 和 Pending CommHandle；
- consumer 使用前调用 wait；
- run 结束前等待所有 outstanding send；
- 捕获 Api 异常，补充 op/value/place 上下文；
- 打印错误并调用 abort_all。

Runtime 不负责：

- topology-aware placement；
- 动态选路；
- capability 查询和协商；
- 通信 fallback；
- 重试、取消或恢复；
- 动态缓存策略。

## 9. Host 与 Device 数据管理

外部 input、plaintext 常量和 ciphertext 输入都可以初始位于 Host。是否需要搬运由 Place 决定，而不是由 Plaintext/Ciphertext 类型决定。

~~~text
当前 Place != consumer Place
    -> 编译器插入通信 action
~~~

例如：

~~~text
%w_host = input { place = Host(rank=0) }

%w_gpu = transfer %w_host {
  source = Host(rank=0),
  destination = Device(rank=0, index=0)
}
~~~

API 可以在 HostToDevice 内部完成：

- pageable/pinned staging；
- CPU/GPU 格式转换；
- allocation；
- cudaMemcpy；
- metadata 转换。

这些细节不进入 Runtime。

### 首期策略

Executable plan 分为：

~~~text
Initialization
Execution
Finalization
~~~

Initialization 把所有本次执行需要的外部值和 plaintext 常量预加载到目标 Device，并一直保留到 run 结束。

### 后续大数据策略

当 plaintext 过大、不能全部驻留 Device 时，由编译器增加 memory planning pass：

- 根据显存容量和 last-use 计算 residency window；
- 插入 Prefetch；
- 插入 Release/Evict；
- 必要时从 Host 重新 materialize；
- 选择跨设备或经 Host 的数据路径。

Runtime 仍然只执行这些 action，不动态猜测缓存策略。

## 10. 错误策略

本项目不实现高可靠错误传播。任何 preflight、compute、communication 或 wait 错误都视为 fatal：

~~~text
detect error
  -> enrich with op/value/place/transfer context
  -> print to stderr
  -> flush logs
  -> Api.abort_all
~~~

MPI Api 可以使用 MPI_Abort。单进程 VecApi 可以抛出 ClusterPanic 或直接终止。

错误至少应包含：

- plan/op ordinal；
- op kind；
- input/output ValueId；
- transfer ID；
- source/destination Place；
- local rank；
- Api 名称；
- 底层错误原因。

不设计 retry、rollback、rank recovery、checkpoint 或部分继续执行。

## 11. 可执行计划不变量

1. 所有 compute op 都是 PureCompute。
2. 每个 compute op 只有一个执行 Place。
3. compute input 已经在执行 Place 上物化。
4. 不存在隐式跨 Place compute operand。
5. 所有数据搬运由通信 action 表示。
6. 通信 action 的数学语义和结果类型明确。
7. source/destination Place 来自编译目标。
8. TransferId 在同一个 plan 中稳定且唯一。
9. 所有 rank 使用同一 executable plan。
10. Runtime 使用前必须把 Pending 输出 wait 为 Ready。
11. Api fallback 不得改变通信 action 的数学语义。
12. 任意错误都会终止整个 execution group。

## 12. Demo 首期范围

### 支持

- 单函数、直线 SSA；
- 固定目标拓扑；
- Host 和 Device Place；
- VecApi；
- MockCommunicationApi；
- 每个模拟 rank 一个独立 Runtime/ValueStore；
- 线程安全的 MockCluster；
- Transfer 和 Replicate；
- Auto、PointToPoint、Broadcast hint；
- Initialization 预加载；
- Ready/Pending 状态；
- 与单 Place 顺序流的最终结果 difftest；
- 可选的运行后逐指令 difftest；
- fail-fast；
- 1、2、4、6、8 个逻辑设备模拟。

### 后续

- Gather；
- 显存 memory planning；
- Prefetch、Release/Evict；
- 真实 Poseidon/MPI/NCCL/CUDA Api；
- shard gather/scatter；
- 更复杂控制流。

### 不做

- 动态 topology；
- 动态 placement；
- retry/cancel/recovery；
- 生产级容错；
- 在线通信调优。
