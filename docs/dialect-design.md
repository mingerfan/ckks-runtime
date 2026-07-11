# Dialect 与 SSA 设计

## 1. 文档范围

本仓库暂不实现 MLIR dialect。本设计用于约束：

- 未来 logical CKKS dialect 的语义；
- Place placement 和 communication materialization pass；
- executable SSA 的输入格式；
- 当前 C++ demo 中 ssa_ops_def.hpp 的演化方向；
- runtime verifier 应当依赖的稳定不变量。

设计重点是保持逻辑计算图纯净，同时生成足够明确、可验证的物理执行计划。

## 2. 表示阶段

### 2.1 Logical CKKS

Logical CKKS 只表达 CKKS 数学语义：

~~~mlir
%2 = ckks.add %0, %1
    : (!ckks.ciphertext, !ckks.ciphertext)
      -> !ckks.ciphertext
~~~

在这一阶段：

- op 没有通信副作用；
- Place 可以尚未确定；
- value 表示数学对象，不表示具体 buffer；
- CKKS level、scale、parameter context 等语义必须能够验证；
- rescale、relinearize、rotate 和 boot 等必要 op 已经显式或可由 semantic legalization 确定。

### 2.2 Placed CKKS

Device placement pass 在同一 CKKS op 上增加 placement：

~~~mlir
%2 = ckks.add %0, %1 {
  compute_place = #place.device<rank=0, index=0>,
  result_places = [
    #place.device<rank=0, index=0>,
    #place.device<rank=0, index=1>,
    #place.device<rank=1, index=0>
  ]
}
~~~

属性语义：

- compute_place：执行该 op 的唯一物理 Place；
- result_places：结果必须可用的 Place 集合；
- compute_place 必须属于 result_places；
- result_places 是 correctness requirement，不是可随意忽略的性能 hint；
- 可选 placement_hint 应使用独立属性表示。

Placed CKKS 中的 add 仍然是纯计算。result_places 不表示 add 在执行时自行调用通信，也不表示同一个物理 ValueId 同时位于多个 Place。它只存在于通信物化之前，是编译器需要满足的中间层 requirement。

### 2.3 Executable CKKS

Communication materialization pass 将远端物化显式化：

~~~mlir
%2 = ckks.add %0, %1 {
  place = #place.device<rank=0, index=0>
}

%3, %4 = dist.replicate %2 {
  source = #place.device<rank=0, index=0>,
  destinations = [
    #place.device<rank=0, index=1>,
    #place.device<rank=1, index=0>
  ],
  hint = #dist.hint<broadcast>
}
~~~

Executable CKKS 的约束：

- 严格采用单位置物理 SSA：每个 ValueId 恰好属于一个 Place；
- compute op 只有一个 Place；
- compute result 初始只存在于该 Place；
- 所有远端物化由 dist.transfer 或 dist.replicate 产生，并使用新的 ValueId；
- Replicate 的第 i 个结果属于 destinations[i]；
- 不存在跨 Place compute operand；
- runtime 不再推导 placement 或补传输。

## 3. Dialect 划分

推荐初期只建立两个语义域：

### CKKS dialect

负责：

- plaintext 和 ciphertext 类型；
- add、sub、mul、negate；
- rotate、rescale、modswitch；
- relinearize、boot；
- CKKS metadata 和语义验证。

### Dist 或 Comm dialect

负责：

- transfer；
- replicate；
- 可选 collective；
- placement effect；
- 可选异步 token。

Device placement pass 不需要新 dialect。它只是为 CKKS op 增加属性并生成 placement analysis。

如果最终 runtime plan 以后需要稳定序列化、独立版本和多个消费者，再引入 ckks-runtime dialect。首期可以把 executable CKKS 与 dist op 直接翻译为 C++ schedule。

## 4. 类型设计

### 4.1 CKKS 语义类型

逻辑类型至少需要区分：

~~~text
Plaintext
Ciphertext
~~~

可选的静态描述包括：

- parameter/context identifier；
- polynomial degree；
- level 或 parms identifier；
- scale 或 log2 scale；
- NTT form；
- ciphertext component count；
- element/encoding kind。

不建议把 Place placement 编入 CKKS 数学类型。Place 是执行属性，而不是密文数学类型的一部分，否则会产生大量 placement cast 和类型组合。

静态类型无法表达或不适合表达的 metadata，可以保存在 value descriptor、op attribute 或分析结果中。

### 4.2 Value 与单位置物理 SSA

Logical SSA ValueId 表示数学值。Placed CKKS 可以在这个数学值上暂时记录 result_places requirement，但它还没有把每个物理 materialization 展开成独立 SSA value。

Executable plan 严格采用单位置物理 SSA：

~~~text
ValueId -> exactly one Place
~~~

Transfer/Replicate 必须产生新的 ValueId，用来表示新的物理 materialization：

~~~text
%3 = dist.transfer %2 {
  source=#place.device<rank=0,index=0>,
  destination=#place.device<rank=1,index=0>
}
~~~

其中 value(%3) 与 value(%2) 数学等价，但 provenance 和 placement 不同。

对于一对多复制，N 个 destination 必须对应 N 个不同的结果 ValueId：

~~~text
%3, %4, %5 = dist.replicate %2 {
  source=#place.device<rank=0,index=0>,
  destinations=[
    #place.device<rank=0,index=1>,
    #place.device<rank=1,index=0>,
    #place.device<rank=1,index=1>
  ]
}
~~~

这里 `%3`、`%4`、`%5` 分别属于 destinations[0]、destinations[1]、destinations[2]。它们与 `%2` 数学等价，但 SSA identity、Place 和 provenance 均独立。

Communication materialization pass 必须把每个远端 consumer 改写为其所在 Place 对应的结果 ValueId。Runtime 仅按 ValueId 建立一个状态项，不使用 Place 作为第二重物理身份；如果同一数学值还需要出现在另一个 Place，计划必须再定义一个新的 ValueId。

## 5. Op 分类与 effect

建议为每个 op 建立静态 trait：

~~~cpp
enum class OpClass {
    PureCompute,
    Communication,
    ExternalIO
};
~~~

### PureCompute

包括：

- AddCC、AddCP；
- SubCC、SubCP；
- MulCC、MulCP；
- Negate；
- Rotate；
- Rescale；
- ModSwitch；
- Relinearize；
- Boot。

纯计算表示逻辑引用透明：

~~~text
outputs = f(inputs)
~~~

Api 在物理上分配和写入输出 buffer 不属于逻辑副作用。计算函数本身不应主动发起通信。

### Communication

包括：

- Transfer；
- Replicate；
- Gather；
- 将来可能增加的 Scatter、AllGather 或 shard conversion。

Transfer/Replicate 在数学值上是 identity，在 placement、资源、错误和完成状态上有 effect。Gather 等操作必须单独定义 input/output 数学关系。通信 op 不能被声明成普通可删除的 NullOp。

### ExternalIO

包括：

- runtime input binding；
- upload/download；
- external result publication。

## 6. 逻辑 hint 与强制 op

必须区分：

- placement_hint：优化建议，可以被 scheduler 修改；
- result_places requirement：placed CKKS 中、通信物化前的目标 Place 集合；
- TransferOp：定义新值的强制执行语义。

一旦 transfer 的结果被后续 op 使用，它就不能只是 hint：

~~~mlir
%3 = dist.transfer %2 {
  source=#place.device<rank=0,index=0>,
  destination=#place.device<rank=1,index=0>
}
%4 = ckks.mul %3, %x {
  place=#place.device<rank=1,index=0>
}
~~~

Planner 可以尊重用户已经插入的 TransferOp，并只补充尚未满足的 result_places requirement。规范化后，每个物理结果都必须具有独立 ValueId，且不得发生重复传输。

## 7. Pass 设计

### 7.1 CKKS Semantic Legalization

在 placement 前完成：

- op 类型合法化；
- scale 和 level 规划；
- rescale 插入；
- relinearize 插入；
- rotate decomposition；
- bootstrap 决策；
- key requirement 分析；
- 目标 Api 能力预检查。

这些 pass 会改变图结构和 ciphertext 大小，应尽量先于 placement。

### 7.2 Device Placement

输入：logical CKKS。

输出：placed CKKS。

职责：

- 为 compute op 选择唯一 compute_place；
- 计算 compiler-only 的 required materialization set；
- 使用编译时已知的固定目标拓扑；
- 验证 Place 与目标 Api 能力；
- 考虑 Host/Device、延迟、带宽、显存和 key placement；
- 不插入实际通信。

### 7.3 Communication Materialization

输入：placed CKKS。

输出：executable CKKS。

职责：

- 将 compute output 规范化到 compute Place；
- 对跨 Place use 插入 Transfer/Replicate；
- 为相同 destination 复用已有的单位置 materialization ValueId；
- 为 Replicate 的每个 destination 生成不同的 output ValueId；
- 将每个 consumer operand 重写为其执行 Place 对应的 ValueId；
- 生成稳定的 transfer provenance；
- 保证 transfer 插入点支配所有相关 use；
- 生成 CommKind 和可选 CommHint；
- 不把具体 MPI/NCCL/CUDA 调用写入通用 IR。

### 7.4 Executable Verification

检查：

- 所有 compute op 都有唯一 Place；
- 每个 executable ValueId 恰好属于一个 Place；
- 所有 compute operand 都在本地物化；
- 不再存在未解决的 result_places requirement；
- Transfer source 已定义且 placement 合法；
- result 类型与 source 数学类型一致；
- source/destination 属于编译目标；
- CommKind 与 input/output 类型关系合法；
- Replicate 的 output 数量等于 destination 数量，且 output[i] 属于 destinations[i]；
- 不允许一个 output ValueId 对应多个 destination。

### 7.5 Translation

把 executable CKKS 转换成 C++ RuntimePlan。首期不要求增加新 dialect。

## 8. 等待与异步表示

首期 executable SSA 可以只包含：

~~~text
Transfer/Replicate
~~~

Runtime 在 consumer 执行前检查对应 Value 状态：

~~~text
Pending -> Api.wait
Ready   -> 直接执行
Error   -> 立即 panic
~~~

因此首期不必显式加入 WaitOp。

如果以后需要编译器精确控制 overlap、prefetch 或 host synchronization，可以降低到：

~~~mlir
%pending, %token = dist.transfer_async %2
%ready = async.await %pending
~~~

MLIR async dialect 可以承担 token 和 await 语义，避免自建重复机制。

## 9. C++ Demo 类型映射

ssa_ops_def.hpp 建议最终演化到类似：

~~~cpp
using ValueId = std::uint64_t;

enum class ValueKind {
    Plaintext,
    Ciphertext
};

enum class PlaceKind {
    Host,
    Device
};

struct Place {
    PlaceKind kind;
    int rank;
    int index;
};

enum class OpKind {
    AddCC,
    AddCP,
    SubCC,
    SubCP,
    MulCC,
    MulCP,
    Negate,
    Rotate,
    Rescale,
    ModSwitch,
    Relinearize,
    Boot,
    Transfer,
    Replicate,
    Gather,
    Input,
    Output
};

// 仅用于 compiler-side placed CKKS；不会进入 RuntimePlan。
struct PlacedComputeAttrs {
    Place compute_place;
    std::vector<Place> required_result_places;
};

struct ComputeAttrs {
    // executable compute result 只位于此 Place。
    Place place;
    // rotate step、target level 等 op-specific attributes
};

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

struct CommunicationAttrs {
    std::vector<Place> sources;
    std::vector<Place> destinations;
    CommHint hint;
    std::uint64_t transfer_ordinal;
};

using OpAttrs = std::variant<ComputeAttrs, CommunicationAttrs>;

struct SsaOp {
    OpKind kind;
    std::vector<ValueId> inputs;
    std::vector<ValueId> outputs;
    OpAttrs attrs;
};

struct ExecutableValueDesc {
    ValueId id;
    ValueKind kind;
    Place place;
};

struct RuntimePlan {
    // Verifier 保证每个 id 只出现一次，因此只对应一个 place。
    std::vector<ExecutableValueDesc> values;
    std::vector<SsaOp> ops;
};
~~~

对于 Replicate，`outputs.size()` 必须等于 `destinations.size()`，并且 `outputs[i]` 的 Value descriptor 必须指向 `destinations[i]`。实际实现可以采用模板或更严格的 typed op，但必须避免所有 op 共享一组含义不明确的平铺字段。

## 10. 编译困难与首期限制

首期需要明确拒绝：

- 一般控制流；
- loop-carried ciphertext；
- 动态未知的 ValueType 或通信结果布局；
- backend 运行时动态改变 ciphertext shape；
- 同一 executable compute op 的多 execution Place；
- 隐式跨 Place operand；
- 未知 effect op。

后续需要重点处理：

- placement 与通信成本的联合优化；
- block argument 和 phi-like value；
- loop iteration 的 transfer identity；
- liveness 与异步 source 生命周期；
- stable schedule/channel ID；
- 目标 Api 支持范围的编译期 legalization；
- 所有 rank 的 executable plan 一致性。
