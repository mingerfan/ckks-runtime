# Dialect 与 SSA 设计

## 1. 文档范围

本仓库暂不实现 MLIR dialect，但 C++ demo 里的 SSA 类型、算子分类、位置分配、通信和验证器，必须和未来 MLIR 下降（lowering）的结果保持一致。本文档约束：

- 未来逻辑 CKKS dialect 的语义；
- 设备分配和通信显式化这两个编译步骤；
- 可执行 SSA 的输入格式；
- 当前 `ssa_ops_def.hpp` 的演化方向；
- runtime 验证器依赖的稳定不变量。

设计重点：让逻辑计算图保持纯净，同时生成足够明确、可验证的物理执行计划。

## 2. 三个表示阶段

### 2.1 逻辑 CKKS

只表达 CKKS 的数学语义：

~~~mlir
%2 = ckks.add %0, %1
    : (!ckks.ciphertext, !ckks.ciphertext)
      -> !ckks.ciphertext
~~~

这一阶段：指令没有通信副作用；Place 尚未确定；值表示数学对象而不是具体 buffer；CKKS 的 level、scale、参数上下文等语义必须可验证；rescale、relinearize、rotate、bootstrap 等必要算子已显式存在，或可由语义合法化步骤确定。

### 2.2 已分配 CKKS

设备分配 Pass 在同一个算子上加位置属性：

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

- `compute_place`：执行这条指令的唯一物理位置；
- `result_places`：结果必须可用的位置集合，`compute_place` 必须包含在内；
- `result_places` 是正确性要求，不是可以随意忽略的性能提示；
- 如果还想给调度器留优化建议，用独立的 `placement_hint` 属性表示。

注意：这时的 add 仍然是纯计算。`result_places` 不表示 add 会自己去通信，也不表示同一个 ValueId 同时存在于多个位置——它只是编译器在下一步需要满足的中间要求，通信显式化之后就消失了。

### 2.3 可执行 CKKS

通信显式化 Pass 把"结果要出现在远端"变成显式的通信指令：

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

可执行 CKKS 的约束：

- 严格单位置：每个 ValueId 恰好属于一个 Place；
- 计算指令只有一个 Place，结果初始只存在于该处；
- 所有远端副本由 `dist.transfer` 或 `dist.replicate` 产生，并使用新的 ValueId；
- Replicate 的第 i 个结果属于 `destinations[i]`；
- 不存在跨 Place 的操作数；
- runtime 不再推导位置或补传输。

## 3. Dialect 划分

初期只需要两个语义域：

**CKKS dialect**：明文和密文类型；add、sub、mul、negate；rotate、rescale、modswitch；relinearize、boot；CKKS 元信息和语义验证。

**Dist（或 Comm）dialect**：transfer、replicate、可选的集合通信；位置效果；可选的异步 token。

设备分配 Pass 不需要新 dialect，它只是给 CKKS 算子加属性。如果最终的执行计划以后需要稳定的序列化格式、独立版本和多个消费者，再引入一个 ckks-runtime dialect；首期直接把可执行 CKKS 翻译成 C++ 执行序列即可。

## 4. 类型设计

### 4.1 CKKS 语义类型

逻辑类型至少区分明文（Plaintext）和密文（Ciphertext）。可选的静态描述包括：参数/上下文标识、多项式阶数、level 或参数层标识、scale、是否 NTT 形式、密文分量个数、编码方式。

**不建议把 Place 编进 CKKS 数学类型。** Place 是执行属性，不是密文数学性质的一部分；混在一起会产生大量的位置转换 cast 和类型组合爆炸。静态类型放不下的元信息，放在值描述符、算子属性或分析结果里。

### 4.2 单位置物理 SSA

逻辑 SSA 的 ValueId 表示数学值。已分配 CKKS 可以在数学值上暂时记录 `result_places` 要求，但还没把每个物理副本展开成独立的 SSA 值。

可执行计划则严格遵守：

~~~text
一个 ValueId -> 恰好一个 Place
~~~

Transfer/Replicate 必须产生新的 ValueId 来表示新的物理副本：

~~~text
%3 = dist.transfer %2 { source=..., destination=... }
~~~

`%3` 和 `%2` 数学上相同，但来历和位置不同。一发多收时，N 个目标对应 N 个不同的结果 ValueId：

~~~text
%3, %4, %5 = dist.replicate %2 { destinations=[d0, d1, d2] }
~~~

通信显式化 Pass 必须把每个远端使用者的操作数改写成它所在 Place 对应的那个 ValueId。Runtime 只按 ValueId 建立状态，不把 Place 当作第二重身份；同一个数学值要出现在另一个位置，计划里就必须再定义一个新的 ValueId。

## 5. 算子分类

给每个算子一个静态类别：

~~~cpp
enum class OpClass {
    PureCompute,     // 纯计算
    Communication,   // 通信
    ExternalIO       // 外部输入输出
};
~~~

**纯计算**：AddCC、AddCP、SubCC、SubCP、MulCC、MulCP、Negate、Rotate、Rescale、ModSwitch、Relinearize、Boot。逻辑上引用透明：`outputs = f(inputs)`。Api 在物理上分配和写入输出 buffer 不算逻辑副作用；计算函数本身不得主动发起通信。

**通信**：Transfer、Replicate、Gather，将来可能加 Scatter、AllGather 或分片转换。Transfer/Replicate 对数学值是原样搬运，但在位置、资源、错误和完成状态上有实际效果，不能被当成可随意删除的空操作。Gather 等必须单独定义输入输出的数学关系。

**外部输入输出**：运行时的输入绑定、上传/下载、结果发布。

## 6. 提示与强制指令的区别

必须区分三样东西：

- `placement_hint`：优化建议，调度器可以改；
- `result_places`：已分配阶段的正确性要求，通信显式化前必须全部满足；
- TransferOp：定义新值的强制指令，有执行语义。

一旦某条 transfer 的结果被后续指令使用，它就不再是建议：

~~~mlir
%3 = dist.transfer %2 { source=..., destination=... }
%4 = ckks.mul %3, %x { place=... }
~~~

规划器可以尊重用户手工插入的 TransferOp，只补充尚未满足的 `result_places`。规范化之后，每个物理结果都有独立 ValueId，且不发生重复传输。

## 7. Pass 设计

### 7.1 CKKS 语义合法化

在设备分配之前完成：算子类型合法化、scale 和 level 规划、rescale 插入、relinearize 插入、rotate 分解、bootstrap 决策、密钥需求分析、目标 Api 能力预检查。这些步骤会改变图结构和密文大小，所以要放在分配之前。

### 7.2 设备分配

输入逻辑 CKKS，输出已分配 CKKS。职责：为每条计算指令选唯一的 `compute_place`；算出各结果需要出现的位置集合；使用编译期已知的固定拓扑；校验位置和目标 Api 能力；权衡 Host/Device、延迟、带宽、显存和密钥位置；**不插入实际通信**。

### 7.3 通信显式化

输入已分配 CKKS，输出可执行 CKKS。职责：

- 把计算输出规范化到计算位置；
- 对跨位置的使用插入 Transfer/Replicate；
- 同一个目标位置只物化一次，复用已有的 ValueId；
- Replicate 的每个目标生成不同的输出 ValueId；
- 把每个使用者的操作数改写成其所在位置对应的 ValueId；
- 生成稳定的传输来历信息；
- 保证传输插入点支配（dominate）所有相关使用；
- 生成 CommKind 和可选的 CommHint；
- 不把具体的 MPI/NCCL/CUDA 调用写进通用 IR。

### 7.4 可执行验证

检查：所有计算指令有唯一 Place；每个 ValueId 恰好属于一个 Place；所有计算操作数都在本地；不再有未满足的 `result_places`；transfer 的源已定义且位置合法；结果类型和源的数学类型一致；来源/目标属于编译目标；通信动作的输入输出关系合法；Replicate 的输出数等于目标数且一一对应；不允许一个输出 ValueId 对应多个目标。

### 7.5 翻译

把可执行 CKKS 转成 C++ 的 RuntimePlan。首期不需要新 dialect。

## 8. 等待与异步的表示

首期可执行 SSA 只包含 Transfer/Replicate，runtime 在消费前检查值状态（Pending 就等，Ready 就用，缺失就报错），所以不需要显式的 WaitOp。

如果以后编译器要精确控制通信计算重叠、预取或 host 同步，可以下降为：

~~~mlir
%pending, %token = dist.transfer_async %2
%ready = async.await %pending
~~~

MLIR 的 async dialect 可以承担 token 和 await 的语义，不必自建一套。

## 9. C++ Demo 类型映射

`ssa_ops_def.hpp` 建议演化为（现有"一个值绑多个设备"的 `id_devices_pair` 写法与单位置原则冲突，重写时删除）：

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
    AddCC, AddCP,
    SubCC, SubCP,
    MulCC, MulCP,
    Negate, Rotate,
    Rescale, ModSwitch,
    Relinearize, Boot,
    Transfer, Replicate, Gather,
    Input, Output
};

// 仅编译器内部（已分配阶段）使用，不进入 RuntimePlan。
struct PlacedComputeAttrs {
    Place compute_place;
    std::vector<Place> required_result_places;
};

struct ComputeAttrs {
    // 可执行阶段：计算结果只位于这一个 Place。
    Place place;
    // rotate 步数、目标 level 等算子专属属性
};

enum class CommHint {
    Auto, PointToPoint, Collective, Broadcast,
    GatherPrimitive, Tree, Ring, HostStaged
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
    // 验证器保证每个 id 只出现一次，因此只对应一个 place。
    std::vector<ExecutableValueDesc> values;
    std::vector<SsaOp> ops;
};
~~~

对 Replicate，`outputs.size()` 必须等于 `destinations.size()`，且 `outputs[i]` 的值描述符必须指向 `destinations[i]`。实现可以用模板或更严格的分类型算子，但要避免所有算子共用一组含义不明的平铺字段。

这套类型与 poseidon::mgpu 的 `MgpuOp`（ops + device_id + 整数属性表）语义兼容：`Place{rank=0, index=d}` 退化后就是 mgpu 的 `device_id = d`，Transfer 对应 mgpu 的 CopyCipher/CopyPlain。迁移对接时可以写一个 MgpuSchedule 到 RuntimePlan 的直接翻译。

## 10. 编译难点与首期限制

首期明确拒绝：一般控制流；跨循环迭代的密文；运行时才能确定的值类型或通信结果布局；后端动态改变密文形状；同一条计算指令有多个执行 Place；隐式跨 Place 操作数；效果未知的算子。

后续需要重点处理：位置分配与通信成本的联合优化；块参数和 phi 类值；循环迭代中的传输身份；活跃性分析与异步发送的生命周期；稳定的调度/通道 ID；目标 Api 支持范围的编译期合法化；所有 rank 的计划一致性。
