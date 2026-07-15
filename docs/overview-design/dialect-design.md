# Dialect 与 SSA 设计

## 1. 文档范围

本仓库暂不实现 MLIR dialect，但 C++ demo 里的 SSA 类型、算子分类、位置分配、通信和验证器，必须和未来 MLIR 下降（lowering）的结果保持一致。本文档约束：

- 未来逻辑 CKKS dialect 的语义；
- 目标相关合法化、设备分配和通信显式化这些编译步骤；
- 可执行 SSA 到 RuntimePlan V1 JSON 的对应关系；
- Runtime 内部 `runtime/plan.hpp` 的目标类型映射；
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

这一阶段：指令没有通信副作用；Place 尚未确定；值表示数学对象而不是具体 buffer；CKKS 的 level、`scale_log2`、参数上下文等语义必须可验证；rescale、relinearize、rotate、bootstrap 等必要算子已显式存在，或可由语义合法化步骤确定。

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

设备分配 Pass 不需要新 dialect，它只是给 CKKS 算子加属性。跨 Dacapo 和 Runtime 的稳定边界已经确定为 RuntimePlan V1 JSON:可执行 CKKS 直接序列化成 JSON,Runtime 再解析成自己的 C++ 结构。V1 不需要额外创建一个 ckks-runtime MLIR dialect;以后只有在确实出现多个 MLIR 消费方时再考虑。

## 4. 类型设计

### 4.1 CKKS 语义类型

逻辑类型至少区分明文（Plaintext）和密文（Ciphertext）。可选的静态描述包括：参数/上下文标识、多项式阶数、level、`scale_log2`、是否 NTT 形式、密文分量个数、编码方式。

两个整数的含义固定如下:

- `level` 是当前模数链层号。较大的值表示还保留更多 RNS 模数;Rescale 或 ModSwitch 后它下降。不要把它解释为浮点数或 `2^x`;
- `scale_log2` 是 scale 的二进制指数,使用非负整数。值 40 表示逻辑 scale 为 `2^40`,计划中不保存浮点 scale。

Dacapo 当前 Earth 类型里的 `scale` 已经是这个整数指数。对外协议改名为 `scale_log2`,只是把原有含义写清楚。

Dacapo 的 Earth 分析阶段用“已经消耗多少层”的方向记录 level,Rescale 后数值增加;下降到 CKKS PolyType 时会换算成“还剩多少层”,Rescale 后数值下降。RuntimePlan V1 固定采用后者。序列化必须发生在这个换算之后。

在逻辑和已分配阶段,这些元信息可以放在类型、值属性或分析结果中;导出可执行计划之前必须全部落到每个值的描述符中,不能再是“可选信息”。

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

**纯计算**：AddCC、AddCP、SubCC、SubCP、MulCC、MulCP、Negate、Rotate、Rescale、ModSwitch、Relinearize、Boot。逻辑上引用透明：`outputs = f(inputs)`。Api 在物理上分配和写入输出 buffer 不算逻辑副作用；计算函数本身不得主动发起通信。计算 Place 可以是 Device,也可以是目标明确支持的 Host;不能在 Runtime 里写死“所有计算只能在 GPU”。

Boot 有两种明确的实现模式:

- `native`:使用目标后端的原生 boot;
- `decrypt_reencrypt`:只允许放在 Host,用 CPU 解密、解码、重新编码和加密来模拟 boot。

第二种模式会暴露明文,只用于测试和联调。它必须由编译 Pass 明确选择,不能由 Api 在原生 Boot 失败后偷偷回退。

**通信**：V1 只有 Transfer 和 Replicate。将来可以增加 Gather、Scatter、AllGather 或分片转换,但要先定义清楚数学关系并升级协议。Transfer/Replicate 对数学值和 CKKS 元信息都是原样搬运,但在位置、资源、错误和完成状态上有实际效果,不能被当成可随意删除的空操作。

**外部输入输出**：运行时的输入绑定、上传/下载、结果发布。

### 5.1 明文常量:双形态 constant

神经网络推理的权重、bias 和 rotate-and-sum 的 mask,在 IR 里统一用一个 `Pure` 的 constant 算子表示,payload 二选一:

- **内联形态**:数据直接放在 attribute 里(用 MLIR 的 `DenseResourceElementsAttr` 存 blob,避免大数据常驻 attribute uniquer),适合小 mask、手写测试;
- **引用形态**:attribute 只存 `{bundle_key, 元信息}`,`bundle_key` 是浮点数据的内容哈希,数据本体在计划旁的[明文数据包](../runtime-plan/v1/plaintext-bundle.md)里。

两种形态对使用者透明,fold/CSE 按 attribute 相等合并——由于引用形态的 key 就是内容哈希,"字节相同 ⇒ key 相同 ⇒ 被 CSE"是免费的去重(重复 mask、低内存测试里复用的占位常量都受益)。不需要单独的 load 算子,constant 本身就是取值点。

导出前由**常量外化 pass**(带字节阈值参数)把超过阈值的内联常量的浮点向量原样写入数据包并原地改写为引用形态;阈值 0 即全部外化。外化只搬字节,**不做 CKKS 编码**——编码由 Runtime 装载时按 ValueDesc 声明的 level/`scale_log2` 执行(见数据包文档),编译器因此不依赖任何同态加密库,数据包体积也停留在浮点向量的量级,而不是编码后多项式的量级。**导出 RuntimePlan 时必须全外化**:每个存活的 constant 映射为一条 Host 上的 plaintext external_input,计划文件里永远不含数据字节。placement 和通信显式化把 constant 的结果当普通 Host 明文值处理,不感知"这是权重"。

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

在设备分配之前完成：算子类型合法化、`scale_log2` 和 level 规划、rescale 插入、relinearize 插入、rotate 分解、bootstrap 决策和密钥需求分析。这些步骤会改变图结构和密文大小，所以要放在分配之前。

### 7.2 目标相关合法化

这个 Pass 读取 TargetSpec 和版本化的 CKKS OperatorSpec,把通用 CKKS 图变成目标真正能执行的图。OperatorSpec 至少给出 context/RNS 限制、算子支持范围、元信息变化、代价、rescale 模式和 boot profile。

rescale 的规则是:

- Poseidon CPU profile 使用 eager 模式,不要求 lazy-rescale;
- 当前低 bit Poseidon GPU profile 固定为 lazy 模式,选中该 profile 时 Dacapo 必须启用相应变换;
- pass 产生的 Rescale 位置、每个值的 level 和 `scale_log2` 都会进入后续计划,Runtime 不再改变。

GPU boot 暂不可用时,可选 Pass 把 Boot 标成 `implementation=decrypt_reencrypt`,并限制 `compute_place` 必须是 Host。通信显式化完成后,它自然形成下面的计划:

~~~mlir
%host_in = dist.transfer %gpu_in {
  source = #place.device<rank=0, index=0>,
  destination = #place.host<rank=0>
}

%host_out = ckks.boot %host_in {
  place = #place.host<rank=0>,
  implementation = #ckks.boot_impl<decrypt_reencrypt>,
  target_level = 6,
  target_scale_log2 = 40,
  target_components = 2,
  operator_profile = "poseidon-cpu-boot-emulation-v1"
}

%gpu_out = dist.transfer %host_out {
  source = #place.host<rank=0>,
  destination = #place.device<rank=0, index=0>
}
~~~

Boot 内部如果也需要 lazy-rescale,其合法输入范围、实际 level 消耗和输出规则由 boot profile 给出。不要把 CPU 的固定消耗直接套到 GPU。

### 7.3 设备分配

输入逻辑 CKKS，输出已分配 CKKS。职责：为每条计算指令选唯一的 `compute_place`；算出各结果需要出现的位置集合；使用编译期已知的固定拓扑；校验位置和目标 Api 能力；权衡 Host/Device、延迟、带宽、显存和密钥位置；**不插入实际通信**。

### 7.4 通信显式化

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

### 7.5 可执行验证

检查：所有计算指令有唯一 Place；该 Place 支持对应算子和 boot implementation；每个 ValueId 恰好属于一个 Place；所有计算操作数都在本地；每个值都有完整的 context、level、`scale_log2`、NTT 和分量数；算子目标元信息与输出值描述一致；OperatorSpec/profile 引用有效；不再有未满足的 `result_places`；transfer 的源已定义且位置合法；传输前后数学类型和 CKKS 元信息一致；来源/目标属于编译目标；通信动作的输入输出关系合法；Replicate 的输出数等于目标数且一一对应；不允许一个输出 ValueId 对应多个目标。

### 7.6 导出 RuntimePlan V1

把可执行 CKKS 序列化成 RuntimePlan V1 JSON。JSON 是 Dacapo 与 Runtime 之间唯一的稳定协议。Runtime 读取后可以转成自己的 C++ `RuntimePlan`,但两边不共享 C++ 类型。首期不需要新 MLIR dialect。

## 8. 等待与异步的表示

首期可执行 SSA 只包含 Transfer/Replicate，runtime 在消费前检查值状态（Pending 就等，Ready 就用，缺失就报错），所以不需要显式的 WaitOp。

如果以后编译器要精确控制通信计算重叠、预取或 host 同步，可以下降为：

~~~mlir
%pending, %token = dist.transfer_async %2
%ready = async.await %pending
~~~

MLIR 的 async dialect 可以承担 token 和 await 的语义，不必自建一套。

## 9. Runtime 内部类型映射

Runtime 读完 V1 JSON 后,内部 C++ 类型至少要能表达下面这些信息。这里描述的是目标形态,不是跨仓库 ABI。当前 `runtime/plan.hpp` 已经有单位置 SSA 和三阶段计划,但还缺 CKKS 元信息、OperatorSpec 引用、整数 `scale_log2` 和 Host Boot 模式。

~~~cpp
using ValueId = std::uint64_t;
using ScaleLog2 = std::uint32_t;

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

struct CkksMetadata {
    std::string context_id;
    std::uint64_t polynomial_degree;
    std::uint32_t level;
    ScaleLog2 scale_log2;
    bool ntt;
    std::uint32_t components;
};

struct ValueDesc {
    ValueId id;
    ValueKind kind;
    Place place;
    CkksMetadata metadata;
};

enum class BootImplementation {
    Native,
    DecryptReencrypt
};

struct RotateAttrs {
    int steps;
};

struct RescaleAttrs {
    std::uint32_t target_level;
    ScaleLog2 target_scale_log2;
};

struct ModSwitchAttrs {
    std::uint32_t target_level;
};

struct BootAttrs {
    BootImplementation implementation;
    std::uint32_t target_level;
    ScaleLog2 target_scale_log2;
    std::uint32_t target_components;
    std::string operator_profile;
};

using ComputeAttrs = std::variant<
    std::monostate, RotateAttrs, RescaleAttrs, ModSwitchAttrs, BootAttrs>;

struct ComputeOp {
    ComputeKind kind;
    std::vector<ValueId> inputs;
    ValueId output;
    Place place;
    ComputeAttrs attrs;
};

struct OperatorSpecRef {
    std::string id;
    std::uint32_t version;
    std::string fingerprint;
};

enum class RescaleMode {
    Eager,
    Lazy
};

enum class BootMode {
    Native,
    DecryptReencrypt
};

struct TargetConfig {
    std::string target_id;
    std::uint32_t capability_version;
    OperatorSpecRef operator_spec;
    RescaleMode rescale_mode;
    BootMode boot_mode;
    int world_size;
    std::vector<int> device_counts;
};

struct RuntimePlan {
    std::uint32_t format_version;
    std::uint64_t plan_id;
    std::string fingerprint;
    TargetConfig target;
    std::vector<ValueDesc> values;
    std::vector<ValueId> external_inputs;
    std::vector<Instruction> initialization;
    std::vector<Instruction> execution;
    std::vector<Instruction> finalization;
    std::vector<ValueId> final_outputs;
};
~~~

Rescale 和 Boot 不再使用 `double scale_divisor` 或 `double scale`;它们直接写目标 level 和整数 `target_scale_log2`。对 Replicate，`outputs.size()` 必须等于 `destinations.size()`，且 `outputs[i]` 的值描述符必须指向 `destinations[i]`。实现可以用模板或更严格的分类型算子，但要避免所有算子共用一组含义不明的平铺字段。

单位置和显式拷贝这两个核心不变量仍与 poseidon::mgpu 的 `MgpuOp` 兼容:`Place{rank=0, index=d}` 退化后就是 mgpu 的 `device_id=d`,Transfer 对应 CopyCipher/CopyPlain。但 mgpu 当前没有完整 CKKS 元信息、Host compute 和 OperatorSpec,所以迁移时只能复用它的拷贝与执行能力,不能把旧的整数属性表原样当成 V1 协议。

## 10. 编译难点与首期限制

首期明确拒绝：一般控制流；跨循环迭代的密文；运行时才能确定的值类型或通信结果布局；后端动态改变密文形状；同一条计算指令有多个执行 Place；隐式跨 Place 操作数；效果未知的算子。

后续需要重点处理：位置分配与通信成本的联合优化；块参数和 phi 类值；循环迭代中的传输身份；活跃性分析与异步发送的生命周期；稳定的调度/通道 ID；目标 Api 支持范围的编译期合法化；所有 rank 的计划一致性。
