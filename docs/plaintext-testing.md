# 明文测试方案

## 1. 目标

本仓库使用 VecApi 和 MockCommunicationApi 验证 runtime，不验证真实 CKKS 密码学或 GPU 性能。

测试需要证明：

- executable plan 的结构和 effect 合法；
- executable ValueId 严格遵守单位置物理 SSA；
- Host/Device Place 正确；
- 每个模拟 rank 都由独立 Runtime 实例解释同一物理计划；
- 不同 Runtime 之间不共享 ValueStore 或 Api::Value；
- Transfer/Replicate 和 Hint 正确交给 Api；
- Pending output 在使用前被 wait；
- Api fallback 不改变结果；
- 多 rank/Device 的最终结果与单 Place 顺序参考流一致；
- 可选逐指令 difftest 能定位第一个错误 Value，且不在执行途中插入同步比较；
- 任意错误都会暴露详细上下文并终止 execution group；
- 未来 PoseidonApi 接入时不需要改变 Runtime。

## 2. VecApi

Plaintext 和 Ciphertext 使用不同类型：

~~~cpp
template <typename T>
struct VecPlaintext {
    std::vector<T> slots;
    PlainMetadata metadata;
};

template <typename T>
struct VecCiphertext {
    std::vector<T> slots;
    CipherMetadata metadata;
};

using VecValue =
    std::variant<VecPlaintext<double>, VecCiphertext<double>>;
~~~

VecApi 实现：

- add/sub/mul；
- add_plain/sub_plain/mul_plain；
- negate；
- rotate；
- rescale；
- modswitch；
- relinearize；
- boot；
- communicate_async；
- wait；
- abort_all。

数值可以逐元素模拟，但 metadata 不应全部 no-op：

- multiply 改变 scale/component count；
- relinearize 恢复 component count；
- rescale 改变 level/scale；
- modswitch 改变 level；
- boot 按测试配置重置 level/scale；
- rotate 规范化 step。

## 3. MockCommunicationApi 与 MockCluster

每个 rank 拥有自己的 MockCommunicationApi。多个 Api 实例只通过一个线程安全的 MockCluster 交换消息：

~~~text
Runtime(rank=0) -- MockCommunicationApi(0) --+
                                                |
Runtime(rank=1) -- MockCommunicationApi(1) -----+-- MockCluster
                                                |   - mailbox/event
Runtime(rank=2) -- MockCommunicationApi(2) -----+   - abort state
                                                    - delay/failure injection
~~~

每个 Runtime 必须拥有独立的 LocalIdentity、ValueStore 和 Api 状态。MockCluster 只模拟传输设施和 execution-group abort，不能直接读取其他 Runtime 的 ValueStore，也不能让 source/destination 共享同一个 VecValue 对象。发送时应复制 payload 和 metadata，避免 C++ 对象别名掩盖通信错误。

Mock Api 使用可控的 mailbox/event 模型：

~~~text
communicate_async
  -> 建立 pending handle
  -> 根据 CommKind 记录 source/destination
  -> 根据测试配置立即或延迟完成

wait
  -> 完成模拟通信
  -> 返回当前 rank 的 outputs
~~~

支持：

- Transfer；
- Replicate；
- HostToDevice；
- DeviceToHost；
- DeviceToDevice；
- Auto、PointToPoint、Broadcast hint；
- Broadcast hint fallback 到多个 point-to-point；
- destination 先到或 source 先到；
- 延迟完成；
- 指定 CommAction 失败；
- output 数量或类型错误注入；
- abort_all 转换为 ClusterPanic。

多 rank 测试使用一个 host thread（建议 std::jthread）运行一个 Runtime。所有线程只在测试开始和结束处汇合，不设置逐 op 全局 barrier；不同 Runtime 可以以不同速度到达同一个 CommAction。MockCluster 支持固定延迟、指定到达顺序或带固定 seed 的随机延迟，用于重复覆盖不同 interleaving。

首期不需要独立 progress thread、真实网络、retry、cancel 或资源恢复。

## 4. 编译目标模拟

测试 plan 使用固定物理 Place：

~~~text
Host(rank=0)
Device(rank=0,index=0)
Device(rank=0,index=1)
Host(rank=1)
Device(rank=1,index=0)
~~~

模拟多 rank 不使用 MPI。测试进程内为每个 rank 构造一个独立 Runtime，并发解释同一 plan：

~~~text
rank 0 -> Runtime0(LocalIdentity{0}, ValueStore0, Api0)
rank 1 -> Runtime1(LocalIdentity{1}, ValueStore1, Api1)
...
rank N -> RuntimeN(LocalIdentity{N}, ValueStoreN, ApiN)
~~~

每个实例通过 Place.rank 判断角色。一个 rank 内可以拥有多个 Device Place，但它们仍由该 rank 的同一个 Runtime 管理。多 rank 测试不得退化为“一个 Runtime 循环切换 local_rank”，因为这种写法无法覆盖独立执行进度、并发 wait、全局 abort 和消息到达顺序。

需要覆盖：

- 单 rank 多 Device；
- 2、4 个独立 Runtime/rank；
- 1、2、4、6、8 个逻辑 Device；
- rank 间执行速度不同；
- source/destination 到达 CommAction 的顺序不同；
- world size 不匹配；
- 本地 Device 数量不匹配。

## 5. 测试层次

### 5.1 类型和 Compute

- Plaintext/Ciphertext 不能混用；
- op arity；
- rotate 正数、负数、零和大步数；
- Vec 输入长度；
- metadata transition；
- unsupported compute op；
- compute error。

### 5.2 Plan Verifier

- 重复 ValueId；
- 一个 ValueId 被绑定到多个 Place；
- use-before-def；
- 未定义输入；
- 错误 ValueType；
- compute op 带通信 effect；
- compute Place 非法；
- 隐式跨 Place operand；
- 重复 TransferId；
- CommKind input/output 数量错误；
- Replicate output 数量与 destination 数量不一致；
- Replicate 的多个 destination 复用同一个 output ValueId；
- source/destination Place 非法；
- Gather result layout 缺失；
- 未知 CommHint；
- world size 与编译目标不匹配。

### 5.3 单 Place Runtime

- 纯线性 add/mul；
- unary op；
- plaintext/ciphertext 混合；
- 同一输入重复使用；
- 多 output；
- metadata mismatch；
- Api 返回错误类型。

### 5.4 Host/Device Initialization

~~~text
%w_host = input @Host0
%w_gpu = transfer %w_host Host0->Device0
%y = mul_plain %x, %w_gpu @Device0
~~~

检查：

- Host input 正确绑定；
- Initialization 执行 transfer；
- Execution 开始前 required Value Ready；
- HostToDevice 保持数学值和 ValueType；
- plaintext 不因为类型而自动上传，必须依赖计划 action；
- 同一常量上传到多个 Device 时，每个目标获得独立 ValueId，且数学值一致。

### 5.5 多 Device Transfer

~~~text
%2 = add %0, %1 @Device(rank0,0)
%3 = transfer %2 Device(rank0,0)->Device(rank0,1)
%4 = mul %3, %x @Device(rank0,1)
~~~

覆盖：

- 同 rank DeviceToDevice；
- 跨 rank DeviceToDevice；
- source/destination 同 rank；
- non-participant no-op；
- consumer 等待 Pending；
- source handle 在 Finalization 被 wait。

### 5.6 Replicate 与 Fallback

~~~text
%3, %4 = replicate %2 {
  source=Device(rank0,0),
  destinations=[
    Device(rank0,1),
    Device(rank1,0)
  ],
  hint=Broadcast
}
~~~

检查：

- `%3` 与 `%4` 分别只属于 destinations[0] 与 destinations[1]；
- 每个 destination 有独立 output ValueId；
- wait 的 output slot 按计划位置映射到对应 ValueId；
- Api 可以按 Broadcast 实现；
- Api 可以 fallback 到多个 point-to-point；
- 两种实现结果一致；
- debug trace 记录实际实现；
- Runtime 不参与 fallback。

### 5.7 异步顺序

- destination 比 source 更早到达 CommAction；
- source 比 destination 更早到达；
- consumer 比通信完成更早到达；
- 独立 op 可以在 Pending 期间执行；
- wait 后 output 进入 Ready；
- source Value 保留到 Finalization；
- 同一 plan 重复 run 不串 TransferId。

### 5.8 Fail-fast

注入：

- compute 抛错；
- communicate_async 抛错；
- wait 抛错；
- output 数量不匹配；
- output ValueType 不匹配；
- Place 不匹配；
- world size 不匹配；
- unsupported CommKind；
- 无等价 fallback。

检查：

- 打印 op ordinal/kind；
- 打印 ValueId/TransferId；
- 打印 source/destination Place；
- 打印 local rank 和 Api 名称；
- abort_all 被调用；
- 所有模拟 Runtime 停止；
- 不继续执行后续 op。

### 5.9 多 Runtime 隔离

检查：

- rank 数量等于 Runtime 实例数量；
- 每个 Runtime 的 LocalIdentity 唯一；
- 每个 Runtime 只持有本 rank 已物化的 Value；
- source 和 destination 的 VecValue 地址不同；
- 删除任意一侧 communicate_async 都会造成明确失败，而不是因共享内存仍得到正确值；
- 一个 Runtime 调用 abort_all 后，其他 Runtime 的 wait 被唤醒并终止；
- 不使用逐 op barrier 维持表面上的正确顺序。

## 6. 代表性图

### 6.1 线性 HostToDevice

~~~text
Host input
  -> Transfer
  -> Device compute
  -> DeviceToHost result
~~~

### 6.2 Fan-out

~~~text
Device0 compute
  -> %3, %4, %5 = Replicate to Device1/2/3
  -> 三条独立分支
~~~

### 6.3 分支汇合

~~~text
branch A @Device0
branch B @Device1
  -> Transfer to Device2
  -> add @Device2
~~~

### 6.4 同一输入两次

~~~text
%3 = add %2, %2
~~~

### 6.5 Fallback

同一 Replicate plan 分别由：

~~~text
MockBroadcast
MultiplePointToPoint
~~~

执行，最终结果必须相同。

## 7. Difftest

### 7.1 两条执行流

每个端到端算例由同一个测试描述构造两条执行流：

~~~text
Sequential reference
  - 单 rank、单 Place
  - VecApi 顺序执行
  - 不包含物理通信

Distributed execution
  - 多 Device 或多 rank
  - 每个 rank 一个 Runtime
  - 执行 placement 后的 executable plan
  - 使用 MockCommunicationApi
~~~

两条流使用内容相同但对象独立的输入。测试侧维护 DiffMap，把 distributed ValueId 映射到具有相同数学语义的 reference ValueId：

~~~cpp
struct DiffPoint {
    std::size_t op_ordinal;
    ValueId distributed_value;
    ValueId reference_value;
};
~~~

DiffMap 是 plan builder 或未来编译器产生的测试/debug artifact，不进入 Runtime/Api 的正式执行语义。对于 Transfer/Replicate 输出，其 reference_value 与 source 对应的参考数学值相同，因此多个物理 ValueId 可以分别与同一个 reference ValueId 比较。

### 7.2 最终结果 diff

最终结果 diff 是所有成功端到端测试的必选项，默认开启：

1. 顺序执行 reference；
2. 并发运行全部 distributed Runtime；
3. 等待全部 Runtime 结束；
4. 按计划声明的最终 output 收集所属 rank 的结果；
5. 与 reference output 比较。

该比较只发生在 distributed execution 完全结束以后，不注册逐 op callback，不增加 Runtime 间 barrier，也不影响通信时序。

最终 diff 检查：

- slot 数量和值；
- ValueKind；
- level/scale/component 等 metadata；
- distributed output 位于计划声明的唯一 Place；
- output 缺失或被两个 Runtime 重复持有；
- NaN/Inf 分类；
- 配置的 absolute/relative tolerance。

### 7.3 可选逐指令 diff

逐指令 diff 默认关闭，通过独立测试选项开启，例如：

~~~text
DiffMode::FinalOnly       // 默认
DiffMode::AllValuesAfterRun
~~~

`AllValuesAfterRun` 不采用 lock-step 执行，也不在每条指令后立即比较。首期 Runtime 不做 last-use 回收，SSA Value 在 run 内不可变并保留到 Finalization，因此可以：

1. 先让所有 Runtime 按正常异步语义完整运行；
2. 在 plan 执行结束后等待所有 source handle，并把本 rank 仍为 Pending 的 diff-point output 全部变为 Ready；
3. 在 ValueStore 清空前，由测试代码复制 test-only RunArtifact；
4. 合并各 rank artifact，并检查每个 ValueId 只来自其唯一 Place；
5. 按 op ordinal 遍历 DiffMap，离线比较每条有结果指令的 output；
6. 报告第一个 mismatch，并可继续汇总后续 mismatch。

这种方式不会把 diff 计算延迟放进 compute、send、receive 或正常 consumer wait 的执行路径。`AllValuesAfterRun` 只可能在所有 plan action 都已发起后增加 finalization wait 和 artifact 拷贝，因此不能用于性能计时；关闭逐指令 diff 时，不生成 RunArtifact，也没有逐 op trace、拷贝或回调开销。

逐指令报告至少包含：

- op ordinal/kind；
- distributed/reference ValueId；
- owner rank 和 Place；
- 第一个错误 slot；
- expected/actual；
- absolute/relative error；
- ValueType 和 metadata 差异。

无结果的指令只检查执行 trace 和错误状态，不做数值 diff。加入 Release/Evict 或 buffer reuse 后，历史 Value 可能不再保留；届时若仍需要逐指令 diff，应由编译器插入显式 debug capture，不能让 Runtime 暗中延长生产执行的生命周期。

### 7.4 比较规则

- 当前 VecApi 可以选择 exact 模式或配置 abs/rel tolerance；
- metadata、ValueKind、slot 数量和 SSA/Place 归属必须精确一致；
- reference Place 与 distributed Place 不需要相同，distributed Place 只和 executable plan 比较；
- 不比较 C++ 地址、buffer layout 或通信 fallback 实现；
- FHE backend 的 ciphertext 表示可能不确定，未来应比较解密后的语义值；逐指令密文 diff 不属于当前 demo 范围。

## 8. 非数值执行检查

除了 difftest，每个端到端测试还应检查：

- CommAction 启动次数；
- Hint 与实际 fallback trace；
- wait 次数；
- source handle 在 Finalization 完成；
- abort_all 是否按预期调用；
- 每个 Runtime 只执行本 rank 的 compute；
- 所有 Runtime 都完成或一起 fail-fast。

## 9. 大 Plaintext 的后续测试

增加 memory planning 后再覆盖：

- plaintext 不能全部驻留 Device；
- Prefetch 在首次 use 前完成；
- Release 后对应 Device ValueId 的物理 materialization 不可用；
- 从 Host 重新 materialize；
- 多次使用的 plaintext 保持合适 residency；
- memory high-water 不超过编译预算。

这些属于第二阶段，不进入首期 runtime。

## 10. MPI 测试定位

核心明文多 rank 测试不通过 MPI 实现，而是在同一测试进程中并发运行多个独立 Runtime。这样可以快速注入乱序、延迟和错误，同时仍然覆盖 Runtime 间的独立执行进度和 ValueStore 隔离。

现有 mpi_test.cpp 和 mpi_test_comm.cpp 只验证：

- MPI 初始化；
- rank/size；
- Allreduce；
- Sendrecv；
- host-buffer 延迟和带宽。

它们不证明 executable plan、Api fallback、Host/Device materialization 或 fail-fast 正确。

未来 MpiApi integration test 应与 MockCommunicationApi 测试分开，并采用：

~~~text
mpiexec -n N
  -> 每个 MPI process 一个 Runtime
  -> 每个 process 的 local rank 来自 MPI_Comm_rank
  -> 所有 process 读取同一 executable plan
~~~

MPI integration test 至少复用最终结果 diff。逐指令 diff 仍应在运行结束后由测试驱动汇总各 rank artifact，不能在每条指令后增加 MPI_Barrier。

## 11. 首期完成标准

1. 可构造并打印严格单位置 ValueId 的 executable plan。
2. Verifier 能拒绝 SSA、Place、单 ValueId 多位置和 CommAction 错误。
3. VecApi 能执行核心计算并维护 metadata。
4. MockCommunicationApi 支持 Transfer、Replicate 和 Broadcast fallback。
5. Initialization 能执行 HostToDevice 预加载。
6. 多 rank 算例为每个 rank 创建独立 Runtime、ValueStore 和 Api 实例。
7. Runtime 能模拟 1、2、4、6、8 个 Device，并覆盖至少 2、4 个 rank。
8. Pending output 在 consumer 前被 wait。
9. Finalization 等待所有 source handle。
10. 任意错误打印详细上下文并调用 abort_all，使全部 Runtime 停止。
11. 最终结果 diff 默认开启，且结果与单 Place 顺序 reference 一致。
12. 可选逐指令 diff 在运行结束后离线执行，不插入逐 op barrier。
