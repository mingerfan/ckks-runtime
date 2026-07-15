# 明文测试方案

> 本文同时记录当前测试设施和 RuntimePlan V1 草案的目标测试。当前已有 VecExecutor、MockVecApi、MockCluster 和 MpiVecApi。待补部分包括显式 Encode、bundle preflight、Host compute、原始文件摘要和调试跳过策略测试，详见[实现状态](implementation-status.md)。

## 1. 目标

本仓库用 `VecExecutor + MockVecApi/MockCluster` 和 `MpiVecApi` 验证 runtime 本身，不验证真实的 CKKS 密码学或 GPU 性能。

测试要证明：

- 可执行计划的结构和效果合法；
- 每个 ValueId 严格只属于一个位置；
- Host/Device 的 Place 处理正确；
- 每个模拟 rank 由独立的 runtime 实例解释同一份计划；
- 不同 runtime 之间不共享值存储或 `Api::Value`；
- Transfer/Replicate 和 hint 被正确交给 Api；
- Pending 的输出在使用前被等待；
- Api 换一种等价的通信实现方式，结果不变；
- 多 rank/多设备的最终结果和单设备顺序执行一致；
- 可选的逐指令对比能定位第一个出错的值，且不在执行途中插入同步；
- 任意错误都带详细上下文，并终止整个执行组；
- 未来接入 PoseidonApi 时不需要改 runtime。

## 2. VecExecutor 与完整 Api

当前实现分成三层:

- `VecExecutor`:只实现明文计算内核,支持同步/异步两种模式;
- `MockVecApi`:组合 VecExecutor 和 MockCluster,实现当前 Runtime 接口需要的模拟 Api;
- `MpiVecApi`:复用 VecExecutor 的计算语义,用 MPI 实现当前 Runtime 接口。

文档旧称的“VecApi”是这套明文参考后端的统称,不是当前代码中的一个类名。

明文和密文用不同类型表示：

~~~cpp
struct VecPayload {
    ValueKind kind;
    std::vector<double> slots;
    VecMetadata metadata;
};

class VecValue {
    // 当前 Pending 句柄只保存 expected kind。
    // 目标接口还要在创建时保存不可变 CKKS 元信息，slots 可以稍后填入。
};
~~~

- 当前 `VecExecutor` 已实现 add/sub/mul、明密文混合运算、negate、rotate、rescale、modswitch、relinearize 和 boot。
- `MockVecApi`/`MpiVecApi` 当前提供内部 64 位摘要 `preflight`、通信、等待、最终同步和全组终止。
- 目标接口还要把 `preflight` 扩展到计划原始字节 SHA-256、摘要调试策略和 target/context/密钥检查，并增加 `validate_value` 和 `encode_plaintext`。Vec 后端不做密码学编码，只把 slots 和输出 ValueDesc 组装成 Host plaintext。

数值部分逐元素模拟即可，但元信息不能全部空转，否则非法的计算图也能通过测试：

- 乘法按 CKKS 规则改变 `scale_log2` 和分量个数；
- relinearize 恢复分量个数；
- rescale 改变 level 和 `scale_log2`；
- modswitch 改变 level；
- boot 按测试配置重置 level 和 `scale_log2`；
- rotate 对步数做规范化。

V1 的元信息测试统一使用整数 `scale_log2`:值 40 表示逻辑 scale 为 `2^40`。当前 VecValue、RuntimePlan 和 MPI 元信息都已经使用整数 `scale_log2`;inline payload 中的浮点数只是待编码的 slot 数据,不是 scale。

还要测试显式的 `Device→Host → Host Boot → Host→Device` 模拟路径。两次 Transfer 必须保持元信息，Host Boot 必须精确产出目标 level 和 `scale_log2`；不能靠 GPU Boot 失败后的自动回退通过测试。

### 2.1 同步与异步两种模式

VecExecutor 提供两种执行模式，完整 Api 的接口不变，测试可以用同一份计划分别跑：

~~~text
VecExecMode::Sync    // 计算在调用线程上当场完成（默认）
VecExecMode::Async   // 每个模拟设备一个工作线程，模拟 GPU 的发起/执行分离
~~~

**同步模式**用于基础正确性测试，行为最简单、出错最好定位。

**异步模式**用来模拟真实 GPU 后端"发起串行、执行并行"的时序：

- 每个模拟设备（Place）一个工作线程和一个先进先出任务队列；
- `compute` 把任务放进目标设备的队列后立即返回，不等计算完成；
- `compute` 返回一个可传递的值句柄，任务完成后结果才写入句柄；后续任务真正用到它时再等待；
- 同一设备队列先进先出，天然满足"输出对后续调用可见"的契约；跨设备的依赖通过值的共享状态传递；
- `communicate_async` 把"取源值并发送"作为任务放进源设备的队列，保证它排在产生该值的计算之后；
- `wait` 是 Runtime 对异步通信的唯一显式等待点;同步模式的 `compute` 本身会占用调用线程；
- 可配置每个任务的固定或随机（带种子）延迟，用来放大不同设备之间的进度差。

异步模式专门要测的东西：

- 多卡独立分支真正并发执行时，最终结果仍与顺序参考一致；
- runtime 在某个值 Pending 期间继续发起后续指令，不会用到未就绪的数据；
- 指令排序的效果可观察：通信早发起晚消费时，模拟的总时间线上出现计算与通信重叠；
- 带种子的延迟注入能稳定复现不同的完成顺序。

同一批端到端算例应在两种模式下各跑一遍，结果必须一致。异步模式只在 VecExecutor 内部，runtime 和完整 Api 接口不感知模式差异——这本身就是对"异步契约"设计的一次验证。

## 3. MockVecApi 与 MockCluster

每个 rank 有自己的 MockVecApi，多个 Api 实例只通过一个线程安全的 MockCluster 交换消息：

~~~text
Runtime(rank=0) -- MockVecApi(0) --+
                                             |
Runtime(rank=1) -- MockVecApi(1) --+-- MockCluster
                                             |   - 消息队列/事件
Runtime(rank=2) -- MockVecApi(2) --+   - 终止状态
                                                 - 延迟/故障注入
~~~

每个 runtime 必须有独立的身份、值存储和 Api 状态。MockCluster 只模拟传输设施和整组终止，不能直接读其他 runtime 的值存储，也不能让发送方和接收方共享同一个 VecValue 对象。发送时必须复制数据和元信息——如果靠 C++ 对象共享就能"传"过去，真正的通信 bug 就被掩盖了。

Mock Api 的行为模型：

~~~text
communicate_async
  -> 建立 Pending 句柄
  -> 按通信类型登记来源/目标
  -> 按测试配置立即或延迟完成

wait
  -> 推进模拟通信
  -> 返回本 rank 的输出
~~~

Mock Api 覆盖：

- 通信：Transfer、Replicate，以及 Host↔Device、Device↔Device；
- hint：Auto、PointToPoint、Broadcast，并允许 Broadcast 降级为多个点对点；
- 时序：发送方或接收方先到、延迟完成；
- 错误：指定通信失败、输出数量/类型错误，以及 `abort_all` 转换为 ClusterPanic。

多 rank 测试用一个 `std::thread` 跑一个 runtime，并显式 join。所有线程只在测试开始和结束处汇合，不加逐指令的全局同步；不同 runtime 可以以不同速度到达同一条通信指令。MockCluster 支持固定延迟或带固定种子的确定性延迟，用来重复覆盖不同的交错情况。

首期不需要独立进度线程、真实网络、重试、取消或资源恢复。

## 4. 编译目标的模拟

测试计划使用固定的物理 Place，例如：

~~~text
Host(rank=0)
Device(rank=0, index=0)
Device(rank=0, index=1)
Host(rank=1)
Device(rank=1, index=0)
~~~

模拟多 rank 不用 MPI。测试进程内为每个 rank 建一个独立 runtime，并发解释同一份计划：

~~~text
rank 0 -> Runtime0(RuntimeEnvironment{rank=0, ...}, ValueStore0, Api0)
rank 1 -> Runtime1(RuntimeEnvironment{rank=1, ...}, ValueStore1, Api1)
...
~~~

每个实例通过 `Place.rank` 判断自己的角色。一个 rank 可以带多个 Device，它们由该 rank 的同一个 runtime 管理。

**多 rank 测试不允许退化成"一个 runtime 循环切换 rank"**——那种写法覆盖不了独立执行进度、并发等待、全组终止和消息到达顺序这些真正的风险点。

需要覆盖：单 rank 多设备；2、4 个独立 runtime；1、2、4、6、8 个逻辑设备；rank 之间执行速度不同；发送方/接收方到达通信指令的顺序不同；节点数不匹配；本地设备数不匹配。

## 5. 测试层次

### 5.1 类型和计算

明文/密文不能混用；参数个数错误；rotate 的正数、负数、零和大步数；向量长度；元信息变化；不支持的算子；计算出错。

### 5.2 计划验证器

按四组覆盖非法计划：

- **Encode/bundle**：阶段、输出类型、payload 二选一、inline 数值、bundle 目录/清单/blob/content 和 OperatorSpec 约束错误；
- **SSA/类型**：重复 ValueId、一值多 Place、先用后定义、未定义输入、错误值类型；
- **计算/Place**：计算带通信效果、计算位置非法、隐式跨 Place 操作数；
- **通信/目标**：重复 TransferId、输入输出数量错误、Replicate 映射错误、非法 Place、未知 hint、不支持的 Gather、节点或设备数不匹配。

### 5.3 单设备 runtime

纯线性的 add/mul；一元算子；明文密文混合；同一输入用两次；多输出；元信息不匹配；Api 返回错误类型。

### 5.4 Host/Device 初始化

~~~text
%w_host = encode inline_or_bundle_payload @Host0
%w_gpu = transfer %w_host Host0->Device0
%y = mul_plain %x, %w_gpu @Device0
~~~

检查分三组：

- **Encode 数据语义**：inline/bundle 都能生成正确 Host 明文；同一 `content` 按不同 level/scale 编码时得到不同 ValueId；external input 与 Encode 输出不混淆；
- **多 rank 行为**：每个 rank 都完成 bundle preflight，只有输出 Host 所属 rank 执行 Encode；
- **显式上传**：初始化 Transfer 完成后所需值才 Ready；Host→Device 保持数学值和元信息；多个设备上的副本各有自己的 ValueId。

### 5.5 多设备 Transfer

~~~text
%2 = add %0, %1 @Device(rank0,0)
%3 = transfer %2 Device(rank0,0)->Device(rank0,1)
%4 = mul %3, %x @Device(rank0,1)
~~~

覆盖：同 rank 设备间；跨 rank 设备间；来源目标在同一 rank；旁观者 rank 空操作；消费方等待 Pending；发送句柄在收尾阶段被等待。

### 5.6 Replicate 与降级

~~~text
%3, %4 = replicate %2 {
  source=Device(rank0,0),
  destinations=[Device(rank0,1), Device(rank1,0)],
  hint=Broadcast
}
~~~

检查：`%3`、`%4` 分别只属于各自的目标；每个目标有独立的输出 ValueId；`wait` 的输出按计划里的下标对应到 ValueId；Api 可以按广播实现，也可以降级成多个点对点；两种实现结果一致；调试日志记录实际用了哪种；runtime 不参与降级。

### 5.7 异步顺序

接收方比发送方先到达通信指令；发送方先到；消费方比通信完成先到；不相关的指令可以在 Pending 期间继续执行；`wait` 之后输出变 Ready；源值保留到收尾；使用新的 Runtime 实例重复运行同一份计划时，传输 ID 不串。当前 `SequentialRuntime` 实例本身是一次性的。

### 5.8 Fail-fast

当前注入：计算抛错；`communicate_async` 抛错；`wait` 抛错；输出数量不匹配；输出类型不匹配；Place 不匹配；节点数不匹配；不支持的通信类型；无等价降级。

加入 Encode 后还要注入：bundle 目录/manifest/blob 读取失败、摘要或内容哈希不符、`encode_plaintext` 失败、跨 rank `preflight` 不一致和最终 `synchronize` 失败。它们必须进入同一条 fail-fast 路径。

摘要策略还要单独测试两种模式:

- 默认模式:计划原始字节、OperatorSpec 或 manifest 摘要不符时必须拒绝;
- `skip_artifact_digest_checks=true`:只允许上述摘要不符继续执行,其他任一结构、语义、id/version 或 blob `content` 错误仍必须拒绝,并检查所有 rank 的开关值一致且日志包含调试警告。

检查：打印指令序号/类型、ValueId/传输 ID、来源/目标 Place、本地 rank 和 Api 名称；`abort_all` 被调用；所有模拟 runtime 停止；不继续执行后续指令。

### 5.9 多 runtime 隔离

检查：

- rank 数与 runtime 实例数一致，每个实例身份和 ValueStore 独立；
- 发送方与接收方得到深拷贝，不能靠共享 VecValue 对象“碰巧正确”；
- 任一 runtime 调用 `abort_all` 后，其他 runtime 的 `wait` 都被唤醒并终止；
- 测试没有靠逐指令全局同步维持顺序。

## 6. 代表性算例

~~~text
6.1 线性流:    Host 输入 -> Transfer -> Device 计算 -> 结果回 Host
6.2 扇出:      Device0 计算 -> Replicate 到 Device1/2/3 -> 三条独立分支
6.3 分支汇合:  分支A@Device0、分支B@Device1 -> Transfer 到 Device2 -> add
6.4 重复输入:  %3 = add %2, %2
6.5 降级对比:  同一个 Replicate 计划分别用广播和多个点对点执行，结果必须相同
~~~

## 7. Difftest（差分测试）

### 7.1 两条执行流

每个端到端算例由同一个测试描述构造两条执行流：

~~~text
顺序参考流:  单 rank、单 Place、VecExecutor 顺序执行、没有物理通信
分布式执行流: 多设备/多 rank、每个 rank 一个 runtime、
             执行分配后的可执行计划、使用 MockVecApi/MockCluster
~~~

两条流的输入内容相同但对象独立。测试侧维护一张 DiffMap，把分布式的 ValueId 映射到数学语义相同的参考 ValueId：

~~~cpp
struct DiffPoint {
    std::size_t op_ordinal;
    ValueId distributed_value;
    ValueId reference_value;
};
~~~

DiffMap 由计划构造器（或未来的编译器）生成，是测试/调试用的产物，不进入 Runtime/Api 的正式语义。Transfer/Replicate 的输出在数学上等于其来源，所以多个物理 ValueId 可以分别对到同一个参考 ValueId。

### 7.2 最终结果对比（默认开启）

1. 顺序执行参考流；
2. 并发运行全部分布式 runtime；
3. 等所有 runtime 结束；
4. 按计划声明的最终输出，从所属 rank 收集结果；
5. 与参考输出比较。

比较只发生在分布式执行完全结束之后：不注册逐指令回调，不加 runtime 间同步，不影响通信时序。

检查内容分三类：

- 数值：槽数量、NaN/Inf 分类，以及配置的绝对/相对容差；
- 类型和元信息：值类别、level、`scale_log2`、分量数；
- 归属：输出位于计划声明的唯一 Place，没有缺失，也没有被两个 runtime 重复持有。

### 7.3 可选的逐指令对比

默认关闭，用独立选项开启：

~~~text
DiffMode::FinalOnly          // 默认
DiffMode::AllValuesAfterRun  // 运行后逐指令对比
~~~

`AllValuesAfterRun` 不逐条同步执行，也不在每条指令后立即比较。首期 runtime 不回收中间值，SSA 值在一次运行内不可变且保留到收尾，因此可以：

1. 让所有 runtime 按正常异步语义完整跑完；
2. 计划执行结束后，等待所有发送句柄，把本 rank 仍是 Pending 的对比点输出全部等成 Ready；
3. 在值存储清空之前，由测试代码拷贝出仅供测试用的 RunArtifact；
4. 合并各 rank 的产物，并检查每个 ValueId 只来自它唯一的 Place；
5. 按指令序号遍历 DiffMap，离线比较每条有结果的指令；
6. 报告第一个不匹配，并可继续汇总后面的不匹配。

这种方式不会把对比开销塞进计算、发送、接收或正常等待的执行路径。它只在所有指令都发起之后增加收尾等待和一次拷贝，所以不能用于性能计时；关闭时没有任何逐指令的记录、拷贝或回调开销。

逐指令报告至少包含：指令序号/类型；分布式/参考 ValueId；所属 rank 和 Place；第一个出错的槽位；期望值/实际值；绝对/相对误差；类型和元信息的差异。

没有结果的指令只检查执行轨迹和错误状态，不做数值对比。将来加了 Release/Evict 或 buffer 复用后，历史值可能不再保留；届时若仍要逐指令对比，应由编译器插入显式的调试捕获指令，不能让 runtime 暗中延长生产执行的值生命周期。

### 7.4 比较规则

- Vec 后端可以选精确比较，或配置绝对/相对容差；
- 元信息、值类别、槽数量和 SSA/Place 归属必须精确一致；
- 参考流的 Place 和分布式的 Place 不需要相同，分布式 Place 只和可执行计划比对；
- 不比较 C++ 地址、buffer 布局或通信的具体实现方式；
- 真实 FHE 后端的密文表示可能不确定，未来应比较解密后的语义值；逐指令的密文级对比不属于当前 demo 范围。

## 8. 非数值的执行检查

除了数值对比，每个端到端测试还应检查：

- 通信动作、hint/实际降级和 `wait` 的调用次数；
- 所有发送句柄都在收尾阶段完成；
- 每个 runtime 只执行本 rank 的计算；
- 全部 runtime 要么完成，要么一起进入 fail-fast；
- `abort_all` 是否按预期调用。

## 9. 大明文的后续测试

等加入内存规划后再覆盖：明文装不下设备内存；预取在首次使用前完成；释放后对应设备上的物理数据不可用；从 Host 重新物化；多次使用的明文保持合适的驻留时间；内存峰值不超过编译预算。这些属于第二阶段。

## 10. MPI 多进程验证（MpiVecApi）

核心的明文多 rank 测试**不**通过 MPI 实现，而是在同一进程里并发跑多个独立 runtime——这样才能方便地注入乱序、延迟和错误。但只靠 Mock 测试不够：有些问题只有真实多进程才能暴露，所以首期还包含一个明文的 MPI 集成测试。

**MpiVecApi**：计算部分直接复用 VecExecutor，通信部分用 MPI 非阻塞收发实现 `communicate_async`/`wait`，最终输出通过 `synchronize` 发布，`abort_all` 用 `MPI_Abort`。Value 只含槽数组和固定元信息，按明确格式序列化。它和 MockVecApi 是同一完整接口的两个实现，也用来验证“换 Api 不需要改 runtime”。

它要验证 Mock 测试覆盖不到的东西：

- 每个进程有独立地址空间时，"所有进程加载同一份计划、各自验证、各自绑定输入"这条路径真的成立；
- 值真正被序列化、经过网络、在对端重建（Mock 里无论怎么强制复制，都仍在同一个地址空间）；
- MPI 的消息匹配、非阻塞请求管理与本框架的 `communicate_async`/`wait` 抽象对得上——这是未来 Poseidon 多机路径的骨架预演；
- `MPI_Abort` 能实现 fail-fast 的全组终止；
- 真实的进程启动、汇合和 world size 校验。

运行方式：

~~~text
mpiexec -n N
  -> 每个 MPI 进程一个 runtime
  -> 本地 rank 来自 MPI_Comm_rank
  -> 所有进程读同一份可执行计划
~~~

MPI 集成测试至少复用最终结果对比。逐指令对比仍在运行结束后汇总，不能在每条指令后加 `MPI_Barrier`。

- Mock 测试负责覆盖交错、延迟和故障注入；
- MpiVecApi 测试负责真实多进程通信和端到端结果，不必复刻全部 Mock 注入场景。

`tools/mpi_env_check.cpp` 和 `benchmarks/mpi_comm_benchmark.cpp` 只负责 MPI 环境与基础通信自检；可执行计划、Api 降级和 fail-fast 由 `tests/mpi_runtime_test.cpp` 验证。

## 11. 当前原型完成标准

下面对应当前 Compute/Transfer/Replicate 原型。显式 Encode、bundle、Host compute 和完整 OperatorSpec 验收项见[实现状态](implementation-status.md)的待实现清单。

1. 能构造并打印严格单位置的可执行计划。
2. 验证器能拒绝 SSA、Place、"一值多位置"和通信动作的各类错误。
3. VecExecutor 能执行核心计算并维护元信息。
4. MockVecApi 支持 Transfer、Replicate 和 Broadcast 降级。
5. 初始化阶段能执行 Host→Device 预加载。
6. 多 rank 算例为每个 rank 创建独立的 runtime、值存储和 Api 实例。
7. 能模拟 1、2、4、6、8 个设备，覆盖至少 2、4 个 rank。
8. Pending 输出在消费前被等待。
9. 收尾阶段等待所有发送句柄。
10. 任意错误打印详细上下文并调用 `abort_all`，全部 runtime 停止。
11. 最终结果对比默认开启，且与单设备顺序参考一致。
12. 可选的逐指令对比在运行结束后离线执行，不插入逐指令同步。
13. MpiVecApi 能在 `mpiexec -n 2/4` 下跑通至少一个端到端算例，最终结果与顺序参考一致，且注入错误时 `MPI_Abort` 终止全组。
14. VecExecutor 异步模式下，同一批端到端算例结果与同步模式和顺序参考一致，且带种子的延迟注入可稳定复现不同完成顺序。
