# 明文测试方案

## 1. 目标

本仓库用 VecApi 和 MockCommunicationApi 验证 runtime 本身，不验证真实的 CKKS 密码学或 GPU 性能。

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

## 2. VecApi

明文和密文用不同类型表示：

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

VecApi 实现：add/sub/mul、add_plain/sub_plain/mul_plain、negate、rotate、rescale、modswitch、relinearize、boot，以及 communicate_async、wait、最终输出 synchronize、abort_all。

数值部分逐元素模拟即可，但元信息不能全部空转，否则非法的计算图也能通过测试：

- 乘法按 CKKS 规则改变 `scale_log2` 和分量个数；
- relinearize 恢复分量个数；
- rescale 改变 level 和 `scale_log2`；
- modswitch 改变 level；
- boot 按测试配置重置 level 和 `scale_log2`；
- rotate 对步数做规范化。

V1 的元信息测试统一使用整数 `scale_log2`:值 40 表示逻辑 scale 为 `2^40`。当前 VecValue 里的浮点 scale 是待迁移实现,不能进入 RuntimePlan V1 或协议指纹。

还要为 `Boot(implementation=decrypt_reencrypt)` 建一条明文模拟路径。测试计划必须显式包含 Device→Host、Host Boot、Host→Device,并检查两次 Transfer 保持元信息、Host Boot 精确产出目标 level 和 `scale_log2`。不能用“GPU Boot 失败后自动改走 CPU”来通过测试。

### 2.1 同步与异步两种模式

VecApi 提供两种执行模式，接口完全相同，测试可以用同一份计划分别跑：

~~~text
VecExecMode::Sync    // 计算在调用线程上当场完成（默认）
VecExecMode::Async   // 每个模拟设备一个工作线程，模拟 GPU 的发起/执行分离
~~~

**同步模式**用于基础正确性测试，行为最简单、出错最好定位。

**异步模式**用来模拟真实 GPU 后端"发起串行、执行并行"的时序：

- 每个模拟设备（Place）一个工作线程和一个先进先出任务队列；
- `compute` 把任务放进目标设备的队列后立即返回，不等计算完成；
- 值内部用共享状态表示"算完了没有"（概念上是一个 future）：任务完成后由工作线程填入结果，后续任务用到某个输入时，如果它还没就绪就在工作线程里等它；
- 同一设备队列先进先出，天然满足"输出对后续调用可见"的契约；跨设备的依赖通过值的共享状态传递；
- `communicate_async` 把"取源值并发送"作为任务放进源设备的队列，保证它排在产生该值的计算之后；
- `wait` 是唯一阻塞调用线程的点，等待通信（及其依赖的计算）完成；
- 可配置每个任务的固定或随机（带种子）延迟，用来放大不同设备之间的进度差。

异步模式专门要测的东西：

- 多卡独立分支真正并发执行时，最终结果仍与顺序参考一致；
- runtime 在某个值 Pending 期间继续发起后续指令，不会用到未就绪的数据；
- 指令排序的效果可观察：通信早发起晚消费时，模拟的总时间线上出现计算与通信重叠；
- 带种子的延迟注入能稳定复现不同的完成顺序。

同一批端到端算例应在两种模式下各跑一遍，结果必须一致。异步模式的实现只在 VecApi 内部，runtime 和接口不感知模式差异——这本身就是对"异步契约"设计的一次验证。

## 3. MockCommunicationApi 与 MockCluster

每个 rank 有自己的 MockCommunicationApi，多个 Api 实例只通过一个线程安全的 MockCluster 交换消息：

~~~text
Runtime(rank=0) -- MockCommunicationApi(0) --+
                                             |
Runtime(rank=1) -- MockCommunicationApi(1) --+-- MockCluster
                                             |   - 消息队列/事件
Runtime(rank=2) -- MockCommunicationApi(2) --+   - 终止状态
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

支持：Transfer、Replicate；Host→Device、Device→Host、Device→Device；Auto/PointToPoint/Broadcast 提示；Broadcast 降级为多个点对点；接收方先到或发送方先到；延迟完成；指定某次通信失败；输出数量或类型的错误注入；`abort_all` 转换为 ClusterPanic。

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
rank 0 -> Runtime0(LocalIdentity{0}, ValueStore0, Api0)
rank 1 -> Runtime1(LocalIdentity{1}, ValueStore1, Api1)
...
~~~

每个实例通过 `Place.rank` 判断自己的角色。一个 rank 可以带多个 Device，它们由该 rank 的同一个 runtime 管理。

**多 rank 测试不允许退化成"一个 runtime 循环切换 rank"**——那种写法覆盖不了独立执行进度、并发等待、全组终止和消息到达顺序这些真正的风险点。

需要覆盖：单 rank 多设备；2、4 个独立 runtime；1、2、4、6、8 个逻辑设备；rank 之间执行速度不同；发送方/接收方到达通信指令的顺序不同；节点数不匹配；本地设备数不匹配。

## 5. 测试层次

### 5.1 类型和计算

明文/密文不能混用；参数个数错误；rotate 的正数、负数、零和大步数；向量长度；元信息变化；不支持的算子；计算出错。

### 5.2 计划验证器

重复 ValueId；一个 ValueId 绑到多个 Place；先用后定义；未定义输入；错误的值类型；计算指令带通信效果；计算 Place 非法；隐式跨 Place 操作数；重复传输 ID；通信类型的输入输出数量错误；Replicate 输出数和目标数不一致；多个目标复用同一个输出 ValueId；来源/目标 Place 非法；V1 中出现不支持的 Gather；未知 hint；节点数与编译目标不符。

### 5.3 单设备 runtime

纯线性的 add/mul；一元算子；明文密文混合；同一输入用两次；多输出；元信息不匹配；Api 返回错误类型。

### 5.4 Host/Device 初始化

~~~text
%w_host = input @Host0
%w_gpu = transfer %w_host Host0->Device0
%y = mul_plain %x, %w_gpu @Device0
~~~

检查：Host 输入正确绑定；初始化阶段执行了 transfer；执行阶段开始前所需的值都 Ready；Host→Device 保持数学值和类型不变；明文不因为"它是明文"就自动上传，必须走计划里的动作；同一个常量上传到多个设备时，每个目标有独立 ValueId，数学值一致。

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

接收方比发送方先到达通信指令；发送方先到；消费方比通信完成先到；不相关的指令可以在 Pending 期间继续执行；`wait` 之后输出变 Ready；源值保留到收尾；同一份计划跑多次，传输 ID 不串。

### 5.8 Fail-fast

注入：计算抛错；`communicate_async` 抛错；`wait` 抛错；输出数量不匹配；输出类型不匹配；Place 不匹配；节点数不匹配；不支持的通信类型；无等价降级。

检查：打印指令序号/类型、ValueId/传输 ID、来源/目标 Place、本地 rank 和 Api 名称；`abort_all` 被调用；所有模拟 runtime 停止；不继续执行后续指令。

### 5.9 多 runtime 隔离

检查：rank 数等于 runtime 实例数；每个实例身份唯一；每个实例只持有本 rank 已物化的值；发送方和接收方的 VecValue 地址不同；删掉任何一侧的 `communicate_async` 都会导致明确失败，而不是靠共享内存"碰巧正确"；一个 runtime 调用 `abort_all` 后，其他 runtime 的 `wait` 被唤醒并终止；没有靠逐指令同步维持表面上的正确顺序。

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
顺序参考流:  单 rank、单 Place、VecApi 顺序执行、没有物理通信
分布式执行流: 多设备/多 rank、每个 rank 一个 runtime、
             执行分配后的可执行计划、使用 MockCommunicationApi
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

检查内容：槽数量和数值；值类别；level/`scale_log2`/分量数等元信息；分布式输出位于计划声明的唯一 Place；输出没有缺失、也没有被两个 runtime 重复持有；NaN/Inf 的分类；配置的绝对/相对容差。

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

- VecApi 可以选精确比较，或配置绝对/相对容差；
- 元信息、值类别、槽数量和 SSA/Place 归属必须精确一致；
- 参考流的 Place 和分布式的 Place 不需要相同，分布式 Place 只和可执行计划比对；
- 不比较 C++ 地址、buffer 布局或通信的具体实现方式；
- 真实 FHE 后端的密文表示可能不确定，未来应比较解密后的语义值；逐指令的密文级对比不属于当前 demo 范围。

## 8. 非数值的执行检查

除了数值对比，每个端到端测试还应检查：通信动作的发起次数；hint 与实际降级记录；`wait` 的调用次数；发送句柄在收尾阶段全部完成；`abort_all` 是否按预期调用；每个 runtime 只执行本 rank 的计算；所有 runtime 都完成，或一起 fail-fast。

## 9. 大明文的后续测试

等加入内存规划后再覆盖：明文装不下设备内存；预取在首次使用前完成；释放后对应设备上的物理数据不可用；从 Host 重新物化；多次使用的明文保持合适的驻留时间；内存峰值不超过编译预算。这些属于第二阶段。

## 10. MPI 多进程验证（MpiVecApi）

核心的明文多 rank 测试**不**通过 MPI 实现，而是在同一进程里并发跑多个独立 runtime——这样才能方便地注入乱序、延迟和错误。但只靠 Mock 测试不够：有些问题只有真实多进程才能暴露，所以首期还包含一个明文的 MPI 集成测试。

**MpiVecApi**：计算部分直接复用 VecApi，通信部分用 MPI 非阻塞收发实现 `communicate_async`/`wait`，最终输出通过 `synchronize` 发布，`abort_all` 用 `MPI_Abort`。Value 就是向量加元信息，序列化是平凡的。它和 MockCommunicationApi 是同一接口的两个实现，顺带验证了"换 Api 不需要改 runtime"这个核心承诺。

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

MPI 集成测试至少复用最终结果对比。逐指令对比仍应在运行结束后由测试驱动汇总各 rank 产物，不能在每条指令后加 `MPI_Barrier`。测试矩阵上的分工：Mock 测试负责覆盖面（交错、延迟、故障注入），MpiVecApi 测试负责真实性（多进程冒烟 + 端到端结果对比），后者不要求复刻前者的全部注入场景。

`tools/mpi_env_check.cpp` 和 `benchmarks/mpi_comm_benchmark.cpp` 只负责 MPI 环境与基础通信自检；可执行计划、Api 降级和 fail-fast 由 `tests/mpi_runtime_test.cpp` 验证。

## 11. 首期完成标准

1. 能构造并打印严格单位置的可执行计划。
2. 验证器能拒绝 SSA、Place、"一值多位置"和通信动作的各类错误。
3. VecApi 能执行核心计算并维护元信息。
4. MockCommunicationApi 支持 Transfer、Replicate 和 Broadcast 降级。
5. 初始化阶段能执行 Host→Device 预加载。
6. 多 rank 算例为每个 rank 创建独立的 runtime、值存储和 Api 实例。
7. 能模拟 1、2、4、6、8 个设备，覆盖至少 2、4 个 rank。
8. Pending 输出在消费前被等待。
9. 收尾阶段等待所有发送句柄。
10. 任意错误打印详细上下文并调用 `abort_all`，全部 runtime 停止。
11. 最终结果对比默认开启，且与单设备顺序参考一致。
12. 可选的逐指令对比在运行结束后离线执行，不插入逐指令同步。
13. MpiVecApi 能在 `mpiexec -n 2/4` 下跑通至少一个端到端算例，最终结果与顺序参考一致，且注入错误时 `MPI_Abort` 终止全组。
14. VecApi 异步模式下，同一批端到端算例结果与同步模式和顺序参考一致，且带种子的延迟注入可稳定复现不同完成顺序。
