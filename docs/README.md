# CKKS Runtime 设计文档

本目录是多设备 CKKS 推理 runtime 的设计基线。这里的 runtime 指的是"拿到一份编译器已经排好的执行计划，在多个设备上把它跑完"的那部分程序。

## 项目要做什么

本仓库计划实现：

- 一个与具体同态加密库解耦的 runtime 原型：runtime 只负责按计划执行，不关心密文在底层长什么样；
- VecApi：用普通 `std::vector` 模拟密文运算的明文参考实现，提供同步和异步（每设备一个工作线程，模拟 GPU 并行时序）两种模式；
- MockCommunicationApi：用线程和消息队列模拟跨设备通信，可以人为注入延迟和错误；
- 多 rank 测试：在一个进程里为每个模拟节点各建一个独立 runtime，并发执行同一份计划；
- difftest：把多设备执行结果和单设备顺序执行的参考结果做对比，验证正确性；
- 计划合法性检查和统一的错误处理；
- MpiVecApi：明文计算 + MPI 非阻塞通信，在 `mpiexec` 多进程下验证 runtime 的可行性；
- MPI 环境与基础通信自检。

本仓库当前不实现：

- MLIR dialect 的正式代码（但 SSA 与算子的设计必须和未来 MLIR 方案对得上，所以设计文档里包含这部分）；
- 真实的 GPU 计算与 CUDA/NCCL/GPU-aware MPI 通信后端；
- 生产级多机调度、容错和恢复；
- RNS 分片级并行（即把一个密文拆到多卡上算，本项目只做"整个密文放在一张卡上"的并行）。

## 与 poseidon::mgpu 的关系

Poseidon 仓库里已有一套单机多卡的静态调度代码（`src/poseidon/mgpu`，包含 schedule IR、placement、copy 插入、verifier、对象级 GPU 拷贝）。**本框架是它的演进目标**：mgpu 目前只覆盖单节点、同步拷贝、无 rank 概念，后续会把其中有价值的部分（GPU 对象拷贝层、verifier 不变量、Dacapo 前端产物）迁移过来，作为本框架下的一个 Api 实现接入；冗余部分会删除。因此本文档中的 SSA 不变量刻意与 mgpu 的做法保持兼容，细节见[总体架构](overview-design/architecture.md)最后一节。

## 常用术语

第一次读建议先过一遍这张表，后文直接使用这些词：

| 术语 | 含义 |
| --- | --- |
| SSA | 静态单赋值形式：每个值只被定义一次，之后只读。计划中的程序就是一串 SSA 指令 |
| ValueId | 一个值的编号。本框架规定**一个 ValueId 只存在于一个位置**；同一份数据被复制到别的设备后，副本用新的 ValueId |
| Place | 物理位置，由三元组（Host 或 Device、rank、设备序号）表示 |
| rank | 一个进程/节点的编号，沿用 MPI 的叫法。一个 rank 可以带多张卡 |
| 可执行计划（executable plan） | 编译器输出的最终产物：每条指令在哪算、数据怎么搬都已经定死，runtime 照着执行即可 |
| 物化（materialize） | 让一个值在某个位置上真实存在（分配内存并填入数据） |
| 通信动作（CommAction） | 计划中的一条搬运指令，如 Transfer（点对点搬运）、Replicate（一发多收） |
| Ready / Pending | 一个值的两种状态：数据已就绪 / 通信还在路上 |
| OperatorSpec | 版本化的 CKKS 算子配置：参数限制、level/`scale_log2` 变化、lazy-rescale、boot profile 和代价 |
| `scale_log2` | scale 的二进制指数；40 表示逻辑 scale 为 `2^40` |
| fail-fast | 出错就立刻带上下文报错并停掉整个执行组，不做重试和恢复 |
| difftest | 差分测试：两种方式算同一个程序，对比结果是否一致 |

## 文档导航

### 协议规范(权威,机器可检查)

- [RuntimePlan V1 规范](runtime-plan/v1/specification.md):计划文件的字段语义、编码规则、SSA/元信息不变量和指纹计算。配套 [schema.json](runtime-plan/v1/schema.json)、[版本兼容规则](runtime-plan/v1/compatibility.md) 和 [合法/非法样例集](runtime-plan/v1/testdata/README.md)。
- [CKKS OperatorSpec V1 规范](operator-spec/v1/specification.md):目标后端的算子能力、level/`scale_log2` 边界、boot profile 和代价模型。配套 [schema.json](operator-spec/v1/schema.json) 和 [占位 profile](operator-spec/v1/profiles/README.md)(CPU eager / GPU lazy)。

协议实现(JSON 读取器、验证器扩展)以规范为准;实现与样例集冲突时,先怀疑实现。

### 总体设计(背景与决策)

- [总体架构](overview-design/architecture.md):目标、分层、编译流水线、核心约束、与 poseidon::mgpu 的对接。
- [Dialect 与 SSA 设计](overview-design/dialect-design.md):逻辑算子、目标合法化、设备分配、通信显式化和 RuntimePlan 映射。
- [Runtime 设计](overview-design/runtime-design.md):执行流程、值状态管理、异步等待和错误终止。
- [通信设计](overview-design/communication-design.md):通信动作、实现提示（hint）、Host/Device 搬运和无死锁论证。
- [合法性验证与错误处理](overview-design/validation-and-errors.md):各层检查清单和错误诊断格式。
- [明文测试方案](overview-design/plaintext-testing.md):VecApi、MockCommunicationApi 和测试矩阵。
- [Dacapo、Runtime 与 Poseidon 集成方案](overview-design/dacapo-runtime-integration.md):仓库关系、RuntimePlan/OperatorSpec、lazy-rescale、CPU boot 模拟、集成测试与迁移顺序。
- [实现状态](overview-design/implementation-status.md):代码路径、构建命令、首期验收项和明确未实现范围。
- [架构设计 v0.1 归档](overview-design/archive/architecture-v0.1.md):最初草案，仅作历史记录。

## 当前核心决策

1. 逻辑层的计算算子保持纯函数语义，不夹带通信副作用。
2. 目标合法化、设备分配（placement）和通信显式化（materialization）是编译器的三个独立步骤。
3. 集群拓扑在编译时完全已知，编译器输出绑定了物理 Place 的可执行计划。
4. Host 和 Device 都是 Place；不允许把 Host 伪装成"0 号 GPU"。
5. 可执行计划中每条计算指令只在一个 Place 执行，结果只出现在本地。
6. 一个 ValueId 恰好属于一个 Place；数据要到多个位置，就有多个 ValueId。
7. 数据搬运一律用显式的通信动作表达，编译器可以附带实现方式的建议（hint）。
8. Runtime 只验证和执行计划，不做分配、选路或 GPU Boot 到 CPU 的隐式回退。
9. Api 层定义实际的值类型、计算函数、`communicate_async`、`wait`、最终输出 `synchronize`、等价的通信降级和 `abort_all`。
10. 首期不引入对象描述符、拓扑查询、取消或重试这类接口。
11. VecApi 与 MockCommunicationApi 是本仓库的参考实现；PoseidonApi 是未来接入目标。
12. 多 rank 测试中每个 rank 有独立的 runtime 和值存储，并发执行，不加逐指令同步。
13. 最终结果 difftest 默认开启；逐指令 difftest 可选，且在运行结束后离线比较。
14. 任何错误都打印详细上下文并终止整个执行组。
15. V1 的 `level` 和 `scale_log2` 都是非负整数；CPU 使用 eager rescale，当前低 bit GPU profile 使用 lazy-rescale。
16. GPU boot 的 CPU 解密再加密模拟必须由编译器生成显式 Device→Host、Host Boot、Host→Device 指令，只用于测试与联调。

## 代码对应关系

| 路径 | 职责 |
| --- | --- |
| `runtime/plan.*` | 单位置 RuntimePlan、强类型算子/通信动作、稳定 fingerprint 与计划打印 |
| `runtime/verifier.*` | SSA、Place、类型、目标配置和 Transfer/Replicate 合法性检查 |
| `runtime/runtime.hpp` | ValueStore、Ready/Pending、SequentialRuntime 和 RunArtifact |
| `api/vec_*` | VecValue 元信息语义及同步/异步计算 |
| `api/mock_api.*` | MockCluster、多 runtime 通信、延迟/故障注入和全组终止 |
| `api/mpi_api.*` | MPI 非阻塞通信、序列化、fingerprint preflight 与 MPI_Abort |
| `testing/` | 计划构造器、顺序参考执行器、DiffMap、比较和 Mock 测试驱动 |
| `tests/` | 普通测试与 MPI 端到端测试 |
| `tools/`、`benchmarks/` | MPI 环境自检与通信基准 |

旧的多位置 SSA、`CommInterface` send/receive 接口和空解释器已经删除，不提供兼容层。

## 设计旁注

> [!PDF|8, 109, 221] [[计算机程序的构造和解释 原书第2版 典藏版 (哈罗德·埃布尔森 杰拉尔德·杰伊·萨斯曼 裘宗燕) (z-library.sk, 1lib.sk, z-lib.sk).pdf#page=19&annotation=2135R|计算机程序的构造和解释 原书第2版 典藏版 (哈罗德·埃布尔森 杰拉尔德·杰伊·萨斯曼 裘宗燕) (z-library.sk, 1lib.sk, z-lib.sk), p.3]]
>
> 这样，任何强有力的程序设计语言都必须能表述基本的数据和基本的过程，还需要提供对过程和数据进行组合和抽象的方法。
