# Runtime Demo 设计文档

本目录记录多设备 CKKS runtime demo 的当前设计基线。

本仓库只计划实现：

- 与具体同态加密库解耦的 runtime 原型；
- 明文 VecApi；
- 可控的 MockCommunicationApi；
- 每个模拟 rank 一个独立 Runtime 的多 rank 测试；
- 与单 Place 顺序参考流比较的最终/逐指令 difftest；
- SSA、可执行计划、合法性验证和错误处理测试；
- MPI 环境与基础通信自检。

本仓库当前不计划实现：

- MLIR dialect 的正式代码；
- Poseidon GPU backend；
- CUDA、NCCL 或 GPU-aware MPI 的正式通信 backend；
- 生产级多机调度器；
- RNS shard 级并行。

Dialect 设计仍然属于本项目范围，因为 C++ demo 中的 SSA 类型、op 分类、placement、transfer 和 verifier 必须与未来 MLIR 下降结果保持一致。

## 文档导航

- [总体架构](architecture.md)：目标、分层、编译流水线、核心约束和范围。
- [Dialect 与 SSA 设计](dialect-design.md)：逻辑 CKKS SSA、设备分配、通信物化和 C++ 类型映射。
- [Runtime 设计](runtime-design.md)：纯执行 runtime、Ready/Pending 状态、执行流程和 fail-fast。
- [通信设计](communication-design.md)：CommAction、Hint、Host/Device Place、Api fallback 和无死锁约束。
- [合法性验证与错误处理](validation-and-errors.md)：编译期 verifier、runtime preflight 和动态保护。
- [明文测试方案](plaintext-testing.md)：VecApi、MockCommunicationApi 和端到端测试矩阵。
- [架构设计 v0.1 归档](archive/architecture-v0.1.md)：项目最初的设计草案，仅作历史记录。

## 当前核心决策

1. 逻辑 CKKS SSA 中的计算 op 保持无逻辑副作用。
2. 设备分配与通信物化是两个独立 pass，但不要求各自对应一个 dialect。
3. 完整集群拓扑在编译时已知，编译器生成绑定物理 Place 的 executable plan。
4. Host 和 Device 都是 Place；Host 不能伪装成 GPU device 0。
5. 可执行计划中的计算 op 只在一个 Place 执行并产生本地结果。
6. 可执行计划严格采用单位置物理 SSA：每个 ValueId 恰好属于一个 Place，多目标通信产生多个不同 ValueId。
7. Transfer、Replicate、Gather 等通信 action 表达通信语义，编译器可以附加实现 hint。
8. Runtime 不做 placement、路由选择或 fallback，只验证并执行计划。
9. Api 定义实际 Value、计算函数、communicate_async、wait、fallback 和 abort_all。
10. 首期不公开 ObjectAdapter、ObjectDescriptor、Topology、cancel 或 retry。
11. VecApi 与 MockCommunicationApi 是本仓库的参考实现。
12. Mock 多 rank 测试为每个 rank 创建独立 Runtime/ValueStore，并发执行且不设置逐 op barrier。
13. 最终结果 difftest 默认开启；逐指令 difftest 可选，并在运行结束后离线比较。
14. 任何错误都打印详细上下文并终止整个 execution group。

## 代码对应关系

当前源码仍处于概念验证阶段，后续建议逐步演化为：

| 当前文件 | 目标职责 |
| --- | --- |
| ../ssa_ops_def.hpp | SSA、Value、Op、placement 与 transfer 类型定义 |
| ../operations.hpp | VecApi 的明文计算实现与通用计算 contract |
| ../comm_interface.hpp | CommAction、communicate_async、wait 与 abort_all contract |
| ../seq_interpreter.hpp | Runtime verifier、Ready/Pending value store 与 executor |
| ../mpi_test.cpp | 环境自检与最小 backend smoke test |
| ../mpi_test_comm.cpp | MPI host-buffer 基准，不作为 runtime 正确性测试 |

## 设计旁注

> [!PDF|8, 109, 221] [[计算机程序的构造和解释 原书第2版 典藏版 (哈罗德·埃布尔森 杰拉尔德·杰伊·萨斯曼 裘宗燕) (z-library.sk, 1lib.sk, z-lib.sk).pdf#page=19&annotation=2135R|计算机程序的构造和解释 原书第2版 典藏版 (哈罗德·埃布尔森 杰拉尔德·杰伊·萨斯曼 裘宗燕) (z-library.sk, 1lib.sk, z-lib.sk), p.3]]
>
> 这样，任何强有力的程序设计语言都必须能表述基本的数据和基本的过程，还需要提供对过程和数据进行组合和抽象的方法。
