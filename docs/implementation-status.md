# 实现状态

## 已实现的首期范围

- C++17 单函数直线 SSA；每个 ValueId 唯一对应一个 Place。
- C++ 内存版的固定 target、三阶段 RuntimePlan、稳定指令序号/TransferId/fingerprint 和计划打印。
- PlanVerifier：版本、目标、SSA 定义顺序、类型、Place、算子属性、Transfer/Replicate 映射和 hint 检查。
- SequentialRuntime：每 rank 独立实例和 ValueStore、Ready/Pending、用前等待、初始化预加载、最终句柄收尾、最终输出同步和 RunArtifact。
- VecValue：明文/密文、槽值、context、degree、level、浮点 scale、NTT 和分量数。这是当前参考实现状态,还不是 V1 最终的 `scale_log2` 表示。
- VecExecutor：全部首期计算算子；同步模式和每 Device 一个 FIFO 工作线程的异步模式。
- MockCluster/MockVecApi：Transfer、Replicate、Broadcast 到点对点的等价实现、深拷贝、确定性延迟、故障注入和全组终止。
- 顺序参考执行、DiffMap、最终/运行后全部值差分比较。
- MpiVecApi：元信息头与槽数组分离的非阻塞点对点通信、request/缓冲区生命周期、tag 上限、计划 fingerprint 集合检查和 MPI_Abort。
- MPI 环境自检和主机内存通信基准。

## 测试覆盖

- 合法代表计划、稳定 fingerprint 和非法计划注入。
- Vec 算子、元信息、rotate 规范化和非法组合。
- 1/2/4/6/8 个逻辑设备，1/2/4 个独立 rank，同步与异步模式。
- 初始化 Host→Device、Transfer、Replicate、分支扇出、最终 Device→Host。
- 带种子延迟、深拷贝隔离、运行后全部值差分。
- compute、communicate、publish、wait、输出数量和输出类型故障。
- 异步最终计算失败会在 `run()` 返回前触发全组终止；超大 TransferId 会在 MPI tag 运算前被拒绝。
- `mpiexec -n 2`、`mpiexec -n 4` 端到端以及 `--inject-error` 非零退出。

## 构建与运行

~~~bash
xmake f -m debug
xmake
./build/runtime_tests
mpiexec -n 2 ./build/mpi_runtime_test
mpiexec -n 4 ./build/mpi_runtime_test
mpiexec -n 2 ./build/mpi_runtime_test --inject-error  # 预期非零

xmake f -m release
xmake
./build/runtime_tests
~~~

## 设计已定但尚未实现

- RuntimePlan V1 JSON 读取器、Schema 和合法/非法协议样例；
- `ValueDesc` 中完整的 context、level、`scale_log2`、NTT 和分量数；
- 用整数 `scale_log2` 替换 `RescaleAttrs`、`BootAttrs` 和 Vec/MPI 元信息里的浮点 scale；
- OperatorSpec 的 id/版本/指纹和 CPU eager、GPU lazy profile；
- Runtime 对 Host compute 的支持。当前验证器仍把计算写死在 Device；
- `Boot(implementation=decrypt_reencrypt)` 及显式的 Device→Host、Host→Device 流程；
- PoseidonGpuApi 内部组合 Poseidon CPU 能力来执行 Host boot 模拟。

## 明确未实现

Gather/Scatter/AllGather、真实 FHE/GPU/NCCL、MLIR、动态拓扑和设备分配、内存规划与值回收、重试/取消/恢复均未实现。验证器会直接拒绝首期不支持的通信类型和 hint。
