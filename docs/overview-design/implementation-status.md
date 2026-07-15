# 实现状态

本文是“当前代码到底做到了什么”的准确信息源。其他 overview 文档主要描述目标设计;两者冲突时,先以本文和代码为准,再修正文档或实现。

## 已实现的首期范围

- C++17 单函数直线 SSA；每个 ValueId 唯一对应一个 Place。
- C++ 内存版的固定 target、三阶段 RuntimePlan、稳定指令序号/TransferId、内部 64 位计划摘要和计划打印。这个摘要供 Mock/MPI 多 rank 一致性检查使用,不是 JSON 协议的 SHA-256 `fingerprint`。
- 基础 PlanVerifier：版本、world size/device count、SSA 定义顺序、类型、Place、算子属性、Transfer/Replicate 映射和已实现 hint 检查。
- SequentialRuntime：每 rank 独立实例和 ValueStore、Ready/Pending、用前等待、初始化预加载、最终句柄收尾、最终输出同步和 RunArtifact。
- VecValue：明文/密文、槽值、context、degree、level、整数 `scale_log2`、NTT 和分量数。
- VecExecutor：全部首期计算算子；同步模式和每 Device 一个 FIFO 工作线程的异步模式。
- MockCluster/MockVecApi：Transfer、Replicate、Broadcast 到点对点的等价实现、深拷贝、确定性延迟、故障注入和全组终止。
- 顺序参考执行、DiffMap、最终/运行后全部值差分比较。
- MpiVecApi：元信息头与槽数组分离的非阻塞点对点通信、request/缓冲区生命周期、tag 上限、内部 64 位计划摘要集合检查和 MPI_Abort。
- MPI 环境自检和主机内存通信基准。
- RuntimePlan V1 上一版 JSON 读取器：严格拒绝 BOM、重复 key、未知/缺失字段、浮点数、非法枚举和错误的 64 位 ID 编码，并用仓库内 SHA-256 实现重算指纹；nlohmann-json 固定在 `third_party/`，不依赖 OpenSSL。它尚未支持当前规范中的显式 Encode 和 inline 浮点 payload。
- `ValueDesc` 已包含 context、level、整数 `scale_log2`、NTT 和分量数；Rescale/Boot、Vec 和 MPI 元信息也已改用整数 `scale_log2`。
- `TargetConfig` 已能保存 capability、OperatorSpec 引用、rescale/boot 模式和能力声明;`RuntimePlan` 顶层已能保存密钥声明。
- 顶层可选 `plaintext_bundle` 引用的结构解析与 external input Host-only 检查(IO-2)。上一版 v005 样例把 bundle 对应的明文描述成 external input;Runtime 本身没有读取 manifest、数据文件或物化 bundle 明文。
- 当前 `SequentialRuntime` 实例按一次运行使用:ValueStore 和通信组在 `run()` 前后不重置,同一实例不能直接重复执行第二份计划。

## 测试覆盖

- 上一版 RuntimePlan V1 五份合法样例(含 plaintext_bundle 引用)、结构类非法样例、IO-2 拒绝、重复 JSON key、BOM 和错误 ID 类型;显式 Encode 的新样例尚未补齐。
- 合法代表计划、稳定内部摘要和非法计划注入。
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
./build/runtime_plan_json_tests
mpiexec -n 2 ./build/mpi_runtime_test
mpiexec -n 4 ./build/mpi_runtime_test
mpiexec -n 2 ./build/mpi_runtime_test --inject-error  # 预期非零

xmake f -m release
xmake
./build/runtime_tests
~~~

## 设计已定但尚未实现

- PlanVerifier 尚未覆盖 V1 的全部语义检查，包括完整元信息变化、能力/密钥声明精确匹配和无孤儿 ValueDesc；
- Runtime 绑定 external input、接收计算/通信输出时目前只核对明文/密文类别,尚未核对实际 `Api::Value` 的 context、level、`scale_log2`、NTT 和 components 是否与 ValueDesc 一致;
- 目标 Api 需要能按 ValueDesc 校验不透明值，并用协议 SHA-256 与 target/context/密钥要求做 preflight；当前接口只用内部 64 位摘要检查多 rank 计划一致性；
- 当前异步 `VecValue` 的 Pending 句柄只提前保存明文/密文类别；为支持非阻塞的完整 `validate_value`，后续还要让句柄在创建时携带不可变 CKKS 元信息；
- OperatorSpec 文件读取、id/版本/指纹匹配、level/rescale/boot profile 检查;
- Runtime 启动入口接收明文数据包目录,以及 manifest 读取、content/长度/哈希核对、小端 float64 解码与 BND-1 preflight;
- `plan.hpp`/Schema/reader/verifier/runtime/Api 对显式 Encode 指令和 inline/bundle payload 的支持;
- 真正符合 RFC 8785 浮点数字规范化的计划指纹实现和跨语言测试向量;
- Runtime 对 Host compute 的支持。当前验证器仍把计算写死在 Device；
- `Boot(implementation=decrypt_reencrypt)` 及显式的 Device→Host、Host→Device 流程；
- PoseidonGpuApi 内部组合 Poseidon CPU 能力来执行 Host boot 模拟。
- 完整错误诊断:当前 compute 不打印全部输入 ValueId,通信只打印第一个来源/目标,加载和 preflight 错误也不一定带指令上下文。

## 明确未实现

Gather/Scatter/AllGather、真实 FHE/GPU/NCCL、MLIR、动态拓扑和设备分配、内存规划与值回收、重试/取消/恢复均未实现。验证器会直接拒绝首期不支持的通信类型和 hint。
