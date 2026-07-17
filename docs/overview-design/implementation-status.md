# 实现状态

## 阶段一已完成

- RuntimePlan V1、OperatorSpec V1 和 plaintext bundle manifest 已有一致的规范、Schema、profiles 和生成样例。
- reader 严格拒绝 BOM、重复 key、未知/缺失字段、错误类型和不支持版本，并对完整原始文件字节计算 SHA-256。
- RuntimePlan 已支持 inline/bundle Encode；bundle loader 每个 rank 只读取本地实际需要的 blob。
- verifier 覆盖 SSA、阶段、Place、完整 CKKS 元信息变化、OperatorSpec 边界、Boot profile、Rotate step 规范化和具体密钥需求。
- Runtime 在执行前完成 OperatorSpec/bundle/preflight，执行 Host/Device compute、Encode 和显式通信，并对输入及 Api 输出做完整元信息检查。
- Mock 和 MPI preflight 使用计划原始字节摘要，并检查所有 rank 的调试开关一致。
- Vec/Mock 测试覆盖同步、异步、多 rank、摘要调试策略、bundle 局部装载和 fail-fast；MPI 测试覆盖端到端和摘要一致性。

仓库内 OperatorSpec 仍是 `placeholder`，只用于协议测试。

## 阶段二进行中

- 已把 `mingerfan/dacapo-modified` 作为可选 submodule 固定在 `third_party/dacapo/`。它不参与默认 xmake 构建，需要时用 `git submodule update --init third_party/dacapo` 显式初始化。
- 已把 Dacapo 的 HEAAN CPU/GPU、SEAL CPU profile 迁到 OperatorSpec V2，保留 CKKS 参数、逐 level 延迟/噪声、来源 SHA-256，并显式完成 Earth→CKKS bootstrap level 换算。V1 保持冻结。
- Dacapo 的 CKKS `PolyType` 已保存 components、`scale_log2` 和 Runtime 方向的 level；全部 CKKS 算子使用纯 SSA result-style，不再携带 `dst` 或生成 `tensor.empty`；Earth 密文乘法下降为独立的 `ckks.mulcc` 和 `ckks.relinearize`。
- `emit-runtime-plan` 支持单 Host 和已 placement 的函数：小 Encode payload 保持 inline，大 float64 payload 按阈值写入内容寻址 bundle，相同内容只保存一次；输出严格的 RuntimePlan V1 JSON，并覆盖 Mul/Relinearize、Upscale、常量、bundle 复用、Boot 层号换算、真实多 rank/device target 和点对点 Transfer。
- 已增加 `dist.transfer`、`assign-ckks-placement` 和 `materialize-ckks-communication`。placement 使用确定性 HEFT、OperatorSpec V2 逐 level 延迟和固定 rank 内/间通信代价；当前是完整值放置，外部输入与 Encode 位于 Host rank 0，同一值到同一目标只复制一次。
- 编译器 fixture 的 `1x8`/`2x8` 测试覆盖全部 device、点对点 hint、本地操作数、设备区间不重叠、依赖到达时间和重复编译确定性。MLP 的 Host/`1x8`/`2x8` 计划已用 MockVecApi `AllValuesAfterRun` 完成每条 Encode、Compute、Transfer 及 final output 的精确差分，均为 0 diff。
- 已删除依赖 destination-style 的旧 `ReuseBuffer` Pass；`RemoveLevel` 仍保留注册但不进入新管线，`emit-hevm` 调用会直接报错。Python frontend 不再生成 `.cst` 索引。
- 待完成：目标合法化、Bootstrap placement、Replicate 合并、分片、显存容量和基于值大小/带宽/链路争用的通信代价。当前 placement 已直接读取 OperatorSpec V2；RuntimePlan V1 无需增加字段。

## 阶段三进行中

- Poseidon 仓库已经通过可选构建路径接入本 Runtime。`PoseidonCpuApi` 同时支持不依赖 MPI 的单进程模式和可选的 MPI 多进程模式；MPI 模式支持 Host rank 间的明文/密文 Transfer、Replicate、计划与 context 一致性检查及全组终止，手工 RuntimePlan 已在 2/4 rank 下通过端到端测试。
- 单进程单卡 `PoseidonGpuApi` 已实现：使用 Poseidon 的 `GpuUploader`、`GpuParameterData` 和 `GpuEvaluator`，支持 Host↔Device Transfer 与当前 GPU 库已有的普通 CKKS 算子。
- Dacapo 已生成多 rank/device placement 和点对点 Transfer；Poseidon CPU MPI 的既有测试仍使用手工 RuntimePlan，Dacapo 产出的 Device 计划目前由 MockVecApi 验证，后续等 GPU 多卡/跨进程通信接入后再交给 PoseidonGpuApi。
- 生产 GPU OperatorSpec 尚未完成。当前 placeholder profile 使用 28-bit 模数且只允许两层 Rescale，不符合实际 30-bit 模数和 lazy Rescale 连续下降 4 层的设计；ModSwitch 和 Boot 也尚未实现，不能被真实 GPUApi 接受。

这些内容不属于 RuntimePlan V1 冻结条件。
