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

- 已把 `mingerfan/dacapo-modified` 作为可选 submodule 固定在 `third_party/dacapo/`。它不参与默认 CMake 构建，需要时用 `git submodule update --init third_party/dacapo` 显式初始化。
- 已把 Dacapo 的 HEAAN CPU/GPU、SEAL CPU profile 迁到 OperatorSpec V2，保留 CKKS 参数、逐 level 延迟/噪声、来源 SHA-256，并显式完成 Earth→CKKS bootstrap level 换算。V1 保持冻结。
- Dacapo 的 CKKS `PolyType` 已保存 components、`scale_log2` 和 Runtime 方向的 level；全部 CKKS 算子使用纯 SSA result-style，不再携带 `dst` 或生成 `tensor.empty`；Earth 密文乘法下降为独立的 `ckks.mulcc` 和 `ckks.relinearize`。
- `emit-runtime-plan` 支持单 Host 和已 placement 的函数：小 Encode payload 保持 inline，大 float64 payload 按阈值写入内容寻址 bundle，相同内容只保存一次；输出严格的 RuntimePlan V1 JSON，并覆盖 Mul/Relinearize、Upscale、常量、bundle 复用、Boot 层号换算、真实多 rank/device target 和点对点 Transfer。
- 已增加 `dist.transfer`、`assign-ckks-placement` 和 `materialize-ckks-communication`。placement 使用确定性 HEFT 和 OperatorSpec V2 逐 level 延迟；通信支持固定 rank 内/间代价，也支持按 CKKS value 大小、链路速率上限和可选 payload/rate 点位表估算。当前是完整值放置，外部输入与 Encode 位于 Host rank 0，同一值到同一目标只复制一次。
- 编译器 fixture 的 `1x8`/`2x8`/`2×CPU` 测试覆盖全部候选 Place、点对点 hint、本地操作数、时间区间不重叠、依赖到达时间和重复编译确定性。MLP 的 Host/`1x8`/`2x8`/`2×CPU` 计划已用 MockVecApi `AllValuesAfterRun` 完成每条 Encode、Compute、Transfer 及 final output 的精确差分，均为 0 diff。
- 已删除依赖 destination-style 的旧 `ReuseBuffer` Pass，以及仅服务 HEVM 的 `RemoveLevel`/`EmitHEVM` Pass、C++ interpreter、Python runner 和示例执行脚本。Python frontend 不再生成 `.cst` 索引。
- 待完成：目标合法化、Bootstrap placement、Replicate 合并、分片、显存容量、具体 device pair 和链路争用。当前粗粒度通信模型已经考虑值大小和带宽上限，但 profile 仍需用真实硬件数据标定；RuntimePlan V1 无需增加字段。

## 阶段三进行中

- Poseidon 仓库已经通过可选构建路径接入本 Runtime。`PoseidonCpuApi` 同时支持不依赖 MPI 的单进程模式和可选的 MPI 多进程模式；MPI 模式支持 Host rank 间的明文/密文 Transfer、Replicate、计划与 context 一致性检查及全组终止。手工 RuntimePlan 已在 2/4 rank 下通过，Dacapo 生成的 2-rank MLP 也已完成 Poseidon、MockVecApi 和 Python 三方验证。
- 单进程 `PoseidonGpuApi` 已从单卡兼容扩展到本机多卡：`Device(rank=0,index=i)` 映射到构造时给出的第 `i` 个物理 CUDA device，每张卡独立持有参数、Evaluator 和密钥缓存；旧单卡构造函数保持兼容。
- 本机 GPU 通信已支持 Host↔Device、Device↔Device、Transfer 和 Replicate。跨卡复制先物化完整目标对象，再按拓扑使用 CUDA P2P 或 pinned Host 中转；V1 仍只接受单 field、单完整 shard 的值。
- GPU compute 结果现在记录 CUDA completion event，不再在每个算子末尾无条件执行 `cudaDeviceSynchronize`。Runtime 继续按 Dacapo 的静态顺序提交，因此不同设备上的独立算子可以重叠；跨卡通信、下载和最终输出会等待源 Value。多级 Rescale 为保护本次调用中的临时对象仍保留一次同步。
- GPU ModSwitch 已接入 `GpuEvaluator` 和 `PoseidonGpuApi`：直接物化目标 q-count 并复制保留的 RNS limbs，保持 scale、NTT 和 components 不变，支持一次下降多个 level。单卡 GPU API 测试已在 RTX 4060 Laptop 上通过，覆盖下降两个 level 后的元信息和解密结果；四次 30-bit lazy Rescale 也已和 CPU 四次 Rescale 做结果差分。双卡用例仍需双卡机器验收。
- Poseidon 中已删除旧 `src/poseidon/mgpu` 调度/解释/通信系统、HEVM/CST frontend、工具、benchmark 和对应测试。现在只保留 RuntimePlan 执行器、`PoseidonCpuApi`/`PoseidonGpuApi` 和它们直接使用的 GPU 通信原语。
- 手工双卡 RuntimePlan 已覆盖反向逻辑/物理设备映射、Replicate、明文/密文跨卡传输和两卡计算；当前单卡开发机只编译并跳过该用例，仍需在真实双卡机器上完成运行验收。
- Dacapo 已生成多 rank/device placement 和点对点 Transfer；Poseidon CPU MPI 的既有测试仍使用手工 RuntimePlan。Dacapo 产出的 Device 计划目前由 MockVecApi 验证，交给真实 PoseidonGpuApi 还需要生产 GPU OperatorSpec、受支持算子合法化和多卡硬件验收。
- Dacapo 已加入 `materialize-ckks-physical-levels`：Earth 继续按 120-bit/1 个逻辑 level 分析，Earth→CKKS 后、placement 前再按目标 OperatorSpec 把一个逻辑 level 展开为 4 个物理 level，并同步改写 ModSwitch down factor。测试覆盖 `L13 -> L9` lazy Rescale、`downFactor 1 -> 4`、placement 使用物理 level 代价、scale bit 不匹配和物理链不足。
- 生产 GPU OperatorSpec 尚未完成。当前 placeholder profile 使用 28-bit 模数且只允许两层 Rescale，不符合实际 30-bit 模数和 lazy Rescale 连续下降 4 层的设计；它对 ModSwitch/Boot 的声明也不符合当前真实能力，生产 spec 应启用 ModSwitch 并关闭尚未实现的 Boot。

这些内容不属于 RuntimePlan V1 冻结条件。
