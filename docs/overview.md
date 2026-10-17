# CKKS Runtime Overview

CKKS Runtime 执行编译器已经排好的同态计算计划。编译器确定每条指令的计算位置、CKKS 元信息和数据搬运；Runtime 验证计划、管理值的生命周期并调用 Api。具体密文对象、计算库和通信实现由 Api 提供。

## 编译器、Runtime 和后端

```text
Dacapo Python tracing / Earth MLIR
  -> scale 与 Boot 规划
  -> CKKS SSA 与目标物理 level
  -> rank/device placement
  -> 显式通信
  -> 可选 Waterline Prepare、Release 与原位规划
  -> RuntimePlan + OperatorSpec 引用 + 可选 plaintext bundle
       |
       v
RuntimePlanReader -> PlanVerifier -> SequentialRuntime<Api>
                                        |
                       Vec/Mock/MPI 或 Poseidon CPU/GPU Api
```

Runtime 和 Dacapo 不共享 C++ 类型。它们通过 [RuntimePlan](runtime-plan/README.md)、[OperatorSpec](operator-spec/README.md) 和 [bundle](runtime-plan/plaintext-bundle.md) 协议对接。默认文件表示是[二进制](runtime-plan/binary.md)，JSON 用于检查指令和测试。

Runtime 不反向依赖 Dacapo 或 Poseidon。Dacapo 是可选 submodule，单独构建 MLIR 编译器；Poseidon 把 Runtime 作为库接入。跨仓库变更按 Dacapo 提交、Runtime 更新指针并验证、Poseidon 更新指针并验证的顺序发布。

## 值和位置

| 名称 | 含义 |
| --- | --- |
| SSA / ValueId | 一个值只定义一次，每个 ValueId 属于一个 Place；复制后使用新的 ValueId |
| Place | Host 是 `(host, rank)`；Device 是 `(device, rank, index)`，index 是 rank 内的逻辑设备编号 |
| level | 剩余 RNS 模数链层号，越大表示剩余模数越多；Rescale/ModSwitch 后下降 |
| scale_log2 | scale 的二进制指数，例如 40 表示逻辑 scale 为 `2^40` |
| Encode | 将 inline 或 bundle 中的 float64 slots 编码为 Host CKKS 明文 |
| Prepare | V4 将同一 payload 完整编码为指定 Device 的 NTT 明文，后端在完整组准入时选择 |
| content | 编码前载荷的 SHA-256 标识；可以被不同 level/scale 的多个 Encode 共享 |
| external input | 调用方每次运行传入的 Host 对象；固定权重通常由 Encode 生成 |
| Ready | 已有本地值句柄；GPU 数据仍可能带有尚未完成的 stream/event 依赖 |
| Pending | 通信或 Prepare 尚未交付本地句柄，使用前等待对应完成点 |

每条 Compute 的输入和输出都在指令指定的 Place。Transfer 和 Replicate 显式生成其他位置的副本；Runtime 不自动补搬运，也不把 Host 伪装成某张 GPU。

## Dacapo 管线

CKKS 运算采用纯 SSA result-style。Earth 层负责逻辑 scale/level 分析，`materialize-ckks-physical-levels` 在 placement 前按目标 OperatorSpec 换算物理 level。Poseidon 的 30-bit lazy rescale 通常将一个 120-bit 逻辑降幅展开为 4 个物理 level；导出器只写已经换算好的类型，不再修改 level。

`assign-ckks-placement` 用确定性 HEFT 分配完整值。算子代价来自 OperatorSpec V2 的逐 level 表；Boot 根据 profile 的 implementation 选择 Host 或设备候选位置。通信代价支持固定成本，也支持值大小、带宽曲线和[具体设备对规则](../integrations/dacapo/communication-profile.md)。

`materialize-ckks-communication` 为跨 Place 操作数插入 `dist.transfer`，同一值到同一目标复用一个副本。Runtime 支持 Replicate，但编译器当前仍生成点对点 Transfer。外部输入和初始 Encode 位于 Host rank 0。

通信插入后可以规划 Release 和安全的 `reuse_input=0`。单 rank Waterline 模式先将直接 Host Encode 上传替换为 Device Prepare，再规划释放和原位计算，不插自动 Fence、不按批克隆权重。完整消费者组同时取得明文准备、输出、工作区和通信额度，最后使用真正完成后回收。编译器和 Runtime 检查单组容量；多 rank、明文 final output 和不可行容量明确拒绝。HEFT 不变；当前 placement 不建模密文分片、链路争用，也不联合搜索设备容量和调度。具体配置见 [V4 Waterline](runtime-plan/v4.md)。

## Runtime 执行

启动时检查结构、SSA、Place、CKKS 元信息、OperatorSpec、能力和具体密钥要求，再准备 bundle、完成 Api preflight 并绑定 external inputs。摘要检查随文件表示不同，见[二进制格式](runtime-plan/binary.md)；不能把 JSON 的整文件摘要保证套到二进制计划上。

计划依次执行 initialization、execution、finalization。V1/V2 Encode 在 initialization；V3/V4 允许 execution Encode。显式 Fence 保留整体边界语义；V4 Prepare 仅在 execution，真实 Ready 决定消费者何时可用。运行结束前完成通信、同步最终输出并调用 `Api::drain()`，再清理内部值表。同一个 Runtime 实例可以重复运行。

`DeviceExecutionMode::Sequential` 按普通计划顺序提交，异步 Api 可让不同设备上的工作重叠。`PerDeviceWorkers` 使用设备工作线程和有序通信提交，并有一条串行 CPU 队列处理 Encode、CPU Prepare 和就绪的 Host compute。Waterline 要求此模式，沿一个拓扑准入前缀滚动补充；每卡已准入工作独立推进，共享 Host/pinned/raw 预算统一计费。显式 Fence 等待边界前的工作和通信，并 drain Api。

ValueStore 管理 Ready、Pending 和 Released。Release 表示该名字不再可用，实际对象要等 Runtime 尚未提交的使用和 Api 异步工作都不再持有它才回收。原位计算使旧输入名字失效，输出使用新 ValueId。需要所有历史值的 `AllValuesAfterRun` 与 Release/reuse 不兼容，也不用于设备工作线程执行。

Api 提供 Encode、compute、通信、等待、元信息校验、preflight 和 abort。支持 `posted_outputs` 的后端可以先交付带事件的 GPU 句柄，消费者在自己的 stream 建立依赖；其他后端通过 Pending/wait 交付。`collect_completed()` 回收已完成的工作，`drain()` 等待所有工作结束。

任何解析、验证、Api 提交或异步执行错误都直接报错并终止执行组。Runtime 不重试、不重新 placement 或改排前台计算、不隐式回退 GPU Boot 到 CPU，也不修复非法计划。Waterline 仅在合法 CPU/GPU 完整准备候选中选择，启动后不迁移或失败重跑。

## 后端与测试

| 后端 | 职责 |
| --- | --- |
| VecExecutor | 明文 slots 运算，同步或每设备异步队列 |
| MockVecApi + MockCluster | 明文计算、进程内模拟通信、延迟/错误注入和多 rank 差分 |
| MpiVecApi | 明文计算与 MPI 非阻塞通信，验证多进程执行 |
| PoseidonCpuApi | 真实 CKKS CPU 计算；可选 MPI Host rank 通信 |
| PoseidonGpuApi | 真实 CKKS GPU 计算、本机多卡通信；可选 MPI/NCCL 跨进程通信 |

PoseidonGpuApi 按构造参数把逻辑 Device 映射到物理 CUDA device。计算和通信通过 stream/event 建立依赖，本机复制可使用 P2P 或 pinned Host 中转。原生 GPU Boot 需要显式配置对应 profile 与设备资源；Host `decrypt_reencrypt` Boot 需要公钥、私钥和显式 Device→Host→Device 通信。这条 Host 路径只用于测试与联调，会暴露明文。

具体支持的算子、位置、上下文和密钥由所选 OperatorSpec 与 Api preflight 共同验证。存在某个后端入口不代表任意参数或硬件配置都已验收。旧 HEVM/CST 与 Poseidon mgpu 执行链已删除。

测试入口是 [`tests/`](../tests/)、[`testing/`](../testing/) 和 Dacapo 自身的 CTest。明文差分用于检查计算与数据搬运；真实后端测试还需要解密数值和实际硬件验证。生产级容错、动态拓扑和 RNS 分片不属于当前执行模型。

## 代码导航

| 路径 | 职责 |
| --- | --- |
| `runtime/plan.*` | 强类型计划、指令与打印 |
| `runtime/plan_reader.*`、`json_plan_reader.*` | 按 `.bin`/`.json` 显式选择读取方式 |
| `runtime/operator_spec_reader.*`、`plaintext_bundle.*` | 后端配置、manifest 和本 rank 载荷读取 |
| `runtime/verifier.*` | SSA、生命周期、CKKS 元信息和资源要求 |
| `runtime/runtime.hpp` | 值存储、顺序/工作线程执行和计时 |
| `api/` | Vec、Mock、MPI 参考后端 |
| `integrations/dacapo/` | profile 迁移、模型产物生成和集成工具 |
| `tools/`、`benchmarks/` | bundle 转换、I/O 工具和 MPI 检查 |

Overview 说明职责和当前流程；字段约束统一维护在规范中。开发细节、测试过程和历史性能数字通过对应代码版本与 Git 历史查询。
