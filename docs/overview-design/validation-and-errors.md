# 合法性验证与错误处理

> 本文列出目标验证规则。当前 `PlanVerifier` 已覆盖基础结构、SSA、Place、类型和 Transfer/Replicate；OperatorSpec、能力/密钥、Encode/bundle 和实际 Api::Value 的完整元信息核对仍未实现。现状见[实现状态](implementation-status.md)。

## 1. 原则

Runtime 在加载可执行计划后，必须重新做一遍合法性检查：

~~~text
编译器：       生成计划
Runtime 验证器：只检查，不修改
Runtime：      执行；任何错误立即终止
~~~

Runtime 不做自动修复、重新分配、降级选择或错误恢复。

## 2. 验证层次

| 层次 | 检查内容 |
| --- | --- |
| MLIR verifier | 单条指令的类型、属性和效果 |
| 编译器各 Pass 的 verifier | 逻辑、已分配、可执行各阶段的不变量 |
| Runtime 加载检查（preflight） | 计划结构、物理 Place、目标配置和输入绑定 |
| Runtime 运行时检查 | Ready/Pending 状态、Api 返回值和异步等待 |

编译期和 runtime 重复做关键检查是有意的：计划可能来自文件、别的编译器版本，甚至是损坏的输入。

## 3. SSA 结构检查

- 格式/版本；
- ValueId 唯一定义，且每个 ValueId 恰好绑定一个 Place；
- 不允许同一个 ValueId 复用于多个 Place；
- 不允许先用后定义或使用未定义的输入；
- 指令参数个数、输入输出值类型、必需属性齐全；
- 每个值都有 context、level、`scale_log2`、NTT 和分量数；`level` 与 `scale_log2` 必须是非负整数；
- 没有未知指令；
- 数据依赖构成有向无环图；
- 当前版本不支持控制流。

注意：MLIR 打印出来的 `%2` 这类名字不是稳定 ID。翻译成可执行计划时必须分配稳定的指令序号和传输 ID。

## 4. Encode 与计算指令检查

Encode 单独检查:

- 只能出现在 initialization;
- 输出必须是 Host plaintext,components 必须为 1;
- payload 只能是 inline 或 bundle,不能两种字段混写;
- inline 数组非空且所有值都是有限 float64,不允许负零,slot 数不超过 context 容量;
- bundle `content` 格式正确，且能在 manifest 中找到；
- manifest 中的每个 blob 文件都存在，长度和 SHA-256 相符；
- blob 能解码成非空的小端 float64 数组，不含 NaN、Infinity 或负零；
- 每个 Encode 再按自己的输出描述检查 slot 容量；
- 输出 level 和 `scale_log2` 满足 OperatorSpec 对所有值的范围和模数预算约束;
- Encode 输出不能同时列为 external_input;
- 同一 `content` 被多个 Encode 使用是合法的,各输出元信息仍分别检查。

对每条计算指令：

- 类别必须是纯计算；
- 只有一个执行 Place，且该 Place 类型是目标 Api 能计算的；
- 每个输入已在执行 Place 上物化，不存在隐式跨 Place 操作数;
- 输出只在执行 Place 产生；
- 指令和属性合法；
- Rescale/Boot 的目标 level、`target_scale_log2`、目标分量数和输出 ValueDesc 一致；
- 算子引用的 OperatorSpec/profile 存在且与计划文件头一致；
- `Boot(implementation=decrypt_reencrypt)` 只能放在 Host,并要求 CPU 解密、编码、加密能力和 secret key；
- CKKS 类型和元信息满足该算子的前置条件。

验证器只能检查计划声明和接口返回值；Api 实现仍必须遵守“计算函数不主动发起通信”的约定。

## 5. CKKS 语义检查

CKKS 检查分四类：

- 值本身：明文/密文、context、多项式阶数、level、`scale_log2`、NTT 和分量数；
- 普通算子：add/sub 输入兼容性、乘法前置条件和 rotate 步数；
- 改变元信息的算子：rescale、modswitch、relinearize 和 boot 的输入输出规则；
- 部署条件：rotate/relinearize/boot 所需密钥，以及 boot profile 的输入范围和实现模式。

`level` 是整数模数链层号，`scale_log2=x` 表示逻辑 scale 为 `2^x`。V1 不接受浮点 scale。Dacapo、Runtime 和 Vec 后端对这两个字段必须使用同一含义。

VecExecutor 也要模拟这些元信息的变化，否则 rescale、modswitch、boot 全是空操作，非法的图也能骗过测试。

## 6. Place 检查

- PlaceKind 是 Host 或 Device；
- rank 在编译目标范围内；
- Device 序号在该 rank 的目标范围内；
- Host 在 JSON 中只有 kind/rank,不允许 index;当前 C++ `Place` 内部用 `index = 0` 表示 Host；
- 计算没有被放到只能存数据的位置；
- 通信的来源/目标属于编译目标；
- 外部输入的初始位置正确；
- 输出的最终位置正确。

Runtime 不重新解析拓扑模型，只验证计划里的物理 Place 和启动环境相符。

## 7. 通信动作检查

- 传输 ID 唯一；
- CommKind 已知，输入输出数量符合该类型；
- 源值已定义，源类型和输出类型的关系合法；
- 来源/目标 Place 数量正确；
- JSON 中的 outputs、destinations、output_kinds 数量一致；C++ 内部对应字段名是 `output_types`；
- 目标不重复，输出 ValueId 不重复；
- `outputs[i]` 的唯一 Place 等于 `destinations[i]`；
- Replicate 输出数等于目标数，且不允许一个输出对应多个目标；
- Transfer/Replicate 不改变数学上的值类型；
- Transfer/Replicate 不改变 context、level、`scale_log2`、NTT 状态和分量数；
- 当前协议版本支持该通信种类；V1 只支持 Transfer/Replicate；
- CommHint 枚举值合法；
- 通信输出必须先定义，后续指令才能使用；
- 所有 Pending 输出最终有消费者或被收尾阶段等待；
- 集合通信在全局计划中的顺序确定。

RuntimePlan V1 只接受 Transfer 和 Replicate。Gather、Scatter、AllGather 即使枚举在代码中预留,也必须在协议升级并定义结果布局之前直接拒绝。

Runtime 不验证 MPI/NCCL 是否支持某个 hint——Api 自己降级，实在不行就报错。

## 8. 启动目标检查

启动检查分四组：

- **实际环境**：world size、本地 rank、本地设备数与计划目标一致；
- **协议与后端**：计划 ID/版本、target、capability、OperatorSpec id/version/source_sha256、rescale/boot 模式一致；
- **部署输入**：本 rank 的 external_inputs 完整绑定，Api 已在启动前配置所需 context 和密钥；
- **多 rank 数据**：默认模式下各 rank 使用相同的计划原始字节 SHA-256；存在 bundle Encode 时，都拿到 id/version/manifest_sha256 和完整 blob 集合一致的数据包。

这些信息分别来自 RuntimeEnvironment、部署方提供的 OperatorSpec/bundle 和已配置的 Api。Runtime 调用 Api 的 `preflight` 检查启动配置，并用 `validate_value` 检查不透明值；缺任何一项都直接终止。

不设计完整的能力协商，部署环境和编译配置不匹配就直接终止。

## 9. 运行时检查

消费方读取输入时：

~~~text
Ready:   Runtime 已拿到可交给 Api 的值句柄
Pending: 通信还没产出本地句柄；调 Api.wait 后安装为 Ready
缺失:    立即终止
~~~

Ready 只表示 Runtime 已经拿到值句柄，不表示 GPU 已经完成。依赖等待由 Api 在句柄内部处理。

bundle I/O、`Api.preflight`、`encode_plaintext`、计算、`communicate_async`、`wait` 或最终 `synchronize` 抛出的任何异常都直接进入顶层的 fail-fast 流程。

Runtime 还要检查：

- Api 返回的输出数量正确；
- 每个值的类型、位置和完整 CKKS 元信息符合 ValueDesc；
- 每个通信句柄的本地输出下标对应一个不同的 ValueId；
- 收尾后没有未等待的发送句柄。

## 10. 诊断信息

错误至少包含：

~~~cpp
struct FatalDiagnostic {
    std::string reason;        // 原因描述

    std::uint64_t plan_id;
    std::string operator_spec_id;
    std::string operator_profile;
    std::size_t op_ordinal;    // 指令序号
    OpKind op_kind;

    ValueId value_id;
    TransferId transfer_id;

    Place source;
    Place destination;

    int level;
    std::uint32_t scale_log2;

    int local_rank;
    std::string api_name;
};
~~~

不要求每种错误都填满所有字段，但必须尽可能打印上下文。

> **当前缺口：** compute 尚未打印全部输入，通信尚未打印全部输入输出和多目标 Place，加载/preflight 错误也不一定带指令序号。

## 11. Fail-fast 流程

~~~text
发现错误
  -> 抛出异常
  -> Runtime 顶层补充当前指令的上下文
  -> 打印到 stderr
  -> 刷新日志
  -> 调用 Api.abort_all
~~~

MPI 版 Api 调 `MPI_Abort`；单进程的 Mock Api 抛出 ClusterPanic 并终止所有模拟执行实例。

不设计：重试、取消、回滚、超时恢复、节点恢复、检查点、把错误值继续往后传、返回部分结果。

## 12. 调试模式

部署方可以显式设置 `skip_artifact_digest_checks=true`。它只跳过计划跨 rank、OperatorSpec 和 bundle manifest 的原始字节 SHA-256 比较;计划不能在 JSON 中自行开启。开启时必须输出警告,并要求所有 rank 使用相同值。

以下检查即使在该模式下也不能跳过:版本和字段结构、SSA、Place、算子元信息、能力和密钥、id/version、bundle 文件存在性和长度、blob `content` SHA-256、float64 合法性以及 Api 实际值校验。也就是说,这是方便调试发布文件摘要的开关,不是“关闭验证器”。

调试模式可以按需记录：

- 执行轨迹：每条指令的输入输出类型、Ready/Pending 变化、Host/Device 物化；
- 通信：实际采用的 hint 降级路径；
- 资源：值分配/释放次数和内存峰值；
- 故障：指定指令或传输的错误注入。

这些信息用于暴露错误和分析实验，不改变执行语义。
