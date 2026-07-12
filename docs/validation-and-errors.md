# 合法性验证与错误处理

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
- 没有未知指令；
- 数据依赖构成有向无环图；
- 当前版本不支持控制流。

注意：MLIR 打印出来的 `%2` 这类名字不是稳定 ID。翻译成可执行计划时必须分配稳定的指令序号和传输 ID。

## 4. 计算指令检查

对每条计算指令：

- 类别必须是纯计算；
- 只有一个执行 Place，且该 Place 类型是目标 Api 能计算的；
- 每个输入已在执行 Place 上物化，不存在隐式跨 Place 操作数;
- 输出只在执行 Place 产生；
- 指令和属性合法；
- CKKS 类型和元信息满足该算子的前置条件。

Runtime 能检查的是声明和接口契约，无法证明任意 Api 的 C++ 实现内部绝对没有副作用；约定是计算函数不主动发起通信。

## 5. CKKS 语义检查

按计划中可获得的信息检查：明文/密文类型；参数/上下文标识；多项式阶数；level；scale；NTT 形式；分量个数；add/sub 的输入兼容性；乘法前置条件；rescale 的 level 要求；rotate 步数；relinearize/rotate/boot 所需的密钥配置。

VecApi 也要模拟这些元信息的变化，否则 rescale、modswitch、boot 全是空操作，非法的图也能骗过测试。

## 6. Place 检查

- PlaceKind 是 Host 或 Device；
- rank 在编译目标范围内；
- Device 序号在该 rank 的目标范围内；
- Host 序号符合当前约定；
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
- outputs、destinations、output_types 数量一致；
- 目标不重复，输出 ValueId 不重复；
- `outputs[i]` 的唯一 Place 等于 `destinations[i]`；
- Replicate 输出数等于目标数，且不允许一个输出对应多个目标；
- Transfer/Replicate 不改变数学上的值类型；
- Gather/Scatter 的结果布局已定义；
- CommHint 枚举值合法；
- 通信动作支配所有使用者；
- 所有 Pending 输出最终有消费者或被收尾阶段等待；
- 集合通信在全局计划中的顺序确定。

Runtime 不验证 MPI/NCCL 是否支持某个 hint——Api 自己降级，实在不行就报错。

## 8. 启动目标检查

启动时检查：计划 ID/版本、节点总数、本地 rank、本地设备数、编译目标标识、所有外部输入已绑定、所有 rank 使用相同计划。

不设计完整的能力协商，部署环境和编译配置不匹配就直接终止。

## 9. 运行时检查

消费方读取输入时：

~~~text
Ready:   直接使用
Pending: 调 Api.wait，成功后安装为 Ready
缺失:    立即终止
~~~

`Api.wait`、计算或 `communicate_async` 抛出的任何异常都直接进入顶层的 fail-fast 流程。

Runtime 还要检查：Api 返回的输出数量、值类型、位置；每个通信句柄的本地输出下标恰好对应一个不同的 ValueId；收尾之后没有未等待的发送句柄。

## 10. 诊断信息

错误至少包含：

~~~cpp
struct FatalDiagnostic {
    std::string reason;        // 原因描述

    std::uint64_t plan_id;
    std::size_t op_ordinal;    // 指令序号
    OpKind op_kind;

    ValueId value_id;
    TransferId transfer_id;

    Place source;
    Place destination;

    int local_rank;
    std::string api_name;
};
~~~

不要求每种错误都填满所有字段，但必须尽可能打印上下文。

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

可选增加：每条指令的输入输出类型记录、Ready/Pending 状态轨迹、实际通信降级的记录、Host/Device 物化轨迹、值分配/释放计数、内存峰值、指定指令或传输的故障注入。

这些信息用于暴露错误和分析实验，不改变执行语义。
