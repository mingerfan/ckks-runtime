# 合法性验证与错误处理

## 1. 原则

Runtime 可以并且必须在加载 executable plan 后重新检查合法性。

~~~text
Compiler:
    生成计划

Runtime Verifier:
    检查，不修改

Runtime:
    执行；任何错误立即 panic
~~~

Runtime 不做自动修复、重新 placement、fallback 选择或错误恢复。

## 2. 验证层次

| 层次 | 内容 |
| --- | --- |
| MLIR verifier | op-local 类型、属性和 effect |
| Compiler pass verifier | logical、placed、executable 各阶段不变量 |
| Runtime preflight | executable plan、物理 Place、目标配置和输入绑定 |
| Runtime dynamic check | Ready/Pending、Api 返回值和异步 wait |

编译期和 Runtime 重复关键检查是合理的，因为计划可能来自文件、其他编译器版本或损坏输入。

## 3. SSA 结构检查

- schema/version；
- ValueId 唯一定义；
- 每个 executable ValueId 恰好绑定一个 Place；
- 不允许同一个物理 ValueId 复用于多个 Place；
- use-before-def；
- 未定义输入；
- op arity；
- input/output ValueType；
- 必需 attribute；
- 未知 op；
- 数据依赖 DAG；
- 当前版本不支持的 control flow。

MLIR 打印出来的百分号名称不是稳定 ID。翻译 executable plan 时必须分配稳定 op ordinal 和 TransferId。

## 4. Compute 检查

对每个 compute op：

- trait 必须是 PureCompute；
- 只有一个执行 Place；
- PlaceKind 必须是该目标 Api 可计算的位置；
- 每个 input 已在执行 Place 物化；
- output 只在执行 Place 产生；
- 不存在隐式跨 Place operand；
- op 和 attributes 合法；
- CKKS ValueType/metadata 满足前置条件。

Runtime 能检查声明和接口契约，不能证明任意 Api 的 C++ 实现内部绝对无副作用。计算 API 不应主动调用通信。

## 5. CKKS 语义检查

按计划中可获得的信息检查：

- plaintext/ciphertext 类型；
- context/parameter identifier；
- polynomial degree；
- level/parms identifier；
- scale；
- NTT form；
- component count；
- add/sub 输入兼容；
- multiply 前置条件；
- rescale level；
- rotate step；
- relinearize/rotate/boot 所需配置。

VecApi 也应模拟 metadata 转换，避免非法图因为 rescale、modswitch、boot 全部 no-op 而通过。

## 6. Place 检查

- PlaceKind 为 Host 或 Device；
- rank 在编译目标范围内；
- Device index 在该 rank 的目标范围内；
- Host index 符合当前约定；
- compute 不被错误放到仅存储 Place；
- CommAction source/destination 属于编译目标；
- external input 的初始 Place 正确；
- output 最终 Place 正确。

Runtime 不重新解析 TopologyModel，只验证计划中的物理 Place 与启动环境匹配。

## 7. CommAction 检查

- TransferId 唯一；
- CommKind 已知；
- input/output 数量符合 CommKind；
- source Value 已定义；
- source ValueType 与 output ValueType 合法；
- source/destination Place 数量正确；
- outputs、destinations 和 output ValueType 数量一致；
- destination 不重复；
- output ValueId 不重复；
- output[i] 的唯一 Place 等于 destinations[i]；
- Replicate output 数量等于 destination 数量；
- Replicate 不得把一个 output ValueId 分配给多个 destination；
- Transfer/Replicate 不改变数学 ValueType；
- Gather/Scatter 的 result layout 已定义；
- CommHint 枚举合法；
- action 支配所有 consumer；
- 所有 Pending output 最终存在 consumer 或 finalization；
- collective action 在全局计划中顺序确定。

Runtime 不验证 MPI/NCCL 是否支持该 hint。Api 自行使用首选实现或等价 fallback；无可用实现时直接抛错。

## 8. 启动目标检查

启动时检查：

- plan ID/version；
- world size；
- local rank；
- 本地 Device 数量；
- 编译目标标识；
- 所有外部输入已绑定；
- 所有 rank 使用相同计划。

不设计完整 capability negotiation。目标部署与编译配置不匹配时直接 panic。

## 9. Dynamic Check

consumer 读取输入时：

~~~text
Ready:
    直接使用

Pending:
    Api.wait
    成功后安装 Ready Value

Missing:
    panic
~~~

Api.wait、compute 或 communicate_async 抛出的任何异常都立即进入顶层 fail-fast。

Runtime 还要检查：

- Api 返回的 output 数量；
- output ValueType；
- output Place；
- CommHandle 的每个本地 output slot 恰好对应一个不同 ValueId；
- Finalization 后没有未等待的 source handle。

## 10. 诊断

错误至少包含：

~~~cpp
struct FatalDiagnostic {
    std::string reason;

    std::uint64_t plan_id;
    std::size_t op_ordinal;
    OpKind op_kind;

    ValueId value_id;
    TransferId transfer_id;

    Place source;
    Place destination;

    int local_rank;
    std::string api_name;
};
~~~

不要求所有字段对所有错误都有值，但必须尽可能打印上下文。

## 11. Fail-fast 流程

~~~text
detect error
  -> 抛出异常
  -> Runtime 顶层补充当前 op 上下文
  -> 打印 stderr
  -> flush
  -> Api.abort_all
~~~

MPI Api 可以调用 MPI_Abort。单进程 Mock Api 可以抛出 ClusterPanic 并终止所有模拟执行实例。

不设计：

- retry；
- cancel；
- rollback；
- timeout recovery；
- rank recovery；
- checkpoint；
- 错误 Value 继续向后传播；
- 部分执行结果。

## 12. Debug 模式

可选增加：

- 每个 op 的 input/output ValueType；
- Ready/Pending 状态 trace；
- 实际通信 fallback trace；
- Host/Device materialization trace；
- Value allocation/free 数量；
- memory high-water；
- 指定 op/transfer 失败注入。

这些信息用于暴露错误和分析实验，不改变执行语义。
