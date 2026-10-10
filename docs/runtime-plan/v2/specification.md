# RuntimePlan V2：Release 与 reuse_input

V2 沿用 [V1](../v1/specification.md) 的顶层结构、阶段顺序、值描述、
算子参数和严格 JSON 规则，只增加下面两项。OperatorSpec 和 plaintext
bundle 的格式不变。V1 继续可读，但不能出现新指令或新字段。

reader、计划打印、Verifier 和执行器已支持 V2 的 Release，API 异步数据
清理与通信交付已完成。原位执行留到 step 4；含 `reuse_input` 的计划仍会
在装载 bundle、绑定输入和提交 API 工作前明确报错。

## Release

```json
{"ordinal": 3, "kind": "release", "value": "7"}
```

Release 没有结果，也不算一次数据使用。它可以出现在初始化、执行或收尾
阶段，释放一个已经定义的值。不能释放最终返回值，也不能再次释放已经
释放或被原位覆盖的旧值。以后任何计算或通信都不能再使用这个名字。

外部输入可以释放 Runtime 持有的引用，调用方持有的对象不受影响。
不要求每个中间值必须写 Release。没有 `wait` 字段。

执行前，Runtime 统计三个阶段中本地值尚待提交的使用次数。计算的每个
操作数出现都计一次；并行通信按实际提交的切片计数，Replicate 的本地
目的地各计一次，跨 rank 的目的地由通信提交线程一起提交、计一次。
初始化结束后继续使用剩余计数，不重新统计。

API 成功接手计算或传输后，Runtime 清掉本次调用的临时输入副本，再减少
次数。Release 先记录释放请求；次数归零时才清掉 Runtime 的数据引用，
保留值表项。未提交的其他线程仍能取到数据，已提交的异步工作由 API 保留。
已释放且不再使用的通信输出晚到时直接丢弃，不重新放入值表。
Release 会调用非阻塞的 `collect_completed()`；整次运行收尾仍调用 `drain()`。

## reuse_input

```json
{
  "ordinal": 4,
  "kind": "compute",
  "op": "negate",
  "place": {"kind": "host", "rank": 0},
  "inputs": ["7"],
  "output": "8",
  "reuse_input": 0
}
```

`reuse_input` 是 Compute 的可选非负整数，遵守 V1 其他整数的 int32 范围
规则，不接受浮点数、字符串、布尔值或 null。目前只有 0 合法。
没有字段时使用普通计算路径。写了字段后不能默默改用普通路径。

允许原位的操作：

| 位置 | 操作 |
| --- | --- |
| Host | Negate、Rotate |
| Device | AddCP、SubCP、Rotate |

被覆盖的输入必须来自本计划中的计算，不能来自 external input、Encode
或通信。它不能是最终返回值，在三个阶段中只能出现一次操作数使用。
传输源也算使用；`AddCC(x, x)` 算两次；Release 不计入次数。

输入和输出必须在相同 Place，kind、context、level、scale、NTT 和
components 保持一致。现有算子类型和参数检查继续生效。实际存储布局
是否允许原位，留给后续 CPU/GPU 原位入口检查。

计算后输入的旧名字失效，新输出可以继续作为下一次原位操作的输入。
`final_outputs` 可以返回新输出，但不能同时保留被覆盖的旧输入。
需要保存所有历史值的 `AllValuesAfterRun` 模式与 Release/reuse_input
不兼容，执行前报错。

## 检查与样例

[schema.json](schema.json) 描述 JSON 结构。使用次数、定义来源、值生命周期
和操作是否允许原位由 Verifier 检查；Schema 不承担这些检查。

- [CPU 三次原位链](testdata/valid/cpu_reuse_chain.json)：初始化计算产生输入，
  执行阶段 Negate/Rotate，收尾阶段 Negate。三个阶段共用使用次数和生命周期。
- [GPU 三次原位链](testdata/valid/gpu_reuse_chain.json)：AddCP/SubCP/Rotate，
  展示传输生成副本与计算生成值的区别。这是解析及验证样例，尚不能执行。

`runtime_plan_memory_tests` 覆盖合法链、V1/V2 版本限制、严格字段解析、
释放错误、原位来源和使用次数、通信使用、元信息检查、原位执行前拒绝，
以及去掉原位标记后的 CPU/GPU Release 计划执行。
`runtime_release_tests` 检查跨阶段和重复操作数计数、拆分发送、失败提交、
晚到输出、Mock 多设备和跨 rank 通信；MPI 用例也有 Release 版本。
