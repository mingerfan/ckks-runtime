# RuntimePlan 版本与兼容规则

## 1. 什么算一个版本

`format_version` 在正式发布后标识文件格式和执行语义的一个**冻结快照**。同一个已冻结版本号下,规范、Schema 和样例集的语义不允许变化——发现规范有歧义可以澄清措辞、补充样例,但不允许改变任何已定字段的含义、取值范围或合法组合。

需要以下任何一种变化时,必须启用新的 `format_version`:

- 增删字段,或改变字段的类型、编码、约束;
- 改变枚举的取值集合;
- 改变原始文件摘要字段或计算规则;
- 改变任何执行语义(算子元信息规则、通信语义、阶段划分)。

**同一个版本号下不允许"看文件内容猜格式"**;也不靠"缺了某字段"或"解析失败"推断这是旧格式文件。版本判断只看 `format_version` 一个字段。

> **V1 已冻结。** V1 使用显式 Encode、inline/bundle payload 和完整原始文件字节 SHA-256。external_inputs 只表示调用方每次运行传入的参数，manifest 不绑定 ValueId。

## 2. 修改流程

1. 先改 Runtime 仓库里的规范和 Schema,同步增补 `testdata/` 的合法/非法样例;
2. Runtime 实现新版本的读取和验证,配独立入口和独立测试;
3. Dacapo 实现新版本的生成;
4. Runtime 更新 Dacapo submodule 指针,在同一个 commit 里锁定"这对组合验证过";
5. Poseidon 等该 Runtime commit 测试通过后再更新自己的指针。

## 3. 多版本共存

是否同时支持多个 `format_version` 由发布需求决定,不是默认义务。支持的每个版本必须:

- 有各自完整的规范、Schema 和样例集目录(`docs/runtime-plan/v2/` 等);
- 有独立的读取入口和独立测试;
- 不做任何隐式回退——V2 读取器读到 V1 文件就报错,让调用方显式选择入口。

## 4. target 与 capability 的兼容

`format_version` 之外还有两个正交的兼容轴:

- `target_id` + `capability_version`:Runtime 的每个 Api 实现声明自己支持哪些组合。计划要求的组合不在支持列表里,执行前直接拒绝;
- `operator_spec` 的 id/version/source_sha256：Runtime 持有一份明确选择的 spec 文件，默认要求三项都相符。`source_sha256` 直接覆盖完整原始文件字节;生产部署还要求其状态为 `validated`，`placeholder` 和 `imported` 只能由测试或迁移工具显式允许。

这两轴升级不需要动 `format_version`:新增一个 target 或一版 OperatorSpec,只是新的取值,文件格式没变。

## 5. 拒绝优先于兼容

所有兼容判断的默认答案是拒绝:未知版本、未知 target、未知 capability、摘要不符、能力集合对不上,一律报错退出并给出具体原因。宁可让用户显式升级工具链,不做任何形式的静默降级。

唯一例外是部署方显式设置 `skip_artifact_digest_checks=true` 的调试运行。它只跳过 RuntimePlan、OperatorSpec 和 bundle manifest 的原始字节摘要比较,不能跳过版本、结构、语义、blob 内容或后端能力检查;生产配置不得启用。
