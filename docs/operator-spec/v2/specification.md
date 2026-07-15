# OperatorSpec V2 规范

OperatorSpec V2 用于保存 Dacapo 旧 profile 中 V1 无法表达的逐 level 噪声和 Boot 延迟。RuntimePlan V1 可以通过 `spec_id`、`version` 和原始文件 SHA-256 引用 V1 或 V2 OperatorSpec；V1 格式保持冻结不变。

## 相比 V1 的变化

- `spec_format_version` 为 `2`。
- `status` 是 `imported` 或 `validated`。`imported` 表示数据已按旧 reader 的实际语义迁移，但还没有按当前 Runtime/Poseidon 后端重新测量验证。
- 新增 `noise_unit`。没有噪声表时必须为 `null`；存在噪声表时必须写明单位或模型名称。
- 新增 `provenance`，记录来源仓库、revision、路径和源文件原始字节 SHA-256。
- 普通算子的 `noise_by_level` 可以是非负有限数数组。
- Boot profile 使用 `latency_us_by_input_level` 和 `noise_by_input_level`，不再用单个 `latency_us`。

其余 context、level、算子支持、rescale 和 Boot 元信息规则与 V1 相同。

## 表的下标

所有逐 level 数组长度必须等于 `levels.upper_bound + 1`，数组下标就是 RuntimePlan 使用的 CKKS level：较大的 level 表示剩余模数更多，Rescale 后 level 下降。

Dacapo 旧 reader 会先在 profile 数组前补一个 level 0，再把数组截断或用最后一个值补齐到 `levelUpperBound + 1`。迁移脚本严格复现这个行为，使 V2 中的代价表等于旧编译器实际使用的表，而不是按源 JSON 的表面下标猜测。

缺少测量时写 `null`，不能把未知代价写成 0。旧 profile 没有对应算子的延迟和噪声时，该算子写 `supported=false`。

## 噪声

Dacapo SEAL profile 的噪声值只在旧 ErrorEstimator 公式中有意义，没有跨后端物理单位。因此导入文件使用：

```json
"noise_unit": "dacapo-legacy-estimator"
```

Runtime 只检查结构，不使用噪声重新规划。后续生产 profile 必须定义自己的稳定单位或模型名称，不能把这组数值当成 Poseidon 噪声数据。

## Bootstrap level 换算

旧 profile 的 `bootstrapLevelLowerBound=B_lo`、`bootstrapLevelUpperBound=B_hi` 服务于 Earth level。Earth level 随 Rescale 增加；Runtime level 方向相反。

Dacapo 当前管线使用：

```text
init_level = B_hi
Earth 合法输入 = [0, B_hi - B_lo]
Runtime level = init_level - Earth level
```

所以迁移后的 Boot 输入范围为 `[B_lo, B_hi]`，Boot 输出的 Earth level 为 0，对应 Runtime `output_level=B_hi`。结果数值看起来和旧上下界相同，但必须通过上述换算得到，不能把字段直接改名复制。

## Dacapo 导入 profile 的边界

旧 profile 只保存 `rescalingFactor`，没有保存每个真实 RNS 模数。导入时按旧编译器代价模型展开成等长的 `rns_moduli_log2`。SEAL 的 14 个 60-bit 模数还能和旧 HEVM 初始化代码对上；HEAAN FVa 的精确模数链不在仓库中，因此两份 HEAAN profile 保持 `status=imported`，不能作为生产 Runtime 的后端参数。

旧管线的常用 waterline 是 40，导入 profile 的 `default_scale_log2` 和 Boot 输出 scale 固定为 40。它们用于复现旧编译配置，不代表所有 Dacapo 命令行 waterline。

## 生成

三份 Dacapo profile 由仓库内脚本生成：

```bash
python3 integrations/dacapo/migrate_profiles.py
python3 integrations/dacapo/migrate_profiles.py --check
```

脚本要求 `third_party/dacapo` 已初始化并处于记录的 revision；来源字段或数组类型不符合预期时直接失败。
