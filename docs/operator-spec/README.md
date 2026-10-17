# OperatorSpec 规范

OperatorSpec 描述一个后端的 CKKS 参数范围、算子支持、Boot profile 和编译期代价。RuntimePlan 通过 id、version 和文件原始字节 SHA-256 引用它。本文持续维护当前 reader 支持的格式；历史说明通过对应 Git commit 查询。

OperatorSpec 没有自嵌摘要。任何字节变化都会改变 RuntimePlan 中的 `source_sha256`；发布后修改内容必须升 version。

## 格式版本

| `spec_format_version` | status | 噪声与 Boot 代价 | Schema |
| --- | --- | --- | --- |
| 1 | placeholder / validated | 无噪声表；Boot 使用单个 latency_us | [V1](v1/schema.json) |
| 2 | imported / validated | 明确 noise_unit、provenance；Boot 按输入 level 保存表 | [V2](v2/schema.json) |

RuntimePlan 的语义版本不限制 OperatorSpec 必须是同号版本。未知 spec 格式版本直接拒绝。以下先说明共有规则与 V1 字段，再说明 V2 差异。

## 顶层字段

```json
{
  "spec_format_version": 1,
  "spec_id": "poseidon-ckks-gpu-v1",
  "version": 1,
  "status": "placeholder",
  "target_id": "poseidon-ckks-gpu",
  "rescale_mode": "lazy",
  "context": {},
  "levels": {},
  "operators": {},
  "boot_profiles": []
}
```

V1 的 `status` 是 `placeholder` 或 `validated`。`v1/profiles/` 中两份 poseidon profile 是协议测试占位数据，不能作为真实 Poseidon 参数或性能测量。真实后端是否接受某种 status，由其 preflight 检查。

`rescale_mode` 是 `eager` 或 `lazy`，只在 OperatorSpec 中声明。RuntimePlan 不重复保存它。

## context 和 levels

```json
{
  "context_id": "ctx-main",
  "poly_degree": 32768,
  "rns_moduli_log2": [28, 28, 28],
  "max_modulus_log2": 28,
  "default_scale_log2": 40
}
```

- `poly_degree` 必须是 2 的幂，slot 容量为 `poly_degree / 2`。
- `rns_moduli_log2` 下标就是 level；每项为正整数且不超过 `max_modulus_log2`。
- `levels.lower_bound <= levels.upper_bound`，上下界都必须落在模数链内。
- RuntimePlan 中每个 value 的 level 必须落在该区间，且 `scale_log2 < sum(rns_moduli_log2[0..level])`。

## operators

`operators` 必须完整列出 RuntimePlan 的 12 个计算 op。普通条目格式：

```json
{
  "supported": true,
  "latency_us_by_level": [0, 10, 20],
  "noise_by_level": null
}
```

- 延迟表可以为 `null`；若为数组，长度必须等于 `levels.upper_bound + 1`。
- `noise_by_level` 在 V1 必须为 `null`。
- Rescale 额外要求正整数 `max_levels_per_op`。
- Boot 条目只包含 `supported`，具体约束放在 `boot_profiles`。

计划使用 `supported=false` 的算子时直接拒绝。Runtime 不根据代价表重新规划。

## boot_profiles

```json
{
  "profile_id": "poseidon-native-boot-v1",
  "implementation": "native",
  "input_level_min": 2,
  "input_level_max": 13,
  "input_components": 2,
  "output_level": 12,
  "output_scale_log2": 40,
  "output_components": 2,
  "latency_us": 1000000,
  "host_requirements": {
    "needs_secret_key": false,
    "needs_host_compute": false
  }
}
```

- `profile_id` 在一份 spec 内唯一。
- 输入和输出 level 必须落在 `levels` 范围内，输入区间不能反向。
- `decrypt_reencrypt` 必须声明需要 secret key 和 Host compute；`native` 两项都必须为 false。
- `operators.boot.supported` 必须与 `boot_profiles` 是否非空一致。
- RuntimePlan 的 Boot 必须逐项匹配 implementation、输入范围和输出元信息。

Runtime 不在 native Boot 失败后改走 `decrypt_reencrypt`。

## Runtime 检查

Runtime 读取完整原始字节并计算 `source_sha256`，严格拒绝 BOM、重复 key、未知字段、错误类型和不支持的版本。然后检查上述结构约束，以及计划的 target、context、level、scale、算子支持、Rescale 降幅和 Boot profile。

## V2 字段差异

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

缺少测量时写 `null`，不能把未知代价写成 0。通常旧 profile 没有对应算子的延迟和噪声时，该算子写 `supported=false`。

Dacapo 导入 profile 的 `relinearize` 是一个明确标注的开发期例外。旧 `earth.mul_double` 已经包含乘法和重线性化语义，但没有独立的重线性化测量；新的 CKKS MLIR 又把两步拆成了 `mul_cc` 和 `relinearize`。为了让这批 `status=imported` 的 profile 能验证开发期 RuntimePlan，迁移脚本把 `relinearize` 标为支持，并在合法 level 上填写固定的 `1 us` 占位延迟，噪声保持 `null`。这个数值不代表测量结果，不能用于生产调度或性能比较；Poseidon profile 必须换成真实支持范围和测量数据。

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

脚本从 provenance 记录的 revision 读取旧 profile；`third_party/dacapo` 已初始化且 Git 中有该 revision 即可，当前 submodule 可以继续前进。来源字段或数组类型不符合预期时直接失败。

## 仓库 profile 的用途

| 目录 / 文件 | target | 状态与用途 |
| --- | --- | --- |
| `v1/profiles/poseidon-ckks-cpu.v1.json` | poseidon-ckks-cpu | placeholder；eager 协议测试 |
| `v1/profiles/poseidon-ckks-gpu.v1.json` | poseidon-ckks-gpu | placeholder；lazy 协议测试 |
| `v2/profiles/dacapo-heaan-cpu.v1.json` | dacapo-heaan-cpu | imported；复现旧编译模型 |
| `v2/profiles/dacapo-heaan-gpu.v1.json` | dacapo-heaan-gpu | imported；复现旧编译模型 |
| `v2/profiles/dacapo-seal-cpu.v1.json` | dacapo-seal-cpu | imported；复现旧编译模型 |

文件名中的 `v1` 是 profile 自身的版本，不是 `spec_format_version`。来源 revision 与摘要以 profile 中的 provenance 为准。HEAAN CPU/GPU 导入数据都是 eager 编译语义，不能因文件名有 GPU 就将它当成 Poseidon 的 30-bit lazy profile。

真实后端配置必须记录真实模数链、支持范围和测量来源；占位或导入的数值不能冒充实测。发布修改后的 spec 时更新 version，并对最终原始字节重新计算摘要，更新引用它的计划。格式化同样会改变摘要。
