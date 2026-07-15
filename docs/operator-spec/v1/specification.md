# OperatorSpec V1 规范

> **状态：V1 已冻结。** OperatorSpec 描述一个后端真实支持的 CKKS 参数范围、算子和 Boot profile。RuntimePlan 通过 id、version 和文件原始字节 SHA-256 引用它。

OperatorSpec 没有自嵌摘要。任何字节变化都会改变 RuntimePlan 中的 `source_sha256`；发布后修改内容必须升 version。

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

`status` 是 `placeholder` 或 `validated`。仓库内阶段一 profile 保持 `placeholder`，只用于协议和 Vec/Mock/MPI 测试，不冒充生产测量数据。

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
