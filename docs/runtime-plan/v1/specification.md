# RuntimePlan V1 规范

> **状态：V1 已冻结。** 本文、`schema.json`、`plaintext-bundle.schema.json` 和 `testdata/` 共同定义 V1。Runtime 不读取旧草案，也不根据缺失字段猜版本。

RuntimePlan 是编译器交给 Runtime 的最终执行文件。它写清楚 SSA 值、CKKS 元信息、执行位置、Encode、计算和通信。Runtime 只验证并执行，不补指令、不改 level/scale、不隐式搬运数据。

## 1. JSON 基本规则

- UTF-8，无 BOM；顶层必须是 object。
- 未知字段、缺字段、同一 object 内的重复 key 都直接拒绝。
- `plan_id`、ValueId、TransferId 使用十进制字符串，必须匹配 `^(0|[1-9][0-9]*)$`，范围是 uint64。
- 其他整数使用 JSON integer，不接受 `40.0`、科学计数法或字符串。
- 只有 inline Encode 的 `payload.values` 可以出现浮点数。每项必须能转换成有限 float64。正零和负零在交给 Api 前都转成正零。
- 所有枚举使用规范列出的固定小写字符串。

## 2. 顶层结构

```json
{
  "format_version": 1,
  "plan_id": "42",
  "target": {},
  "plaintext_bundle": {},
  "values": [],
  "external_inputs": [],
  "initialization": [],
  "execution": [],
  "finalization": [],
  "final_outputs": []
}
```

`plaintext_bundle` 是可选字段，只在存在 bundle Encode 时出现。V1 没有计划自摘要、`required_keys`、`required_capabilities`、`rescale_mode` 或 `boot_mode` 字段。Runtime 从 OperatorSpec 和实际指令推导这些要求，避免重复声明互相打架。

## 3. target 和外部产物摘要

```json
{
  "target_id": "poseidon-ckks-gpu",
  "capability_version": 1,
  "operator_spec": {
    "id": "poseidon-ckks-gpu-v1",
    "version": 1,
    "source_sha256": "sha256:..."
  },
  "world_size": 2,
  "device_counts": [1, 1]
}
```

- `device_counts` 长度必须等于 `world_size`。
- `source_sha256` 是 OperatorSpec 文件完整原始字节的 SHA-256，不做 JSON 重排或 JCS。
- 计划文件本身不保存自摘要。reader 返回 `plan_source_sha256`，多 rank preflight 默认要求它完全一致。

bundle 引用格式：

```json
{
  "id": "model-weights",
  "version": 1,
  "manifest_sha256": "sha256:..."
}
```

`manifest_sha256` 同样覆盖 `manifest.json` 的完整原始字节。

## 4. ValueDesc 和 Place

Host Place 是 `{"kind":"host","rank":0}`；Device Place 是 `{"kind":"device","rank":0,"index":0}`。Host 不允许 `index`。

ValueDesc：

```json
{
  "id": "7",
  "kind": "ciphertext",
  "place": {"kind": "device", "rank": 0, "index": 0},
  "context": "ctx-main",
  "level": 5,
  "scale_log2": 40,
  "ntt": true,
  "components": 2
}
```

- plaintext 的 `components` 必须为 1；ciphertext 至少为 2。
- `context`、level 范围和 scale 的模数预算必须符合 OperatorSpec。
- `values` 必须恰好覆盖所有实际出现的 ValueId，未使用的描述也拒绝。
- external input 必须在 Host，由调用方提供完整 CKKS 值。进入 Device 必须经过显式通信。

## 5. 指令、阶段和 SSA

三个阶段按 initialization、execution、finalization 拼接。`ordinal` 从 0 开始全局连续。每个 ValueId 只能由 external input 或一条指令定义一次；任何输入必须先定义后使用。TransferId 全局唯一。

### 5.1 Encode

Encode 只能在 initialization，输出必须是 Host plaintext。

```json
{"ordinal":0,"kind":"encode","payload":{"kind":"inline","values":[1,0,-1.5]},"output":"2"}
```

```json
{"ordinal":0,"kind":"encode","payload":{"kind":"bundle","content":"sha256:..."},"output":"2"}
```

inline 数组非空，长度不能超过 `poly_degree / 2`。bundle content 必须出现在 manifest 中。同一个 content 可以被多个 Encode 以不同 level、scale 或 NTT 状态编码。

### 5.2 计算

计算格式：

```json
{
  "ordinal": 3,
  "kind": "compute",
  "op": "rotate",
  "place": {"kind":"device","rank":0,"index":0},
  "inputs": ["7"],
  "output": "8",
  "attrs": {"steps": -1}
}
```

合法 op：`add_cc`、`add_cp`、`sub_cc`、`sub_cp`、`mul_cc`、`mul_cp`、`negate`、`rotate`、`rescale`、`mod_switch`、`relinearize`、`boot`。

普通计算既可在 Host 也可在 Device，但输入、输出和指令 Place 必须完全相同，Api 也必须支持该位置。Runtime 不自动搬运。

元信息规则：

| op | 规则 |
| --- | --- |
| add/sub cc | 两个 ct 的 level、scale、components 相同；输出与输入相同 |
| add/sub cp | pt 与 ct 的 level、scale 相同；输出与 ct 相同 |
| mul_cc | level 不变；scale 相加；components = c1 + c2 - 1 |
| mul_cp | level 不变；scale 相加；components 与 ct 相同 |
| negate | 元信息不变 |
| rotate | 元信息不变，输入 components=2 |
| rescale | target level 更小；输出 level/scale 等于 attrs；components 不变 |
| mod_switch | target level 更小；只改变 level |
| relinearize | components 从 3 变 2，其余不变 |
| boot | 输出 level、scale、components 等于 attrs，并匹配 OperatorSpec profile |

所有计算都保持 context 和 NTT 状态。需要 attrs 的 op 缺 attrs，或不需要 attrs 的 op 出现 attrs，都拒绝。

Rotate 的 `steps` 先按 `poly_degree / 2` 个 slot 取模，负数转成等价正 step。规范化结果为 0 时拒绝。Runtime 用规范化 step 请求具体 Galois key，不接受只有“某种 Galois key”的模糊声明。

Boot 的 `implementation` 是 `native` 或 `decrypt_reencrypt`。后者只能在 Host 执行，需要 secret key 和 Host compute 能力。原生 Boot 失败后不允许自动改走解密再加密。

### 5.3 通信

通信是 `transfer`（1→1）或 `replicate`（1→N，N≥2）。`inputs/sources` 长度为 1；`outputs/destinations/output_kinds` 等长。目的 Place 不能重复，也不能等于源 Place。

hint 合法值为 `auto`、`point_to_point`、`broadcast`、`tree`、`ring`、`host_staged`。hint 只是实现建议，不限制它与 Transfer/Replicate 的组合。Api 可以使用等价实现，但必须完成计划规定的逻辑通信。

通信只改变 id 和 Place，不能改变 kind、context、level、scale、NTT 或 components。

## 6. OperatorSpec 和启动要求

Runtime 必须检查：

- spec id/version/target/context 与计划一致；默认模式下原始字节摘要一致。
- `poly_degree` 为 2 的幂，level 位于模数链内，每个模数不超过上限，延迟表长度正确。
- 计划使用的算子受支持；Rescale 降幅不超过 `max_levels_per_op`。
- Boot profile 唯一，输入范围、implementation 和输出元信息一致。

Runtime 从实际指令推导能力：Encode、Transfer、Replicate、Host compute、native Boot、decrypt/re-encrypt Boot；并推导 relin key、带 step 的 Galois key和 secret key。Api preflight 必须明确确认这些资源可用。

## 7. bundle 的分 rank 装载

每个 rank 都读取同一逻辑 bundle 的 manifest，并检查 format、id/version；默认模式还检查 `manifest_sha256`。

每个 rank 只读取并校验本 rank 实际执行的 bundle Encode 所引用的 blob。manifest 中未被本 rank 使用的 blob 不读取，也不要求本地存在。被使用的 blob 必须检查文件存在、长度、content SHA-256、小端 float64、有限值和 slot 容量。详细格式见 [plaintext-bundle.md](plaintext-bundle.md)。

## 8. Runtime 启动顺序和调试开关

启动顺序固定为：

1. 若启用摘要调试开关，先打印警告；随后验证计划和 OperatorSpec，推导能力与密钥。
2. 读取 manifest，装载本 rank 需要的 blob。
3. Api preflight。
4. 绑定并完整验证 external inputs。
5. 顺序执行 Encode、计算和通信。
6. 等待通信并同步最终输出。

`skip_artifact_digest_checks=true` 只跳过三类比较：rank 间计划原始字节摘要、OperatorSpec 原始字节摘要、manifest 原始字节摘要。它不跳过结构、id/version、SSA、元信息、能力、密钥、blob content/长度或 Api 值校验。所有 rank 的开关必须一致，每个 rank 都要打印明确警告。

Api 的 `validate_value` 必须检查 kind、context、poly degree、level、scale、NTT 和 components。任何一项不符都立即终止。

## 9. 兼容性

V1 不提供旧字段别名、默认值补全、自动降级或旧格式 reader。未知版本、字段、算子和能力都直接报错。版本规则见 [compatibility.md](compatibility.md)。
