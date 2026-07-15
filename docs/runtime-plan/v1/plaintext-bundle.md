# 明文数据包 V1

> **状态：V1 已冻结。** bundle 保存 Encode 前的小端 float64 slot 数组，不保存 CKKS 明文对象。

目录结构：

```text
<bundle>/
  manifest.json
  data/<64位小写十六进制>.bin
```

manifest 格式由 [plaintext-bundle.schema.json](plaintext-bundle.schema.json) 定义：

```json
{
  "bundle_format_version": 1,
  "bundle_id": "model-weights",
  "version": 1,
  "blobs": [
    {"content": "sha256:...", "byte_length": 32}
  ]
}
```

- `content` 是 blob 完整字节的 SHA-256。
- 文件名是去掉 `sha256:` 前缀后的 64 位摘要，加 `.bin`。
- `byte_length` 在 `[8, 2^53-1]` 内且是 8 的倍数。
- 同一 content 在 manifest 中只能出现一次。
- blob 是 IEEE 754 float64 小端序数组，只允许有限实数。正零和负零在交给 Api 前都转成正零。

同一个 content 可以被多个 Encode 引用。每条 Encode 仍按自己的输出 ValueDesc 决定 context、level、scale 和 NTT，因此共享的是原始 slot 数据，不是编码后的明文对象。

## 分 rank 装载

每个 rank 都读取 manifest，检查 format、bundle id/version；默认模式还核对 RuntimePlan 中的 `manifest_sha256`。

随后每个 rank 只装载本 rank 实际执行的 bundle Encode 所引用的 blob：

- 未被本 rank 使用的 blob 不读取，也不要求本地存在。
- 被使用的 blob 必须存在，长度和 content SHA-256 必须匹配。
- 解码后必须是非空有限 float64 数组，slot 数不能超过 OperatorSpec 的容量。
- 相同 content 被多条本地 Encode 引用时，文件只需装载一次，但每条 Encode 的输出约束都要单独验证。

这允许不同 rank 使用不同本地目录，并只部署本 rank 需要的数据文件；各目录里的 manifest 仍必须代表同一个 bundle id/version。

`skip_artifact_digest_checks=true` 只跳过 manifest 原始字节摘要比较，不跳过 id/version、blob 文件、长度、content SHA-256、float64 或 slot 容量检查。
