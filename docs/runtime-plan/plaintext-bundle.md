# 明文数据包规范

bundle 保存 Encode 前的小端 float64 slots，不保存 CKKS 明文对象。多个 Encode 可以共享同一个 content，但各自按输出 ValueDesc 的 context、level、scale 和 NTT 编码。`content` 是导出时对原始载荷计算的 SHA-256 标识。

`bundle_format_version` 是存储布局版本，与 RuntimePlan `format_version`、二进制容器版本和 bundle 身份中的 `version` 分开。当前默认是存储 V2 pack 加二进制 manifest；需要 JSON 或逐文件布局时显式选择格式。

## Manifest 与载荷

JSON manifest 使用 UTF-8、严格字段规则，计划通过 `manifest_sha256` 引用其完整原始字节。二进制 manifest 使用 `manifest_format="binary"` 的引用并选择 `manifest.bin`；它的记录和摘要规则见[二进制格式](binary.md)。两种 manifest 都必须与计划中的 bundle id/version 一致。

所有载荷必须为非空、有限的 IEEE 754 float64 小端数组，长度是 8 的倍数，解码后的 slot 数不超过 OperatorSpec 的 `poly_degree / 2`。传给 Api 前将负零归一化为正零。

JSON manifest 路径在读取载荷时检查 content SHA-256；二进制 manifest 路径使用 content 作为标识，不重算 payload 哈希。`skip_artifact_digest_checks` 只跳过 JSON manifest 原始字节摘要比较，不跳过其 blob content 校验，也不关闭结构、长度、有限值或容量检查。

## 存储 V1：逐 blob 文件

```text
<bundle>/
  manifest.json
  data/<64位小写十六进制>.bin
```

[JSON Schema](v1/plaintext-bundle.schema.json)：

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

文件名是去掉 `sha256:` 前缀的 content 加 `.bin`。同一 content 只能出现一次；`byte_length` 在 `[8, 2^53-1]` 内且是 8 的倍数。每个 rank 都读取 manifest，但只要求本 rank 使用的 blob 在本地存在。

选择该布局需要同时设置 JSON 计划和 files bundle：

```bash
hecate-opt ... --runtime-plan-format=json --runtime-plan-bundle-format=files
```

export pass 对应 `plan-format=json bundle-format=files`。默认 binary 导出要求 pack，不能只切换 bundle 为 files。

## 存储 V2：单文件 pack

```text
<bundle>/
  manifest.bin 或 manifest.json
  data.bin
```

[JSON Schema](v2/plaintext-bundle.schema.json)：

```json
{
  "bundle_format_version": 2,
  "bundle_id": "model-weights",
  "version": 1,
  "pack_byte_length": 32,
  "blobs": [
    {"content": "sha256:...", "offset": 0, "byte_length": 32}
  ]
}
```

每个 content 只能出现一次。offset 和 byte_length 是字节数，均为 8 的倍数；JSON 数值在安全整数范围内。条目可以按任意顺序出现，但必须无重叠、无空洞地覆盖整个 pack，实际文件长度必须等于 `pack_byte_length`。每个 rank 必须部署完整 pack，仍只解码本 rank 实际使用的内容。

DaCapo 用 staging 文件和目录发布产物，内容去重与计划调度不因存储布局改变。已有 bundle 只有 manifest 和 payload 匹配时才复用；不同表示使用独立输出 prefix。旧 reader 不支持 V2 pack。

已有 JSON/V1 文件布局可以合并为 JSON/V2 pack：

```bash
python3 tools/pack_plaintext_bundle.py OLD.bundle NEW.bundle \
  --plan OLD.runtime-plan.json --output-plan NEW.runtime-plan.json --progress
```

工具复制原始载荷、检查长度并登记 offset，不重新计算 blob 哈希。它计算新旧 manifest 摘要；可选计划复制保留原始字节，仅替换唯一且未转义的 `manifest_sha256`。其他表示直接报错，且不支持二进制计划/manifest。新计划的整文件摘要随之变化。工具拒绝已有输出，失败清理自己创建的 staging；不要让多个进程同时写同一输出路径。

## 计划版本与装载时机

V1/V2 计划在启动时读取并保留本 rank 使用的解码 slots；相同 content 只装载一次。V3 启动时检查 manifest 和 Encode 引用，在 Encode 执行时按需读载荷，不保留跨 Encode 的解码 slots 缓存。重复编码可能重复读取，生命周期由 Release/Fence 控制。

默认 `BundleReadOptions{}` 保持一个 pack 文件句柄，按 offset 读取所需范围；loader 副本共享句柄，seek/read 加锁，不为每个 blob 重新 open/stat。

## 显式 raw pack 驻留

设置 `BundleReadOptions.resident_byte_limit` 可在 open 时预载完整 raw pack：

```cpp
fhegpu::RuntimeResources resources{
    loaded_spec, bundle_dir, false,
    fhegpu::BundleReadOptions{16ULL * 1024 * 1024 * 1024}};
```

预算 0 表示按范围读取文件；正值是 raw 字节驻留上限。超预算、分配失败或对 V1 逐文件布局请求驻留均报错，不自动换模式。预载只读取 raw pack，不进行全量 blob 哈希遍历；随后按需从不可变范围转换为 slots，并按所选 manifest 路径检查载荷。

V3 在运行期间保留 raw pack，slots 和 RNS 明文仍按 Encode/Release/Fence 生命周期管理。`RuntimeTiming.bundle_resident_bytes` 和 `bundle_resident_load_nanoseconds` 记录驻留量和加载时间。V1/V2 eager load 后保留解码 slots，raw 预载缓冲随 loader 销毁而释放。

驻留预算不包括 manifest 索引、计划、slots、RNS 对象或 GPU 显存。各独立 rank 进程分别占用内存；同一 loader 的副本共享 raw 缓冲。它不是全量 RNS 预编码缓存，也不代替编译器的 streaming 预算。
