# 明文数据包存储 V2：单文件 pack

V2 将去重后的 float64 原始载荷依次追加到一个 `data.bin`。计划中 Encode 的 content ID、slot 值和 V3 Fence 语义不变。这里的 `bundle_format_version` 是存储版本，与 RuntimePlan V1/V2/V3 和 bundle 身份中的 `version` 分开。

```text
<bundle>/
  manifest.json
  data.bin
```

[schema](plaintext-bundle.schema.json) 定义如下字段：

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

每个 content 只能出现一次。offset 和长度以字节计，均为 8 的倍数，且在 JSON 安全整数范围内。条目可以按任意顺序出现，但其范围必须无重叠、无空洞地覆盖整个 pack；文件实际长度必须等于 `pack_byte_length`。载荷仍为小端、有限 float64 数组，读取所需 blob 时检查 content SHA-256 和 CKKS slot 容量。manifest 原始字节摘要继续绑定在计划引用中。

## 导出和兼容

DaCapo 默认生成 pack。顶层参数 `--runtime-plan-bundle-format=files` 或 pass 参数 `bundle-format=files` 可生成旧 V1 逐 blob 文件。两种输出均采用 staging 发布，内容去重和计划调度相同。已有 prefix 若对应另一种存储格式，会明确报错，应使用新 prefix。

新 reader 显式支持 V1 和 V2。旧 reader 不支持存储 V2。V1 允许各 rank 只部署所需的 blob 文件；V2 要求部署完整 pack，每个 rank 仍只解码自己实际使用的内容。

已有编译器生成的 V1 产物无需重新编译即可合并：

```bash
python3 tools/pack_plaintext_bundle.py OLD.bundle NEW.bundle \
  --plan OLD.runtime-plan.json --output-plan NEW.runtime-plan.json --progress
```

工具按 manifest 顺序复制原始载荷并登记 offset，检查长度，不重新计算 blob 哈希。它只计算新旧 manifest 摘要；可选计划复制保留所有字节，仅替换 `manifest_sha256`。此替换要求编译器输出中有唯一、未转义的原摘要；其他表示直接报错。计划原始字节摘要因此改变，外部摘要记录需更新。工具拒绝已存在的输出，失败清理自己创建的 staging；输入保持原样。不支持多个进程同时写入相同输出路径。

## 按 offset 读取与内存驻留

默认 `BundleReadOptions{}` 保持一个 pack 文件句柄，按 offset 读取所需范围。loader 的副本共享句柄，文件 seek/read 有锁；不会为每个 blob 再次 open/stat。

显式设置原始载荷内存预算，即可在 open 时顺序预载整个 pack：

```cpp
fhegpu::RuntimeResources resources{
    loaded_spec, bundle_dir, false,
    fhegpu::BundleReadOptions{16ULL * 1024 * 1024 * 1024}};
```

预算为零表示文件读取模式；正值是允许驻留的原始字节上限。超过预算、分配失败或对 V1 请求驻留均报错，不自动切换模式。预载只读一次 raw pack，不进行全量 blob 哈希遍历。随后读取直接访问不可变内存范围，再按需校验和转换为 slot 数组，不复制临时 raw 字节。

V3 在运行期间持有 raw pack；slot 缓冲和 RNS 明文仍遵守原有按需 Encode/Fence 生命周期。`RuntimeTiming.bundle_resident_bytes` 和 `bundle_resident_load_nanoseconds` 记录 V3 预载量和时间。V1/V2 计划原有 eager load 会保留已解码的本地 slot 数组；raw 预载缓冲在 eager load 完成后释放。

驻留预算不包含 manifest 索引、执行计划、解码 slot、RNS 或 GPU 对象。14.70 GB 的 pack 需要约 13.69 GiB raw 内存。各独立 rank 进程分别预载并占用内存；同一 loader 的副本共享 raw 缓冲。不要将此选项当作全量 RNS 预编码缓存。

只测一次 raw 预载、避免逐 blob 解码或哈希，可使用 Linux 实验工具：

```bash
cmake --build BUILD --target runtime_bundle_io_benchmark -j4
BUILD/runtime_bundle_io_benchmark NEW.bundle BUNDLE_ID BUNDLE_VERSION MANIFEST_SHA256 17179869184
```

报告将 manifest 加载与 raw pack 预载分开；缓存未控制时不能将结果标作冷磁盘速度。
