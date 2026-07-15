# 明文数据包(plaintext bundle)V1 格式

> **状态：设计草案。** 当前 C++ Runtime 还没有 manifest/blob loader、bundle 目录参数或显式 Encode 执行。本文定义的是目标格式，现状见[实现状态](../../overview-design/implementation-status.md)。

Encode 指令有两种 payload:小数据可以直接内联在 RuntimePlan JSON 中;大数据只在指令里保存 `content` 哈希,实际浮点数组装在这里定义的**明文数据包**中,由编译器随计划一并生成和分发。

`content` 是编码前“小端 float64 数组原始字节”的 SHA-256,用来标识“是哪份浮点数据”;Encode 的 `output` ValueId 标识编码后的 CKKS 明文。同一个 `content` 可以被多个 Encode 引用,各自产生不同 ValueId 和不同 level/`scale_log2` 的明文。manifest 不保存 ValueId,也不决定数据编码成什么样。

例如,同一份原始数据可以编码两次:

~~~json
{
  "ordinal": 0,
  "kind": "encode",
  "payload": {"kind": "bundle", "content": "sha256:abcd…"},
  "output": "17"
}
~~~

~~~json
{
  "ordinal": 1,
  "kind": "encode",
  "payload": {"kind": "bundle", "content": "sha256:abcd…"},
  "output": "29"
}
~~~

`values` 中的 ValueId `17` 和 `29` 可以声明不同的 level、`scale_log2` 或 NTT 状态。Runtime 因此会读取同一份浮点字节,但按两组输出描述分别编码。这里共享的是原始数据,不是编码后的明文对象。

包里存的是**编码前的浮点 slot 向量**,不是编码后的 CKKS 明文多项式。编码(浮点向量 → 明文多项式)由 Runtime 执行 Encode 指令时完成,理由:

- **体积**:编码后的明文在 level L 有 (L+1) 份模数下的 `poly_degree` 个系数,比 `poly_degree/2` 个 slot 的浮点向量大一个数量级;
- **依赖**:编译器只做数值搬运,不需要链接任何同态加密库;
- **复用**:同一个数据包既可以喂真实后端(执行 Encode 时做 CKKS 编码),也可以原样喂 Vec 明文参考后端做 difftest,两条路径吃同一份字节。

## 目录结构

~~~
<bundle>/
  manifest.json
  data/<64位小写十六进制>.bin
~~~

- `manifest.json`:清单,格式见下;
- `data/`:数据文件,**按内容寻址**:文件名就是文件字节的 SHA-256(64 位小写十六进制,不带 `sha256:` 前缀),后缀 `.bin`。

内容寻址是去重机制:两个逻辑权重的浮点向量字节相同(全零 bias、重复 mask、低内存测试里复用的占位常量),就指向同一个数据文件。打包器写文件前先算哈希,已存在即跳过。

## manifest.json

基础编码规则与 RuntimePlan 相同:UTF-8 无 BOM、重复 key/未知字段/缺字段都拒绝。manifest 没有 Encode inline payload,因此**不允许任何浮点数字面量**。`bundle_format_version`、`version` 和 `byte_length` 都写成 JSON 整数;浮点数据只存在于 `data/*.bin` 的原始字节中。

~~~json
{
  "bundle_format_version": 1,
  "bundle_id": "resnet20-weights",
  "version": 1,
  "blobs": [
    {
      "content": "sha256:…",
      "byte_length": 131072
    }
  ]
}
~~~

- `bundle_format_version`:本文档对应 `1`;
- `bundle_id` / `version`:计划 `plaintext_bundle` 字段引用的就是这两项。同一 `bundle_id` + `version` 发布后,manifest 和 blob 集合都不能改变;包括只改 manifest 格式在内的任何字节变化都必须升 `version`;
- `blobs`:数据包包含的原始浮点数据清单,同一 `content` 在清单中必须恰好出现一次:
  - `content`:数据文件字节的 SHA-256(`sha256:` 前缀 + 64 位小写十六进制),对应 `data/` 下同名文件;
  - `byte_length`:数据文件应有的字节数,用于在解码前发现截断、错文件或多余数据。它必须是 `[8, 2^53-1]` 范围内的 JSON 整数,并且是 8 的倍数;slot 数就是 `byte_length / 8`。限制在 JSON 安全整数范围内,避免不同语言把大整数转成 double 后丢精度。

## 数据文件的字节格式

`data/*.bin` 是 **IEEE 754 float64 的小端序数组**,slot 数 = `byte_length / 8`。V1 只支持有限的实数 slot 向量,NaN、Infinity 和负零都拒绝,复数编码不在范围内。

把 inline payload 外化成 bundle 时,编译器从 MLIR 中的 f64 payload 取值,把负零规范成正零,再按数组顺序写成小端 8 字节并计算 `content`。如果以后提供“读取现有 RuntimePlan 再外化”的工具,它也必须先把 JSON number 解码成同样的 float64 字节。这样同一组 slot 的 inline 和 bundle 形态才有明确的等价关系。

装载语义:Runtime 在 preflight 中按每条 bundle Encode 的 `payload.content` 找到清单条目和数据文件,校验并解码成 float64 向量。真正执行 Encode 时,Runtime 再把这段向量交给 Api,由 Api 按输出 ValueDesc 声明的 `level`、`scale_log2`、`ntt` 和 context 参数编码成 Host 上的明文(真实后端调用 CKKS encode;Vec 明文参考后端直接把数组当 slots 用,不做密码学编码)。

manifest 不保存自己的摘要。RuntimePlan 的 `plaintext_bundle.manifest_sha256` 直接对 `manifest.json` 的完整原始字节计算,不解析、不重排字段。数据文件不参与 manifest 摘要,它们由各自的 `content` SHA-256 锚定。

数据包路径不写进 RuntimePlan。启动每个 rank 的 Runtime 时,部署方另外传入一个本 rank 可读的完整 bundle 目录;不同 rank 可以使用不同本地路径,但内容必须是计划引用的同一个逻辑数据包。缺目录、缺 manifest 或缺文件都直接报错,V1 不做下载或搜索路径兜底。

## 与计划的一致性

计划中存在 bundle 形态的 Encode 时,Runtime 加载前必须检查:

1. `bundle_format_version` 必须为 1;清单的 `bundle_id`、`version` 分别与计划 `plaintext_bundle.id`、`version` 相符;默认模式下,manifest 完整原始字节的 SHA-256 还必须等于计划的 `manifest_sha256`(BND-1);
2. 每个 Encode 引用的 `content` 都在 `blobs` 中恰好出现一次;清单可以包含本计划未使用的其他数据,方便同一个模型数据包供多份计划复用;
3. `blobs` 中每一项对应的数据文件都存在、字节数等于 `byte_length`、SHA-256 等于 `content`;即使当前计划没有引用某个 blob,它既然写进这份 manifest,就必须真实存在;
4. 每个文件都能完整解码为非空的小端 float64 数组,每个值都有限且不是负零;
5. 对引用该 `content` 的每个 Encode,slot 数都不超过其输出 context 的容量,输出 level 和 `scale_log2` 也满足 OperatorSpec。相同 `content` 被引用多次时,这些输出约束要逐条检查,不能只检查第一次。

external_inputs 与 manifest 没有对应关系：external_inputs 是本次运行由调用方传入的参数，manifest 是 Encode 使用的固定数据清单。动态传入的权重可以是 external input；随计划发布的固定权重走 Encode。值的 CKKS 元信息（level、`scale_log2`、`ntt`、`components`）只在 Encode 输出的 ValueDesc 里声明——清单管原始字节，计划管编码语义。

`skip_artifact_digest_checks=true` 时可以跳过第 1 项中的 `manifest_sha256` 比较,但不能跳过 id/version、blob `content` SHA-256、长度或 float64 合法性检查。这个开关只用于调试,不改变内容寻址语义。
