# 明文数据包(plaintext bundle)V1 格式

计划文件只引用数据、不含数据(见规范 3.1 节)。明文常量(权重、bias、mask)的数值装在这里定义的**明文数据包**里,由编译器在导出计划时一并生成,随计划分发。

包里存的是**编码前的浮点 slot 向量**,不是编码后的 CKKS 明文多项式。编码(浮点向量 → 明文多项式)由 Runtime 在装载时执行,理由:

- **体积**:编码后的明文在 level L 有 (L+1) 份模数下的 `poly_degree` 个系数,比 `poly_degree/2` 个 slot 的浮点向量大一个数量级;
- **依赖**:编译器只做数值搬运,不需要链接任何同态加密库;
- **复用**:同一个数据包既可以喂真实后端(装载时编码),也可以原样喂 VecApi 明文 difftest,两条路径吃同一份字节。

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

编码规则与 RuntimePlan 相同(UTF-8 无 BOM、未知字段拒绝、无浮点数字面量、64 位标识符用十进制字符串)。注意"无浮点数"只约束 JSON 文本本身——浮点数据全部在 `data/*.bin` 里以原始字节存在,不经过 JSON。

~~~json
{
  "bundle_format_version": 1,
  "bundle_id": "resnet20-weights",
  "version": 1,
  "fingerprint": "sha256:…",
  "context": "ctx-main",
  "entries": [
    {
      "value_id": "1",
      "name": "conv1.diag3.mask",
      "content": "sha256:…",
      "byte_length": 131072
    }
  ]
}
~~~

- `bundle_format_version`:本文档对应 `1`;
- `bundle_id` / `version`:计划 `plaintext_bundle` 字段引用的就是这两项。同一 `bundle_id` + `version` 的内容永远不变,改内容必须升 `version`;
- `fingerprint`:清单自身的指纹,按 RuntimePlan 规范第 8 节计算(删除顶层 `fingerprint` 字段后 JCS 序列化取 SHA-256)。数据文件不参与清单指纹——它们由 `content` 哈希各自锚定,清单指纹间接覆盖了全部数据;
- `context`:必须等于计划使用的 OperatorSpec 的 `context_id`;
- `entries`:每项对应计划中一个明文 external_input:
  - `value_id`:计划中的 ValueId(64 位标识符编码)。同一清单内不允许重复;
  - `name`:人类可读的逻辑名,只用于调试和日志,允许多个 entry 的 `content` 相同(去重后多对一);
  - `content`:数据文件字节的 SHA-256(`sha256:` 前缀 + 64 位小写十六进制),对应 `data/` 下同名文件;
  - `byte_length`:数据文件的字节数,必须是 8 的倍数(见下),加载时核对。

## 数据文件的字节格式

`data/*.bin` 是 **IEEE 754 float64 的小端序数组**,slot 数 = `byte_length / 8`。V1 只支持实数 slot 向量,复数编码不在范围内。

装载语义:Runtime 物化一个明文 external_input 时,把对应数据文件的浮点向量交给 Api,由 Api 按该值 ValueDesc 声明的 `level`、`scale_log2`、`ntt` 和 context 参数**编码**成 Host 上的明文(真实后端调用 CKKS encode;VecApi 直接把数组当 slots 用,不编码)。slot 数与目标 context 的兼容性(真实后端通常要求 ≤ `poly_degree / 2`)由 Api 在编码时检查。

## 与计划的一致性

计划引用了 bundle 时,Runtime 加载前必须检查:

1. 清单指纹与计划 `plaintext_bundle.fingerprint` 相符(BND-1);
2. 清单 `entries` 的 `value_id` 集合**恰好等于**计划中 `kind` 为 `plaintext` 的 external_inputs 集合——缺了没数据可加载,多了说明计划和包错配;
3. 每个数据文件存在、字节数等于 `byte_length`(且为 8 的倍数)、SHA-256 等于 `content`。

值的 CKKS 元信息(level、`scale_log2`、`ntt`、`components`)**只在计划的 ValueDesc 里声明**,清单不重复记账——清单管字节,计划管语义,编码参数以计划为准。
