# RuntimePlan V1 规范

> **状态:设计草案。** 本文已经采用显式 Encode 指令、inline/bundle 双 payload 和原始文件字节摘要;当前 `schema.json`、样例集和 C++ 读取/执行代码仍是上一版实现,尚未同步。完成 Schema、reader、verifier、Runtime、Api 和端到端测试之后,V1 才会正式冻结。

本文档是 RuntimePlan V1 的目标定义。草案迁移期间,对新格式的理解以本文为准;正式冻结时,本文、Schema 和 `testdata/` 必须完全一致并共同构成协议依据。Dacapo 按规范生成计划文件,Runtime 按规范读取和验证,两边不共享任何代码。

设计背景和职责划分见 [集成方案](../../overview-design/dacapo-runtime-integration.md);本文只写"文件里有什么、每个字段什么意思、什么样的文件必须被拒绝"。

规范用词:**必须**(违反即拒绝整份计划)、**不允许**(出现即拒绝)、**可选**(字段可缺席,缺席有明确定义的含义)。除非明确标注"可选",所有字段都是必填的——**缺字段就拒绝,不做默认值补全**。

## 1. 文件整体

- 编码:UTF-8,不允许 BOM;
- 顶层是一个 JSON object;
- **不允许出现本规范未定义的字段**。读取方遇到不认识的字段必须拒绝,而不是跳过——跳过会让拼写错误和新旧版本混用悄悄溜过去;
- 数组顺序全部有语义(指令顺序就是发起顺序,列表的第 i 项之间互相对应),序列化时不允许重排;
- 同一个 object 内不允许重复的 key。

### 1.1 整数编码

协议里的整数分两类,编码方式不同:

| 类别 | 字段 | JSON 编码 | 理由 |
| --- | --- | --- | --- |
| 64 位标识符 | `plan_id`、ValueId、TransferId | **十进制字符串**,如 `"42"` | 很多 JSON 工具把大整数当 double 处理,超过 2^53 会悄悄丢精度 |
| 小整数 | `format_version`、`level`、`scale_log2`、`components`、`rank`、`index`、`world_size`、`device_counts[i]`、`ordinal`、`steps` 等 | JSON 数字 | 取值远小于 2^31,不存在精度问题 |

标识符字符串必须匹配正则 `^(0|[1-9][0-9]*)$`:纯十进制、不允许前导零、不允许正负号,取值范围 [0, 2^64)。

小整数**必须是 JSON 整数**。`40.0`、`4e1`、`"40"` 都不允许。`level`、`scale_log2`、`components` 还必须非负。

### 1.2 浮点数只用于 Encode 的内联数据

除 `encode.payload.values` 外,协议不允许 JSON 浮点数。CKKS 的 scale 仍用整数指数 `scale_log2` 表达(`scale_log2 = 40` 表示逻辑 scale 为 2^40),level、scale 和标识符都不能写成浮点数。

`encode.payload.values` 是编码前的 slot 数组,每项按 IEEE 754 float64 解释。整数形式的 JSON number(如 `0`、`1`)也允许出现在该数组中并转换为 float64;NaN、Infinity、超出 float64 范围的数、解析后非有限的值和负零一律拒绝。生成端如果遇到 `-0.0`,必须先规范成 `0.0` 再写计划。

### 1.3 枚举

所有枚举用固定的小写下划线字符串,不用数字。各枚举的合法取值在对应章节列出,出现清单之外的字符串即拒绝。

## 2. 顶层结构

~~~json
{
  "format_version": 1,
  "plan_id": "42",
  "target": { … },
  "plaintext_bundle": { … },
  "values": [ … ],
  "external_inputs": [ … ],
  "required_keys": [ … ],
  "initialization": [ … ],
  "execution": [ … ],
  "finalization": [ … ],
  "final_outputs": [ … ]
}
~~~

- `format_version`:本规范对应 `1`。读取方遇到其他值必须直接拒绝,不猜测语义,不尝试按旧格式解析;
- `plan_id`:这份计划的标识符(64 位标识符编码);
- `target`:目标环境描述,见第 3 节;
- `plaintext_bundle`:**可选**,明文数据包(权重包)的引用,见 3.1 节。缺席表示计划不依赖任何预置数据包;
- `values`:计划中出现的**所有**值的描述,见第 4 节;
- `external_inputs`：调用方在本次运行开始时提供的参数 ValueId 列表，可以是 plaintext 或 ciphertext。它与明文数据包没有对应关系。随计划固定发布的权重由 Encode 产生；每次调用动态传入的权重也可以列为 external input。**每个外部输入的 place 必须是 Host**——调用方只向 Host 绑定已经物化好的 CKKS 值，数据进设备的唯一途径是计划中的显式通信指令（通常放在 `initialization` 阶段）；
- `required_keys`:计划需要的密钥种类和位置声明,见 4.3 节;
- `initialization` / `execution` / `finalization`:三个阶段的指令列表,见第 5、6 节;
- `final_outputs`:最终输出的 ValueId 列表,每个都必须在 `values` 中有描述。

## 3. target:目标环境

~~~json
{
  "target_id": "poseidon-ckks-gpu",
  "capability_version": 1,
  "operator_spec": {
    "id": "poseidon-ckks-gpu-v1",
    "version": 1,
    "source_sha256": "sha256:…"
  },
  "rescale_mode": "lazy",
  "boot_mode": "decrypt_reencrypt",
  "world_size": 2,
  "device_counts": [2, 2],
  "required_capabilities": ["encode", "transfer", "host_compute", "boot_decrypt_reencrypt"]
}
~~~

- `target_id`:目标后端家族的固定名字;
- `capability_version`:Runtime Api 能力集合的版本,整数;
- `operator_spec`:编译这份计划时使用的 [OperatorSpec](../../operator-spec/v1/specification.md) 的 id、版本和原始文件摘要。`source_sha256` 对 OperatorSpec 文件的**完整原始字节**直接计算 SHA-256,不解析、不重排字段、不做 JCS。Runtime 必须核对实际读取文件的摘要,不符即拒绝;
- `rescale_mode`:`"eager"` 或 `"lazy"`。**这是冗余摘要,权威在 OperatorSpec**——两处不一致即拒绝;
- `boot_mode`:`"native"` 或 `"decrypt_reencrypt"`。同样是冗余摘要,权威在每条 Boot 指令的 `implementation` 属性:计划中出现任何一条 `implementation` 与 `boot_mode` 不符的 Boot 指令即拒绝。计划中没有 Boot 指令时,`boot_mode` 必须写 `"native"`;
- `world_size`:进程(rank)数,≥ 1;
- `device_counts`:长度必须等于 `world_size`,第 i 项是 rank i 的设备数,≥ 0;
- `required_capabilities`:计划实际用到的能力集合,合法取值:`"encode"`、`"transfer"`、`"replicate"`、`"host_compute"`、`"boot_native"`、`"boot_decrypt_reencrypt"`。列表必须与计划内容一致:用到了没声明、声明了没用到,都拒绝(这让 Runtime 只看头部就能做能力预检,且头部撒谎会被抓住)。存在 Encode 指令就必须声明 `encode`;`host_compute` 只指一般计算指令放在 Host,不包含 Encode。

### 3.1 plaintext_bundle:编码前明文数据包引用(可选)

~~~json
{
  "id": "resnet20-weights",
  "version": 1,
  "manifest_sha256": "sha256:…"
}
~~~

神经网络推理的权重、bias 和 rotate-and-sum mask 由 Encode 指令产生。小数据可以直接放在 Encode 的 inline payload 中;大数据以编码前的 float64 slot 数组放进随计划分发的**明文数据包**(bundle),Encode 的 bundle payload 只保存 `content` 哈希。浮点 → CKKS 明文的编码由 Runtime 执行 Encode 时完成。

- `id` / `version`:数据包的标识和版本,同一 `id` + `version` 发布后,manifest 和 blob 集合的字节都不再改变;
- `manifest_sha256`:对数据包 `manifest.json` 的**完整原始字节**直接计算的 SHA-256,见第 8 节。

数据包的文件格式(manifest、数据文件、内容寻址去重)见 [plaintext-bundle.md](plaintext-bundle.md)。计划只要出现一个 `payload.kind = "bundle"` 的 Encode,顶层 `plaintext_bundle` 就必须存在;没有 bundle Encode 时该字段必须缺席。Runtime 必须在执行前取得指定 id/version 的数据包并完成 BND-1 检查。inline Encode、bundle Encode 和调用方提供的 external_inputs 可以同时存在,三者互不冒充。

计划只保存数据包的身份和 manifest 原始字节摘要,**不保存本机文件路径**。启动每个 Runtime 实例时,部署方必须另外传入一个该 rank 可读的完整 bundle 目录;各 rank 的目录路径可以不同,但 manifest 的 id、version 和 `manifest_sha256` 必须与计划一致。计划含 bundle Encode 却没有传目录,或目录不完整,都在执行任何指令前直接报错。V1 不支持“缺文件时再向别处下载”之类的兜底行为。

## 4. 值、位置和密钥

### 4.1 Place(位置)

~~~json
{"kind": "host", "rank": 0}
{"kind": "device", "rank": 0, "index": 1}
~~~

- `kind`:`"host"` 或 `"device"`;
- `rank`:0 ≤ rank < `world_size`;
- `index`:仅 `device` 有,0 ≤ index < `device_counts[rank]`。**Host 不允许出现 `index` 字段**(不是"忽略",是出现即拒绝)。

### 4.2 值描述(values 数组元素)

~~~json
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
~~~

- `id`:全局唯一。重复定义即拒绝;
- `kind`:`"ciphertext"` 或 `"plaintext"`;
- `place`:这个值存在的**唯一**位置。同一份数据要出现在第二个位置,必须由通信指令产生新的 ValueId;
- `context`:CKKS 参数环境的标识字符串,必须等于 OperatorSpec 中声明的 `context_id`(V1 一份计划只允许一个 context);
- `level`:模数链层号,非负整数。**方向定义:较大的值表示还保留更多 RNS 模数**;Rescale/ModSwitch 后 level 变小。注意这与 Dacapo Earth 分析阶段"随 rescale 递增"的 level 方向相反,导出器必须写换算后的最终 CKKS 层号;
- `scale_log2`:scale 的二进制指数,非负整数;
- `ntt`:是否处于 NTT 形式;
- `components`:多项式分量数。`plaintext` 必须为 1;`ciphertext` 通常为 2,乘法之后、重线性化之前为 3。

`values` 必须**恰好**覆盖计划中出现的每一个 ValueId(external_inputs、每条指令的输入输出、final_outputs):缺了拒绝,多出没人用的描述也拒绝。

### 4.3 required_keys(密钥声明)

~~~json
{"kind": "galois", "place": {"kind": "device", "rank": 0, "index": 0}}
~~~

`kind` 合法取值:`"secret"`、`"relin"`、`"galois"`。**密钥本身不进计划文件**——这里只声明"哪个位置需要哪类密钥",让 Runtime 在执行前对照 Api 的实际配置把关。列表必须与计划内容一致:

- 存在 Rotate 指令的 place → 该 place 必须声明 `galois`;
- 存在 Relinearize 指令的 place → 该 place 必须声明 `relin`;
- 存在 `implementation = "decrypt_reencrypt"` 的 Boot 指令的 place → 该 place 必须声明 `secret`;
- 声明了但没有指令需要 → 拒绝。

## 5. Encode 与计算指令

三个阶段(`initialization`、`execution`、`finalization`)的数组元素统一为"指令"。指令用 `kind` 区分:`"encode"`、`"compute"`、`"transfer"`、`"replicate"`。

`ordinal` 是指令的稳定序号:在**整份计划**(三个阶段按 initialization → execution → finalization 拼接)中从 0 开始、严格递增、连续。它只用于报错定位和交叉引用,执行顺序由数组顺序决定(两者必须一致)。

### 5.1 Encode 指令

Encode 把编码前的 float64 slot 数组物化成 Host 上的 CKKS plaintext。它没有普通 SSA 输入,只定义一个输出 ValueId,并且只能出现在 `initialization`。

内联 payload:

~~~json
{
  "ordinal": 0,
  "kind": "encode",
  "payload": {
    "kind": "inline",
    "values": [1, 0, -1.5, 2.25]
  },
  "output": "17"
}
~~~

bundle payload:

~~~json
{
  "ordinal": 1,
  "kind": "encode",
  "payload": {
    "kind": "bundle",
    "content": "sha256:abcd…"
  },
  "output": "29"
}
~~~

共同规则:

- `output` 的 ValueDesc 必须是 Host 上的 plaintext,`components` 必须为 1;Encode 的执行 rank、context、level、`scale_log2` 和 NTT 全部从该 ValueDesc 读取,指令中不重复保存;
- `payload.kind` 只能是 `"inline"` 或 `"bundle"`,两种字段严格二选一:inline 只有 `values`,bundle 只有 `content`;
- inline `values` 必须非空,每项按 1.2 节解码成有限且不是负零的 float64,长度不能超过目标 context 的 slot 容量;
- bundle `content` 必须是 `sha256:` 加 64 位小写十六进制,并且能在 `plaintext_bundle` 指向的 manifest 中找到;
- Encode 使用 bundle payload 时顶层 `plaintext_bundle` 必须存在;inline Encode 不依赖 bundle;
- 无论 payload 是 inline 还是 bundle,输出 ValueDesc 的 level 和 `scale_log2` 都必须满足 OperatorSpec 对所有值的范围和模数预算约束;
- 相同 `content` 只表示原始浮点字节相同。不同 Encode 可以引用同一个 `content`,再按各自输出 ValueDesc 编码成不同 ValueId、level 或 `scale_log2`;
- Encode 输出是普通 SSA 定义,不能同时出现在 `external_inputs`,也不能由其他指令再次定义;
- 所有 rank 都按同一顺序看到 Encode,并且在执行前完成同样的 payload 和 bundle preflight。真正执行这条指令时,只有 `output` 的 Host Place 所属 rank 调用 Api 并定义本地值;其他 rank 明确跳过,不能把“不是本 rank 的输出”当成错误。

### 5.2 计算指令

计算指令格式:

~~~json
{
  "ordinal": 3,
  "kind": "compute",
  "op": "rotate",
  "place": {"kind": "device", "rank": 0, "index": 0},
  "inputs": ["7"],
  "output": "8",
  "attrs": {"steps": 3}
}
~~~

- `op` 合法取值:`add_cc`、`add_cp`、`sub_cc`、`sub_cp`、`mul_cc`、`mul_cp`、`negate`、`rotate`、`rescale`、`mod_switch`、`relinearize`、`boot`;
- `place`:唯一执行位置。所有输入值的 place 必须等于它,输出值的 place 也必须等于它。Runtime 不根据输入位置推断,也不代为搬运;
- `inputs` / `output`:ValueId。输入可以重复(如 `x+x`);
- `attrs`:按 `op` 决定,见 5.3。**需要 attrs 的算子缺 attrs、不需要 attrs 的算子出现 attrs,都拒绝。**

这个算子集合是封闭的:可执行 CKKS 里的编译器内部算子(如 `upscale`)必须在导出前消除(Dacapo 已有 `UpscaleToMulcp` 之类的下沉),出现集合之外的算子名即拒绝。

### 5.3 各计算算子的参数和元信息规则

下表中"输入"指 `inputs` 数组各项的 ValueDesc,"输出"指 `output` 的 ValueDesc。所有算子都要求:输入输出 `context` 相同、`ntt` 相同(V1 不表达 NTT 形式转换);所有 ciphertext 输入 `level` 相同。对 `*_cp` 混合算子,plaintext 的 `level` 还必须等于 ciphertext 的 `level`,不能让 Runtime 临时 mod-switch。违反任意一条即拒绝。

| op | inputs | attrs | 元信息规则 |
| --- | --- | --- | --- |
| `add_cc` `sub_cc` | 2 个 ct | 无 | 两输入 `scale_log2`、`components` 相同;输出与输入完全同元信息 |
| `add_cp` `sub_cp` | ct, pt(顺序固定) | 无 | pt 与 ct 的 `level`、`scale_log2` 相同;输出与 ct 输入同元信息 |
| `mul_cc` | 2 个 ct | 无 | 输出 `scale_log2` = 两输入之和;`components` = c₁+c₂−1;`level` 不变 |
| `mul_cp` | ct, pt | 无 | pt 与 ct 的 `level` 相同;输出 `scale_log2` = 两输入之和;`components` = ct 的;`level` 不变 |
| `negate` | 1 个 ct | 无 | 输出与输入同元信息 |
| `rotate` | 1 个 ct | `{"steps": n}`,非零整数 | 输出与输入同元信息;输入 `components` 必须为 2;place 需 `galois` 密钥 |
| `rescale` | 1 个 ct | `{"target_level": l, "target_scale_log2": s}` | `target_level` < 输入 `level`;输出 `level` = `target_level`、`scale_log2` = `target_scale_log2`、`components` 不变。允许一次降多层(lazy 模式合并的 rescale);单次合法降幅由 OperatorSpec 约束 |
| `mod_switch` | 1 个 ct | `{"target_level": l}` | `target_level` < 输入 `level`;输出 `level` = `target_level`,`scale_log2`、`components` 不变 |
| `relinearize` | 1 个 ct | 无 | 输入 `components` = 3,输出 = 2;其余不变;place 需 `relin` 密钥 |
| `boot` | 1 个 ct | 见 5.4 | 输出 `level` = `target_level`、`scale_log2` = `target_scale_log2`、`components` = `target_components`;`context` 不变(V1 Boot 不换 context、不换多项式度数) |

**算子的目标参数与输出 ValueDesc 必须逐字段一致**(如 Boot 声明 `target_level: 6` 而输出值描述写 `level: 5`,拒绝)。这是故意的双重记账:两处由导出器独立写出,不一致说明导出器有 bug。

### 5.4 Boot 的 attrs

~~~json
{
  "target_level": 6,
  "target_scale_log2": 40,
  "target_components": 2,
  "operator_profile": "poseidon-cpu-boot-emulation-v1",
  "implementation": "decrypt_reencrypt"
}
~~~

- `implementation`:`"native"` 或 `"decrypt_reencrypt"`;
- `operator_profile`:必须是 OperatorSpec `boot_profiles` 中存在的 `profile_id`,且该 profile 的 `implementation`、输出元信息、合法输入 level 范围都必须与本指令相符;
- `implementation = "decrypt_reencrypt"` 的 Boot,`place` 必须是 Host。这是测试和联调用的模拟路径,会使用 secret key、让明文短暂出现在主机内存,**不是安全的生产 boot**。Runtime 不会在原生 Boot 失败后自动改走这条路——用不用它,由编译器写死在计划里。

## 6. 通信指令

~~~json
{
  "ordinal": 5,
  "kind": "transfer",
  "transfer_id": "10",
  "hint": "auto",
  "inputs": ["8"],
  "outputs": ["9"],
  "sources": [{"kind": "device", "rank": 0, "index": 0}],
  "destinations": [{"kind": "host", "rank": 1}],
  "output_kinds": ["ciphertext"]
}
~~~

- `kind`:`"transfer"`(1 源 → 1 目的)或 `"replicate"`(1 源 → N 目的,N ≥ 2);
- `transfer_id`:全局唯一(64 位标识符编码),与 ValueId 是两个独立的编号空间;
- `hint`:实现方式建议,合法取值 `"auto"`、`"point_to_point"`、`"broadcast"`、`"tree"`、`"ring"`、`"host_staged"`。**hint 只是建议**:Api 可以选任何等价实现,但通信完成后"哪些值出现在哪些位置"必须与计划一致;
- `inputs` 和 `sources` 长度必须为 1;`outputs`、`destinations`、`output_kinds` 三个列表长度相同(transfer 为 1,replicate ≥ 2),**第 i 项互相对应**;
- `sources[0]` 必须等于输入值的 place;`destinations[i]` 必须等于 `outputs[i]` 的 place;同一条指令的目的位置之间、目的位置与源位置之间都不允许重复(把值"搬"到原地没有意义,是导出器 bug);
- `output_kinds[i]` 必须等于输入值的 `kind`,也必须等于 `outputs[i]` 的 ValueDesc 的 `kind`。

**通信不改变数学值和元信息**:每个输出值的 `context`、`level`、`scale_log2`、`ntt`、`components` 必须与输入值完全相同,只有 `id` 和 `place` 不同。CPU/GPU 表示的转换是 Api 内部的事,不在协议中出现。

## 7. SSA 与生命周期不变量

Runtime 的执行前验证必须覆盖以下全部检查(编号供测试样例引用):

1. **[SSA-1] 每个 ValueId 恰好被定义一次**:要么在 `external_inputs` 中,要么是恰好一条指令的输出。既是外部输入又是指令输出、或被两条指令输出,都拒绝;
2. **[SSA-2] 先定义后使用**:按 ordinal 顺序,任何指令的输入必须已被更早的指令定义,或是外部输入;
3. **[SSA-3] 无孤儿描述**:`values` 中的每个条目都被实际使用(定义或引用);
4. **[PLACE-1] 单一位置**:值的 place 由 `values` 声明,所有引用处(计算的 place、通信的 sources/destinations)必须一致;Encode 的执行位置就是其输出 ValueDesc 的 Host place;
5. **[ENC-1] Encode 合法**:只在 initialization,输出为 Host plaintext 且 components=1,payload 严格符合 5.1 节;
6. **[META-1] 计算算子元信息规则成立**(5.3 表);
7. **[META-2] 算子目标参数与输出 ValueDesc 一致**;
8. **[META-3] 通信保持元信息不变**(第 6 节);
9. **[TGT-1] 所有 place 在 `world_size`/`device_counts` 范围内**;
10. **[TGT-2] `required_capabilities`、`required_keys` 与计划内容精确一致**(3、4.3 节);
11. **[TGT-3] `rescale_mode`/`boot_mode` 与 OperatorSpec 及指令属性一致**;
12. **[SPEC-1] OperatorSpec 的 id/version/source_sha256 与 Runtime 实际读取的文件相符**,计划中的 level 范围、boot profile 引用、rescale 降幅都在 spec 声明的合法范围内;
13. **[ORD-1] ordinal 连续递增且与数组顺序一致;TransferId 全局唯一**;
14. **[IO-1] `final_outputs` 非空、无重复、都有定义;`external_inputs` 无重复**;
15. **[IO-2] 每个 external_input 的 place 必须是 Host**:它们只由调用方绑定,进设备必须走显式通信指令;
16. **[BND-1] 出现 bundle Encode 时,每个 Runtime 实例都必须取得顶层引用的完整数据包;`bundle_format_version = 1`,id/version/manifest_sha256 必须相符,每个 Encode 引用的 content 必须在 manifest 中。manifest `blobs` 列出的每个数据文件都必须存在,长度、内容哈希和小端 float64 数据合法性全部通过检查;再按引用它的每个 Encode 输出分别检查 slot 容量。不存在 bundle Encode 时不允许出现 `plaintext_bundle`**;
17. **[ART-1] 默认模式下,所有 rank 读取的 RuntimePlan 原始字节 SHA-256 必须相同;OperatorSpec 和 bundle manifest 的原始字节摘要必须分别与计划引用相符**。调试模式只能按第 8.2 节显式跳过这组摘要比较。

Dacapo 生成端应实现同样的检查以便尽早报错,但 Runtime **不信任**生成端的结果,一律重查。

## 8. 原始文件字节摘要

V1 不定义 JSON 语义指纹,也不使用 RFC 8785/JCS。摘要回答的是“是不是同一份发布文件”,不是“两个不同写法的 JSON 是否表达相同语义”。

### 8.1 计算与使用

对 RuntimePlan、OperatorSpec 或 bundle manifest 计算摘要时:

1. 以二进制方式读取文件的完整原始字节;
2. 不解析 JSON,不删除字段,不调整缩进、换行或字段顺序;
3. 直接计算 SHA-256,文本形式为 `"sha256:" + 64 位小写十六进制`。

RuntimePlan 不在自身 JSON 中保存摘要,避免文件包含自身摘要的循环定义。Runtime 读取计划时计算 `plan_source_sha256`,用于日志、缓存键和多 rank 一致性检查。OperatorSpec 和 bundle manifest 是独立文件,所以计划分别保存它们的 `source_sha256` 和 `manifest_sha256`。

原始字节不同就视为不同发布文件。只改缩进、字段顺序、行尾或末尾换行也会改变摘要,这是 V1 的明确语义。计划和配置由生成器产出后应原样分发,不应在部署阶段重新格式化。

摘要只能发现误修改、文件损坏和版本漂移,不能证明文件来自可信发布者。需要防止恶意篡改时应在发布层增加数字签名,不能把 SHA-256 摘要当成签名。

### 8.2 调试时跳过摘要检查

Runtime 启动选项提供 `skip_artifact_digest_checks`,默认值为 `false`。它是部署方传入的调试选项,**不写进 RuntimePlan**,计划文件不能自行要求跳过校验。

设为 `true` 时只跳过以下比较:

- 各 rank 的 `plan_source_sha256` 是否相同;
- OperatorSpec 原始字节 SHA-256 是否等于计划的 `operator_spec.source_sha256`;
- bundle manifest 原始字节 SHA-256 是否等于计划的 `plaintext_bundle.manifest_sha256`。

它不会跳过 JSON 严格解析、Schema/版本、SSA、Place、元信息、能力、密钥、id/version、bundle 文件存在性、长度、blob `content` SHA-256、小端 float64 合法性或 Api 值校验。开启时必须打印醒目的调试警告;同一执行组的所有 rank 必须使用相同设置。多 rank 开启后无法提前发现“各 rank 拿到不同但各自合法的计划”,可能造成通信错位或挂起,因此优先只用于单 rank 调试。生产启动配置必须保持 `false`。

## 9. 与样例集的关系

V1 正式冻结后,`testdata/valid/` 中的每份文件都必须被接受,`testdata/invalid/` 中的每份都必须被拒绝,且每份 invalid 只含一个错误。当前样例集仍对应 Encode 被降成 external_input 的上一版实现,尚未迁移到本文的显式 Encode 格式;迁移任务见[实现状态](../../overview-design/implementation-status.md)。完成新 Schema、样例和 reader 之前,不能宣称 V1 已冻结或实现已符合本规范。
