# RuntimePlan V1 规范

本文档是 RuntimePlan V1 文件格式的**唯一权威定义**。Dacapo 按本规范生成计划文件,Runtime 按本规范读取和验证。两边不共享任何代码;对协议的理解有分歧时,以本文档和 `testdata/` 样例集为准。

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

### 1.2 没有浮点数

**整个协议不出现任何 JSON 浮点数。** CKKS 的 scale 用整数指数 `scale_log2` 表达(`scale_log2 = 40` 表示逻辑 scale 为 2^40)。任何字段出现非整数的数字字面量,整份计划拒绝。

### 1.3 枚举

所有枚举用固定的小写下划线字符串,不用数字。各枚举的合法取值在对应章节列出,出现清单之外的字符串即拒绝。

## 2. 顶层结构

~~~json
{
  "format_version": 1,
  "plan_id": "42",
  "fingerprint": "sha256:…",
  "target": { … },
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
- `fingerprint`:计划语义内容的指纹,计算规则见第 8 节。读取方必须重算并比对,不符即拒绝;
- `target`:目标环境描述,见第 3 节;
- `values`:计划中出现的**所有**值的描述,见第 4 节;
- `external_inputs`:由调用方在启动时提供的 ValueId 列表;
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
    "fingerprint": "sha256:…"
  },
  "rescale_mode": "lazy",
  "boot_mode": "decrypt_reencrypt",
  "world_size": 2,
  "device_counts": [2, 2],
  "required_capabilities": ["transfer", "host_compute", "boot_decrypt_reencrypt"]
}
~~~

- `target_id`:目标后端家族的固定名字;
- `capability_version`:Runtime Api 能力集合的版本,整数;
- `operator_spec`:编译这份计划时使用的 [OperatorSpec](../../operator-spec/v1/specification.md) 的 id、版本和指纹。Runtime 必须核对自己持有的同 id spec 的指纹,不符即拒绝;
- `rescale_mode`:`"eager"` 或 `"lazy"`。**这是冗余摘要,权威在 OperatorSpec**——两处不一致即拒绝;
- `boot_mode`:`"native"` 或 `"decrypt_reencrypt"`。同样是冗余摘要,权威在每条 Boot 指令的 `implementation` 属性:计划中出现任何一条 `implementation` 与 `boot_mode` 不符的 Boot 指令即拒绝。计划中没有 Boot 指令时,`boot_mode` 必须写 `"native"`;
- `world_size`:进程(rank)数,≥ 1;
- `device_counts`:长度必须等于 `world_size`,第 i 项是 rank i 的设备数,≥ 0;
- `required_capabilities`:计划实际用到的能力集合,合法取值:`"transfer"`、`"replicate"`、`"host_compute"`、`"boot_native"`、`"boot_decrypt_reencrypt"`。列表必须与计划内容一致:用到了没声明、声明了没用到,都拒绝(这让 Runtime 只看头部就能做能力预检,且头部撒谎会被抓住)。`host_compute` 指存在 place 为 Host 的计算指令。

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

## 5. 计算指令

三个阶段(`initialization`、`execution`、`finalization`)的数组元素统一为"指令"。指令用 `kind` 区分:`"compute"`、`"transfer"`、`"replicate"`。

`ordinal` 是指令的稳定序号:在**整份计划**(三个阶段按 initialization → execution → finalization 拼接)中从 0 开始、严格递增、连续。它只用于报错定位和交叉引用,执行顺序由数组顺序决定(两者必须一致)。

计算指令:

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
- `attrs`:按 `op` 决定,见 5.1。**需要 attrs 的算子缺 attrs、不需要 attrs 的算子出现 attrs,都拒绝。**

这个算子集合是封闭的:可执行 CKKS 里的编译器内部算子(如 `upscale`)必须在导出前消除(Dacapo 已有 `UpscaleToMulcp` 之类的下沉),出现集合之外的算子名即拒绝。

### 5.1 各算子的参数和元信息规则

下表中"输入"指 `inputs` 数组各项的 ValueDesc,"输出"指 `output` 的 ValueDesc。所有算子都要求:输入输出 `context` 相同、`ntt` 相同(V1 不表达 NTT 形式转换);所有 ciphertext 输入 `level` 相同。违反任意一条即拒绝。

| op | inputs | attrs | 元信息规则 |
| --- | --- | --- | --- |
| `add_cc` `sub_cc` | 2 个 ct | 无 | 两输入 `scale_log2`、`components` 相同;输出与输入完全同元信息 |
| `add_cp` `sub_cp` | ct, pt(顺序固定) | 无 | 两输入 `scale_log2` 相同;输出与 ct 输入同元信息 |
| `mul_cc` | 2 个 ct | 无 | 输出 `scale_log2` = 两输入之和;`components` = c₁+c₂−1;`level` 不变 |
| `mul_cp` | ct, pt | 无 | 输出 `scale_log2` = 两输入之和;`components` = ct 的;`level` 不变 |
| `negate` | 1 个 ct | 无 | 输出与输入同元信息 |
| `rotate` | 1 个 ct | `{"steps": n}`,非零整数 | 输出与输入同元信息;输入 `components` 必须为 2;place 需 `galois` 密钥 |
| `rescale` | 1 个 ct | `{"target_level": l, "target_scale_log2": s}` | `target_level` < 输入 `level`;输出 `level` = `target_level`、`scale_log2` = `target_scale_log2`、`components` 不变。允许一次降多层(lazy 模式合并的 rescale);单次合法降幅由 OperatorSpec 约束 |
| `mod_switch` | 1 个 ct | `{"target_level": l}` | `target_level` < 输入 `level`;输出 `level` = `target_level`,`scale_log2`、`components` 不变 |
| `relinearize` | 1 个 ct | 无 | 输入 `components` = 3,输出 = 2;其余不变;place 需 `relin` 密钥 |
| `boot` | 1 个 ct | 见 5.2 | 输出 `level` = `target_level`、`scale_log2` = `target_scale_log2`、`components` = `target_components`;`context` 不变(V1 Boot 不换 context、不换多项式度数) |

**算子的目标参数与输出 ValueDesc 必须逐字段一致**(如 Boot 声明 `target_level: 6` 而输出值描述写 `level: 5`,拒绝)。这是故意的双重记账:两处由导出器独立写出,不一致说明导出器有 bug。

### 5.2 Boot 的 attrs

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
4. **[PLACE-1] 单一位置**:值的 place 由 `values` 声明,所有引用处(计算的 place、通信的 sources/destinations)必须一致;
5. **[META-1] 算子元信息规则成立**(5.1 表);
6. **[META-2] 算子目标参数与输出 ValueDesc 一致**;
7. **[META-3] 通信保持元信息不变**(第 6 节);
8. **[TGT-1] 所有 place 在 `world_size`/`device_counts` 范围内**;
9. **[TGT-2] `required_capabilities`、`required_keys` 与计划内容精确一致**(3、4.3 节);
10. **[TGT-3] `rescale_mode`/`boot_mode` 与 OperatorSpec 及指令属性一致**;
11. **[SPEC-1] OperatorSpec 的 id/version/fingerprint 与 Runtime 持有的副本相符**,计划中的 level 范围、boot profile 引用、rescale 降幅都在 spec 声明的合法范围内;
12. **[ORD-1] ordinal 连续递增且与数组顺序一致;TransferId 全局唯一**;
13. **[IO-1] `final_outputs` 非空、无重复、都有定义;`external_inputs` 无重复**;
14. **[FP-1] 重算指纹与 `fingerprint` 字段一致**。

Dacapo 生成端应实现同样的检查以便尽早报错,但 Runtime **不信任**生成端的结果,一律重查。

## 8. 指纹(fingerprint)

指纹回答"两份文件是不是同一份计划":同一语义内容,无论缩进、字段顺序、是否美化打印,指纹必须相同。多进程执行时各 rank 比对指纹,确认拿到的是同一份计划。

计算步骤:

1. 把 JSON 文本解析成数据模型;
2. 删除顶层的 `fingerprint` 字段;
3. 按 **RFC 8785(JSON Canonicalization Scheme)** 序列化。由于本协议的数字全部是整数且绝对值 < 2^53,JCS 的数字规则退化为普通十进制整数,无需实现浮点规范化。等价的参考实现(Python):

   ~~~python
   import json, hashlib
   canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
   digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
   fingerprint = "sha256:" + digest
   ~~~

4. 对 UTF-8 字节算 SHA-256,字段值写成 `"sha256:" + 64 位小写十六进制`。

OperatorSpec 文件的指纹用同样的规则计算(同样在计算前删除其顶层 `fingerprint` 字段)。

**注意**:指纹的输入是删除 `fingerprint` 后的**全部**剩余内容。改任何字段——包括 `plan_id` 或 hint——指纹都会变。这是刻意的:指纹校验的是"同一份文件",不是"数学上等价的计划"。

## 9. 与样例集的关系

`testdata/valid/` 中的每份文件,任何符合本规范的读取实现都必须接受;`testdata/invalid/` 中的每份必须拒绝,且每份文件只含一个错误(文件清单和错误原因见 [testdata/README.md](testdata/README.md))。修改本规范时必须同步增补样例;实现与样例冲突时,先怀疑实现,再怀疑样例,最后才怀疑规范。
