# CKKS OperatorSpec V1 规范

> 本文描述目标规范。当前 C++ Runtime 只保存 OperatorSpec 引用,尚未读取 spec 文件或按本文做完整校验;现状见[实现状态](../../overview-design/implementation-status.md)。

OperatorSpec 回答一个问题：**在某个具体后端上，哪些 CKKS 算子可用、参数边界是什么、代价是多少**。普通算子的元信息变化规则由 RuntimePlan 规范统一定义；OperatorSpec 只补充目标相关的范围、rescale 限制和 Boot profile。它既是 Dacapo 的编译输入，也是 Runtime 的验证输入。

字段含义和 Schema 由本仓库(Runtime)维护,因为 Runtime 要按同一规则验证计划。具体数值来自 Poseidon 的实现和测量;本仓库 `profiles/` 目前只有用于格式和测试的 placeholder 副本,还不是已联调验证的生产参数。Dacapo 读取 spec,不拥有也不猜测这些数值。

基础编码规则与 [RuntimePlan 规范](../../runtime-plan/v1/specification.md)第 1 节一致:UTF-8、未知字段拒绝、缺字段拒绝、枚举用固定小写字符串。RuntimePlan 只在 Encode 的 inline payload 中允许浮点数,OperatorSpec **任何位置都不允许浮点数**(延迟等测量值用整数微秒/纳秒表示)。OperatorSpec 文件本身不保存摘要;RuntimePlan 用 `operator_spec.source_sha256` 引用其完整原始文件字节,规则见 RuntimePlan 第 8 节。

## 1. 顶层结构

~~~json
{
  "spec_format_version": 1,
  "spec_id": "poseidon-ckks-cpu-v1",
  "version": 1,
  "status": "placeholder",
  "target_id": "poseidon-ckks-cpu",
  "rescale_mode": "eager",
  "context": { … },
  "levels": { … },
  "operators": { … },
  "boot_profiles": [ … ]
}
~~~

- `spec_format_version`:本规范对应 `1`;
- `spec_id`:spec 的稳定标识。**同一个 `spec_id` + `version` 发布后,完整文件字节永远不允许变**——数值、字段或格式要改,都升 `version`;
- `version`:整数,从 1 开始;
- `status`：`"placeholder"` 表示数值未经实测，只能用于测试；`"validated"` 表示数值已经测量并联调通过。MockVecApi/MpiVecApi 这类参考后端可以显式接受 placeholder；PoseidonApi 必须拒绝 placeholder，生产部署只接受 `validated`；
- `target_id`:该 spec 服务的目标后端家族,必须与引用它的计划的 `target.target_id` 一致;
- `rescale_mode`:`"eager"` 或 `"lazy"`。这是权威声明:引用本 spec 的计划,头部 `rescale_mode` 必须与此相同。CPU profile 用 eager;当前受低 bit RNS 限制的 Poseidon GPU profile 用 lazy——选了这个 profile,Dacapo 的 lazy-rescale pass 就是必选项。

## 2. context

~~~json
{
  "context_id": "ctx-main",
  "poly_degree": 32768,
  "rns_moduli_log2": [60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60],
  "max_modulus_log2": 60,
  "default_scale_log2": 40
}
~~~

- `context_id`:计划中每个值的 `context` 字段必须等于它;
- `poly_degree`:多项式度数,2 的幂;
- `rns_moduli_log2`:模数链上每个 RNS 模数的 bit 数,**下标 = level**(下标 0 是链的最底层)。在 level `L` 时,协议使用的粗粒度总模数预算定义为 `sum(rns_moduli_log2[0..L])`;链长决定 level 的合法上界;
- `max_modulus_log2`:单模数 bit 数上限。低 bit GPU 实现的限制就体现在这里,它是"GPU profile 为什么必须 lazy"的物理原因;
- `default_scale_log2`:编译器规划新明文时优先采用的 scale 指数。RuntimePlan 中每个 Encode 输出仍必须把 `scale_log2` 明确写在 ValueDesc 里,Runtime 不靠这个字段补默认值。

## 3. levels

~~~json
{
  "lower_bound": 2,
  "upper_bound": 13
}
~~~

计划中任何明文或密文的 `level` 都必须落在 `[lower_bound, upper_bound]`。这也包括 Encode 产生的 plaintext。**方向与 RuntimePlan 一致:大 level = 剩余模数多**。注意 Dacapo 现有 profile JSON(`profiled_*.json`)里的 bootstrap 范围是 Earth 方向的数字,迁移到本格式时必须做层号换算,不能照抄。

## 4. operators

`operators` 对 RuntimePlan 的 12 个计算算子逐一声明,**一个都不能缺**。其中 11 个普通算子保存支持状态和按 level 的代价;`boot` 在这里只保存总开关,具体实现和代价放在 `boot_profiles`。

~~~json
{
  "add_cc": {
    "supported": true,
    "latency_us_by_level": [50, 98, 153, 209, 269, 335, 409, 472, 561, 638, 709, 800, 950, 1100],
    "noise_by_level": null
  },
  "rescale": {
    "supported": true,
    "max_levels_per_op": 1,
    "latency_us_by_level": [ … ],
    "noise_by_level": null
  },
  …
}
~~~

普通算子条目的字段:

- `supported`:布尔。计划中出现 `supported: false` 的算子即拒绝;
- `latency_us_by_level`:整数微秒数组,**下标 = 执行时输入的 level**,长度 = `levels.upper_bound + 1`(用不到的低下标照样占位,保持下标即 level 的直观性)。允许为 `null`,表示该 profile 未提供延迟数据(此时 Dacapo 不能用它做代价驱动的决策);
- `noise_by_level`:V1 草案尚未冻结跨后端可解释的噪声单位,因此目前**必须为 `null`**。字段仍必须出现,明确表示“未提供”;在单位和缩放规则确定前,Dacapo 不得消费非空噪声数据。以后若支持非空表,必须先修改规范和 Schema;
- `rescale` 独有 `max_levels_per_op`:单条 rescale 指令允许的最大降层数。eager 模式通常为 1;lazy 模式可以更大(合并的 rescale)。

计算算子怎样改变元信息(scale 相加、components 变化等)由 RuntimePlan 规范 5.3 节统一定义,**不在 spec 中重复**——spec 只提供数值边界和代价,规则本身是协议的一部分。

## 5. boot_profiles

`operators.boot.supported` 只是 Boot 的总开关;具体实现不放在该条目中,因为一个后端可能同时提供原生、CPU 解密再加密模拟等多个 profile,每种合法范围和代价都不同:

~~~json
[
  {
    "profile_id": "poseidon-cpu-boot-emulation-v1",
    "implementation": "decrypt_reencrypt",
    "input_level_min": 2,
    "input_level_max": 13,
    "input_components": 2,
    "output_level": 12,
    "output_scale_log2": 40,
    "output_components": 2,
    "latency_us": 2500000,
    "host_requirements": {
      "needs_secret_key": true,
      "needs_host_compute": true
    }
  }
]
~~~

- `profile_id`:计划中 Boot 指令的 `operator_profile` 必须引用一个存在的 id;
- `implementation`:`"native"` 或 `"decrypt_reencrypt"`,必须与引用它的 Boot 指令一致;
- `input_level_min` / `input_level_max` / `input_components`:合法输入范围;
- `output_level` / `output_scale_log2` / `output_components`:引用该 profile 的 Boot 指令,其 `target_*` 属性必须与这三项完全一致。boot 内部若用 lazy-rescale、受 RNS bit 数限制多消耗 level,都必须体现在 `output_level` 的数值里——**GPU boot 的消耗不能沿用 CPU 数值**。内部实现变了就升 spec `version`,不允许同一版本下悄悄改含义;
- `latency_us`:整数微秒;
- `host_requirements`:`decrypt_reencrypt` 实现必填其能力要求(secret key、Host 计算);`native` 实现写 `{"needs_secret_key": false, "needs_host_compute": false}`。

`operators.boot.supported` 必须与 `boot_profiles` 是否非空完全一致:

- `supported = false` 时 `boot_profiles` 必须为空,计划中出现 Boot 即拒绝;
- `supported = true` 时 `boot_profiles` 必须至少有一项;
- 同一份 spec 内的 `profile_id` 必须唯一。

## 6. 一致性检查(Runtime 侧)

Runtime 验证计划时,对照 spec 至少检查:

1. 计划 `target.operator_spec` 的 id/version 与实际读取的 spec 一致;默认模式下,该文件完整原始字节的 SHA-256 还必须等于 `source_sha256`;
2. 计划 `target.target_id` = spec `target_id`,计划 `rescale_mode` = spec `rescale_mode`;
3. 每个值的 `context` = `context_id`,`level` 在 `[lower_bound, upper_bound]` 内,且 `scale_log2 < sum(rns_moduli_log2[0..level])`。这是格式级粗检查,不替代后端更严格的精度/噪声判断;
4. 计划用到的每种普通算子 `supported: true`;rescale 降幅 ≤ `max_levels_per_op`;
5. `operators.boot.supported` 与 `boot_profiles` 非空状态一致,且 `profile_id` 唯一;
6. 每条 Boot 指令的 profile 存在、`implementation` 匹配、输入 level/components 在范围内、`target_*` 与 profile 输出三项一致。

OperatorSpec 是编译输入,不是让 Runtime 现场重新规划的配置:Runtime 只做以上验证,不会根据 spec 帮编译器补 Rescale 或重算 boot 位置。

## 7. profiles/ 目录

`profiles/` 存放计划样例引用的 spec 副本。当前两份都是 `status: "placeholder"`——结构完整、数值虚构,用于设计和测试数据;当前 C++ reader 还不会加载它们做联动校验。等 Poseidon 实测后替换数值并把 `status` 改为 `validated`(同时升 `version`)。目标格式不在 spec 内保存自摘要;当前 profile JSON 仍保留上一版 `fingerprint` 字段,会和 Schema、样例一起迁移。见 [profiles/README.md](profiles/README.md)。
