# CKKS OperatorSpec V1 规范

OperatorSpec 回答一个问题:**在某个具体后端上,每种 CKKS 算子的合法输入是什么、会怎样改变值的元信息、代价是多少**。它是 Dacapo 的编译输入——bootstrap 安排、scale 管理和 placement 都要照着它算;也是 Runtime 的验证输入——计划里的 level、`scale_log2` 和 boot profile 引用都要对着它检查。

字段含义和 Schema 由本仓库(Runtime)维护,因为 Runtime 要按同一规则验证计划。具体数值来自 Poseidon 的实现和测量;本仓库只固定一份已联调验证的副本(`profiles/`)。Dacapo 读取 spec,不拥有也不猜测这些数值。

基础编码规则与 [RuntimePlan 规范](../../runtime-plan/v1/specification.md)第 1 节一致:UTF-8、未知字段拒绝、缺字段拒绝、枚举用固定小写字符串。RuntimePlan 只在 Encode 的 inline payload 中允许浮点数,OperatorSpec **任何位置都不允许浮点数**(延迟等测量值用整数微秒/纳秒表示)。指纹计算规则同 RuntimePlan 第 8 节。

## 1. 顶层结构

~~~json
{
  "spec_format_version": 1,
  "spec_id": "poseidon-ckks-cpu-v1",
  "version": 1,
  "fingerprint": "sha256:…",
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
- `spec_id`:spec 的稳定标识。**同一个 `spec_id` + `version` 的内容永远不允许变**——数值要改,就升 `version`;
- `version`:整数,从 1 开始;
- `fingerprint`:对删除本字段后的全部内容按 RuntimePlan 第 8 节规则计算;
- `status`:`"placeholder"`(数值是占位,未经实测,不得用于真实编译决策)或 `"validated"`(数值经过测量并联调通过)。计划引用 `placeholder` spec 时 Runtime 应给出警告;
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
- `rns_moduli_log2`:模数链上每个 RNS 模数的 bit 数,**下标 = level**(下标 0 是链的最底层)。链长决定 level 的合法上界;
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

对 RuntimePlan 的 12 种算子逐一声明,**一个都不能缺**(不支持就写 `"supported": false`,不允许用缺字段表达"不支持"):

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

每个算子条目的字段:

- `supported`:布尔。计划中出现 `supported: false` 的算子即拒绝;
- `latency_us_by_level`:整数微秒数组,**下标 = 执行时输入的 level**,长度 = `levels.upper_bound + 1`(用不到的低下标照样占位,保持下标即 level 的直观性)。允许为 `null`,表示该 profile 未提供延迟数据(此时 Dacapo 不能用它做代价驱动的决策);
- `noise_by_level`:同上形状的整数数组(噪声模型的定点表示,单位由 profile 自己在数值上保持一致即可)或 `null`。**字段必须出现**,`null` 是显式的"未提供"——现状 CPU profile 有噪声表而 GPU 没有是漂移不是设计,新格式不允许再用缺字段表达含义;
- `rescale` 独有 `max_levels_per_op`:单条 rescale 指令允许的最大降层数。eager 模式通常为 1;lazy 模式可以更大(合并的 rescale)。

计算算子怎样改变元信息(scale 相加、components 变化等)由 RuntimePlan 规范 5.3 节统一定义,**不在 spec 中重复**——spec 只提供数值边界和代价,规则本身是协议的一部分。

## 5. boot_profiles

Boot 不走 `operators` 表,因为一个后端可能同时提供多种 boot 实现(原生、CPU 解密再加密模拟),每种的合法范围和代价都不同:

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

`boot_profiles` 允许为空数组,表示该后端完全不支持 Boot(计划中出现任何 Boot 指令即拒绝)。

## 6. 一致性检查(Runtime 侧)

Runtime 验证计划时,对照 spec 至少检查:

1. 计划 `target.operator_spec` 的 id/version/fingerprint 与持有副本一致;
2. 计划 `target.target_id` = spec `target_id`,计划 `rescale_mode` = spec `rescale_mode`;
3. 每个值的 `context` = `context_id`,`level` 在 `[lower_bound, upper_bound]` 内,`scale_log2` 不超过模数链在该 level 下的总预算;
4. 计划用到的每种算子 `supported: true`;rescale 降幅 ≤ `max_levels_per_op`;
5. 每条 Boot 指令的 profile 存在、`implementation` 匹配、输入 level/components 在范围内、`target_*` 与 profile 输出三项一致。

OperatorSpec 是编译输入,不是让 Runtime 现场重新规划的配置:Runtime 只做以上验证,不会根据 spec 帮编译器补 Rescale 或重算 boot 位置。

## 7. profiles/ 目录

`profiles/` 存放本仓库集成测试固定使用的 spec 副本。当前两份都是 `status: "placeholder"`——结构完整、数值虚构,用于冻结格式和驱动测试;等 Poseidon 实测后替换数值并把 `status` 改为 `validated`(同时升 `version`)。见 [profiles/README.md](profiles/README.md)。
