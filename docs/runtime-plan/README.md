# RuntimePlan 规范

本文统一维护当前 reader 和执行器支持的 RuntimePlan。旧版文字和开发过程由 Git 历史保存；需要复现某次实现时，使用同一 commit 的代码、规范和测试数据。

| `format_version` | 指令与阶段 | JSON Schema |
| --- | --- | --- |
| 1 | Encode、Compute、Transfer/Replicate；Encode 仅在 initialization | [V1](v1/schema.json) |
| 2 | V1 加 Release、Compute `reuse_input` | [V2](v2/schema.json) |
| 3 | V2 加 execution Encode 和 Fence；仅支持 `world_size=1` | [V3](v3/schema.json) |
| 4 | V3 加 execution Device Prepare 和 Waterline 预算；Waterline 仅支持单 rank 的设备工作线程 | [V4 规范](v4.md) |

计划语义版本与文件表示分开：二进制容器版本和 bundle 存储版本见[二进制格式](binary.md)与[明文数据包](plaintext-bundle.md)。编译器默认导出二进制；下面用 JSON 展示字段，二进制记录具有相同的计划语义。

RuntimePlan 是编译器交给 Runtime 的最终执行文件。它写清楚 SSA 值、CKKS 元信息、执行位置、Encode、计算和通信。Runtime 只验证并执行，不补指令、不改 level/scale、不隐式搬运数据。

## 1. JSON 基本规则

- UTF-8，无 BOM；顶层必须是 object。
- 未知字段、缺字段、同一 object 内的重复 key 都直接拒绝。
- `plan_id`、ValueId、TransferId 使用十进制字符串，必须匹配 `^(0|[1-9][0-9]*)$`，范围是 uint64。
- level、scale、rank、index、ordinal 等整数使用 JSON integer，受对应字段的 int32 范围约束，不接受 `40.0`、科学计数法、布尔值或字符串。
- inline Encode/Prepare 的 `payload.values` 与 V4 `waterline.device_fractions` 可以出现浮点数。载荷必须能转换成有限 float64；正零和负零在交给 Api 前都转成正零。fraction 必须严格介于 0 和 1。
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

`plaintext_bundle` 是可选字段，只在存在 bundle Encode 时出现。计划没有计划自摘要、`required_keys`、`required_capabilities`、`rescale_mode` 或 `boot_mode` 字段。Runtime 从 OperatorSpec 和实际指令推导这些要求，避免重复声明互相打架。

V4 Prepare 复用 inline/bundle 载荷表示；存在 bundle Prepare 时也必须声明 `plaintext_bundle`。V4 可带计划级 `waterline` 配置，Prepare 要求该配置，字段见 [V4 Waterline](v4.md)。准备依赖来自现有 ValueId 与指令输入，不复制每消费者依赖表。

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
- 计划文件本身不保存自摘要。JSON reader 返回完整原始字节的 `source_sha256`，多 rank preflight 默认比较它；二进制 reader 不计算整文件摘要，preflight 只比较小型 metadata 的摘要，不保证指令或载荷逐字节相同。

bundle 引用格式：

```json
{
  "id": "model-weights",
  "version": 1,
  "manifest_sha256": "sha256:..."
}
```

`manifest_sha256` 同样覆盖 `manifest.json` 的完整原始字节。二进制 manifest 使用互斥的引用形式 `{"id":"model-weights","version":1,"manifest_format":"binary"}`，选择 `manifest.bin`，不携带 manifest 原始字节摘要。其记录规则见[二进制格式](binary.md)。

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

V1/V2 的 Encode 只能在 initialization；V3/V4 还允许在 execution。所有版本的 Encode 都要求输出是 Host plaintext，finalization 不允许 Encode。V4 `prepare_plaintext` 输出指定 Device 的单组件 NTT plaintext，且仅允许在 execution，见 [V4](v4.md)。

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

| op | attrs 字段 |
| --- | --- |
| rotate | `steps`（有符号 int32） |
| rescale | `target_level`、`target_scale_log2` |
| mod_switch | `target_level` |
| boot | `target_level`、`target_scale_log2`、`target_components`、`operator_profile`、`implementation` |
| 其他 | 不允许 attrs |

Rotate 的 `steps` 先按 `poly_degree / 2` 个 slot 取模，负数转成等价正 step。规范化结果为 0 时拒绝。Runtime 用规范化 step 请求具体 Galois key，不接受只有“某种 Galois key”的模糊声明。

Boot 的 `implementation` 是 `native` 或 `decrypt_reencrypt`。后者只能在 Host 执行，需要 secret key 和 Host compute 能力。原生 Boot 失败后不允许自动改走解密再加密。

### 5.3 通信

通信是 `transfer`（1→1）或 `replicate`（1→N，N≥2）。`inputs/sources` 长度为 1；`outputs/destinations/output_kinds` 等长。目的 Place 不能重复，也不能等于源 Place。

hint 合法值为 `auto`、`point_to_point`、`broadcast`、`tree`、`ring`、`host_staged`。hint 只是实现建议，不限制它与 Transfer/Replicate 的组合。Api 可以使用等价实现，但必须完成计划规定的逻辑通信。

通信只改变 id 和 Place，不能改变 kind、context、level、scale、NTT 或 components。

## 6. OperatorSpec 和启动要求

RuntimePlan 不把 OperatorSpec 的格式版本写死在计划结构中。当前 Runtime 明确支持 OperatorSpec V1 和 V2；遇到其他 `spec_format_version` 直接拒绝。

Runtime 必须检查：

- spec id/version/target/context 与计划一致；默认模式下原始字节摘要一致。
- `poly_degree` 为 2 的幂，level 位于模数链内，每个模数不超过上限，延迟表长度正确。
- 计划使用的算子受支持；Rescale 降幅不超过 `max_levels_per_op`。
- Boot profile 唯一，输入范围、implementation 和输出元信息一致。

Runtime 从实际指令推导能力：Encode、Transfer、Replicate、Host compute、native Boot、decrypt/re-encrypt Boot；并推导 relin key、带 step 的 Galois key和 secret key。Api preflight 必须明确确认这些资源可用。

## 7. bundle 的分 rank 装载

每个 rank 都读取同一逻辑 bundle 的 manifest，并检查 format、id/version；JSON manifest 默认还检查 `manifest_sha256`。

V1/V2 在启动时装载本 rank 实际执行的 bundle Encode 载荷；V3/V4 在启动时检查 manifest 和引用，在 Encode/Prepare 执行时按需读取。逐 blob 布局允许只部署本 rank 使用的文件；pack 布局要求完整 data.bin。两种表示都检查长度、小端 float64、有限值和 slot 容量；JSON manifest 路径还检查所读取载荷的 content SHA-256，二进制 manifest 路径不重算载荷哈希。详细规则见[明文数据包](plaintext-bundle.md)。

## 8. Runtime 启动顺序和调试开关

启动顺序固定为：

1. 若启用摘要调试开关，先打印警告；随后验证计划和 OperatorSpec，推导能力与密钥。
2. 读取 manifest；V1/V2 装载本 rank 需要的载荷，V3/V4 准备按需读取的索引。
3. Api preflight；Waterline 验证后端持久资源并取得 allocator capacity。
4. 绑定并完整验证 external inputs。
5. Waterline 在提交 initialization 前检查全图单组基线；再按所选执行模式提交 Encode/Prepare、计算、通信和版本允许的内存指令。
6. 完成通信、同步最终输出，调用 Api::drain 并清理 Runtime 内部值和句柄。

`skip_artifact_digest_checks=true` 只跳过三类比较：rank 间计划原始字节摘要、OperatorSpec 原始字节摘要、manifest 原始字节摘要。它不跳过结构、id/version、SSA、元信息、能力、密钥、载荷长度或 Api 值校验，也不关闭 JSON manifest 路径的 blob content 校验。二进制路径的摘要行为见上文。所有 rank 的开关必须一致，每个 rank 都要打印明确警告。

Api 的 `validate_value` 必须检查 kind、context、poly degree、level、scale、NTT 和 components。任何一项不符都立即终止。

## 9. Release（V2/V3/V4）

```json
{"ordinal": 3, "kind": "release", "value": "7"}
```

Release 没有结果，也不算一次数据使用。它可以出现在初始化、执行或收尾
阶段，释放一个已经定义的值。不能释放最终返回值，也不能再次释放已经
释放或被原位覆盖的旧值。以后任何计算或通信都不能再使用这个名字。

外部输入可以释放 Runtime 持有的引用，调用方持有的对象不受影响。
不要求每个中间值必须写 Release。没有 `wait` 字段。

执行前，Runtime 统计三个阶段中本地值尚待提交的使用次数。计算的每个
操作数出现都计一次；并行通信按实际提交的切片计数，Replicate 的本地
目的地各计一次，跨 rank 的目的地由通信提交线程一起提交、计一次。
初始化结束后继续使用剩余计数，不重新统计。

API 成功接手计算或传输后，Runtime 清掉本次调用的临时输入副本，再减少
次数。Release 先记录释放请求；次数归零时才清掉 Runtime 的数据引用，
保留值表项。未提交的其他线程仍能取到数据，已提交的异步工作由 API 保留。
已释放且不再使用的通信输出晚到时直接丢弃，不重新放入值表。
Release 会调用非阻塞的 `collect_completed()`；整次运行收尾仍调用 `drain()`。

## 10. reuse_input（V2/V3/V4）

```json
{
  "ordinal": 4,
  "kind": "compute",
  "op": "negate",
  "place": {"kind": "host", "rank": 0},
  "inputs": ["7"],
  "output": "8",
  "reuse_input": 0
}
```

`reuse_input` 是 Compute 的可选非负整数，遵守 V1 其他整数的 int32 范围
规则，不接受浮点数、字符串、布尔值或 null。目前只有 0 合法。
没有字段时使用普通计算路径。写了字段后不能默默改用普通路径。

允许原位的操作：

| 位置 | 操作 |
| --- | --- |
| Host | Negate、Rotate |
| Device | AddCP、SubCP、Rotate |

被覆盖的输入必须来自本计划中的计算，不能来自 external input、Encode
或通信。它不能是最终返回值，在三个阶段中只能出现一次操作数使用。
传输源也算使用；`AddCC(x, x)` 算两次；Release 不计入次数。

输入和输出必须在相同 Place，kind、context、level、scale、NTT 和
components 保持一致。现有算子类型和参数检查继续生效。实际存储布局
是否允许原位，由 Api 的原位入口检查。执行前检查 Api 是否支持该操作，实际调用 `compute_reuse`；不支持则报错。

计算后输入的旧名字失效，新输出可以继续作为下一次原位操作的输入。
`final_outputs` 可以返回新输出，但不能同时保留被覆盖的旧输入。
需要保存所有历史值的 `AllValuesAfterRun` 模式与 Release/reuse_input
不兼容，执行前报错。

## 11. 显式 Fence 与 Waterline（V3/V4）

```json
{"ordinal": 123, "kind": "fence"}
```

Fence 没有输入、输出或额外属性，只能在 execution。它完成边界前的工作与通信，交付 Pending 结果并调用 `Api::drain()`，然后才执行边界后的工作。已完成的通信组在 Pending 引用替换后清理；跨边界值、使用次数和 final outputs 保留。Fence 不是 MPI/NCCL barrier；V3 和 Waterline 均要求 `world_size=1`。Waterline 编译器不自动插 Fence。

顺序执行和设备工作线程模式都支持普通 V3 Encode/Transfer/Fence。Waterline 要求设备工作线程模式，使用同一计算/通信执行器；串行 CPU 队列处理完整 CPU Encode 与已就绪的 Host compute，包括 `decrypt_reencrypt` Boot，每卡 GPU 准备至多一份。消费者等待真实 Ready；显式 Fence 等待边界前已提交工作。失败会唤醒依赖等待者并传播原始错误，不切换后端重跑。

V3/V4 不保留已解码 slots 的跨 Encode 缓存。默认按需从文件读取；可显式配置整个 raw pack 的驻留预算，见[明文数据包](plaintext-bundle.md)。JSON manifest 路径校验实际读取的内容哈希，二进制 manifest 路径不重算 payload 哈希。有限值、容量、长度和负零归一化检查始终保留。

Dacapo 的 `--runtime-plan-plaintext-schedule=waterline` 启用 V4 Prepare 规划，保持已有 HEFT placement 与前台顺序，要求以下四个预算为正：

| 顶层选项 | 计入范围 |
| --- | --- |
| `--runtime-plan-gpu-budget-bytes` | 每卡执行器总容量 M；reserve 和动态对象在该容量内一起检查 |
| `--runtime-plan-host-budget-bytes` | 共享 Host 对象、raw、CPU 编码临时量及 pinned staging |
| `--runtime-plan-pinned-budget-bytes` | 共享 pinned 上传及通信暂存 |
| `--runtime-plan-raw-budget-bytes` | 准备期间读取和解码的 raw 数据额度 |

`--runtime-plan-reserve-bytes` 覆盖密钥、参数表、retained workspace、pool 外开销与余量；`--runtime-plan-waterline-fraction` 必须严格介于 0 和 1。`--runtime-plan-workspace-bytes-per-op` 是单组前台计算临时空间上界，0 只做对象容量检查，不是实际显存保证。`--runtime-plan-prepare-backend=cpu|gpu|auto` 选择完整 Encode 路径；`--runtime-plan-lookahead-groups` 限制滚动消费者窗口，不按字节静态提前或预测 Encode 时间。

Pass 顺序是 placement、communication、Waterline Prepare、Release/reuse。Prepare 后不再运行 Encode CSE。一份共享权重 Prepare 保留到最后使用实际完成，不按批克隆或重编码。密文计算与通信顺序保留；多 rank、明文 final output 和超出预算的完整必需工作组明确报错。

`*.waterline.json` 检查同一拓扑顺序的完整单组基线，包含配置的工作区上界；`*.memory.json` 仅按顺序完成假设估算 RNS 对象。Runtime 在初始化提交前重复全图基线，查询实际持久资源、编码准备峰值、消费者工作区及 allocator capacity。在线沿一个准入前缀原子取得完整组额度，按真实完成事件回收，不能先取半组或越过阻塞组。每卡明文水位和硬容量、共享 Host/pinned/raw 同时受约束。具体账本与支持边界见 [V4](v4.md)；对象报告不能当成整个进程的内存测量。

RuntimeTiming 记录 Encode/read 次数、字节与时间、raw 读取上界，以及显式 Fence 次数与时间；显式 raw pack 驻留另有对应计时。API 提供实际 CUDA Encode/H2D/consumer-wait events 与 allocator 指标。工作线程模式的 Fence 时间包含边界前工作和汇合；这些主机或设备时间可能重叠，不能相加当作端到端 GPU 时间。`POSEIDON_RUNTIME_TRACE` 输出 Prepare/Encode/read/Fence 与准入峰值，`POSEIDON_THREAD_TRACE` 还记录工作线程与上传 packing。

## 12. 版本与兼容

`RuntimePlanReader::read_file` 按 `.json` 或 `.bin` 显式选择文件表示，随后按 `format_version` 检查 V1/V2/V3/V4。未知扩展名、版本、字段、算子和不匹配的能力直接报错；不会在解析失败后改用其他表示。V1 不能含 Release/reuse，V1/V2 不能含 Fence 或 execution Encode，旧版本不能含 Prepare。旧静态分批编译选项已删除，普通 Encode/Transfer 和显式 Fence 继续由统一执行器支持。

计划语义版本、二进制容器版本、bundle 存储版本、target/capability 与 OperatorSpec id/version 是不同的兼容轴。文档与当前代码一起演进；修改执行语义或文件布局时明确更新相关版本、Schema、reader、导出器和样例，不通过删除规范文字掩盖兼容性变化。

JSON Schema 描述结构，SSA、CKKS 元信息、使用次数、生命周期和后端资源由 reader/Verifier/Api 检查。[V1 样例](v1/testdata/README.md)和 [CPU](v2/testdata/valid/cpu_reuse_chain.json)/[GPU 原位链](v2/testdata/valid/gpu_reuse_chain.json)保留为测试输入；原位链已由 Mock 执行测试覆盖。`AllValuesAfterRun` 不兼容 Release/reuse，设备工作线程模式也不支持这个历史值捕获模式。
