# profiles 目录说明

这里保存 RuntimePlan 样例引用的 OperatorSpec 副本。目标协议中计划保存 id/version/source_sha256,其中 `source_sha256` 直接对 spec 文件完整原始字节计算。**当前 Schema、profile 和样例仍是上一版自嵌 fingerprint 格式**,C++ reader 也尚未实现 OperatorSpec 联动读取;它们会随 RuntimePlan 新样例一起迁移。

| 文件 | spec_id | rescale_mode | 状态 |
| --- | --- | --- | --- |
| `poseidon-ckks-cpu.v1.json` | `poseidon-ckks-cpu-v1` | eager | placeholder |
| `poseidon-ckks-gpu.v1.json` | `poseidon-ckks-gpu-v1` | lazy | placeholder |

**两份都是 placeholder**:结构完整、符合 Schema,但所有延迟数值都是虚构的占位,`noise_by_level` 按当前 V1 草案要求全部为 `null`。它们只用于设计和样例,不能用于真实的编译代价决策。

等 Poseidon 实测数据可用后:

1. 替换数值,`status` 改为 `"validated"`;
2. `version` 升为 2(同一 `spec_id` + `version` 发布后连格式也不再改变);
3. 生成最终 spec 文件后,对完整原始字节计算 `source_sha256`;
4. 同步更新引用这些 spec 的 RuntimePlan 样例。开发中只要文件字节变化,哪怕只是格式化,摘要也必须更新;已经发布的版本则必须升 `version`。

GPU profile 的 `max_modulus_log2` 是 28(低 bit RNS 限制),这是它 `rescale_mode: "lazy"` 的物理原因;CPU profile 是 60,用 eager。GPU 的 boot 走 Host `decrypt_reencrypt` 模拟 profile(GPU 原生 boot 尚不可用,`operators.boot.supported` 仍为 true 是因为计划里的 Boot 指令放在 Host 执行)。
