# profiles 目录说明

这里保存 RuntimePlan 样例引用的 OperatorSpec 副本。计划 JSON 只保存 id/version/fingerprint;目标 Runtime 应读取对应 spec 文件并核对三项。**当前 C++ reader 尚未实现这一步**,所以改动 profile 后必须由维护者主动升 `version`、重算 spec 指纹并更新所有引用样例,不能依赖现有测试自动发现。

| 文件 | spec_id | rescale_mode | 状态 |
| --- | --- | --- | --- |
| `poseidon-ckks-cpu.v1.json` | `poseidon-ckks-cpu-v1` | eager | placeholder |
| `poseidon-ckks-gpu.v1.json` | `poseidon-ckks-gpu-v1` | lazy | placeholder |

**两份都是 placeholder**:结构完整、符合 Schema,但所有延迟数值都是虚构的占位,`noise_by_level` 按当前 V1 草案要求全部为 `null`。它们只用于设计和样例,不能用于真实的编译代价决策。

等 Poseidon 实测数据可用后:

1. 替换数值,`status` 改为 `"validated"`;
2. `version` 升为 2(同一 `spec_id` + `version` 的内容永远不变);
3. 按 RuntimePlan 规范第 8 节重算 `fingerprint`;
4. 同步更新引用这些 spec 的 testdata 样例并重算它们的指纹。

GPU profile 的 `max_modulus_log2` 是 28(低 bit RNS 限制),这是它 `rescale_mode: "lazy"` 的物理原因;CPU profile 是 60,用 eager。GPU 的 boot 走 Host `decrypt_reencrypt` 模拟 profile(GPU 原生 boot 尚不可用,`operators.boot.supported` 仍为 true 是因为计划里的 Boot 指令放在 Host 执行)。
