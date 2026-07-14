# profiles 目录说明

这里是本仓库集成测试**固定使用**的 OperatorSpec 副本。`testdata/` 里的合法计划样例通过 `target.operator_spec` 的 id/version/fingerprint 引用它们,所以这两份文件的内容一变,样例计划的指纹核对就会失败——这是刻意的,提醒你同步升 `version` 并重算指纹。

| 文件 | spec_id | rescale_mode | 状态 |
| --- | --- | --- | --- |
| `poseidon-ckks-cpu.v1.json` | `poseidon-ckks-cpu-v1` | eager | placeholder |
| `poseidon-ckks-gpu.v1.json` | `poseidon-ckks-gpu-v1` | lazy | placeholder |

**两份都是 placeholder**:结构完整、符合 Schema,但所有延迟数值都是虚构的占位,`noise_by_level` 全部为 `null`。它们的用途是冻结格式、驱动协议测试,不能用于真实的编译代价决策。

等 Poseidon 实测数据可用后:

1. 替换数值,`status` 改为 `"validated"`;
2. `version` 升为 2(同一 `spec_id` + `version` 的内容永远不变);
3. 按 RuntimePlan 规范第 8 节重算 `fingerprint`;
4. 同步更新引用这些 spec 的 testdata 样例并重算它们的指纹。

GPU profile 的 `max_modulus_log2` 是 28(低 bit RNS 限制),这是它 `rescale_mode: "lazy"` 的物理原因;CPU profile 是 60,用 eager。GPU 的 boot 走 Host `decrypt_reencrypt` 模拟 profile(GPU 原生 boot 尚不可用,`operators.boot.supported` 仍为 true 是因为计划里的 Boot 指令放在 Host 执行)。
