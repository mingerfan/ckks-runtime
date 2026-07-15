# profiles 目录说明

这里保存 RuntimePlan V1 样例引用的 OperatorSpec 副本。计划保存 id/version/source_sha256，其中 `source_sha256` 直接覆盖 spec 文件完整原始字节。

| 文件 | spec_id | rescale_mode | 状态 |
| --- | --- | --- | --- |
| `poseidon-ckks-cpu.v1.json` | `poseidon-ckks-cpu-v1` | eager | placeholder |
| `poseidon-ckks-gpu.v1.json` | `poseidon-ckks-gpu-v1` | lazy | placeholder |

**两份都是 placeholder**：结构完整、符合 Schema，但所有延迟数值都是虚构占位，`noise_by_level` 按 V1 要求为 `null`。它们只用于测试，不能用于真实的编译代价决策。

等 Poseidon 实测数据可用后:

1. 替换数值,`status` 改为 `"validated"`;
2. `version` 升为 2(同一 `spec_id` + `version` 发布后连格式也不再改变);
3. 生成最终 spec 文件后,对完整原始字节计算 `source_sha256`;
4. 同步更新引用这些 spec 的 RuntimePlan 样例。开发中只要文件字节变化,哪怕只是格式化,摘要也必须更新;已经发布的版本则必须升 `version`。

两份 profile 都保持 `placeholder`，只用于阶段一协议测试。生产测量数据不在阶段一范围内。
