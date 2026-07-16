# Dacapo 导入 profiles

这些文件由 `integrations/dacapo/migrate_profiles.py` 从 submodule 固定版本生成。

| 文件 | 来源 | target | 状态 |
| --- | --- | --- | --- |
| `dacapo-heaan-cpu.v1.json` | `profiled_HEAAN_CPU.json` | `dacapo-heaan-cpu` | imported |
| `dacapo-heaan-gpu.v1.json` | `profiled_HEAAN_GPU.json` | `dacapo-heaan-gpu` | imported |
| `dacapo-seal-cpu.v1.json` | `profiled_SEAL_CPU.json` | `dacapo-seal-cpu` | imported |

统一来源：

- 仓库：`mingerfan/dacapo-modified`
- revision：`a2c9ce41a57062cdf77ea4bac5b02be747109448`
- 源文件 SHA-256 已写入各 profile 的 `provenance.source_sha256`

`imported` 只表示旧数据已经按明确规则迁移，不表示它们能作为 Poseidon 生产配置：

- HEAAN 的真实 FVa 模数链不在 Dacapo 仓库中，当前 `rns_moduli_log2` 是按旧 profile 的 `rescalingFactor=51` 展开的编译模型；
- SEAL 的模数链可由旧 HEVM 代码确认是 14 个 60-bit 模数；
- SEAL 噪声使用旧 Dacapo estimator 的内部数值，不能跨后端比较；
- HEAAN CPU/GPU profile 都来自同一套 eager-rescale 编译语义，GPU 文件不因此变成 Poseidon lazy-rescale profile；
- `earth.mul_double` 的旧测量包含乘法和重线性化的整体语义，没有独立的 Relinearize 测量。为了验证新的拆分式 RuntimePlan，导入 profile 把 `relinearize` 标为支持，并在合法 level 上填写固定的 `1 us` 开发期占位延迟；这个数值不是测量结果，不能用于生产调度或性能比较；
- SEAL profile 没有 Boot 延迟测量，V2 中保持 `null`，不把未知代价伪装成 0。

校验生成结果：

```bash
python3 integrations/dacapo/migrate_profiles.py --check
```
