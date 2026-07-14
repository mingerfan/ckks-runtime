# 协议测试样例集

`valid/` 中的每份文件,任何符合规范的实现**必须接受**;`invalid/` 中的每份**必须拒绝**。每份 invalid 文件只含一个错误,错误编号对应 [specification.md](../specification.md) 第 7 节的检查项。

所有文件由 `generate.py` 生成(`python3 generate.py`),指纹按规范第 8 节实算。**不要手改 JSON**——改了指纹就不符了;要改样例就改生成脚本重跑。valid 计划引用的 OperatorSpec 是 [`profiles/`](../../../operator-spec/v1/profiles/) 下的两份占位 spec,它们的指纹也由同一脚本维护。

## valid

| 文件 | 覆盖内容 |
| --- | --- |
| `v001_minimal_single_device.json` | 最小计划:单 rank 单设备,一条 `add_cc`(输入自加),CPU eager spec |
| `v002_mul_rescale_transfer.json` | GPU lazy spec:`mul_cc`(scale 相加、分量 2+2−1=3)→ `relinearize`(需 relin key)→ `rescale`(显式 target)→ 跨卡 `transfer` |
| `v003_replicate_multi_rank.json` | 2 个 rank:`replicate` 一发两收(Device(1,0) + Host(1)),broadcast hint |
| `v004_host_boot_emulation.json` | `decrypt_reencrypt` boot 全流程:Device→Host transfer、Host boot(需 secret key、引用 boot profile)、Host→Device 回传,能力声明齐全 |

## invalid

| 文件 | 错误(仅一处) | 对应检查 |
| --- | --- | --- |
| `i001_unknown_format_version.json` | `format_version: 2` | 第 2 节 |
| `i002_float_scale_log2.json` | `scale_log2: 40.5`(浮点) | 1.2 节 |
| `i003_duplicate_value_id.json` | 同一 ValueId 两条描述 | SSA-1 |
| `i004_comm_list_length_mismatch.json` | replicate 的 `output_kinds` 长度 1,`outputs`/`destinations` 是 2 | 第 6 节 |
| `i005_boot_mode_mismatch.json` | 头部 `boot_mode: "native"`,指令是 `decrypt_reencrypt` | TGT-3 |
| `i006_boot_target_desc_mismatch.json` | Boot 声明 `target_level: 12`,输出值描述是 `level: 11` | META-2 |
| `i007_use_before_define.json` | 指令使用晚于自己才定义的值 | SSA-2 |
| `i008_bad_fingerprint.json` | 指纹与内容不符(末位翻转) | FP-1 |
| `i009_unknown_field.json` | 顶层多了未定义字段 `comment` | 第 1 节 |
| `i010_place_mismatch.json` | 值声明在 Host,指令在 Device 上用它 | PLACE-1 |
| `i011_missing_required_key.json` | 有 `decrypt_reencrypt` boot 却没声明 Host secret key | TGT-2 |
| `i012_operator_spec_fingerprint_mismatch.json` | `operator_spec.fingerprint` 与持有的 spec 副本不符 | SPEC-1 |

除 `i008`(错误本身就是指纹)外,所有 invalid 文件的指纹对其内容都是**正确**的——保证实现拒绝它们的原因恰好是表中那一个错误,而不是顺带的指纹不符。

结构类错误(i001/i002/i004/i009)应当已被 `schema.json` 拦截;语义类错误(SSA、元信息、一致性)需要读取实现自己检查——Schema 只是第一道网,不是全部。
