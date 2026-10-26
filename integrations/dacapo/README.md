# Dacapo 集成

这里维护 Runtime 与可选 `third_party/dacapo` 编译器的构建和使用说明。执行协议见 [RuntimePlan](../../docs/runtime-plan/README.md)、[OperatorSpec](../../docs/operator-spec/README.md) 和 [bundle](../../docs/runtime-plan/plaintext-bundle.md)。开发计划和历史测试过程由 Git 历史保存。

以下 Python、Runtime 构建和差分命令从 CKKS Runtime 仓库根目录执行。

## 编译器与 profile

```bash
git submodule update --init third_party/dacapo
python3 integrations/dacapo/migrate_profiles.py --check
```

迁移脚本从 provenance 固定的 revision 读取三份旧 profile，并检查 `docs/operator-spec/v2/profiles/` 中的数据。submodule 可以继续前进；迁移数据仍来自其记录的 revision。`imported` 只表示复现旧编译模型，不表示 Poseidon 实测配置。

Dacapo 单独构建，下面的括号使后续命令仍从 Runtime 根目录执行：

```bash
(
  cd third_party/dacapo
  nix develop --command cmake --preset nix
  nix develop --command cmake --build --preset nix --parallel 2
  nix develop --command ctest --test-dir build/nix --output-on-failure
)
```

Python 模型 tracing 需要 Dacapo 的 `requirements.txt`；C++ 编译器与 Runtime 核心库不依赖 PyTorch。

## 默认产物与语义版本

`generate_model_artifacts.py` 和 `emit-runtime-plan` 默认导出 `.runtime-plan.bin`，大常量写到 `.bundle/manifest.bin` 与 `.bundle/data.bin`。JSON 模式使用 `.runtime-plan.json` 和 `manifest.json`，仍默认 pack。格式与兼容规则见[二进制规范](../../docs/runtime-plan/binary.md)。

| 规划模式 | RuntimePlan 语义版本 |
| --- | --- |
| 普通 eager 计划 | V1 |
| Release / reuse 内存规划 | V2 |
| execution Encode 或显式 Fence | V3 |
| Waterline Device Prepare 与预算配置 | V4 |

Encode 的 inline 阈值默认 4096 字节，超过阈值的 float64 payload 按 content 去重写入 bundle；`--inline-payload-max-bytes=0` 外化所有非空 payload。不同文件表示使用独立输出 prefix。

## 生成模型

```bash
python3 integrations/dacapo/generate_model_artifacts.py mlp \
  --operator-spec docs/operator-spec/v2/profiles/dacapo-heaan-cpu.v1.json

python3 integrations/dacapo/generate_model_artifacts.py resnet20 \
  --operator-spec docs/operator-spec/v2/profiles/dacapo-heaan-gpu.v1.json
```

默认输出到 `third_party/dacapo/review_artifacts/<model>/`。用 `--traced-mlir /path/to/model.mlir` 跳过 tracing；`--dry-run` 校验配置并打印编译命令，不运行编译器。

脚本从 OperatorSpec V1/V2 读取 target、context、id/version、Boot 配置，并对原始文件字节计算摘要。V2 默认校验 provenance 对应的旧 profile；没有该来源时显式传入 `--compiler-profile`。多个 Boot profile 用 `--boot-profile` 选定。placement 要求 OperatorSpec V2。

`--boot-placement=optimized|greedy|depth-dp` 选择 Boot 规划算法，默认 optimized。lazy spec 自动在 Earth→CKKS 后、placement 前加入物理 level 换算；默认倍率 4，可用 `--lazy-rescale-level-factor` 修改。倍率必须与逻辑 scale 降幅和真实模数链匹配。

native spec 使用一条逻辑 Rescale 对应一个物理 Q，并保留实际浮点 scale。`context.rns_moduli` 必须给出完整的实际 Q 数组，`rns_moduli_log2` 必须逐项匹配其位宽；支持等位宽和混合位宽链。数组从底部保留素数向上排列，Rescale 从当前有效前缀的末端删除素数。compiler profile 的正值 `rescalingFactor` 保留为兼容字段，不用于 native 的降 scale 计算。所有规划算法都从所选 Boot profile 传入输出 scale。optimized 的绕行、覆盖范围和延迟规划使用相同的阈值规则与逐值 Q 状态；Boot 计划保留边界和目标，同一值可在不同边界多次刷新。预算不足直接报错，不切换算法。

native 的 rescale 时机由当前位置的实际素数决定：只有 `s - log2(q_last) >= waterline` 才继续删除一个 Q。greedy、depth-dp 和 optimized 使用相同的状态推导；分段候选保留实际浮点 scale。整数 IR 类型用于类型检查和编码、对齐的整数部分，Rescale 结果类型取实际指数的向上取整；计划和运行时保留实际浮点指数。物理转换重新计算并核对出口 scale、实际模数容量和规划所用的 Q 链。native 路径不允许 EarlyModswitch 越过 Rescale，也不执行用固定步长抵消 Rescale 的改写。

手动运行 native `--dacapo-precompile` 时，必须同时传入 `--runtime-plan-physical-level-operator-spec-path`、`--runtime-plan-boot-profile` 和匹配 profile 的 `--bootstrap-scale`，使 Boot 规划阶段就能读取实际 Q 链和出口约定；完整模型生成命令自动传入这三项。未指定 native OperatorSpec 的逻辑预编译继续使用 compiler profile 的固定步长。生成的 native 检查点保留规划 Q 链、Boot 出口位置和实际 scale，后续导出必须匹配。

可用 `python3 third_party/dacapo/test/runtime-plan/check-native-planning.py --hecate-opt third_party/dacapo/build/nix/bin/hecate-opt --source-dir third_party/dacapo` 检查实际素数对 rescale 次数和 Boot 候选可行性的影响，以及混合链的物理转换和错误位宽拒绝行为。

可用 `python3 tests/dacapo_native_artifact_tests.py --hecate-opt third_party/dacapo/build/nix/bin/hecate-opt` 检查三个规划算法在全 30 位、30/31/32 位混合链和手动 ResNet20 Q50 链上的标准生成路径，并验证旧标量因子不影响 native 计划。手动 Q50 链来自 `tests/fixtures/resnet20-manual-q50.json`，该用例只编译普通算子，要求不出现 Boot；这里的输入锚点 profile 不用于验证真实 Boot 的输出层数或精度。测试使用显式 trace，无需运行模型前端或 GPU。

重复调整 placement、通信或内存参数时，可以直接用 `hecate-opt --dacapo-precompile --emit-bytecode` 显式生成未分配设备的 CKKS 检查点，再用 `hecate-opt --ckks-to-runtime-plan` 显式读取它；前半段 Boot 规划只在生成检查点时执行。完整命令和重新生成条件见编译器的[手动检查点说明](../../third_party/dacapo/README.md#manual-compilation-checkpoints)。该接口不自动查找或更新缓存，当前模型生成脚本仍执行完整编译。

`--device-counts=8` 表示一 rank 八卡，`8x8` 表示两 rank 各八卡，`0x0` 表示两个 Host-only CPU rank；零与正数不能混用。不传该参数时生成单 Host 计划。`--communication-profile` 选择[通信代价模型](communication-profile.md)，包括 V2 拓扑与设备对规则。

## 生成并差分检查 MLP

为了便于审阅，下面显式生成 JSON；计划不启用 Release/reuse，以便捕获所有历史值。traced MLIR 来自上面的 MLP 生成步骤。

```bash
python3 integrations/dacapo/generate_model_artifacts.py mlp \
  --operator-spec docs/operator-spec/v2/profiles/dacapo-heaan-cpu.v1.json \
  --traced-mlir third_party/dacapo/review_artifacts/mlp/mlp.traced.earth.mlir \
  --plan-format=json \
  --output-dir third_party/dacapo/review_artifacts/mlp/host

python3 integrations/dacapo/generate_model_artifacts.py mlp \
  --operator-spec docs/operator-spec/v2/profiles/dacapo-heaan-cpu.v1.json \
  --traced-mlir third_party/dacapo/review_artifacts/mlp/mlp.traced.earth.mlir \
  --device-counts=8x8 \
  --plan-format=json \
  --output-dir third_party/dacapo/review_artifacts/mlp/2x8

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target dacapo_plan_vec_diff --parallel 2

build/dacapo_plan_vec_diff \
  third_party/dacapo/review_artifacts/mlp/host/mlp.optimized._hecate_MLP.runtime-plan.json \
  third_party/dacapo/review_artifacts/mlp/2x8/mlp.optimized._hecate_MLP.runtime-plan.json \
  docs/operator-spec/v2/profiles/dacapo-heaan-cpu.v1.json \
  third_party/dacapo/review_artifacts/mlp/host/mlp.optimized._hecate_MLP.bundle \
  third_party/dacapo/review_artifacts/mlp/2x8/mlp.optimized._hecate_MLP.bundle \
  third_party/dacapo/review_artifacts/mlp/2x8/vec.diff.txt
```

差分工具也接受 `.bin`，但文件名必须与所选生成格式一致。它比较原 CKKS 指令与 Transfer 谱系，再用 Mock 的 `AllValuesAfterRun` 比较每条 Encode/Compute/Transfer 和 final output；没有 bundle 时对应参数传 `-`。它验证明文语义，不代替真实后端验收。

## 内存规划与 Waterline

模型生成脚本的 `--runtime-plan-memory --runtime-plan-memory-report` 插入 Release 和安全的 `reuse_input=0`，输出 V2 与 `.memory.json`、规划前的 `.memory.before.json`。`--runtime-plan-memory-no-reuse` 只插 Release；单独打开 report 只估算，不自动释放。`hecate-opt` 的关闭原位选项是 `--runtime-plan-memory-reuse=false`。

CPU 原位算子为 Negate/Rotate，GPU 为 AddCP/Rotate；Runtime 的手写计划还支持 GPU SubCP 原位。被覆盖值必须是同 Place、元信息相同、仅一次使用且非 final output 的内部计算结果。具体 Api 不支持时执行前报错。

V4 Waterline 通过 `hecate-opt` 的 `--runtime-plan-plaintext-schedule=waterline` 和显式 GPU/Host/pinned/raw 正预算开启；同时配置 `--runtime-plan-reserve-bytes`、`--runtime-plan-waterline-fraction`、`--runtime-plan-prepare-backend=cpu|gpu|auto` 与 `--runtime-plan-lookahead-groups`。当前通用模型生成脚本没有这些预算选项。完整字段及约束见 [V4 规范](../../docs/runtime-plan/v4.md)。Pass 顺序是保持不变的 HEFT placement、communication、Waterline Prepare、Release/reuse；Prepare 后不再运行 Encode CSE。不会按批克隆权重或插自动 Fence，普通 Encode/Transfer 和显式 Fence 仍可执行。

`.memory.json` 按顺序完成估算 RNS 对象，Host 系数按 8 字节、GPU 按 4 字节；`.waterline.json` 检查完整依赖单组基线，包含配置的消费者工作区上界。`--runtime-plan-workspace-bytes-per-op=0` 只做对象检查，不能保证算子实际空间。Runtime 再查询后端持久资源、准备峰值和 allocator capacity，在同一账本中检查单组及滚动准入。密钥、参数表、retained workspace、pool 外开销与余量必须被 reserve 覆盖；对象报告不能替代这些检查或整个进程的显存测量。

## Poseidon 联调数据

`generate_mlp_e2e_fixture.py`、`generate_mlp_gpu_e2e_profiles.py` 和 ResNet-20 对应脚本生成联调输入与 profile；查看各脚本 `--help` 选择参数和输出目录。`poseidon-8192`、`poseidon-65536` preset 含有实测、插值或代理代价，不能当作全部 level 都已实测。检查生成文件中的来源和支持范围。

历史跑分、某台机器的验收过程和当时的计划编号不作为当前使用说明维护；复现时查询对应提交及其产物记录。
