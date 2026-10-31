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

optimized 对每个候选切分点只保留累计估算延迟最低的一份方案，并携带所有存活值（包括绕行值）的实际 level/scale 继续推导。不按不同出口状态保留多份前缀，也不保留旧多状态模式或失败时的备用搜索。较贵但更利于后续计算的前缀可能被丢弃，因此可能漏掉可行方案；目标仍是估算延迟，不是直接最小化 Boot 数。

默认 `threshold` 策略的 native rescale 时机由当前位置的实际素数决定：只有 `s - log2(q_last) >= waterline` 才继续删除一个 Q。greedy、depth-dp 和 optimized 使用相同的状态推导；分段候选保留实际浮点 scale。整数 IR 类型用于类型检查和编码、对齐的整数部分，Rescale 结果类型取实际指数的向上取整；计划和运行时保留实际浮点指数。物理转换重新计算并核对出口 scale、实际模数容量和规划所用的 Q 链。native 路径不允许 EarlyModswitch 越过 Rescale，也不执行用固定步长抵消 Rescale 的改写。

规划、物理转换和 RuntimePlan 导出共用 `hecate/Support/ModulusChain.h` 的实际素数计算。每条链预计算一次 `log2(q)` 和前缀和，耗时、内存均为 O(L)；容量和指定区间的 Rescale 降幅查询为 O(1)。阈值归约按实际插入顺序逐次推导，耗时为 O(1 + RS 次数)。Boot 入口对齐共用 `BootstrapInputAlignment.h`，用 O(log L) 找到满足编码精度的最小 Q 前缀，并统一检查容量；已对齐的输入也必须满足目标 Q 的容量要求。`DacapoModulusChain` 测试用逐项计算和穷举核对这些结果。

native 导出在物理 Q 和实际 scale 确定后、设备分配前，默认运行 `coalesce-reduction-rescales`。对于相同来源 Q、实际 scale、类型和 Rescale 序列的 MulCP 乘积，识别直接的 AddCC 归约，把各分支的 Rescale 合并到归约出口。k 个乘积、每支 r 次 RS，可从 k×r 次变为 r 次；出口 Q、实际 scale 和加法顺序保留，路径上的 Q 消耗不因此减少。Boot 位置规划在这之前完成，这个 pass 不与 Boot 联合搜索；逻辑 CKKS 检查点仍保留前面 scale 管理阶段生成的 RS 布局。

该 pass 处理逐 Q 的 RS 链，不越过旋转、ModSwitch、Boot、其他乘法或 `sm_fixed` 操作。共享的 RS 支路保持原样；共享加法子树可以在自身出口归约，再供多个用户使用。每个部分和检查实际模数容量，并额外为 k 个归约项预留 `log2(k)` 位；这项检查沿用当前 scale 容量模型，不替代输入值域和完整误差分析。只有省下的 RS 估算延迟大于高 Q 加法增加的延迟，才按 OperatorSpec 代价表合并。更少的 RS 不保证任意多 GPU 调度的总耗时或峰值内存都下降；RS 的舍入位置改变，密态结果也不要求逐位相同。

可用 `--runtime-plan-coalesce-rescales=false` 做对照；独立运行时使用 `coalesce-reduction-rescales{operator-spec=PATH}`，输入必须是未分配设备的物理 native CKKS。扫描缓存每个节点的固定大小摘要，修改归约树时使用显式栈，不展开所有前缀或递归遍历长加法链；时间和额外内存为 O(V+E+L)。`DacapoReductionRescales` 覆盖兼容性、容量、共享用户及三种 Boot 策略的完整导出；`test/runtime-plan/benchmark-reduction-rescales.py --hecate-opt PATH --source-dir third_party/dacapo --output-dir /tmp/reduction-bench` 可单独测约一万、十万和百万条已管理指令，耗时包含解析、验证和输出，pass 自身耗时另见生成的 timing 日志。

手动运行 native `--dacapo-precompile` 时，必须同时传入 `--runtime-plan-physical-level-operator-spec-path`、`--runtime-plan-boot-profile` 和匹配 profile 的 `--bootstrap-scale`，使 Boot 规划阶段就能读取实际 Q 链和出口约定；完整模型生成命令自动传入这三项。未指定 native OperatorSpec 的逻辑预编译继续使用 compiler profile 的固定步长。生成的 native 检查点保留规划 Q 链、Boot 出口位置和实际 scale，后续导出必须匹配。

可用 `python3 third_party/dacapo/test/runtime-plan/check-native-planning.py --hecate-opt third_party/dacapo/build/nix/bin/hecate-opt --source-dir third_party/dacapo` 检查实际素数对 rescale 次数和 Boot 候选可行性的影响，以及混合链的物理转换和错误位宽拒绝行为。

可用 `python3 tests/dacapo_native_artifact_tests.py --hecate-opt third_party/dacapo/build/nix/bin/hecate-opt` 检查三个规划算法在全 30 位、30/31/32 位混合链和手动 ResNet20 Q50 链上的标准生成路径，并验证旧标量因子不影响 native 计划。手动 Q50 链来自 `tests/fixtures/resnet20-manual-q50.json`，该用例只编译普通算子，要求不出现 Boot；这里的输入锚点 profile 不用于验证真实 Boot 的输出层数或精度。测试使用显式 trace，无需运行模型前端或 GPU。

手动 ResNet20 Boot 可用 `python3 integrations/dacapo/generate_resnet20_manual_profiles.py /tmp/manual-q50` 生成编译参数和 OperatorSpec。真实入口是 Q6（Runtime level 5）、`input_scale_log2=40`，出口是 Q31（level 30）、scale40。代价表是测试占位数，不是性能测量。GPU 资源必须由 Poseidon 的 `GpuBootstrapProfileBuilder::build_resnet20_manual()` 按同一 Q50/P25 fixture 创建；它执行 S2C-first、degree59/DA2 和系数折叠后的 scale40 出口。

对于声明输入 scale 的 native profile，规划按每个值的实际 level/scale 检查对齐可行性，不固定预留两个 Q。Q31 到 Q6 因而最多可用于 25 次逐 Q Rescale；只有末端 scale 已对齐时才能用满。物理转换与规划共用入口对齐计算，在 Boot 前插入必要的 ModSwitch，以及编码值为 1 的明文乘法和实际除 Q 的 Rescale；编码 scale 至少保留 30 位。已对齐时不插入乘法/Rescale，否则按实际需要消耗一个或更多 Q。若 Q 或模数容量不足则拒绝。可用 `python3 tests/dacapo_manual_boot_tests.py --hecate-opt PATH --output-dir /tmp/manual-boot-check` 检查三种规划算法和入口对齐边界。

默认初始密文与 Boot 出口使用相同 Q 前缀。`generate_model_artifacts.py --input-level=33`（或 hecate-opt 的 `--native-input-level=33`）可让初始密文使用同一实际链的 Q34，Boot 出口仍按 profile 使用 Q31。该值不得低于 Boot 出口或超过 OperatorSpec 上界，实际加密输入必须匹配。检查点独立保存初始锚点和 Boot 出口；此选项不会延长每次 Boot 后的 25Q 预算。可运行 `python3 tests/dacapo_boot_budget_tests.py --hecate-opt PATH --output-dir /tmp/boot-budget-check` 检查满 25Q、按需入口对齐、不同初始前缀和已删除选项的明确拒绝。

上述测试还生成 `scaled-depth-dp` 用例：保留非零的 `0.5*x` 支路，使它在 Boot 前确实需要 scale 对齐。可用 Poseidon 的 GPU 测试程序执行该计划：

```bash
POSEIDON_RUNTIME_BOOTSTRAP_MANUAL_Q50="$PWD/tests/fixtures/resnet20-manual-q50.json" \
  /path/to/poseidon_runtime_gpu_plan_e2e \
  /tmp/manual-boot-check/scaled-depth-dp/mlp.optimized.squares.runtime-plan.json \
  /tmp/manual-boot-check/operator-spec.json /tmp/manual-gpu-report.json
```

手动 profile 保留 `arcsine_degree=0`，其输出是正弦近似。测试使用独立的实数运算参考，分别报告相对该映射的执行误差和相对恒等 Boot 的近似误差；前者阈值为 `1e-4`。对齐 scale40 不代表消除了这项正弦近似误差，也不等于已经通过完整 ResNet20 的密态精度验证。

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
