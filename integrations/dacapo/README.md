# Dacapo 集成

这里放 Runtime 与 `third_party/dacapo` 的集成脚本和后续端到端测试。

当前已有 profile 迁移脚本：

```bash
git submodule update --init third_party/dacapo
python3 integrations/dacapo/migrate_profiles.py --check
```

脚本从 provenance 固定的 Dacapo revision 读取三份旧 profile，再转成 `docs/operator-spec/v2/profiles/` 下的版本化 OperatorSpec。当前 submodule 可以继续前进，不会改变这批已迁移 profile 的来源或输出。

Dialect 和 RuntimePlan 管线可以独立构建、测试：

```bash
cd third_party/dacapo
nix develop --command cmake --preset nix
nix develop --command cmake --build --preset nix
nix develop --command ctest --test-dir build/nix --output-on-failure
```

`emit-runtime-plan` 输出 RuntimePlan V1，文件名是 `<prefix>.<func>.runtime-plan.json`。Dacapo 的 CKKS MLIR 使用纯 SSA result-style，不再生成 `dst` 和 `tensor.empty`。target、OperatorSpec 引用和 context 必须通过 Pass option 明确提供。Encode payload 默认以 4096 字节为界：不超过阈值的常量内联到 JSON，超过阈值的 float64 常量写入 `<prefix>.<func>.bundle/`；相同内容按 SHA-256 复用同一个 blob。

当 OperatorSpec 声明 `rescale_mode=lazy` 时，生成脚本会在 Earth→CKKS 之后、placement 之前自动加入 `materialize-ckks-physical-levels`。默认一个逻辑 level 展开为 4 个物理 RNS level，可用 `--lazy-rescale-level-factor` 显式调整。Pass 使用目标 spec 的 `levels.upper_bound` 作为物理起始 level，并检查每条 Rescale 的 scale 降幅是否等于被丢弃模数的 bit 数之和；因此 Poseidon GPU 的 Dacapo compiler profile 必须把 `rescalingFactor` 设为 120，目标 OperatorSpec 必须记录真实的 30-bit 模数链并允许一次 Rescale 至少下降 4 层。只改 JSON writer 或只把 profile 标成 lazy 都不够。

不传 `--device-counts` 时生成原来的单 Host 计划。传入后，编译器用 OperatorSpec V2 的逐 level 延迟做确定性 HEFT placement，再为跨 Place 操作数插入 1 对 1 的 `point_to_point` Transfer。`8` 表示 1 rank × 8 devices，`8x8` 表示 2 ranks × 8 devices，`0x0` 表示 2 个只有 Host 的 CPU rank。零和正数不能混用。

placement 可通过 `--communication-profile` 使用按 CKKS value 大小估算的通信代价。每类链路配置启动延迟、速率上限和曲线饱和尺寸，也可以提供 `payload_bytes`/`rate_bytes_per_us` 点位表做线性插值。完整格式和公式见 [Placement 通信 Profile](communication-profile.md)。不传该选项时保留原来的两个固定整数，只用于兼容已有命令和测试。

GPU MLP 的 8192 实机校准 profile 可用下面的 preset 生成。它保留
`synthetic` 默认 preset，不覆盖现有 32768/65536 产物：

```bash
python3 integrations/dacapo/generate_mlp_gpu_e2e_profiles.py \
  --poly-degree 8192 \
  --profile-preset poseidon-8192 \
  --output-dir third_party/dacapo/review_artifacts/mlp/8192/profiles
```

`poseidon-8192` 的最高物理 level 来自 RTX 4060 Laptop GPU 上的 Release
实测：degree=8192、Q=17、P=2，每轮 20 次预热和 100 次计时，取三轮 GPU
wall time 中位数后按 OperatorSpec 的整数微秒精度取整。L1-L16 仍按温和曲线
估计，最低 level 约为最高 level 的 80%，不代表逐 level 实测。ModSwitch
当前没有独立 benchmark，暂用实测的 HYBRID BConv/ModUp 作为代理。

同一台 GPU 上的 8192 单卡 MLP 校准使用最终 1×1 RuntimePlan，包含 375 次
Compute、122 次 Host→Device plaintext Transfer 和 111 个 rotation key。该 cost
profile 下 Dacapo 选择了 0-Boot 图，因此运行时需显式设置
`POSEIDON_GPU_MLP_ALLOW_NO_BOOT=1`；默认 E2E 行为仍要求至少一次 Host Boot。
三轮均通过 Python/Mock 数值校验：

Runtime 现在把时间拆成 setup、initialization 和 online execution。initialization
包含 Encode 以及 122 次 plaintext upload，并在边界同步完成；online execution
包含 execution、finalization、通信完成和最终输出同步。外层 Runtime wall 仍保留为
完整冷启动墙钟时间。使用 `sm_89` Release 构建重新运行三轮：

| run | Runtime wall (ms) | Setup (ms) | Initialization (ms) | Online (ms) | Compute (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 504.762 | 31.118 | 115.044 | 358.593 | 350.667 |
| 2 | 425.653 | 4.885 | 101.751 | 319.012 | 312.238 |
| 3 | 406.849 | 4.932 | 112.317 | 289.595 | 283.658 |
| 中位数 | 425.653 | 4.932 | 112.317 | 319.012 | 312.238 |

最终 placement MLIR 的最大 `dist.schedule_finish` 是 61,688 us。正确的比较对象是
319.012 ms 的 online 中位数，实测约为静态跨度的 5.17 倍，而不是原先用完整
Runtime wall 得到的 6.87 倍。生成阶段打印的 `Estimated Latency: 0.143722 sec`
是 Earth 优化阶段的旧 latency 汇总，不是最终 placement 跨度。

剔除 plaintext upload 后差距仍然较大，且 online 与 Compute 只差约 6.8 ms。
因此主要误差在逐 Compute 路径：算子微基准复用预分配输出，而 Runtime 为每个
SSA value 分配 GPU storage、记录 completion event。不能再把初始化上传解释为全部
差距，也不能用统一缩放因子回填单算子 profile；后续应分别建模 allocation/event
和 Host→Device 传输。

## 32768 CPU 与 2-rank MPI 复测

CPU 计划使用 degree 32768、14 个 60-bit Q limb 和 398 次 Compute。单 rank
计划有 122 次初始化 Encode，没有 Transfer；2-rank 计划有 62 次初始化期 Transfer
和 223 次执行期 Transfer。两者使用同一个 `sm_89` Release+MPI 构建顺序运行三轮，
均通过 Python/Mock 数值校验：

| 拓扑 | run | Runtime wall (ms) | Setup (ms) | Initialization (ms) | Online (ms) | Critical compute (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 rank | 1 | 3101.351 | 18.636 | 215.540 | 2867.173 | 2865.750 |
| 1 rank | 2 | 3102.800 | 5.873 | 216.722 | 2880.202 | 2878.861 |
| 1 rank | 3 | 3152.117 | 5.800 | 218.978 | 2927.337 | 2925.870 |
| 2 ranks | 1 | 3238.153 | 17.403 | 248.764 | 2972.418 | 1570.701 |
| 2 ranks | 2 | 3085.930 | 6.137 | 236.462 | 2843.467 | 1553.760 |
| 2 ranks | 3 | 3084.814 | 6.008 | 241.424 | 2837.570 | 1547.781 |

单 CPU placement profile 的 398 个算子成本串行求和是 2,234,355 us；2-rank
placement MLIR 的最大 `dist.schedule_finish` 是 1,728,637 us。与 online 中位数
比较如下：

| 拓扑 | 模拟 (ms) | 实测 online (ms) | 实测 / 模拟 |
| --- | ---: | ---: | ---: |
| 1 rank | 2234.355 | 2880.202 | 1.289x |
| 2 ranks | 1728.637 | 2843.467 | 1.645x |

模拟预计 1.293x 加速，实测只有 1.013x。2-rank 中位数的 online 比 critical
compute 多 1,289.707 ms，这部分主要是通信和依赖等待。当前 CPU MPI placement
仍为每条跨 rank 依赖使用固定 10,000 us，不按 ciphertext 的 level、component
和传输字节数计算，也不建模 MPI 阻塞、链路争用或大量小传输的串行化。

计算 profile 本身也不是统一倍率误差：静态调度分给 rank 0/1 的算子成本分别是
1,407.383/826.972 ms，而中位数实测为 1,553.760/1,419.564 ms；模拟 Boot 是
500 ms，实测约 29 ms。应补测 Poseidon CPU 的逐 level 算子曲线，并为 CPU MPI
增加 payload-aware 通信 profile，不能用一次 MLP wall time 整体缩放当前 profile。

GPU MLP 的 65536 degree 估算 profile 可用下面的 preset 生成：

```bash
python3 integrations/dacapo/generate_mlp_gpu_e2e_profiles.py \
  --poly-degree 65536 \
  --profile-preset poseidon-65536 \
  --output-dir third_party/dacapo/review_artifacts/mlp/65536/profiles
```

`poseidon-65536` 使用 GPU 表格中 L40 的算子 wall time 作为最高物理 level；
表格的 L1-L40 对应 Runtime 的 level 0-39（即 `level + 1` 个 RNS limb）。
L1-L39 按温和曲线填充，最低 level 约为最高 level 的 80%，用于 placement
估算而不是宣称已逐 level 实测。`multiply_relinearize_rescale` 的 500 us
拆成 `mul_cc=20 us`、`relinearize=370 us`、`rescale=110 us`；表格没有直接
给出 ModSwitch，因此暂以 `keyswitch_bconv_modup=40 us` 作为代理。

通信 profile 采用 count=32 带宽表中的 `object_loop_a2e` 点位，payload 按
MiB 转换为 bytes，`contiguous_buffer` 的 395.7 GB/s 作为速率上限。该 profile
仍是多 GPU placement 的粗粒度估算：点位来自批量 count=32 测试，当前调度器
尚未建模具体 transfer batch、链路争用和异步重叠。

## 生成模型审阅产物

`generate_model_artifacts.py` 从选定的 OperatorSpec V1 或 V2 读取 target、spec id/version、context 和 Boot 配置，并根据 OperatorSpec 文件的完整原始字节计算 RuntimePlan 使用的 SHA-256。V2 默认校验 `provenance` 指向的旧 Dacapo profile；没有这份 provenance 时必须显式传入 `--compiler-profile`。多个 Boot profile 必须用 `--boot-profile` 选定一个。placement 仍只接受 V2。

生成 MLP：

```bash
python3 integrations/dacapo/generate_model_artifacts.py mlp \
  --operator-spec docs/operator-spec/v2/profiles/dacapo-heaan-cpu.v1.json
```

生成 ResNet-20：

```bash
python3 integrations/dacapo/generate_model_artifacts.py resnet20 \
  --operator-spec docs/operator-spec/v2/profiles/dacapo-heaan-gpu.v1.json
```

默认输出到 `third_party/dacapo/review_artifacts/<model>/`，该目录不会进入 Git。大常量同时生成 `.bundle/manifest.json` 和 `.bundle/data/*.bin`。可以用 `--inline-payload-max-bytes` 调整阈值；设为 `0` 会外化所有非空 float64 Encode payload。Python tracing 需要安装 `requirements.txt` 中的模型依赖；Dacapo C++ 编译器仍可独立离线构建，不把 PyTorch 加入默认构建依赖。

如果已经有 Earth MLIR，可以跳过 Python tracing：

```bash
python3 integrations/dacapo/generate_model_artifacts.py mlp \
  --operator-spec docs/operator-spec/v2/profiles/dacapo-heaan-cpu.v1.json \
  --traced-mlir /path/to/MLP.mlir
```

使用 `--dry-run` 可以只检查 OperatorSpec、旧 profile 摘要并打印完整的 `hecate-opt` 命令，不执行编译。

## 生成并检查多 rank MLP

先保留一份 Host 参考计划，再把两个拓扑输出到不同目录：

```bash
python3 integrations/dacapo/generate_model_artifacts.py mlp \
  --operator-spec docs/operator-spec/v2/profiles/dacapo-heaan-cpu.v1.json \
  --traced-mlir third_party/dacapo/review_artifacts/mlp/mlp.traced.earth.mlir \
  --output-dir third_party/dacapo/review_artifacts/mlp/host

python3 integrations/dacapo/generate_model_artifacts.py mlp \
  --operator-spec docs/operator-spec/v2/profiles/dacapo-heaan-cpu.v1.json \
  --traced-mlir third_party/dacapo/review_artifacts/mlp/mlp.traced.earth.mlir \
  --device-counts 8 \
  --communication-profile /path/to/communication-profile.json \
  --output-dir third_party/dacapo/review_artifacts/mlp/1x8

python3 integrations/dacapo/generate_model_artifacts.py mlp \
  --operator-spec docs/operator-spec/v2/profiles/dacapo-heaan-cpu.v1.json \
  --traced-mlir third_party/dacapo/review_artifacts/mlp/mlp.traced.earth.mlir \
  --device-counts 8x8 \
  --output-dir third_party/dacapo/review_artifacts/mlp/2x8
```

构建通用 VecApi 差分程序：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target dacapo_plan_vec_diff --parallel 2
```

程序依次接收 Host 计划、分布式计划、OperatorSpec、两份 bundle 目录和报告路径。它先检查原 CKKS ValueId/算子/操作数与 Transfer 谱系，再用 `DiffMode::AllValuesAfterRun` 执行两份计划，比较每条 Encode、Compute、Transfer，最后单独再比较 final output。没有 bundle 时对应参数写 `-`。

```bash
build/dacapo_plan_vec_diff \
  third_party/dacapo/review_artifacts/mlp/host/mlp.optimized._hecate_MLP.runtime-plan.json \
  third_party/dacapo/review_artifacts/mlp/2x8/mlp.optimized._hecate_MLP.runtime-plan.json \
  docs/operator-spec/v2/profiles/dacapo-heaan-cpu.v1.json \
  third_party/dacapo/review_artifacts/mlp/host/mlp.optimized._hecate_MLP.bundle \
  third_party/dacapo/review_artifacts/mlp/2x8/mlp.optimized._hecate_MLP.bundle \
  third_party/dacapo/review_artifacts/mlp/2x8/vec.diff.txt
```
