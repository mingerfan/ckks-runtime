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

不传 `--device-counts` 时生成原来的单 Host 计划。传入后，编译器用 OperatorSpec V2 的逐 level 延迟做确定性 HEFT placement，再为跨 Place 操作数插入 1 对 1 的 `point_to_point` Transfer。`8` 表示 1 rank × 8 devices，`8x8` 表示 2 ranks × 8 devices，`0x0` 表示 2 个只有 Host 的 CPU rank。零和正数不能混用。通信代价目前是与算子延迟同一调度单位的两个固定整数，只是占位值，不按字节数计算，也不是实测传输耗时。

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
  --output-dir third_party/dacapo/review_artifacts/mlp/1x8

python3 integrations/dacapo/generate_model_artifacts.py mlp \
  --operator-spec docs/operator-spec/v2/profiles/dacapo-heaan-cpu.v1.json \
  --traced-mlir third_party/dacapo/review_artifacts/mlp/mlp.traced.earth.mlir \
  --device-counts 8x8 \
  --output-dir third_party/dacapo/review_artifacts/mlp/2x8
```

构建通用 VecApi 差分程序：

```bash
xmake build dacapo_plan_vec_diff
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
