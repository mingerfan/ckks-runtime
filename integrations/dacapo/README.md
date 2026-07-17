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

当前 `emit-runtime-plan` Pass 输出单 Host RuntimePlan V1，文件名是 `<prefix>.<func>.runtime-plan.json`。Dacapo 的 CKKS MLIR 使用纯 SSA result-style，不再生成 `dst` 和 `tensor.empty`。target、OperatorSpec 引用和 context 必须通过 Pass option 明确提供。Encode payload 默认以 4096 字节为界：不超过阈值的常量内联到 JSON，超过阈值的 float64 常量写入 `<prefix>.<func>.bundle/`；相同内容按 SHA-256 复用同一个 blob。直接读取 OperatorSpec、placement 和通信仍留给后续 Pass。

## 生成模型审阅产物

`generate_model_artifacts.py` 从选定的 OperatorSpec V2 读取 target、spec id/version、context 和 Boot 配置，并根据 OperatorSpec 文件的完整原始字节计算 RuntimePlan 使用的 SHA-256。它还会校验 `provenance` 指向的旧 Dacapo profile，摘要不一致时直接失败。

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
