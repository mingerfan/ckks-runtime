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

当前 `emit-runtime-plan` Pass 输出单 Host RuntimePlan V1，文件名是 `<prefix>.<func>.runtime-plan.json`。target、OperatorSpec 引用和 context 必须通过 Pass option 明确提供。大常量 bundle 外化、OperatorSpec 读取、placement 和通信仍留给后续 Pass。
