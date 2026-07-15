# Dacapo 集成

这里放 Runtime 与 `third_party/dacapo` 的集成脚本和后续端到端测试。

当前已有 profile 迁移脚本：

```bash
git submodule update --init third_party/dacapo
python3 integrations/dacapo/migrate_profiles.py --check
```

脚本把固定 Dacapo revision 中的三份旧 profile 转成 `docs/operator-spec/v2/profiles/` 下的版本化 OperatorSpec。它只读取 submodule，不编译 Dacapo，也不参与 Runtime 默认构建。
