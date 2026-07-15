# 实现状态

## 阶段一已完成

- RuntimePlan V1、OperatorSpec V1 和 plaintext bundle manifest 已有一致的规范、Schema、profiles 和生成样例。
- reader 严格拒绝 BOM、重复 key、未知/缺失字段、错误类型和不支持版本，并对完整原始文件字节计算 SHA-256。
- RuntimePlan 已支持 inline/bundle Encode；bundle loader 每个 rank 只读取本地实际需要的 blob。
- verifier 覆盖 SSA、阶段、Place、完整 CKKS 元信息变化、OperatorSpec 边界、Boot profile、Rotate step 规范化和具体密钥需求。
- Runtime 在执行前完成 OperatorSpec/bundle/preflight，执行 Host/Device compute、Encode 和显式通信，并对输入及 Api 输出做完整元信息检查。
- Mock 和 MPI preflight 使用计划原始字节摘要，并检查所有 rank 的调试开关一致。
- Vec/Mock 测试覆盖同步、异步、多 rank、摘要调试策略、bundle 局部装载和 fail-fast；MPI 测试覆盖端到端和摘要一致性。

仓库内 OperatorSpec 仍是 `placeholder`，只用于协议测试。

## 后续阶段

- 阶段二：Dacapo submodule 和 RuntimePlan 生成 Pass。
- 阶段三：PoseidonCpuApi、PoseidonGpuApi 和生产测量 profile。

这些内容不属于 RuntimePlan V1 冻结条件。
