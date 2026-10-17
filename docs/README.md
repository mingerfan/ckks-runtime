# CKKS Runtime 文档

这里维护当前代码的总体说明和执行协议。开发方案、阶段进度、实验过程和旧版文档由 Git 历史保存，不在当前目录另建归档。修改代码时同步修改相关 overview 和规范；规范中的版本表说明当前 reader 实际支持的格式。

| 内容 | 文档 |
| --- | --- |
| 项目职责、编译与执行流程、后端边界 | [Overview](overview.md) |
| RuntimePlan 字段、算子、Release/reuse、显式 Fence、V4 Waterline 和兼容规则 | [RuntimePlan 规范](runtime-plan/README.md)、[V4](runtime-plan/v4.md) |
| 默认二进制计划、manifest 编码和摘要行为 | [二进制格式](runtime-plan/binary.md) |
| float64 载荷、逐文件与 pack 布局、装载方式 | [明文数据包规范](runtime-plan/plaintext-bundle.md) |
| CKKS 参数、算子能力、Boot profile 与代价表 | [OperatorSpec 规范](operator-spec/README.md) |
| Dacapo 构建、生成模型和差分检查 | [Dacapo 集成](../integrations/dacapo/README.md) |
| 编译期 placement 通信代价和拓扑规则 | [通信 profile 规范](../integrations/dacapo/communication-profile.md) |

Schema、测试样例和 profile 是程序与测试使用的数据，保留在各自的版本目录中。文字规范统一维护在上表中的文档，不为每个开发步骤复制一份。

阅读代码可从 [`plan.hpp`](../runtime/plan.hpp)、[`plan_reader.cpp`](../runtime/plan_reader.cpp)、[`verifier.cpp`](../runtime/verifier.cpp) 和 [`runtime.hpp`](../runtime/runtime.hpp) 开始。构建命令见[项目 README](../README.md)。
