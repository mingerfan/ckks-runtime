# CKKS Runtime

这是一个面向多设备 CKKS 推理的执行器原型。编译器负责生成已经确定计算位置和数据搬运方式的计划，Runtime 负责验证并执行，具体计算和通信由 Api 实现。

> 当前代码已经能执行 Compute、Transfer 和 Replicate，并提供 Vec/Mock/MPI 测试后端。显式 Encode、inline/bundle 双 payload、OperatorSpec 完整校验和 Host compute 仍是目标设计，尚未实现。准确边界见[实现状态](docs/overview-design/implementation-status.md)。

目标数据流如下；当前 reader 仍读取上一版计划格式，也不会装载 OperatorSpec 或 bundle：

~~~text
Dacapo / 其他编译器
        |
        | RuntimePlan JSON + OperatorSpec + 可选 plaintext bundle
        v
RuntimePlanJsonReader -> PlanVerifier -> SequentialRuntime<Api>
                                             |
                            +----------------+----------------+
                            |                                 |
                        MockVecApi                        MpiVecApi
                     明文计算 + 模拟通信              明文计算 + MPI 通信

目标后续接入: PoseidonCpuApi / PoseidonGpuApi
~~~

## 从哪里开始看

- [设计文档总入口](docs/README.md)：术语、架构、协议和代码导航。
- [当前实现状态](docs/overview-design/implementation-status.md)：已经实现、尚未实现和测试覆盖。
- [目标总体架构](docs/overview-design/architecture.md)：编译器、Runtime 和 Api 的职责边界。
- [RuntimePlan V1 草案](docs/runtime-plan/v1/specification.md)：目标 JSON 协议；当前 Schema、样例和 reader 仍在迁移。
- [当前计划结构](runtime/plan.hpp)、[当前验证器](runtime/verifier.cpp)、[当前执行主线](runtime/runtime.hpp)：直接对照代码。

## 构建与测试

~~~bash
xmake f -m debug
xmake
./build/runtime_tests
./build/runtime_plan_json_tests
~~~

MPI 可用时还可以运行：

~~~bash
mpiexec -n 2 ./build/mpi_runtime_test
mpiexec -n 4 ./build/mpi_runtime_test
~~~

## 可选的 Dacapo 集成

Dacapo 只用于编译器集成开发和端到端测试，不参与 Runtime 的默认构建。普通开发不需要下载它；需要时再显式初始化固定版本：

~~~bash
git submodule update --init third_party/dacapo
~~~

Dacapo 是独立的 CMake/MLIR 工程，初始化 submodule 不会让 xmake 自动编译它。当前固定的是 `mingerfan/dacapo-modified` fork；后续阶段二的生成 Pass 和集成测试会基于这个目录接入。
