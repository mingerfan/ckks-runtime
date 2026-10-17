# CKKS Runtime

这是面向多设备 CKKS 推理的执行器。编译器确定计算位置和数据搬运，Runtime 验证并执行计划，具体计算与通信由 Api 提供。

当前支持 RuntimePlan V1/V2/V3/V4、inline/bundle Encode、Host/Device compute、Transfer/Replicate、Release、受支持算子的原位执行、显式 Fence 与按需明文装载。[V4 Waterline](docs/runtime-plan/v4.md) 使用 Device Prepare 和完整消费者组准入；CPU/GPU 完整 Encode 与简单 Auto 选择复用同一设备工作线程执行器，不插自动批末 Fence。计划读取入口支持二进制和 JSON；Dacapo 默认导出二进制计划和 pack bundle。Vec/Mock/MPI 提供参考后端，真实 CPU/GPU 后端在 Poseidon 中实现。

## 文档与代码

- [文档入口](docs/README.md)与 [Overview](docs/overview.md)：项目职责、管线和后端边界。
- [RuntimePlan](docs/runtime-plan/README.md)、[二进制格式](docs/runtime-plan/binary.md)、[bundle](docs/runtime-plan/plaintext-bundle.md) 和 [OperatorSpec](docs/operator-spec/README.md)：当前执行协议。
- [`plan.hpp`](runtime/plan.hpp)、[`plan_reader.cpp`](runtime/plan_reader.cpp)、[`verifier.cpp`](runtime/verifier.cpp)、[`runtime.hpp`](runtime/runtime.hpp)：计划、读取、验证与执行。
- [Dacapo 集成](integrations/dacapo/README.md)：模型生成、内存规划和差分检查。

文档随代码维护；开发草案、阶段记录和旧文档保存在 Git 历史中。

## 构建与测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

独立构建默认开启单元测试和实验工具。作为其他 CMake 工程的子目录时，默认生成 `ckks_runtime::core` 和 `ckks_runtime::plan_json` 两个库；用 `CKKS_RUNTIME_BUILD_TESTS`、`CKKS_RUNTIME_BUILD_EXPERIMENTS` 显式控制额外目标。核心库使用 C++17、Threads 和仓库内的 nlohmann-json；测试还需要 Python。

MPI 是可选依赖：

```bash
cmake -S . -B build-mpi \
  -DCMAKE_BUILD_TYPE=Release \
  -DCKKS_RUNTIME_BUILD_MPI=ON
cmake --build build-mpi --parallel 2
ctest --test-dir build-mpi --output-on-failure
```

## 可选 Dacapo 编译器

Dacapo 只用于编译器集成和端到端测试，不参与 Runtime 默认构建。需要时初始化：

```bash
git submodule update --init third_party/dacapo
```

固定的是 `mingerfan/dacapo-modified` fork。它是单独的 CMake/MLIR 工程，构建步骤见 [Dacapo README](third_party/dacapo/README.md)。
