# CKKS Runtime

这是面向多设备 CKKS 推理的执行器。编译器确定计算位置和数据搬运，Runtime 验证并执行计划，具体计算与通信由 Api 提供。

当前支持 RuntimePlan V1/V2/V3/V4、inline/bundle Encode、Host/Device compute、Transfer/Replicate、Release、受支持算子的原位执行、显式 Fence 与按需明文装载。[V4 Waterline](docs/runtime-plan/v4.md) 使用 Device Prepare 和完整消费者组准入；CPU/GPU 完整 Encode 与简单 Auto 选择复用同一设备工作线程执行器，不插自动批末 Fence。计划读取入口支持二进制和 JSON；Dacapo 默认导出二进制计划和 pack bundle。Vec/Mock/MPI 提供参考后端，真实 CPU/GPU 后端在 Poseidon 中实现。

`SequentialRuntime` 的默认模式仍为 `Sequential`。选择 `PerDeviceWorkers` 且 `local_devices=0` 时，设备 worker 数量保持为零，额外的 Host 队列由一个计算 worker 执行；`device_worker_count` 应保留为零。纯 CPU 的 initialization、execution、finalization 都复用该执行器，不要求 Waterline。各队列严格按计划顺序执行，输入未发布时等待队首任务，CPU Api 同步返回即完成。

跨 rank worker 通信支持 Host→Host 和 Device→Device。未声明后台通信提交能力的 Api 由调用线程依次提交并完成通信组，接收结果由该线程发布；MPI Api 的调用线程必须是初始化 MPI 的线程。该路径不把所有发送先提交后统一等待：CPU 接收只有在 `wait()` 中才提交正文接收，先等待某个发送输入可能使其依赖的早先接收一直得不到正文。逐组完成时，各 rank 的通信序列都遵循相同的计划顺序；当前组的输入生产者只依赖更早定义的值，更早的跨 rank 接收已由主线程完成。因此有序计算队列与通信队列之间不会形成循环等待，显式 Fence 和阶段边界也会先完成当前组。

Release 在未提交用途归零后才清除 runtime 引用；异步 Api 必须自行持有已接受发送所需的数据。Poseidon CPU 在提交时同步序列化，通信句柄保留发送字节直到 MPI 完成，随后释放原计算值不会改变消息。verifier 要求 `reuse_input` 的输入在整个计划中恰好只有一次使用，禁止发送后再原位覆盖同一个值。异常会唤醒依赖等待者并在收拢 worker 后由主线程报告/终止分布式作业；当前通信组中的阻塞 MPI wait 仍沿用原有语义，不提供本地取消或通信轮询。

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
