# RuntimePlan 与 manifest 二进制读取实验

这是独立实验工具 `runtime_binary_io_experiment`，生产 JSON reader、DaCapo 导出器和 Poseidon 执行入口继续使用已有协议。没有自动格式探测或读取失败后的切换，也没有新增第三方依赖。

目的是区分文本解析、typed plan 构建、元数据摘要、全局 Verifier 和文件读取的成本。二进制仍解码成现有 `RuntimePlan` 和 manifest 内容索引，没有改变执行器的数据结构或算子语义。

## 实验编码

两个容器采用显式小端编码，不写入 C++ struct、variant、指针或内存填充字节：

| 部分 | 编码 |
| --- | --- |
| 共同头部 | 8 字节 magic、uint32 实验版本 1、uint32 元数据 JSON 长度、少量根元数据 JSON |
| plan magic | `PLNEXP01` |
| manifest magic | `MNFEXP01` |
| plan 字符串表 | uint64 条数；每条 uint32 字节长度和 UTF-8 字节 |
| plan 各数组 | values、external_inputs、initialization、execution、finalization、final_outputs；每个数组先写 uint64 条数 |
| ValueDesc | 35 字节：整数 ID、枚举 kind/place、rank/index、字符串表编号、level/scale、NTT、components |
| 指令 | uint64 ordinal、uint8 指令类型，再按 Encode/Compute/Comm/Release/Fence 写整数、属性、数组与字符串表编号 |
| manifest 条目 | 32 字节原始 content 摘要、uint64 offset、uint64 byte_length，共 48 字节；条目前写 uint64 数量 |

plan 的重复 context、bundle content 和 Boot profile 共用字符串表。操作码和 variant 类型使用当前模型定义的顺序，严格受实验版本限制。描述符和指令不再保存重复字段名，也不再将整数 ID 转成十进制文本。根元数据保留小型 JSON 以复用已有约束；它不是主要体积或时间来源。

读取按预先知道的数量分配 typed 数组和 manifest 哈希表，再直接填入成员。仍校验 magic/version、计数、短读、尾部字节、枚举、布尔值、字符串编号、有限浮点和 manifest 覆盖范围。实验限制每个 typed 数组最多 2 GiB、每个字符串最多 1 MiB。完整不可信输入的协议审计、稳定操作码分配、节目录、外部摘要绑定和可配置总内存预算尚未完成，不能作为发布格式。

## 构建与运行

```bash
cmake -S . -B build-binary -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE='-O2 -DNDEBUG' \
  -DCKKS_RUNTIME_BUILD_TESTS=ON -DCKKS_RUNTIME_BUILD_EXPERIMENTS=ON
cmake --build build-binary --target runtime_binary_io_experiment -j4
build-binary/runtime_binary_io_experiment self-test . OPTIONAL_SMALL_V3_PLAN

build-binary/runtime_binary_io_experiment convert-plan PLAN.json PLAN.bin
build-binary/runtime_binary_io_experiment plan-json PLAN.json OPERATOR_SPEC.json
build-binary/runtime_binary_io_experiment plan-binary PLAN.bin OPERATOR_SPEC.json

build-binary/runtime_binary_io_experiment convert-manifest manifest.json manifest.bin
build-binary/runtime_binary_io_experiment manifest-json manifest.json
build-binary/runtime_binary_io_experiment manifest-binary manifest.bin
```

转换命令只处理计划/manifest 元数据，不访问权重 blob。plan 转换先用严格 JSON reader 构建 typed plan，再写二进制。manifest 转换逐条比较读回的内容标识、offset、长度和根元数据。工具是实验入口，输出路径应使用独立目录；没有生产导出器的 staging 发布功能。

加载时间包含文件读取、同一 SHA-256 实现的原始元数据摘要、解析和 typed 构建。`verify_seconds` 单独运行同一 PlanVerifier。manifest 两种加载方式均建立相同 key/value 类型的索引并检查完整范围；二进制的数量信息允许提前 reserve。没有读取或校验权重载荷。

小型正确性测试将读回计划投影为 JSON，与源样例逐项比较，涵盖六个 V1 fixture 和可选真实 V3 计划。包括 inline/bundle、Compute 属性、Transfer/Replicate、Release/reuse/Fence。测试也拒绝截断、尾部垃圾和未知 magic。大样本由独立计算图复制得到，JSON/二进制均通过同一 PlanVerifier；不将它当作完整模型执行或完整 Qwen 格式支持。

## 后续接入的边界

实验 plan 保留原来指向 JSON manifest 的引用；二进制 manifest 单独测量，不是已经绑定到生产 runtime 的新 bundle 入口。若正式采用，需要先明确存储格式与摘要绑定，再同时接入 DaCapo 和 runtime，提供显式兼容入口和诊断导出。现有 pack、内容 ID 与 Encode/Fence 语义可以继续复用。

二进制改善元数据加载和文件体积；同一 typed plan 上的 Verifier、索引、任务构建、CKKS Encode、GPU 计算仍需各自测量。不能将加载阶段的倍率外推为完整推理加速。
