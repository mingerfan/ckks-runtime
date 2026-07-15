# Dacapo、Runtime 与 Poseidon 集成方案

> 本文描述目标集成结构和迁移顺序。当前仓库还没有 Dacapo submodule、`integrations/dacapo/` 或 Poseidon 后端；Runtime 自身的完成情况见[实现状态](implementation-status.md)。

## 1. 这份方案要解决什么问题

现在有三份代码,各自都在做"执行计划"这件事,职责重叠:

- **Dacapo**(`utils/dacapo`,即 hecate 编译器的 fork):把上层程序编译成 CKKS 程序,自带一套 HEVM 字节码和两个解释器(`lib/Runtime` 里的 SEAL/HEAAN 版本);
- **Poseidon**(FHE GPU 库):`src/poseidon/mgpu` 里长出了一整套独立的执行系统——自己的调度中间表示、设备分配、拷贝插入、验证器、执行器,外加 `frontends/dacapo` 里的 HEVM 导入和第三套执行器;
- **本仓库**(CKKS Runtime):一套干净的"照计划执行"框架,有验证器、值状态管理和明文参考实现,但还没接真实 GPU。

三套执行逻辑长期并存,每改一处要同步三处,谁也不敢删谁。本方案的目标是收敛成一条链,每个仓库只干一件事:

- **Dacapo 只做编译**:从上层程序生成一份"每条指令在哪算、数据怎么搬都定死了"的静态计划文件;
- **Runtime 只做执行**:读入计划文件,验证,然后照着执行。单卡、多卡、多机用同一套执行代码;计算和通信的具体实现通过一个抽象接口(下文称 Api)注入,明文、CPU、GPU 各实现一份;
- **Poseidon 只做算子和通信**:不再维护自己的计划解释器,改为实现 Runtime 需要的 CPU/GPU Api;
- Dacapo 和 Runtime 之间**只通过一个版本化的计划文件交流**,不共享任何 C++ 类型;
- Runtime 仓库固定一个已验证兼容的 Dacapo 版本(git submodule),用 MockVecApi/MpiVecApi 明文参考后端做"编译器产物真的能被执行器跑对"的端到端验证。

依赖方向是单向的:

~~~text
Dacapo
  CKKS 优化 / bootstrap 安排 / scale 管理
  -> 按目标配置做 CKKS 合法化
     |- CPU:普通 rescale
     |- GPU:按 OperatorSpec 启用 lazy-rescale
     `- 可选的 GPU boot -> CPU 解密再加密模拟
  -> 决定每条指令放在哪个进程、哪张卡(placement)
  -> 把所有数据搬运写成显式指令
  -> 输出 RuntimePlan 文件
                 |
                 v
Runtime
  读文件 -> 验证 -> 逐条执行
  -> 调 Api 完成实际计算和通信
       |- MockVecApi      (明文计算 + 模拟通信)
       |- MpiVecApi       (明文计算 + MPI 通信)
       |- PoseidonCpuApi  (Poseidon CPU 算子)
       `- PoseidonGpuApi  (GPU 算子 + 计划明确要求的 Host 算子)
~~~

Runtime 不反向依赖 Dacapo 或 Poseidon。Poseidon 只依赖 Runtime;Dacapo 只在生成计划和跑编译集成测试时出现。

## 2. 仓库怎么摆

目标仓库结构如下。当前仓库尚未加入 `integrations/dacapo/` 和 Dacapo submodule,它们属于迁移阶段二:

~~~text
ckks-runtime/
|- docs/runtime-plan/           # 计划文件格式的规范文档
|- docs/operator-spec/          # CKKS 算子参数、约束和代价模型的规范
|- runtime/                     # 计划类型、验证器、执行器
|- api/                         # VecExecutor、MockVecApi、MpiVecApi 等实现
|- testing/                     # 计划构造器、参考执行、差分比较
|- integrations/dacapo/         # 和 Dacapo 对接的胶水与集成测试
`- third_party/dacapo/          # submodule,只给集成测试用

poseidon/
|- src/poseidon/runtime_api/    # Runtime Api 的 Poseidon 实现
|  |- poseidon_cpu_api.*
|  |- poseidon_gpu_api.*
|  `- poseidon_communication.*
`- third_party/ckks-runtime/    # submodule
~~~

这样形成一条确定的版本链:Poseidon 的每个 commit 固定一个 Runtime 版本,Runtime 的每个 commit 固定一个 Dacapo 版本。任何时刻检出 Poseidon,都能精确复现出配套的整套工具链。

几点必须说清楚的现实约束:

- **Dacapo submodule 不参与日常构建**。平时编 Runtime 或 Poseidon 都不需要下载和编译 Dacapo(它带着 MLIR,很重);只有显式打开 Dacapo 集成测试的开关时才拉取和构建。
- **submodule 要指向我们的 fork**。实际在用的编译器是 `mingerfan/dacapo-modified`,不是上游 `corelab-src/dacapo`。Poseidon 现有的 `third_party/dacapo` submodule 指向上游,迁移时要改掉;而且迁移完成后 Poseidon 不应再直接挂 Dacapo submodule——它只认 Runtime,Dacapo 的版本由 Runtime 这一层锁定,避免出现两个各自漂移的 Dacapo 指针。
- **三层 submodule 链是有维护成本的**。一次跨仓库的修改要按"Dacapo 提交 → Runtime 更新指针并测试 → Poseidon 更新指针并测试"三步走,对一两个人的团队来说不轻。在协议(见下节)稳定之前,允许先用普通目录摆在一起开发,协议冻结后再正式拆成 submodule 链;但目录结构从现在起就按上面的样子摆,避免以后大挪移。

在 submodule 里直接改代码是允许的,但子仓库的文件必须提交回子仓库自己的 git;父仓库只记录"我验证过这个版本能配套工作"的指针。

## 3. 计划文件是一份契约,不是一组共享的 struct

RuntimePlan 文件要当成 Runtime 对外承诺的**指令集**来对待:文件里每个字段的含义、取值范围、合法组合,都白纸黑字写进规范;Dacapo 照规范生成,Runtime 照规范读取。就像 CPU 厂商和编译器厂商之间只共享指令集手册,不共享源代码。

~~~text
Dacapo 内部的计划表示(服务于 MLIR pass 和分析)
        |
        |  Dacapo 自己实现"写文件"
        v
RuntimePlan 文件(唯一的契约)
        |
        |  Runtime 自己实现"读文件"
        v
Runtime 内部的计划表示(服务于验证和执行)
~~~

明确禁止两种省事的做法:

- **不允许两边 include 同一个 plan.hpp**。一旦共享类型,编译器就被迫链接执行器的代码,两个仓库的发布节奏也被绑死;
- **不允许把 C++ struct 的内存直接 dump 进文件**。文件格式必须是显式定义的,和任何一方的内存布局无关。

两边各写各的序列化/反序列化,看起来是重复劳动,实际是安全网:双方对协议的理解有出入时,交叉测试(Dacapo 写的文件让 Runtime 读,构造的边角文件让双方各自判断合法性)能把分歧暴露出来,而不是被共享代码悄悄掩盖。

## 4. 规范放在哪,谁说了算

规范放在 Runtime 仓库,由 Runtime 维护。理由很直接:计划文件描述的是"Runtime 能执行什么",这个能力边界只有 Runtime 自己说了算。建议目录:

~~~text
docs/runtime-plan/v1/
|- specification.md       # 每个字段的含义、取值和必须满足的约束
|- schema.json            # 机器可检查的 JSON 结构定义
|- compatibility.md       # 版本升级和目标兼容规则
`- testdata/
   |- valid/              # V1 冻结后,所有实现都必须接受的样例文件
   `- invalid/            # V1 冻结后,所有实现都必须拒绝的样例文件

docs/operator-spec/v1/
|- specification.md       # CKKS 参数边界、算子支持和代价模型
|- schema.json            # OperatorSpec 的 JSON Schema
`- profiles/              # 当前为占位配置，联调后再换成实测版本
~~~

OperatorSpec 的字段含义和 Schema 由 Runtime 仓库维护，因为 Runtime 也要按同一规则验证计划。最终发布时，Runtime 仓库固定一份经过 Poseidon 实测和联调的副本；当前 `profiles/` 仍是 placeholder，只用于格式和测试。Dacapo 负责读取 spec，不拥有也不猜这些数值。

两边的分工:

| 事项 | Dacapo | Runtime |
| --- | --- | --- |
| 编译器内部的中间表示 | 自己定义维护 | 不关心 |
| RuntimePlan 文件规范 | 照着生成 | 制定规范,照着读取 |
| 写文件(序列化) | 必须实现 | 只在测试/调试时需要 |
| 读文件(反序列化) | 可选,用于自测回读 | 必须实现 |
| 生成时检查 | 必须做,尽早在编译端报错 | 不信任其结果 |
| 执行前检查 | 替代不了 Runtime 的检查 | 必须做,是最终把关 |
| 指令放哪、数据怎么搬 | 必须全部决定好 | 照单执行,不推断不修补 |
| 实际执行 | 不负责 | 负责 |

注意"生成时检查"和"执行前检查"是**按同一份规范各写各的**,不共享检查代码。Runtime 要把收到的计划文件当成不可信输入:哪怕 Dacapo 声称已经验证过,Runtime 也从头再查一遍。这不是不信任对方的人,而是不给"两边共用同一个带 bug 的检查"留机会。

## 5. V1 协议要写清楚哪些内容

本节只解释跨仓库为什么需要这些信息,不再充当字段规范。字段名称、合法取值和拒绝条件以 [RuntimePlan V1 草案](../runtime-plan/v1/specification.md) 与 [OperatorSpec V1 规范](../operator-spec/v1/specification.md)为准。

V1 RuntimePlan 对应 `dialect-design.md` 里的**可执行 CKKS**阶段:placement 已完成,通信也已经显式化。逻辑 CKKS、已分配 CKKS 和 Dacapo 自己的分析属性都属于编译器内部表示,不直接进入协议。Dacapo 把可执行 CKKS 写成 JSON,Runtime 再把 JSON 读成自己的 C++ 结构。C++ `plan.hpp` 只是 Runtime 的内部类型,不是跨仓库契约。

V1 只需要一份 JSON 协议,不要求再造一个 MLIR runtime dialect。如果以后确实有多个 MLIR 消费方,再讨论是否增加该 dialect。

### 5.1 文件头和目标环境

- `format_version`:文件格式版本号;
- `plan_id`:这份计划的标识;
- 计划文件原始字节摘要:Runtime 读取 JSON 时直接计算 `plan_source_sha256`,多进程执行时默认比较该摘要,确认大家拿的是完全相同的发布文件;
- `target_id`:目标后端家族,如 `"poseidon-ckks-gpu"`;
- `capability_version`:Runtime Api 支持的算子、Place 和通信能力版本;
- `operator_spec`:包含 id、version 和 `source_sha256`,指向编译这份计划时使用的 CKKS 算子规则,见 5.5;
- `rescale_mode`:本计划采用 `eager` 还是 `lazy` rescale;
- `boot_mode`:使用原生 boot,还是明确选择 CPU 解密再加密模拟;
- `world_size`:进程(rank)数量;
- 每个 rank 的设备数量;
- 计划用到的 Api 能力集合。

`format_version` 管文件长什么样；`target_id` + `capability_version` 管“Runtime 能执行什么”；RuntimePlan 规范定义普通算子的元信息变化；OperatorSpec 管目标相关的参数边界、支持范围和代价。Runtime 只接受自己明确认识的组合，遇到不认识的版本、目标、OperatorSpec 或模式就直接报错退出——不猜，不降级，不尝试用旧逻辑碰运气。

`rescale_mode` 和 `boot_mode` 这两个头部字段是**冗余摘要,不是权威**:rescale 模式的权威是 OperatorSpec,boot 实现方式的权威是每条 Boot 指令自己的 `implementation` 属性。Runtime 必须交叉校验三处一致——头部说 eager 而 spec 说 lazy,或头部说原生 boot 而某条指令写着 `decrypt_reencrypt`,都直接拒绝计划。留着这两个字段只是为了让人和工具不用扫全文件就能看出计划的形态。

### 5.2 每个值(Value)的描述

计划里每个值至少要写明:

- 全局唯一的 `ValueId`;
- 是明文还是密文;
- 它在哪(唯一的 Place,见 5.3);
- 属于哪套 CKKS 参数(context 标识);
- `level`:当前在模数链中的层号,使用非负整数。较大的值表示还保留更多 RNS 模数;Rescale 或 ModSwitch 后 level 下降。它不是浮点数,也不写成 `2^x`;
- `scale_log2`:scale 的二进制指数,使用非负整数。`scale_log2 = 40` 表示逻辑 scale 为 `2^40`,协议里不传 `2^40` 的浮点值;
- 是否处于 NTT 形式;
- 密文分量(components)数量;
- 其他会影响"这个算子能不能作用在这个值上"或"搬运时要复制多少数据"的元信息。

这里必须把两个词分清楚。Dacapo 当前的 Earth 类型本来就分别保存整数 `scale` 和整数 `level`;它的 `scale` 实际表示指数,SEAL 解释器也是到执行时才计算 `pow(2, scale)`。V1 把这个字段明确命名为 `scale_log2`,避免把它误认为浮点 scale,也避免和 level 混淆。

Dacapo 内部还有一个容易踩坑的方向差异:Earth 分析阶段的 level 会随着 rescale 增加,而下降到 CKKS PolyType 后会换算成“剩余模数链层号”,rescale 后下降。RuntimePlan V1 采用后者。导出器必须写最终 CKKS 层号,不能把 Earth 的原始 level 数字直接抄进 JSON。

**和现状的差距要心里有数**:本仓库的 `ValueDesc`、VecValue 和 MPI 元信息已经包含 context、level、整数 `scale_log2`、NTT 和分量数。当前真正缺的是显式 Encode 指令、inline/bundle payload、bundle 装载入口和完整的 OperatorSpec 校验。V1 协议中的 scale 仍只能用整数 `scale_log2`;允许出现的浮点数只限 Encode 的 inline slot 数据。计划摘要直接覆盖文件原始字节,不要求对浮点 JSON 做 JCS 规范化。

### 5.3 位置(Place)

主机内存和显卡都是平等的"位置":

~~~text
Host(rank=N)             # 第 N 个进程的主机内存
Device(rank=N,index=M)   # 第 N 个进程的第 M 张卡
~~~

协议要规定 rank 和 index 的编号范围,以及 Host 是否省略 index 字段。核心规则:**一个 ValueId 只存在于一个位置**。同一份数据要出现在第二个地方,必须由一条显式的通信指令产生一个新的 ValueId。这条规则让"数据在哪"永远可以静态回答,验证器和执行器都因此变简单。

### 5.4 计算指令

每条计算指令至少包含:

- 稳定的指令序号(报错时用来定位);
- 算什么(算子种类);
- 输入输出的 ValueId;
- 在哪算(唯一的 Place);
- 按算子类型定义的参数,不能用一张含义不明的通用整数表;
- 该目标后端特有、但已被协议和 OperatorSpec 明确定义的 profile 引用。

V1 至少把这些参数写死:

- Rotate:`steps`;
- Rescale:`target_level` 和 `target_scale_log2`;
- ModSwitch:`target_level`;
- Boot:`target_level`、`target_scale_log2`、`target_components`、`operator_profile` 和 `implementation`。`implementation` 至少区分原生 boot 与 `decrypt_reencrypt` 模拟。V1 Boot 保持 context 和 polynomial degree 不变。

算子输出的 `ValueDesc` 必须和这些目标参数一致。例如 Boot 声明输出到 level 6、`scale_log2` 40、2 个分量,输出值描述也必须是同一个 context、level 6、`scale_log2` 40、2 个分量。Runtime 发现两处不一致时直接拒绝计划。

V1 规范还要列出一份**导出前必须消除的计算算子清单**:可执行 CKKS 的 Encode、计算和通信指令集合必须和 RuntimePlan 的指令集合严格相等。`ckks.encode` 一对一导出成 Encode 指令;像 `upscale` 这类只服务编译器内部的算子(Dacapo 已有 `UpscaleToMulcp` 把它下沉成明文乘法)必须在导出前消掉。导出器遇到集合之外的算子就是它自己的 bug,必须报错而不是发明新编码。

Runtime 拿到指令就照着执行:不看输入在哪来猜测该在哪张卡上算,也不发现"输入不在本地"就好心帮忙搬一趟。所有搬运必须是编译器写下的显式指令,否则就是编译器的 bug,应该报错暴露而不是被运行时兜底掩盖。

### 5.5 CKKS OperatorSpec

Dacapo 现在的 profile JSON 混合保存了 scale、level 范围、bootstrap 范围、延迟和噪声数据。接 Poseidon GPU 后，boot 的合法输入、输出 level 和实际代价也需要稳定配置。继续把这些数值散落在 pass 参数或代码常量里，编译结果很难复现。

因此把目标相关的 CKKS 算子数据独立成一份有版本的 OperatorSpec。它至少要描述:

- `spec_id`、版本;计划另外保存 spec 完整原始文件字节的 `source_sha256`；
- context、poly degree、RNS 模数链和单模数 bit 数限制；
- eager/lazy rescale 模式；
- 普通算子的支持状态、按 level 的代价和额外边界；
- Boot profile 的合法输入、目标 level/`scale_log2`、实现方式和代价；
- CPU 解密再加密模拟需要的 Host 能力和密钥要求。

普通算子怎样改变 level、`scale_log2` 和 components，由 RuntimePlan 规范统一定义，不在 OperatorSpec 里重复。V1 草案还没有统一的噪声单位，因此 `noise_by_level` 目前必须显式为 `null`。

CPU 和 GPU 使用不同的 OperatorSpec。Poseidon CPU 不要求 lazy-rescale,CPU profile 使用 eager 模式;当前受低 bit RNS 限制的 Poseidon GPU profile 明确使用 lazy 模式。也就是说,lazy-rescale Pass 对整个 Dacapo 编译器是可选项,但选中这个 GPU profile 后就是必选项。GPU boot 的 level 消耗不能沿用 CPU 数值,必须以所选 GPU boot profile 为准。如果内部实现或 RNS bit 限制改变了消耗,就更新 OperatorSpec 版本或 profile,不能悄悄改变同一版本的含义。

OperatorSpec 是 Dacapo 的编译输入,不是让 Runtime 临场重新规划的配置。RuntimePlan 文件头记录它的 id、版本和完整原始文件 `source_sha256`,每个值和算子再记录已经算好的具体 level、`scale_log2` 和 profile。Runtime 只验证这些结果与自己支持的 profile 相符,不会根据 OperatorSpec 帮编译器补 Rescale 或重算 boot 位置。

### 5.6 通信指令

V1 支持两种:

- `Transfer`:一个源,搬到一个目的地;
- `Replicate`:一个源,复制到多个目的地。

每条通信指令至少包含:

- 全局唯一的 `TransferId`;
- 输入输出 ValueId;
- 源位置和目的位置列表;
- 每个输出的类型;
- 通信语义(是 Transfer 还是 Replicate);
- 实现方式建议(hint,如点对点、广播、经主机中转)。

`outputs[i]`、`destinations[i]`、输出类型三个列表必须一一对应。hint 只是建议——Api 可以选别的等价实现,但不管怎么实现,通信完成后"哪些值出现在哪些位置"必须和计划写的一致。

Transfer 和 Replicate 必须保持 CKKS 数学值及其元信息不变。CPU/GPU 表示可以在 Api 内部转换,但 context、level、`scale_log2`、NTT 状态和分量数不能在搬运时悄悄变化。

### 5.7 输入输出和生命周期

计划还要写明:

- 外部输入怎么绑定(哪些函数参数 ValueId 由调用方提供);
- 三个执行阶段:初始化(预加载)、执行、收尾;
- 最终输出是哪些值;
- 常量和模型权重这类大块数据的引用方式。

计划文件只描述“计划”本身，不规定 Poseidon GPU 对象在显存里的布局。external_inputs 表示本次运行由调用方传入的参数。随计划发布的固定权重由 initialization 中的 Encode 产生：小 payload 直接放进 JSON，大 payload 通过 `content` 引用[明文数据包](../runtime-plan/v1/plaintext-bundle.md)。动态权重如果由调用方每次传入，也可以是 external input。Encode 输出和 external input 都先位于 Host，进设备必须走显式 Transfer/Replicate。

**密钥不进计划文件**。secret key(`decrypt_reencrypt` boot 在 Host 上要用)、galois key(Rotate 要用)、relinearization key 这些都是 Api 实例的构造配置,由部署方在创建 Api 时提供,协议里不出现密钥内容或路径。计划最多声明"这份计划需要哪几类密钥、在哪些位置可用",让 Runtime 的执行前检查能对着 Api 的实际配置把关——缺 key 要在执行前发现,而不是跑到一半才炸。

## 6. V1 用什么文件格式

首版用严格定义的 JSON,不要一上来就设计二进制格式:

- Dacapo 好生成,Runtime 好解析;
- 人能直接读,出了"编译器和执行器理解不一致"的问题,肉眼 diff 就能定位;
- 可以配 JSON Schema 做机器检查;
- 适合当测试基准文件(golden file)和代码评审的对象。

结构示意(最终以 V1 规范为准):

~~~json
{
  "format_version": 1,
  "plan_id": "42",
  "target": {
    "target_id": "poseidon-ckks-gpu",
    "capability_version": 1,
    "operator_spec": {
      "id": "poseidon-ckks-gpu-v1",
      "version": 1,
      "source_sha256": "sha256:..."
    },
    "rescale_mode": "lazy",
    "boot_mode": "decrypt_reencrypt",
    "world_size": 2,
    "device_counts": [2, 2]
  },
  "values": [],
  "external_inputs": [],
  "initialization": [],
  "execution": [],
  "finalization": [],
  "final_outputs": []
}
~~~

定规范时还必须把这些细节钉死,它们每一个都出过真实事故:

- 64 位整数是否用十进制字符串编码(很多 JSON 工具把大整数当 double 处理,会悄悄丢精度);
- 枚举用固定名字还是固定数字;
- 读到不认识的字段怎么办,缺字段怎么办;
- `level` 和 `scale_log2` 只接受非负整数,拒绝浮点数和字符串形式的 `2^x`;
- 字符串编码;
- 原始文件摘要覆盖哪些字节——V1 明确覆盖完整文件,所以缩进、空格、字段顺序、行尾和末尾换行变化都会改变摘要;
- 数组的顺序有没有含义。

如果以后确认 JSON 的解析速度或文件体积真成了瓶颈,再为新的 `format_version` 定义二进制格式。同一个版本号下不允许"看文件内容猜格式"。

## 7. Dacapo 这边要做什么

### 7.1 目标形态

Dacapo 新增一条可选的 Poseidon 编译管线:

~~~text
CKKS 逻辑优化
-> bootstrap 安排 / scale 管理
-> 读取 TargetSpec 和 OperatorSpec
-> 按目标选择 eager 或 lazy rescale
-> 按需把 GPU 原生 boot 改成 CPU 解密再加密模拟
-> 决定每条指令的 rank 和设备(placement)
-> 插入显式的 Transfer/Replicate
-> 生成端自检
-> 输出 RuntimePlan 文件
~~~

lazy-rescale 不是所有目标都必须打开的全局规则:

- Poseidon CPU 不要求 lazy-rescale,CPU OperatorSpec 使用 eager 模式;
- 当前低 bit Poseidon GPU OperatorSpec 声明 `rescale_mode=lazy`,选中该 profile 时 Dacapo 必须启用对应 pass;
- 同一个 Dacapo 编译入口通过 spec 选择模式,不维护两套互相漂移的编译器代码。

关于 lazy-rescale 放哪一层,判断标准是:**它改不改计划的可观察内容**。如果它会移动 Rescale 的位置、改变 level/`scale_log2`、增减算子或改变值的生存期,就必须在 placement 之前、在 Dacapo 里做——因为这些变化影响计算量、显存占用和通信量,placement 需要看到真实代价。如果它只是一个不改变这些内容的 GPU kernel 实现细节,才留在 PoseidonGpuApi 内部。

Mul、Rescale 等算子的外部元信息变化由计划明确写出。Boot 内部的 lazy-rescale 不会展开成一长串 RuntimePlan 指令,但它造成的合法输入范围、实际 level 消耗和输出元信息必须由所选 boot profile 明确给出。Dacapo 按该 profile 规划,Runtime 和 PoseidonApi 按同一个 profile 校验和执行。

Dacapo 通过两份声明式配置了解目标:

- TargetSpec 描述 CPU/GPU、进程和设备数量、拓扑、带宽以及 `target_id`/`capability_version`;
- OperatorSpec 描述 CKKS 参数、算子支持边界、代价、lazy-rescale 模式和 boot profile。普通算子的元信息变化规则仍由 RuntimePlan 规范定义。

Dacapo 不 include 任何 Poseidon 头文件,也不调用 Poseidon 的运行时接口。

### 7.2 可选的 GPU boot CPU 模拟 Pass

Poseidon GPU 原生 boot 还不完善时,Dacapo 提供一个显式开关,例如 `boot_mode=decrypt_reencrypt`。打开后,目标合法化 Pass 把 GPU 上的原生 Boot 改成一个只能放在 Host 的 Boot,其 `implementation` 明确写成 `decrypt_reencrypt`。这个 Pass 要在 placement 之前运行,让 placement 和通信显式化看到真实的数据搬运成本。

通信显式化之后,计划形态应当类似:

~~~text
%host_in  = Transfer %gpu_in  Device -> Host
%host_out = Boot %host_in {
  place = Host,
  implementation = decrypt_reencrypt,
  target_level = L,
  target_scale_log2 = S,
  target_components = 2,
  operator_profile = "poseidon-cpu-boot-emulation-v1"
}
%gpu_out  = Transfer %host_out Host -> Device
~~~

Host 上的实现步骤是:用 Poseidon CPU 路径解密和解码,再按目标 context、`target_level` 和 `target_scale_log2` 编码并重新加密。最后一条 Transfer 只负责把这个已经符合 GPU 参数的密文转成 GPU 表示并上传,不能再次改变 level 或 `scale_log2`。

这里有四条硬规则:

1. 这是测试和联调用的 boot 模拟,会用到 secret key,并让明文短暂出现在 Host 内存中,不能当成安全的生产 boot;
2. Runtime 不会在 GPU Boot 失败后偷偷走这条路。是否使用模拟必须由 Dacapo Pass 明确写进计划;
3. Device→Host 和 Host→Device 都是普通的显式 Transfer,各自产生新的 ValueId;
4. Runtime 执行一份计划时只持有一个 Api 实例。GPU 目标的 PoseidonGpuApi 需要组合 Poseidon CPU 能力来执行这个 Host Boot,而不是让 Runtime 同时调度两个互不相关的 Api。

不开该 Pass 时,计划保留原生 GPU Boot。如果目标 capability 或 boot profile 不支持它,Dacapo 应在编译期报错;漏过编译期检查时,Runtime 在执行前再次拒绝。

### 7.3 已定决策:MLIR 原生产出,HEVM 整条路退役

Dacapo 现有的产出路径是编译成 HEVM 字节码(`.hevm` + `.cst` 常量文件),再由 Poseidon 的 `frontends/dacapo` 解析成自己的调度表示。**本方案决定不走这条路**:RuntimePlan 的产出管线在 MLIR 层新写,placement 和通信插入作为 MLIR pass 实现,直接输出 RuntimePlan JSON;HEVM 作为中间产物被彻底抛弃。

理由:

- placement 需要在编译器还持有完整信息(代价模型、值的生存期、scale 管理决策)时做,HEVM 字节码已经把这些信息扔掉了,在它上面补 placement 是先降级再猜回来;
- HEVM 是又一个需要维护规范的序列化格式。留着它,系统里就同时存在 HEVM 和 RuntimePlan 两份契约,和"只有一个协议"的原则直接冲突;
- HEVM 解释器(Dacapo 的 `lib/Runtime`)和 HEVM 解析器(Poseidon 的 `frontends/dacapo`)都是围绕这个格式的存量投资,格式退役后它们没有继续存在的理由。

要正视的代价:这是在 fork 里新做一个编译目标,工作量集中在编译器侧,端到端打通的时间点比"外挂转换器"晚。作为缓冲,阶段二应该按管线顺序增量交付——先让 MLIR 管线在单卡配置(不做 placement、无通信指令)下产出合法的 RuntimePlan 并通过 MockVecApi 端到端测试,再逐步加入多卡 placement 和通信插入。这样协议和 Runtime 的磨合不必等整条编译管线完工。

连带决定:`frontends/dacapo` 里的 HEVM/CST 解析、常量装载和输入输出绑定代码不再迁移,随 HEVM 一起退役(见第 11 节);其中仍有参考价值的部分(如 CKKS 元信息的对账逻辑)以规范文字和测试样例的形式沉淀进协议文档,而不是搬运代码。

### 7.4 常量外化与明文数据包

HEVM 的 `.cst` 常量文件退役后,权重走这条新路:

1. 编译期,现有 `earth.constant` 先表示逻辑常量,其 `value` 仍是旧 `.cst` 索引。迁移后的 lowering 负责解析该索引对应的浮点 slots,再生成携带真实 payload 的 `ckks.encode`(通用 MLIR 输入也可以来自 `arith.constant`)。`ckks.encode` 声明按输出值的 level/`scale_log2`/NTT 把数据编码成 CKKS 明文。这里不新增 `ckks.constant`,设计见 [Dialect 设计 5.1 节](dialect-design.md);
2. 小 payload 保持内联,直接写进 RuntimePlan Encode 指令;明文数据外化 pass 只把超过阈值的大 payload 按内容哈希写入**明文数据包**,再把 op 改成 `content` 引用。外化只搬 float64 字节,不做 CKKS 编码;
3. 导出 RuntimePlan 时,每个 `ckks.encode` 一对一变成 initialization 中的 Encode 指令,原 MLIR 结果编号成为 Encode 的 `output` ValueId。Encode 不是 external input;旧 `.cst` 索引随 HEVM 一起退役;
4. Runtime preflight 先核对 bundle id/version/manifest_sha256,再检查 manifest `blobs` 中所有文件的存在性、长度、内容哈希和小端 float64 合法性,最后确认每个 Encode 引用的 `content` 及其输出约束。manifest 只列原始数据 blob,不保存 ValueId;
5. Runtime 执行 Encode 时,Api 按输出 ValueDesc 把 inline 或 bundle 浮点向量编码成 Host 明文(真实后端调用 CKKS encode;Vec 明文参考后端直接当 slots 用),之后计划中的 Transfer/Replicate 再把它铺到设备。

低内存联调模式由此免费获得:测试编码器把所有权重改为引用少数几个占位 `content`,多个 Encode output 共享同一份原始 blob,但 Encode 指令数、输出 ValueId 和 SSA 图与真实权重完全一致——difftest 走同一份计划逻辑,只换数据。

## 8. Runtime 与 Api

单卡、多卡、多机不写三套执行器。它们是同一份执行逻辑在不同目标配置下的三个实例:

~~~text
单卡:      world_size=1, device_counts=[1]
单机多卡:  world_size=1, device_counts=[N]
多机多卡:  world_size=R, device_counts=[N0,N1,...]
~~~

统一执行器和多 rank 解释框架已经由本仓库的 `SequentialRuntime<Api>` 验证;显式 Encode 与 `encode_plaintext` 接口仍待实现,见实现状态文档。

Runtime 只面对一个窄接口：

- `preflight` 默认检查计划原始字节摘要、target、context 和密钥配置,并接收部署方统一设置的 `skip_artifact_digest_checks`；
- `validate_value` 按 ValueDesc 核对后端实际值；
- `encode_plaintext`、`compute`、`communicate_async`、`wait` 和 `synchronize` 完成执行；
- `abort_all` 在出错时终止整个执行组。

`encode_plaintext` 接收 Encode 输出的 ValueDesc 和 float64 slots，返回 Host plaintext。context 和密钥在 Api 构造或 Runtime 启动前由部署方配置，Runtime 只校验，不在 initialization 中临时上传。Api 内部怎么编码、计算和通信，Runtime 不关心：

`skip_artifact_digest_checks=true` 只允许调试时跳过计划、OperatorSpec 和 manifest 的原始字节摘要比较。它不写进计划,不能跳过 id/version、Schema、SSA、元信息、能力、密钥或 blob `content` 检查;生产配置必须关闭。

~~~text
MockVecApi      = VecExecutor 明文计算 + MockCluster 模拟通信
MpiVecApi       = VecExecutor 明文计算 + MPI 通信
PoseidonCpuApi  = Poseidon CPU 算子 + 主机内存/MPI 通信
PoseidonGpuApi  = Poseidon GPU 算子 + 必要的 Host 算子
                  + CUDA P2P/NCCL/GPU-aware MPI 通信
~~~

`PoseidonGpuApi` 不是说所有指令都必须在 GPU 上执行,而是说这份 Api 能执行 GPU 目标计划。普通计算仍在 Device;只有计划明确放到 Host 的算子,例如 `decrypt_reencrypt` Boot,才调用它内部组合的 Poseidon CPU 实现。Runtime 仍然只调用一个 Api,也不会根据算子失败情况临时换后端。

Poseidon 现有 mgpu 代码里,GPU 对象的物化和分段拷贝、设备上下文、跨卡通信这些是真金白银的积累,全部迁进 PoseidonApi 继续用。而 mgpu 自己那套调度表示、placement、验证器和执行器,在迁移完成后删除——不长期养两套执行系统。具体哪些删哪些留,见第 11 节的分级清单。

## 9. 测试怎么布防

一类端到端测试打天下是不够的:端到端挂了不知道该怪谁,端到端过了也可能是两边的错误互相抵消。至少四层:

| 层级 | 输入与执行 | 主要验证什么 |
| --- | --- | --- |
| Dacapo 单元测试 | 上层程序 → 计划文件 | CPU eager/GPU lazy 两种 rescale、boot profile、CPU 模拟 Pass、placement、通信插入和元信息生成对不对 |
| Runtime 单元测试 | 手工构造的计划 → MockVecApi | 解析、验证器、值状态管理、多 rank、出错即停 |
| 两边集成测试 | Dacapo → JSON 文件 → Runtime → MockVecApi | 双方对协议的理解一致;真实编译产物能跑对 |
| Poseidon 后端测试 | 同一份计划 → Vec/CPU/GPU 三种 Api | 三种后端算出同样的结果,元信息和同步语义一致;GPU 计划的 Host boot 模拟能正确往返 |

Runtime 仓库的协议测试集按四组组织：

- **Encode/bundle**：inline 与 bundle、同一 `content` 的多次编码、payload 混写、缺 content、非法浮点和错误阶段；
- **计算/通信**：每种指令的最小合法计划、元信息不一致、ValueId/Place 错误和 Transfer/Replicate 映射错误；
- **目标配置**：CPU eager、GPU lazy、两种 Boot、未知 OperatorSpec、原始文件摘要不符和能力不匹配；
- **协议编码**：单卡/多卡/多进程代表计划、边界值、原始字节 SHA-256、只改格式也改变摘要,以及调试跳过摘要但不跳过其他错误。

Dacapo 集成测试的主路径:

~~~text
Dacapo 测试模型
-> 生成 RuntimePlan JSON
-> JSON Schema 检查
-> Runtime 读入并验证
-> SequentialRuntime<MockVecApi> 执行
-> 和逻辑参考结果比对
~~~

这条路能同时抓编译器和执行器的互操作问题,但不能替代两边各自的单元测试。

## 10. 版本怎么演进

1. Runtime 仓库里的协议规范是唯一权威;
2. 要改字段或语义,先改规范、先加测试样例,再动实现;
3. Dacapo 和 Runtime 各自更新自己的实现;
4. Runtime 更新 Dacapo submodule 指针,在同一个 commit 里锁定"这对组合验证过";
5. Poseidon 等那个 Runtime commit 测试通过后才更新自己的指针;
6. Runtime 遇到不支持的版本或能力,直接报错,不猜;
7. 不靠"缺了某字段"或"解析失败"来推断这是旧格式文件;
8. 要不要同时支持多个版本由发布需求决定;支持的每个版本都要有独立入口和独立测试,不做隐式回退。

## 11. 迁移顺序与删除清单

### 阶段一:冻结协议

1. 在 Runtime 仓库建 `docs/runtime-plan/v1/`,写规范、Schema、原始文件字节摘要和调试策略;
2. 建 `docs/operator-spec/v1/`,先定义 CPU eager 和 GPU lazy 两个最小 profile,再补 GPU boot profile;
3. **把 5.2 节的 CKKS 元信息(context、level、`scale_log2`、NTT、分量数)补进 `ValueDesc` 和验证器**,并删掉协议侧的浮点 scale;
4. 把 Rescale/Boot 的 C++ 属性改成整数目标 level 和 `target_scale_log2`,加入 boot implementation/profile;
5. 建立合法/非法样例文件集;
6. 给 `RuntimePlan` 写独立的 JSON 读取器,支持显式 Encode 和 inline/bundle 双 payload;
7. 验证器覆盖规范要求的全部执行前检查,包括 Encode、bundle content、OperatorSpec 匹配和 Host compute 能力。

### 阶段二:接入 Dacapo

1. Dacapo 作为可选 submodule 进入 Runtime;
2. 把 Dacapo 现有 profile JSON 中的 CKKS 参数、算子延迟/噪声和 bootstrap 范围迁到版本化 OperatorSpec。注意 profile 里的 bootstrap level 上下界是 Earth 方向的数字,迁移时要做和 5.2 节一样的层号换算,不能照抄;
3. 在 Dacapo 里搭 MLIR 层的 RuntimePlan 产出管线,先支持单卡配置,把 `ckks.encode` 一对一导出成 Encode 指令,实现生成端自检和 JSON 写出;
4. 接 CPU eager 和 GPU lazy 两种 rescale 管线,用 spec 决定是否启用 lazy-rescale;
5. 加入可选的 GPU Boot→Host `decrypt_reencrypt` 合法化 Pass;
6. 用 MockVecApi 建立真实编译产物的单卡端到端测试;
7. 加入多卡 placement 和 Transfer/Replicate 插入,扩展端到端测试到多卡、多进程计划。

### 阶段三:接入 Poseidon CPU/GPU

1. Poseidon 以可选构建路径引入 Runtime;
2. 先做单进程单设备的 PoseidonCpuApi;
3. 再做单进程单卡的 PoseidonGpuApi;
4. 让 PoseidonGpuApi 支持计划明确要求的 Host Boot 模拟:Device→Host、CPU 解密再加密、Host→Device;
5. 迁移同进程跨卡的对象拷贝;
6. 接入跨进程通信;
7. 同一份计划在 Vec/CPU/GPU 三种 Api 下跑,结果必须一致。

### 阶段四:删除重复实现

删除必须等前三个阶段完成,因为旧代码是目前唯一能跑的路径,也是新路径正确性的对照组。按处置方式分三类:

**立即可清理(不是源码,不用等):**

- Poseidon 仓库里的十来个陈旧构建目录(`build-codex-mgpu-*`、`build-mgpu-*-bench` 等)。

**有价值,迁入 PoseidonApi 后继续用:**

- `mgpu/comm/`:GPU 对象物化与分段拷贝、CUDA peer 通信、拓扑探测——密文在显存中不连续、需分段搬运的问题在这里已经解决;
- `mgpu/runtime/type_def/object_store`:不透明 GPU 对象存储,改造成 PoseidonApi 的值存储;
- `mgpu/runtime/backend/poseidon_gpu_execution_backend`:GPU 算子派发,改造成 `PoseidonGpuApi::compute`。

**冗余,迁移验证完成后删除:**

- `mgpu/ir/`:调度中间表示,被 RuntimePlan 取代;
- `mgpu/compiler/`:placement、拷贝插入、调度器、验证器、管线,职责移交计划生成侧;
- `mgpu/runtime/executor/` 与 `mgpu/runtime/preflight/`:被 `SequentialRuntime` + `PlanVerifier` 取代;
- `frontends/dacapo/` 整个目录:HEVM/CST 解析、常量装载、输入输出绑定和第三套执行器(`poseidon_gpu_hevm_executor.*`)。HEVM 格式退役后整条链没有存在理由,不迁移(见 7.3);其中的 CKKS 元信息对账逻辑以规范文字和测试样例形式沉淀进协议,而非搬运代码;
- Dacapo 仓库的 `lib/Runtime/{HEAAN,SEAL}_HEVM.cpp`:Dacapo 自带的 HEVM 解释器。注意它同时是差分测试的 CPU 参照实现,且被 `python/hecate/runner.py` 加载——要等新路径有了替代参照(PoseidonCpuApi 跑通并对过账)之后再删;
- 相关的约 38 个测试文件随各自被删的模块一起退役,其中验证的不变量(SSA、位置一致性、拷贝合法性)要确认在 Runtime 测试集中有对应覆盖。

删除完成后,所有执行入口只接受明确版本的 RuntimePlan,不保留任何自动降级的旧路径。

## 12. 结论

- RuntimePlan 是 Runtime 定义并拥有的版本化执行契约,规范、Schema 和测试样例都放在 Runtime 仓库;
- RuntimePlan V1 是可执行 CKKS 的 JSON 序列化;Dacapo 的逻辑/已分配 MLIR 和 Runtime 的 C++ `plan.hpp` 都不是跨仓库协议;
- `ckks.encode` 在 RuntimePlan 中保留为 initialization 的显式 Encode 指令；external_inputs 表示每次运行由调用方传入的参数；
- Dacapo 和 Runtime 不共享 C++ 类型,各自实现文件的写和读,靠交叉测试对齐;
- 两边各有一个按同一规范独立实现的检查器,Runtime 的执行前检查是最终把关;
- `level` 是整数模数链层号,`scale_log2` 是整数 scale 指数;V1 不传浮点 scale;
- CPU 和 GPU 使用不同的 OperatorSpec。CPU 默认 eager rescale,当前低 bit GPU profile 使用 lazy-rescale;GPU boot 的内部 level 消耗也由独立 boot profile 给出;
- GPU 原生 boot 不可用时,只能由 Dacapo 的可选 Pass 明确生成 Device→Host、CPU 解密再加密、Host→Device 的计划,Runtime 不做隐式回退;
- Dacapo 是 Runtime 的可选 submodule(指向 `dacapo-modified` fork),只用于锁版本和集成测试;
- 计划产出走 MLIR 原生管线直接输出 RuntimePlan,HEVM 字节码及其解释器/解析器整条链退役;
- Poseidon 只引入 Runtime,通过 PoseidonCpuApi/PoseidonGpuApi 提供计算和通信;mgpu 中的 GPU 拷贝与通信代码迁入 Api 层复用,调度和执行代码在迁移验证后删除;
- 单卡、多卡、多机由同一套 Runtime 代码执行,只是目标配置不同;
- 版本、目标或能力对不上时直接报错,永远不猜。
