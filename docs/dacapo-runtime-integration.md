# Dacapo、Runtime 与 Poseidon 集成方案

## 1. 这份方案要解决什么问题

现在有三份代码,各自都在做"执行计划"这件事,职责重叠:

- **Dacapo**(`utils/dacapo`,即 hecate 编译器的 fork):把上层程序编译成 CKKS 程序,自带一套 HEVM 字节码和两个解释器(`lib/Runtime` 里的 SEAL/HEAAN 版本);
- **Poseidon**(FHE GPU 库):`src/poseidon/mgpu` 里长出了一整套独立的执行系统——自己的调度中间表示、设备分配、拷贝插入、验证器、执行器,外加 `frontends/dacapo` 里的 HEVM 导入和第三套执行器;
- **本仓库**(Runtime demo):一套干净的"照计划执行"框架,有验证器、值状态管理和明文参考实现,但还没接真实 GPU。

三套执行逻辑长期并存,每改一处要同步三处,谁也不敢删谁。本方案的目标是收敛成一条链,每个仓库只干一件事:

- **Dacapo 只做编译**:从上层程序生成一份"每条指令在哪算、数据怎么搬都定死了"的静态计划文件;
- **Runtime 只做执行**:读入计划文件,验证,然后照着执行。单卡、多卡、多机用同一套执行代码;计算和通信的具体实现通过一个抽象接口(下文称 Api)注入,明文、CPU、GPU 各实现一份;
- **Poseidon 只做算子和通信**:不再维护自己的计划解释器,改为实现 Runtime 需要的 CPU/GPU Api;
- Dacapo 和 Runtime 之间**只通过一个版本化的计划文件交流**,不共享任何 C++ 类型;
- Runtime 仓库固定一个已验证兼容的 Dacapo 版本(git submodule),用明文 Api(VecApi)做"编译器产物真的能被执行器跑对"的端到端验证。

依赖方向是单向的:

~~~text
Dacapo
  CKKS 优化 / bootstrap 安排 / scale 管理
  -> 面向 Poseidon 的 lazy-rescale 下沉
  -> 决定每条指令放在哪个进程、哪张卡(placement)
  -> 把所有数据搬运写成显式指令
  -> 输出 RuntimePlan 文件
                 |
                 v
Runtime
  读文件 -> 验证 -> 逐条执行
  -> 调 Api 完成实际计算和通信
       |- VecApi          (明文 std::vector,用来测试)
       |- PoseidonCpuApi  (Poseidon CPU 算子)
       `- PoseidonGpuApi  (Poseidon GPU 算子)
~~~

Runtime 不反向依赖 Dacapo 或 Poseidon。Poseidon 只依赖 Runtime;Dacapo 只在生成计划和跑编译集成测试时出现。

## 2. 仓库怎么摆

建议把本仓库更名为能表达实际职责的名字(远程已经叫 `mgpu-runtime-demo`),结构如下:

~~~text
mgpu-runtime/
|- docs/runtime-plan/           # 计划文件格式的规范文档
|- runtime/                     # 计划类型、验证器、执行器
|- api/                         # VecApi、Mock、MPI 等接口实现
|- testing/                     # 计划构造器、参考执行、差分比较
|- integrations/dacapo/         # 和 Dacapo 对接的胶水与集成测试
`- third_party/dacapo/          # submodule,只给集成测试用

poseidon/
|- src/poseidon/runtime_api/    # Runtime Api 的 Poseidon 实现
|  |- poseidon_cpu_api.*
|  |- poseidon_gpu_api.*
|  `- poseidon_communication.*
`- third_party/mgpu-runtime/    # submodule
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
   |- valid/              # 所有实现都必须接受的样例文件
   `- invalid/            # 所有实现都必须拒绝的样例文件
~~~

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

### 5.1 文件头和目标环境

- `format_version`:文件格式版本号;
- `plan_id`:这份计划的标识;
- 计划内容的指纹(fingerprint):对计划的语义内容算一个稳定的哈希,多进程执行时各进程比对指纹,确认大家拿的是同一份计划;
- `target_id`:目标后端家族,如 `"poseidon-ckks-gpu"`;
- `capability_version`:该后端支持的算子集合、lazy-rescale 行为和元信息规则的版本;
- `world_size`:进程(rank)数量;
- 每个 rank 的设备数量;
- 计划用到的 Api 能力集合。

`format_version` 管文件长什么样,`target_id` + `capability_version` 管"这个后端会算什么"。Runtime 只接受自己明确认识的组合,遇到不认识的版本或目标就直接报错退出——不猜,不降级,不尝试用旧逻辑碰运气。

### 5.2 每个值(Value)的描述

计划里每个值至少要写明:

- 全局唯一的 `ValueId`;
- 是明文还是密文;
- 它在哪(唯一的 Place,见 5.3);
- 属于哪套 CKKS 参数(context 标识);
- level(还剩几层乘法深度);
- scale(编码放大系数);
- 是否处于 NTT 形式;
- 密文分量(components)数量;
- 其他会影响"这个算子能不能作用在这个值上"或"搬运时要复制多少数据"的元信息。

**和现状的差距要心里有数**:本仓库现在的 `ValueDesc` 只有 id、明密文类型和位置三个字段,level/scale/NTT/分量数这些目前都活在 VecApi 的值对象里,没进计划协议。所以第一阶段的工作不只是"加个 JSON 读写",还包括把这些元信息字段提升到 `ValueDesc` 和验证器里。

scale 是浮点数,协议必须规定它的精确编码方式,不能依赖 JSON 库默认的浮点打印(不同库打印同一个 double 的文本可能不一样,会导致指纹不稳定)。候选方案:按 IEEE-754 位模式存整数、用十六进制浮点文本,或者如果目标语义允许,只存 scale 的指数。定 V1 规范时必须三选一。

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
- 算子参数:旋转步数、rescale 目标、bootstrap 目标等;
- 该目标后端特有、但已被协议明确定义的属性。

Runtime 拿到指令就照着执行:不看输入在哪来猜测该在哪张卡上算,也不发现"输入不在本地"就好心帮忙搬一趟。所有搬运必须是编译器写下的显式指令,否则就是编译器的 bug,应该报错暴露而不是被运行时兜底掩盖。

### 5.5 通信指令

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

### 5.6 输入输出和生命周期

计划还要写明:

- 外部输入怎么绑定(哪些 ValueId 由调用方在启动时提供);
- 三个执行阶段:初始化(预加载)、执行、收尾;
- 最终输出是哪些值;
- 常量和模型权重这类大块数据的引用方式。

计划文件只描述"计划"本身,不规定 Poseidon GPU 对象在显存里的内存布局。常量数据(比如模型权重)可以放在独立的文件里,计划中用稳定的 ID 引用;那些文件的格式单独定义,不塞进本协议。

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
- 浮点数的精确表示(见 5.2);
- 字符串编码;
- 指纹以什么为输入计算——必须基于解析后的语义内容,和缩进、空格、字段顺序无关,否则格式化一下文件指纹就变了;
- 数组的顺序有没有含义。

如果以后确认 JSON 的解析速度或文件体积真成了瓶颈,再为新的 `format_version` 定义二进制格式。同一个版本号下不允许"看文件内容猜格式"。

## 7. Dacapo 这边要做什么(含一个必须先做的决策)

### 7.1 目标形态

Dacapo 新增一条可选的 Poseidon 编译管线:

~~~text
CKKS 逻辑优化
-> bootstrap 安排 / scale 管理
-> 面向 Poseidon 的 lazy-rescale 下沉
-> 决定每条指令的 rank 和设备(placement)
-> 插入显式的 Transfer/Replicate
-> 生成端自检
-> 输出 RuntimePlan 文件
~~~

关于 lazy-rescale 放哪一层,判断标准是:**它改不改计划的可观察内容**。如果它会移动 Rescale 的位置、改变 level/scale、增减算子或改变值的生存期,就必须在 placement 之前、在 Dacapo 里做——因为这些变化影响计算量、显存占用和通信量,placement 需要看到真实的代价。如果它只是不改变语义的 GPU kernel 提交技巧,就留在 PoseidonGpuApi 内部,协议里根本不出现。

Dacapo 通过一份声明式的目标描述(TargetSpec)了解硬件:CPU 还是 GPU、几个进程几张卡、拓扑和带宽、各算子和通信的代价、支持的算子集合,以及对应的 `target_id` 和 `capability_version`。Dacapo 不 include 任何 Poseidon 头文件,也不调用 Poseidon 的运行时接口。

### 7.2 已定决策:MLIR 原生产出,HEVM 整条路退役

Dacapo 现有的产出路径是编译成 HEVM 字节码(`.hevm` + `.cst` 常量文件),再由 Poseidon 的 `frontends/dacapo` 解析成自己的调度表示。**本方案决定不走这条路**:RuntimePlan 的产出管线在 MLIR 层新写,placement 和通信插入作为 MLIR pass 实现,直接输出 RuntimePlan JSON;HEVM 作为中间产物被彻底抛弃。

理由:

- placement 需要在编译器还持有完整信息(代价模型、值的生存期、scale 管理决策)时做,HEVM 字节码已经把这些信息扔掉了,在它上面补 placement 是先降级再猜回来;
- HEVM 是又一个需要维护规范的序列化格式。留着它,系统里就同时存在 HEVM 和 RuntimePlan 两份契约,和"只有一个协议"的原则直接冲突;
- HEVM 解释器(Dacapo 的 `lib/Runtime`)和 HEVM 解析器(Poseidon 的 `frontends/dacapo`)都是围绕这个格式的存量投资,格式退役后它们没有继续存在的理由。

要正视的代价:这是在 fork 里新做一个编译目标,工作量集中在编译器侧,端到端打通的时间点比"外挂转换器"晚。作为缓冲,阶段二应该按管线顺序增量交付——先让 MLIR 管线在单卡配置(不做 placement、无通信指令)下产出合法的 RuntimePlan 并通过 VecApi 端到端测试,再逐步加入多卡 placement 和通信插入。这样协议和 Runtime 的磨合不必等整条编译管线完工。

连带决定:`frontends/dacapo` 里的 HEVM/CST 解析、常量装载和输入输出绑定代码不再迁移,随 HEVM 一起退役(见第 11 节);其中仍有参考价值的部分(如 CKKS 元信息的对账逻辑)以规范文字和测试样例的形式沉淀进协议文档,而不是搬运代码。

## 8. Runtime 与 Api

单卡、多卡、多机不写三套执行器。它们是同一份执行逻辑在不同目标配置下的三个实例:

~~~text
单卡:      world_size=1, device_counts=[1]
单机多卡:  world_size=1, device_counts=[N]
多机多卡:  world_size=R, device_counts=[N0,N1,...]
~~~

这不是愿景,本仓库的 `SequentialRuntime<Api>` 已经这么工作了。

Runtime 只面对一个窄接口:创建值、计算(`compute`)、发起异步通信(`communicate_async`)、等待(`wait`)、最终同步(`synchronize`)、出错全组终止(`abort_all`)。Api 内部怎么组合计算和通信,Runtime 不关心:

~~~text
VecApi          = 明文 vector 计算 + Mock/MPI 通信
PoseidonCpuApi  = Poseidon CPU 算子 + 主机内存/MPI 通信
PoseidonGpuApi  = Poseidon GPU 算子 + CUDA P2P/NCCL/GPU-aware MPI 通信
~~~

Poseidon 现有 mgpu 代码里,GPU 对象的物化和分段拷贝、设备上下文、跨卡通信这些是真金白银的积累,全部迁进 PoseidonApi 继续用。而 mgpu 自己那套调度表示、placement、验证器和执行器,在迁移完成后删除——不长期养两套执行系统。具体哪些删哪些留,见第 11 节的分级清单。

## 9. 测试怎么布防

一类端到端测试打天下是不够的:端到端挂了不知道该怪谁,端到端过了也可能是两边的错误互相抵消。至少四层:

| 层级 | 输入与执行 | 主要验证什么 |
| --- | --- | --- |
| Dacapo 单元测试 | 上层程序 → 计划文件 | lazy-rescale、placement、通信插入、元信息生成对不对 |
| Runtime 单元测试 | 手工构造的计划 → VecApi/MockApi | 解析、验证器、值状态管理、多 rank、出错即停 |
| 两边集成测试 | Dacapo → JSON 文件 → Runtime → VecApi | 双方对协议的理解一致;真实编译产物能跑对 |
| Poseidon 后端测试 | 同一份计划 → Vec/CPU/GPU 三种 Api | 三种后端算出同样的结果,元信息和同步语义一致 |

Runtime 仓库的协议测试集应包含:每种算子和通信指令的最小合法计划;单卡、单机多卡、多进程的代表计划;合法的边界值;以及成对的"必须拒绝"样例——未知版本、未知枚举、缺字段、重复定义 ValueId、位置对不上、Transfer/Replicate 列表长度不匹配、目标能力不符,等等。指纹的稳定性(同一语义、不同排版 → 同一指纹)也要有专门测试。

Dacapo 集成测试的主路径:

~~~text
Dacapo 测试模型
-> 生成 RuntimePlan JSON
-> JSON Schema 检查
-> Runtime 读入并验证
-> SequentialRuntime<VecApi> 执行
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

1. 在 Runtime 仓库建 `docs/runtime-plan/v1/`,写规范、Schema、指纹计算规则;
2. **把 5.2 节的 CKKS 元信息(context、level、scale、NTT、分量数)补进 `ValueDesc` 和验证器**——这是现状和协议之间最大的实现缺口;
3. 建立合法/非法样例文件集;
4. 给 `RuntimePlan` 写独立的 JSON 读取器(当前完全没有 JSON 代码);
5. 验证器覆盖规范要求的全部执行前检查。

### 阶段二:接入 Dacapo

1. Dacapo 作为可选 submodule 进入 Runtime;
2. 在 Dacapo 里搭 MLIR 层的 RuntimePlan 产出管线,先支持单卡配置(无 placement、无通信指令),实现生成端自检和 JSON 写出;
3. 用 VecApi 建立真实编译产物的单卡端到端测试;
4. 加入多卡 placement 和 Transfer/Replicate 插入,扩展端到端测试到多卡、多进程计划。

### 阶段三:接入 Poseidon CPU/GPU

1. Poseidon 以可选构建路径引入 Runtime;
2. 先做单进程单设备的 PoseidonCpuApi;
3. 再做单进程单卡的 PoseidonGpuApi;
4. 迁移同进程跨卡的对象拷贝;
5. 接入跨进程通信;
6. 同一份计划在 Vec/CPU/GPU 三种 Api 下跑,结果必须一致。

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
- `frontends/dacapo/` 整个目录:HEVM/CST 解析、常量装载、输入输出绑定和第三套执行器(`poseidon_gpu_hevm_executor.*`)。HEVM 格式退役后整条链没有存在理由,不迁移(见 7.2);其中的 CKKS 元信息对账逻辑以规范文字和测试样例形式沉淀进协议,而非搬运代码;
- Dacapo 仓库的 `lib/Runtime/{HEAAN,SEAL}_HEVM.cpp`:Dacapo 自带的 HEVM 解释器。注意它同时是差分测试的 CPU 参照实现,且被 `python/hecate/runner.py` 加载——要等新路径有了替代参照(PoseidonCpuApi 跑通并对过账)之后再删;
- 相关的约 38 个测试文件随各自被删的模块一起退役,其中验证的不变量(SSA、位置一致性、拷贝合法性)要确认在 Runtime 测试集中有对应覆盖。

删除完成后,所有执行入口只接受明确版本的 RuntimePlan,不保留任何自动降级的旧路径。

## 12. 结论

- RuntimePlan 是 Runtime 定义并拥有的版本化执行契约,规范、Schema 和测试样例都放在 Runtime 仓库;
- Dacapo 和 Runtime 不共享 C++ 类型,各自实现文件的写和读,靠交叉测试对齐;
- 两边各有一个按同一规范独立实现的检查器,Runtime 的执行前检查是最终把关;
- Dacapo 是 Runtime 的可选 submodule(指向 `dacapo-modified` fork),只用于锁版本和集成测试;
- 计划产出走 MLIR 原生管线直接输出 RuntimePlan,HEVM 字节码及其解释器/解析器整条链退役;
- Poseidon 只引入 Runtime,通过 PoseidonCpuApi/PoseidonGpuApi 提供计算和通信;mgpu 中的 GPU 拷贝与通信代码迁入 Api 层复用,调度和执行代码在迁移验证后删除;
- 单卡、多卡、多机由同一套 Runtime 代码执行,只是目标配置不同;
- 版本、目标或能力对不上时直接报错,永远不猜。
