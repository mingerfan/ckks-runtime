-- xmake 构建脚本：MPI 测试程序（C++）。
-- 所有产物输出到 build/，保持根目录干净。
--
--   xmake f -m release        # 配置（默认 release）
--   xmake                     # 编译 -> build/mpi_test, build/mpi_test_comm
--   xmake run mpi_test_comm   # 经 mpirun 运行通信基准（-np 2）
--   xmake run mpi_test        # 经 mpirun 运行可用性自检（-np 4，含 VecOps 自检）
--   xmake clean               # 清理 build/
--
-- 源码为 C++(.cpp)，mpi_test 内的 VecOps 工具类见 operations.hpp。

add_rules("mode.release", "mode.debug")

set_project("mpi_test")
set_languages("c11", "cxx17")
set_warnings("all", "extra")

-- 二进制直接落到 build/ 根下，而不是默认的 build/<plat>/<arch>/<mode>/
set_targetdir("$(builddir)")

-- 通过 pkg-config 找 OpenMPI（系统自带 ompi.pc；nix devshell 里 nixpkgs 的
-- openmpi 也带 .pc），自动取得头文件和库，比硬编码路径更通用。
add_requires("pkgconfig::ompi", {alias = "mpi"})

-- OpenMPI 的 .pc 里 "-Wl,-rpath" 与路径分成两个 token，xmake 的 flag 校验会误判并忽略，
-- 关掉自动忽略，让这些链接标志原样传给链接器。
set_policy("check.auto_ignore_flags", false)

target("mpi_test")
    set_kind("binary")
    add_files("mpi_test.cpp")
    add_packages("mpi")
    on_run(function (target)
        os.execv("mpirun", {"-np", "4", target:targetfile()})
    end)

target("mpi_test_comm")
    set_kind("binary")
    add_files("mpi_test_comm.cpp")
    add_packages("mpi")
    on_run(function (target)
        os.execv("mpirun", {"-np", "2", target:targetfile()})
    end)
