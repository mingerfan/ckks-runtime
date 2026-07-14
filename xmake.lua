add_rules("mode.release", "mode.debug")
add_rules("plugin.compile_commands.autoupdate", {outputdir = ".", lsp = "clangd"})

set_project("ckks_runtime")
set_languages("cxx17")
set_warnings("all", "extra", "pedantic")
set_targetdir("$(builddir)")

add_requires("pkgconfig::ompi", {alias = "mpi"})
set_policy("check.auto_ignore_flags", false)

target("runtime_core")
    set_kind("static")
    add_includedirs(".", {public = true})
    add_files(
        "runtime/plan.cpp",
        "runtime/verifier.cpp",
        "api/vec_value.cpp",
        "api/vec_api.cpp",
        "api/mock_api.cpp",
        "testing/testing.cpp")
    add_syslinks("pthread")

target("runtime_tests")
    set_kind("binary")
    add_files("tests/runtime_tests.cpp")
    add_deps("runtime_core")

target("mpi_runtime_test")
    set_kind("binary")
    add_files("tests/mpi_runtime_test.cpp", "api/mpi_api.cpp")
    add_deps("runtime_core")
    add_packages("mpi")

target("mpi_env_check")
    set_kind("binary")
    add_files("tools/mpi_env_check.cpp")
    add_packages("mpi")

target("mpi_comm_benchmark")
    set_kind("binary")
    add_files("benchmarks/mpi_comm_benchmark.cpp")
    add_packages("mpi")
