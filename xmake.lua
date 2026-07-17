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
    add_includedirs("third_party")
    add_files(
        "runtime/plan.cpp",
        "runtime/verifier.cpp",
        "runtime/plaintext_bundle.cpp",
        "runtime/utils/sha256.cpp",
        "api/vec_value.cpp",
        "api/vec_api.cpp",
        "api/mock_api.cpp",
        "testing/testing.cpp")
    add_syslinks("pthread")

target("runtime_plan_json")
    set_kind("static")
    add_includedirs(".", {public = true})
    add_includedirs("third_party")
    add_files(
        "runtime/json_plan_reader.cpp",
        "runtime/operator_spec_reader.cpp")
    add_deps("runtime_core")

target("runtime_tests")
    set_kind("binary")
    add_files("tests/runtime_tests.cpp")
    add_deps("runtime_plan_json")
    add_defines("CKKS_RUNTIME_SOURCE_DIR=\"$(projectdir)\"")

target("runtime_plan_json_tests")
    set_kind("binary")
    add_files("tests/runtime_plan_json_tests.cpp")
    add_deps("runtime_plan_json")
    add_defines("CKKS_RUNTIME_SOURCE_DIR=\"$(projectdir)\"")

target("dacapo_mlp_vec_experiment")
    set_kind("binary")
    add_files("experiments/dacapo_mlp_vec.cpp")
    add_deps("runtime_plan_json")

target("dacapo_mlp_mpi_isolation_experiment")
    set_kind("binary")
    add_files("experiments/dacapo_mlp_mpi_isolation.cpp", "api/mpi_api.cpp")
    add_deps("runtime_plan_json")
    add_packages("mpi")

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
