cmake_minimum_required(VERSION 3.16)

project(ckks_runtime LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CKKS_RUNTIME_STANDALONE ON)
else()
    set(CKKS_RUNTIME_STANDALONE OFF)
endif()

option(CKKS_RUNTIME_BUILD_TESTS "Build CKKS Runtime unit tests"
       ${CKKS_RUNTIME_STANDALONE})
option(CKKS_RUNTIME_BUILD_EXPERIMENTS
       "Build CKKS Runtime Dacapo review tools"
       ${CKKS_RUNTIME_STANDALONE})
option(CKKS_RUNTIME_BUILD_MPI "Build CKKS Runtime MPI targets" OFF)

find_package(Threads REQUIRED)

function(ckks_runtime_enable_warnings target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    elseif(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    endif()
endfunction()

add_library(ckks_runtime_core STATIC
    runtime/plan.cpp
    runtime/verifier.cpp
    runtime/plaintext_bundle.cpp
    runtime/utils/sha256.cpp
)
add_library(ckks_runtime::core ALIAS ckks_runtime_core)

target_compile_features(ckks_runtime_core PUBLIC cxx_std_17)
target_include_directories(ckks_runtime_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_link_libraries(ckks_runtime_core PUBLIC Threads::Threads)
set_target_properties(ckks_runtime_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME runtime_core
)
ckks_runtime_enable_warnings(ckks_runtime_core)

add_library(ckks_runtime_plan_json STATIC
    runtime/json_plan_reader.cpp
    runtime/operator_spec_reader.cpp
)
add_library(ckks_runtime::plan_json ALIAS ckks_runtime_plan_json)

target_compile_features(ckks_runtime_plan_json PUBLIC cxx_std_17)
target_include_directories(ckks_runtime_plan_json
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_link_libraries(ckks_runtime_plan_json PUBLIC ckks_runtime_core)
set_target_properties(ckks_runtime_plan_json PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME runtime_plan_json
)
ckks_runtime_enable_warnings(ckks_runtime_plan_json)

if(CKKS_RUNTIME_BUILD_TESTS OR CKKS_RUNTIME_BUILD_EXPERIMENTS OR
   CKKS_RUNTIME_BUILD_MPI)
    add_library(ckks_runtime_test_support STATIC
        api/vec_value.cpp
        api/vec_api.cpp
        api/mock_api.cpp
        testing/testing.cpp
    )
    add_library(ckks_runtime::test_support ALIAS ckks_runtime_test_support)
    target_compile_features(ckks_runtime_test_support PUBLIC cxx_std_17)
    target_include_directories(ckks_runtime_test_support
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )
    target_link_libraries(ckks_runtime_test_support
        PUBLIC ckks_runtime_core Threads::Threads)
    set_target_properties(ckks_runtime_test_support PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME runtime_test_support
    )
    ckks_runtime_enable_warnings(ckks_runtime_test_support)
endif()

if(CKKS_RUNTIME_BUILD_TESTS)
    enable_testing()

    add_executable(runtime_tests tests/runtime_tests.cpp)
    target_link_libraries(runtime_tests
        PRIVATE ckks_runtime_plan_json ckks_runtime_test_support)
    target_compile_definitions(runtime_tests
        PRIVATE CKKS_RUNTIME_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    ckks_runtime_enable_warnings(runtime_tests)
    add_test(NAME runtime_tests COMMAND runtime_tests)

    add_executable(runtime_plan_json_tests tests/runtime_plan_json_tests.cpp)
    target_link_libraries(runtime_plan_json_tests
        PRIVATE ckks_runtime_plan_json)
    target_compile_definitions(runtime_plan_json_tests
        PRIVATE CKKS_RUNTIME_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    ckks_runtime_enable_warnings(runtime_plan_json_tests)
    add_test(NAME runtime_plan_json_tests COMMAND runtime_plan_json_tests)
endif()

if(CKKS_RUNTIME_BUILD_EXPERIMENTS)
    add_executable(dacapo_mlp_vec_experiment
        experiments/dacapo_mlp_vec.cpp)
    target_link_libraries(dacapo_mlp_vec_experiment
        PRIVATE ckks_runtime_plan_json ckks_runtime_test_support)
    ckks_runtime_enable_warnings(dacapo_mlp_vec_experiment)

    add_executable(dacapo_plan_vec_diff
        experiments/dacapo_plan_vec_diff.cpp)
    target_link_libraries(dacapo_plan_vec_diff
        PRIVATE ckks_runtime_plan_json ckks_runtime_test_support)
    ckks_runtime_enable_warnings(dacapo_plan_vec_diff)

    add_executable(dacapo_mlp_vec_e2e
        experiments/dacapo_mlp_vec_e2e.cpp)
    target_link_libraries(dacapo_mlp_vec_e2e
        PRIVATE ckks_runtime_plan_json ckks_runtime_test_support)
    ckks_runtime_enable_warnings(dacapo_mlp_vec_e2e)
endif()

if(CKKS_RUNTIME_BUILD_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)

    add_library(ckks_runtime_mpi_api STATIC api/mpi_api.cpp)
    add_library(ckks_runtime::mpi_api ALIAS ckks_runtime_mpi_api)
    target_compile_features(ckks_runtime_mpi_api PUBLIC cxx_std_17)
    target_include_directories(ckks_runtime_mpi_api
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )
    target_link_libraries(ckks_runtime_mpi_api
        PUBLIC ckks_runtime_core MPI::MPI_CXX)
    set_target_properties(ckks_runtime_mpi_api PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME runtime_mpi_api
    )
    ckks_runtime_enable_warnings(ckks_runtime_mpi_api)

    add_executable(mpi_runtime_test tests/mpi_runtime_test.cpp)
    target_link_libraries(mpi_runtime_test
        PRIVATE ckks_runtime_mpi_api ckks_runtime_test_support)
    ckks_runtime_enable_warnings(mpi_runtime_test)

    add_executable(mpi_env_check tools/mpi_env_check.cpp)
    target_link_libraries(mpi_env_check PRIVATE MPI::MPI_CXX)
    ckks_runtime_enable_warnings(mpi_env_check)

    add_executable(mpi_comm_benchmark benchmarks/mpi_comm_benchmark.cpp)
    target_link_libraries(mpi_comm_benchmark PRIVATE MPI::MPI_CXX)
    ckks_runtime_enable_warnings(mpi_comm_benchmark)

    if(CKKS_RUNTIME_BUILD_TESTS)
        add_test(
            NAME mpi_runtime_2
            COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:mpi_runtime_test>
                ${MPIEXEC_POSTFLAGS}
        )
        add_test(
            NAME mpi_runtime_4
            COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:mpi_runtime_test>
                ${MPIEXEC_POSTFLAGS}
        )
    endif()
endif()
