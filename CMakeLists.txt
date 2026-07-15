cmake_minimum_required(VERSION 3.16)

project(ckks_runtime LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CKKS_RUNTIME_BUILD_TESTS_DEFAULT ON)
else()
    set(CKKS_RUNTIME_BUILD_TESTS_DEFAULT OFF)
endif()

option(CKKS_RUNTIME_BUILD_TESTS "Build CKKS Runtime unit tests"
       ${CKKS_RUNTIME_BUILD_TESTS_DEFAULT})
option(CKKS_RUNTIME_BUILD_MPI "Build CKKS Runtime MPI targets" OFF)

find_package(Threads REQUIRED)

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
set_target_properties(ckks_runtime_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
set_target_properties(ckks_runtime_plan_json PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CKKS_RUNTIME_BUILD_TESTS OR CKKS_RUNTIME_BUILD_MPI)
    add_library(ckks_runtime_test_support STATIC
        api/vec_value.cpp
        api/vec_api.cpp
        api/mock_api.cpp
        testing/testing.cpp
    )
    target_compile_features(ckks_runtime_test_support PUBLIC cxx_std_17)
    target_include_directories(ckks_runtime_test_support
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )
    target_link_libraries(ckks_runtime_test_support PUBLIC ckks_runtime_core Threads::Threads)
endif()

if(CKKS_RUNTIME_BUILD_TESTS)
    enable_testing()

    add_executable(ckks_runtime_tests tests/runtime_tests.cpp)
    target_link_libraries(ckks_runtime_tests
        PRIVATE
            ckks_runtime_plan_json
            ckks_runtime_test_support
    )
    target_compile_definitions(ckks_runtime_tests
        PRIVATE CKKS_RUNTIME_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME ckks_runtime_tests COMMAND ckks_runtime_tests)

    add_executable(ckks_runtime_plan_json_tests tests/runtime_plan_json_tests.cpp)
    target_link_libraries(ckks_runtime_plan_json_tests PRIVATE ckks_runtime_plan_json)
    target_compile_definitions(ckks_runtime_plan_json_tests
        PRIVATE CKKS_RUNTIME_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME ckks_runtime_plan_json_tests COMMAND ckks_runtime_plan_json_tests)
endif()

if(CKKS_RUNTIME_BUILD_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)

    add_executable(ckks_runtime_mpi_test
        tests/mpi_runtime_test.cpp
        api/mpi_api.cpp
    )
    target_link_libraries(ckks_runtime_mpi_test
        PRIVATE
            ckks_runtime_core
            ckks_runtime_test_support
            MPI::MPI_CXX
    )
endif()
