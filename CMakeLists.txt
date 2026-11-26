cmake_minimum_required(VERSION 3.22...4.2)

# Require TT_METAL_HOME
if(NOT DEFINED TT_METAL_HOME)
    if(DEFINED ENV{TT_METAL_HOME})
        set(TT_METAL_HOME $ENV{TT_METAL_HOME})
    else()
        message(FATAL_ERROR "TT_METAL_HOME must be set (environment or -DTT_METAL_HOME=...)")
    endif()
endif()
message(STATUS "Using TT_METAL_HOME: ${TT_METAL_HOME}")

# Verify tt-metal is built
set(TT_METAL_BUILD_DIR "${TT_METAL_HOME}/build")
if(NOT EXISTS "${TT_METAL_BUILD_DIR}")
    message(FATAL_ERROR "tt-metal build directory not found: ${TT_METAL_BUILD_DIR}")
endif()

project(tt_zkml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Dependencies (fmt, umd, spdlog, ...) are found through the build prefix
list(PREPEND CMAKE_PREFIX_PATH "${TT_METAL_BUILD_DIR}")
# Point straight at the installed config; build/tt-metalium-config.cmake is an incomplete intermediate copy
set(TT-Metalium_DIR "${TT_METAL_BUILD_DIR}/lib/cmake/tt-metalium")
set(TT-NN_DIR "${TT_METAL_BUILD_DIR}/lib/cmake/tt-nn")

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium 0.81 REQUIRED)
endif()
if(NOT TARGET TTNN::TTNN)
    find_package(TT-NN 0.81 REQUIRED)
endif()

# Editor setup: generate .clangd from .clangd.in with this machine's absolute paths. 
# Kernels need -I<repo>/src, which clangd cannot take as a relative path (see .clangd.in).
if(PROJECT_IS_TOP_LEVEL)
    set(ZKML_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/.clangd.in ${CMAKE_CURRENT_SOURCE_DIR}/.clangd @ONLY)
endif()

# Device operations. Kernels are JIT-compiled by tt-metal at runtime from their source files;
# the factories locate them relative to their own __FILE__ (see src/common/kernel_path.hpp).
add_library(
    zkml_ops
    src/device/operations/pot_quantization/per_tensor/pot_quantize/pot_quantize_per_tensor.cpp
    src/device/operations/pot_quantization/per_tensor/pot_quantize/device/pot_quantize_per_tensor_device_operation.cpp
    src/device/operations/pot_quantization/per_tensor/pot_quantize/device/pot_quantize_per_tensor_program_factory.cpp
    src/device/operations/pot_quantization/per_tensor/pot_quantize_m31/pot_quant_m31.cpp
    src/device/operations/pot_quantization/per_tensor/pot_quantize_m31/device/pot_quant_m31_device_operation.cpp
    src/device/operations/pot_quantization/per_tensor/pot_quantize_m31/device/pot_quant_m31_program_factory.cpp
    src/device/operations/layer_execution/execute_layer.cpp
    src/device/operations/layer_execution/device/execute_layer_device_operation.cpp
    src/device/operations/layer_execution/device/execute_layer_program_factory.cpp
)
target_include_directories(zkml_ops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(zkml_ops PUBLIC TTNN::TTNN)

add_executable(zkml_quant src/bin/quantization.cpp)
target_link_libraries(zkml_quant PRIVATE TT::Metalium)

add_executable(zkml_pot_quant_per_tensor src/bin/pot_quant_per_tensor.cpp)
target_link_libraries(zkml_pot_quant_per_tensor PRIVATE zkml_ops)
target_compile_definitions(zkml_pot_quant_per_tensor PRIVATE ZKML_TT_METAL_HOME="${TT_METAL_HOME}")

# NoC / DRAM microbenchmark (latency per bank and per core distance, bandwidth); see benchmarks/noc_bench/
add_executable(zkml_noc_bench benchmarks/noc_bench/noc_bench.cpp)
target_include_directories(zkml_noc_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(zkml_noc_bench PRIVATE TT::Metalium)
target_compile_definitions(zkml_noc_bench PRIVATE ZKML_TT_METAL_HOME="${TT_METAL_HOME}")

# Tests (GoogleTest; every one needs a Tenstorrent device).
# tt-metal builds GoogleTest but installs no CMake config for it, so use its build products directly.
enable_testing()
find_package(GTest QUIET)
if(NOT TARGET GTest::gtest_main)
    file(GLOB ZKML_GTEST_INCLUDE_DIRS "${TT_METAL_HOME}/.cpmcache/gtest/*/googletest/include")
    find_package(Threads REQUIRED)
    foreach(lib gtest gtest_main)
        add_library(GTest::${lib} STATIC IMPORTED)
        set_target_properties(
            GTest::${lib}
            PROPERTIES
                IMPORTED_LOCATION "${TT_METAL_BUILD_DIR}/lib/lib${lib}.a"
                INTERFACE_INCLUDE_DIRECTORIES "${ZKML_GTEST_INCLUDE_DIRS}"
        )
    endforeach()
    target_link_libraries(GTest::gtest INTERFACE Threads::Threads)
    target_link_libraries(GTest::gtest_main INTERFACE GTest::gtest)
endif()

# zkml_add_test(<name> <source> <libs...>): one gtest binary, run as a single ctest test. Run the binary
# directly with --gtest_filter to pick cases. Not gtest_discover_tests, which would run it at build time.
function(zkml_add_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE ${ARGN} GTest::gtest_main)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp)
    # Default value for TT_METAL_RUNTIME_ROOT when the binary is run directly
    target_compile_definitions(${name} PRIVATE ZKML_TT_METAL_HOME="${TT_METAL_HOME}")
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES ENVIRONMENT "TT_METAL_RUNTIME_ROOT=${TT_METAL_HOME}")
endfunction()

zkml_add_test(test_pot_quantize_per_tensor tests/cpp/operations/pot_quantization/test_pot_quantize_per_tensor.cpp zkml_ops)
# check_if_cbs_can_fit_in_l1 on its own; deliberately not part of the op tests
zkml_add_test(test_pot_quant_m31 tests/cpp/operations/pot_quantization/test_pot_quant_m31.cpp zkml_ops)
zkml_add_test(test_execute_layer tests/cpp/operations/layer_execution/test_execute_layer.cpp zkml_ops)
zkml_add_test(test_l1_check tests/cpp/common/test_l1_check.cpp TT::Metalium)

# Does keeping the tensor in L1 between the two passes help? Profiler-based bench; see
# benchmarks/quant_tensor_fit_in_l1_bench/README.md
add_executable(zkml_quant_l1_bench benchmarks/quant_tensor_fit_in_l1_bench/bench.cpp)
target_include_directories(zkml_quant_l1_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(zkml_quant_l1_bench PRIVATE TT::Metalium)
target_compile_definitions(zkml_quant_l1_bench PRIVATE ZKML_TT_METAL_HOME="${TT_METAL_HOME}")

# SFPU against FPU multiply in pot_quantize_per_tensor: cycles, wall time, power. See
# Wall-clock benchmark of the execution-trace operations (pot_quant_m31); see benchmarks/quant_tensor_trace_bench/
add_executable(zkml_quant_trace_bench benchmarks/quant_tensor_trace_bench/bench.cpp)
target_link_libraries(zkml_quant_trace_bench PRIVATE zkml_ops)
target_compile_definitions(zkml_quant_trace_bench PRIVATE ZKML_TT_METAL_HOME="${TT_METAL_HOME}")

# benchmarks/quant_tensor_fpu_vs_sfpu/README.md
add_executable(zkml_quant_fpu_vs_sfpu_bench benchmarks/quant_tensor_fpu_vs_sfpu/bench.cpp)
target_include_directories(
    zkml_quant_fpu_vs_sfpu_bench
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp
)
target_link_libraries(zkml_quant_fpu_vs_sfpu_bench PRIVATE TT::Metalium)
target_compile_definitions(zkml_quant_fpu_vs_sfpu_bench PRIVATE ZKML_TT_METAL_HOME="${TT_METAL_HOME}")

# Probe for the FPU primitives (max pool, mul_tiles) used to explain the benchmark above; see
# benchmarks/quant_tensor_fpu_vs_sfpu/probe/
add_executable(zkml_fpu_probe benchmarks/quant_tensor_fpu_vs_sfpu/probe/probe.cpp)
target_include_directories(zkml_fpu_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(zkml_fpu_probe PRIVATE TT::Metalium)
target_compile_definitions(zkml_fpu_probe PRIVATE ZKML_TT_METAL_HOME="${TT_METAL_HOME}")

# int8 against bf16 matmul on the FPU (throughput and exactness); see benchmarks/fpu_int8_vs_bf16/README.md
add_executable(zkml_fpu_int8_vs_bf16_bench benchmarks/fpu_int8_vs_bf16/bench.cpp)
target_include_directories(zkml_fpu_int8_vs_bf16_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(zkml_fpu_int8_vs_bf16_bench PRIVATE TT::Metalium)
target_compile_definitions(zkml_fpu_int8_vs_bf16_bench PRIVATE ZKML_TT_METAL_HOME="${TT_METAL_HOME}")

# Two's complement against sign-magnitude int8 output of the SFPU quantizer: cycles and time; see
# benchmarks/sign_mag_fpu_compatibillity/README.md
add_executable(zkml_sign_mag_fpu_compat_bench benchmarks/sign_mag_fpu_compatibillity/bench.cpp)
target_include_directories(zkml_sign_mag_fpu_compat_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(zkml_sign_mag_fpu_compat_bench PRIVATE TT::Metalium)
target_compile_definitions(zkml_sign_mag_fpu_compat_bench PRIVATE ZKML_TT_METAL_HOME="${TT_METAL_HOME}")

# Cycles and wall time of execute_layer; see benchmarks/execute_layer/README.md
add_executable(zkml_execute_layer_bench benchmarks/execute_layer/bench.cpp)
target_link_libraries(zkml_execute_layer_bench PRIVATE zkml_ops)
target_compile_definitions(zkml_execute_layer_bench PRIVATE ZKML_TT_METAL_HOME="${TT_METAL_HOME}")

# Two's-complement against sign-magnitude accumulator of execute_layer; see benchmarks/execute_layer_2s_vs_sm_acc/README.md
add_executable(zkml_execute_layer_2s_vs_sm_acc_bench benchmarks/execute_layer_2s_vs_sm_acc/bench.cpp)
target_include_directories(zkml_execute_layer_2s_vs_sm_acc_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(zkml_execute_layer_2s_vs_sm_acc_bench PRIVATE TT::Metalium)
target_compile_definitions(zkml_execute_layer_2s_vs_sm_acc_bench PRIVATE ZKML_TT_METAL_HOME="${TT_METAL_HOME}")

# API documentation from the Doxygen comments in src/: `cmake --build .build/default --target docs`,
# output in <build dir>/docs/html/index.html. Optional: the target exists only if Doxygen is installed.
# Graphviz (dot) adds include and call graphs when it is installed.
find_package(Doxygen OPTIONAL_COMPONENTS dot)
if(DOXYGEN_FOUND)
    set(DOXYGEN_PROJECT_NAME "tt-zkml")
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs")
    set(DOXYGEN_RECURSIVE YES)
    set(DOXYGEN_FILE_PATTERNS *.hpp *.cpp *.md)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    set(DOXYGEN_STRIP_FROM_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
    set(DOXYGEN_EXTRACT_ALL YES)  # also list entities that have no doc comment yet
    set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
    set(DOXYGEN_SOURCE_BROWSER YES)
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_QUIET YES)
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
    if(DOXYGEN_DOT_FOUND)
        set(DOXYGEN_HAVE_DOT YES)
    endif()
    doxygen_add_docs(
        docs
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/README.md
        COMMENT "Generating API documentation with Doxygen"
    )
endif()

# Formatting with the repo's .clang-format: `cmake --build .build/default --target format` rewrites every tracked
# C++ file in place, `--target format-check` only reports (non-zero exit if a file would change).
# Optional: the targets exist only if clang-format is installed.
find_program(CLANG_FORMAT_EXE NAMES clang-format-20 clang-format)
if(CLANG_FORMAT_EXE)
    set(ZKML_FORMAT_FILES "git ls-files -z '*.cpp' '*.hpp' '*.h' '*.cc'")
    add_custom_target(
        format
        COMMAND sh -c "${ZKML_FORMAT_FILES} | xargs -0 ${CLANG_FORMAT_EXE} -i"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Formatting C++ sources with clang-format"
        VERBATIM
    )
    add_custom_target(
        format-check
        COMMAND sh -c "${ZKML_FORMAT_FILES} | xargs -0 ${CLANG_FORMAT_EXE} --dry-run -Werror"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Checking C++ formatting with clang-format"
        VERBATIM
    )
endif()
