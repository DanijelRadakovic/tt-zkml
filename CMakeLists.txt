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
    src/operations/pot_quantization/per_tensor/pot_quantize_per_tensor.cpp
    src/operations/pot_quantization/per_tensor/device/pot_quantize_per_tensor_device_operation.cpp
    src/operations/pot_quantization/per_tensor/device/pot_quantize_per_tensor_program_factory.cpp
)
target_include_directories(zkml_ops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(zkml_ops PUBLIC TTNN::TTNN)

add_executable(zkml_quant src/host/quantization.cpp)
target_link_libraries(zkml_quant PRIVATE TT::Metalium)

add_executable(zkml_pot_quant_per_tensor src/host/pot_quant_per_tensor.cpp)
target_link_libraries(zkml_pot_quant_per_tensor PRIVATE zkml_ops)
target_compile_definitions(zkml_pot_quant_per_tensor PRIVATE ZKML_TT_METAL_HOME="${TT_METAL_HOME}")

# Tests (need a Tenstorrent device)
enable_testing()
add_executable(test_pot_quantize_per_tensor tests/cpp/operations/pot_quantization/test_pot_quantize_per_tensor.cpp)
target_link_libraries(test_pot_quantize_per_tensor PRIVATE zkml_ops)
# Set default value for TT_METAL_RUNTIME_ROOT when the binary is run directly
target_compile_definitions(test_pot_quantize_per_tensor PRIVATE ZKML_TT_METAL_HOME="${TT_METAL_HOME}")
add_test(NAME pot_quantize_per_tensor COMMAND test_pot_quantize_per_tensor)
set_tests_properties(pot_quantize_per_tensor PROPERTIES ENVIRONMENT "TT_METAL_RUNTIME_ROOT=${TT_METAL_HOME}")
