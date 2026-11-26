cmake_minimum_required(VERSION 3.22...4.2)
project(tt_zkml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#Require TT_METAL_HOME
if(NOT DEFINED TT_METAL_HOME)
    if(DEFINED ENV{TT_METAL_HOME})
        set(TT_METAL_HOME $ENV{TT_METAL_HOME})
    else()
        message(FATAL_ERROR "TT_METAL_HOME must be set (environment or -DTT_METAL_HOME=...)")
    endif()
endif()

message(STATUS "Using TT_METAL_HOME: ${TT_METAL_HOME}")

#Verify tt-metal is built
set(TT_METAL_BUILD_DIR "${TT_METAL_HOME}/build")
if(NOT EXISTS "${TT_METAL_BUILD_DIR}")
    message(FATAL_ERROR "tt-metal build directory not found: ${TT_METAL_BUILD_DIR}")
endif()

# Dependencies (fmt, umd, spdlog, ...) are found through the build prefix
list(PREPEND CMAKE_PREFIX_PATH "${TT_METAL_BUILD_DIR}")
# Point straight at the installed config; build/tt-metalium-config.cmake is an incomplete intermediate copy
set(TT-Metalium_DIR "${TT_METAL_BUILD_DIR}/lib/cmake/tt-metalium")

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

add_executable(zkml_quant host/quantazation.cpp)
target_link_libraries(zkml_quant PRIVATE TT::Metalium)
