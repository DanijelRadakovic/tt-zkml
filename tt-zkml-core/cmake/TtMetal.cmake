# Finds tt-metal (TT_METAL_HOME) and provides TT::Metalium, TTNN::TTNN and GTest::gtest(_main).
# Self-contained on purpose: abi/cmake/TtMetal.cmake is a copy, so each project can live in its own repository.
include_guard(GLOBAL)

if(NOT DEFINED TT_METAL_HOME)
    if(DEFINED ENV{TT_METAL_HOME} AND NOT "$ENV{TT_METAL_HOME}" STREQUAL "")
        set(TT_METAL_HOME $ENV{TT_METAL_HOME})
    else()
        message(FATAL_ERROR "TT_METAL_HOME must be set (environment or -DTT_METAL_HOME=...)")
    endif()
endif()
message(STATUS "Using TT_METAL_HOME: ${TT_METAL_HOME}")

set(TT_METAL_BUILD_DIR "${TT_METAL_HOME}/build")
if(NOT EXISTS "${TT_METAL_BUILD_DIR}")
    message(FATAL_ERROR "tt-metal build directory not found: ${TT_METAL_BUILD_DIR}")
endif()

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

# tt-metal builds GoogleTest but installs no CMake config for it, so use its build products directly.
find_package(GTest QUIET)
if(NOT TARGET GTest::gtest_main)
    file(GLOB ZKML_GTEST_INCLUDE_DIRS "${TT_METAL_HOME}/.cpmcache/gtest/*/googletest/include")
    find_package(Threads REQUIRED)
    foreach(lib gtest gtest_main)
        add_library(GTest::${lib} STATIC IMPORTED GLOBAL)
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
