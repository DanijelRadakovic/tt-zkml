cmake_minimum_required(VERSION 3.22...4.2)
project(tt_zkml_abi VERSION 0.1.0 LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(TtMetal)  # TT_METAL_HOME, GTest

# The core is built as a subproject unless a parent (e.g. the repository's root CMakeLists.txt) has added it already.
# Where it lives is configurable so this project does not depend on the monorepo layout: -DTT_ZKML_CORE_SOURCE_DIR=...
set(TT_ZKML_CORE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tt-zkml-core" CACHE PATH "Source directory of tt-zkml-core")
if(NOT TARGET zkml_api)
    add_subdirectory(${TT_ZKML_CORE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/tt-zkml-core)
endif()

# The C ABI of tt-zkml-core as one shared library: Rust (tt-zkml-sys) and Python (abi/python) bind to it.
add_library(tt_zkml_abi SHARED src/tt_zkml.cpp)
target_include_directories(
    tt_zkml_abi
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)
target_link_libraries(tt_zkml_abi PRIVATE zkml_api)
target_compile_definitions(tt_zkml_abi PRIVATE TTZK_VERSION_STRING="${PROJECT_VERSION}")
set_target_properties(
    tt_zkml_abi
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        VERSION ${PROJECT_VERSION}
        SOVERSION 0
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
install(TARGETS tt_zkml_abi LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

# Tests of the ABI (GoogleTest). The one with a device needs a Tenstorrent device.
enable_testing()
add_executable(test_abi tests/test_abi.cpp)
target_link_libraries(test_abi PRIVATE tt_zkml_abi GTest::gtest_main)
target_compile_definitions(test_abi PRIVATE ZKML_TT_METAL_HOME="${TT_METAL_HOME}")
add_test(NAME test_abi COMMAND test_abi)
set_tests_properties(test_abi PROPERTIES ENVIRONMENT "TT_METAL_RUNTIME_ROOT=${TT_METAL_HOME}")
