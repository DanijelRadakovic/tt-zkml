// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <string>

namespace zkml::common {

/**
 * @brief Absolute path of a kernel source file, given relative to the calling source file.
 *
 * tt-metal compiles kernels at runtime from their source files, so the host must pass a path it can
 * find. Pass `__FILE__` as `source_file`. CMake gives the compiler absolute source paths, so
 * `__FILE__` is absolute too. The result does not depend on the current working directory.
 *
 * Example: called from
 * `src/operations/pot_quantization/per_tensor/device/pot_quantize_per_tensor_program_factory.cpp` as
 * `kernel_path(__FILE__, "kernels/dataflow/reader.cpp")`, it returns
 * `<repo>/src/operations/pot_quantization/per_tensor/device/kernels/dataflow/reader.cpp`.
 *
 * @param source_file   Path of the calling source file (`__FILE__`).
 * @param relative_path Kernel path relative to the directory of `source_file`.
 * @return Normalized path to the kernel (`..` segments resolved).
 */
inline std::string kernel_path(const char* source_file, const char* relative_path) {
    const std::filesystem::path source_dir = std::filesystem::path(source_file).parent_path();
    return (source_dir / relative_path).lexically_normal().string();
}

/**
 * @brief Absolute path of this repo's `src/` directory, for use as a kernel include path.
 *
 * tt-metal compiles kernels at runtime and does not search `src/` for includes. Passing this
 * directory in a kernel config's `compiler_include_paths` lets kernels include shared headers such
 * as `#include "common/kernel_debug.hpp"`.
 *
 * Here `__FILE__` is the path of this header, `<repo>/src/common/kernel_path.hpp`. It is absolute
 * because CMake puts `src/` on the include path as an absolute path.
 *
 * @return `<repo>/src`.
 */
inline std::string kernel_include_dir() {
    return std::filesystem::path(__FILE__).parent_path().parent_path().lexically_normal().string();
}

}  // namespace zkml::common
