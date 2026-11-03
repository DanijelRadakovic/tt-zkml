// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace zkml::prim {

/**
 * @brief Per-tensor power-of-two (PoT) symmetric quantization:
 *        `q = clamp(round_half_even(x * 2^k))` into the int8 range.
 *
 * The host chooses `k` (the number of fractional bits, scale = `2^-k`). The device only applies it.
 * Launches exactly one device program (see device/pot_quantize_per_tensor_device_operation.hpp).
 *
 * @param input                TILE layout, interleaved, FLOAT32 or BFLOAT16, on device.
 * @param k                    Fractional bits. `2^k` must be a normal fp32 value (`-126 <= k <= 127`).
 * @param output_dtype         INT32 (values in int8 range) or INT8.
 * @param output_memory_config Output memory config. Defaults to the input's.
 * @return The quantized tensor, same shape as `input`, TILE layout.
 */
ttnn::Tensor pot_quantize_per_tensor(
    const ttnn::Tensor& input,
    int32_t k,
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INT32,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);

}  // namespace zkml::prim
