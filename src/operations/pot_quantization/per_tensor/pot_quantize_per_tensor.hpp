// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace zkml::prim {

/**
 * @brief Per-tensor power-of-two (PoT) symmetric quantization with the scaler computed on device:
 *        `Q = 2^(bit_width - 1) - 1`, `k = max{k : absmax(x) * 2^k <= Q}`,
 *        `q = clamp(round_half_even(x * 2^k), -Q, Q)`.
 *
 * The device finds `absmax(x)` and derives `k` from it, so the largest-magnitude element maps
 * into [-Q, Q] without clipping. `k` is capped at 127 so `2^k` stays a finite fp32 (an all-zero
 * input gets `k = 127`). Launches exactly one device program, which reads the input twice
 * (see device/pot_quantize_per_tensor_device_operation.hpp).
 *
 * @param input                TILE layout, interleaved, FLOAT32 or BFLOAT16, on device. Must be finite.
 * @param bit_width            8, 16 or 32.
 * @param output_dtype         INT32 (any bit width) or INT8 (bit_width 8 only).
 * @param output_memory_config Memory config of both outputs. Defaults to the input's.
 * @return `(quantized, scaler)`: the quantized tensor (same shape as `input`, TILE layout) and a
 *         FLOAT32 [1, 1] tensor holding the scaler `2^k`, so `x ~= quantized / scaler`.
 */
std::tuple<ttnn::Tensor, ttnn::Tensor> pot_quantize_per_tensor(
    const ttnn::Tensor& input,
    uint32_t bit_width,
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INT32,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);

}  // namespace zkml::prim
