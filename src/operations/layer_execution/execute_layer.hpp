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

/// The activation of a layer, applied on the integer accumulator of the matmul (exact for ReLU).
enum class Activation : uint32_t {
    /// No activation: the accumulator is requantized as it is.
    None = 0,
    /// `max(0, acc)`.
    ReLU = 1,
};

/**
 * @brief One quantized layer on one core: `out = quantize(activation(quantize(x) @ quantize(w)))`.
 *
 * Both inputs are quantized per tensor to 8 bits with a power-of-two scaler (the formulas of `pot_quantize_per_tensor`,
 * `Q = 127`, round half to even) into sign-magnitude int8 bytes, the format of the FPU's Int8 type. The FPU multiplies
 * them at HiFi4 into an exact int32 accumulator, the activation runs on it (overlapped with the next output block of the
 * matmul), and the accumulator is requantized per tensor to 8 bits, again with a power-of-two scaler.
 *
 * With `kx`, `kw` the exponents of x and w and `k_acc` the one of the requantization, `x @ w ~= out * 2^-(kx + kw + k_acc)`.
 *
 * @param x `[M, K]`, FLOAT32, TILE layout, interleaved, on device; finite values; `M`, `K` multiples of 32.
 * @param w `[K, N]`, FLOAT32, TILE layout, interleaved, on device; finite values; `N` a multiple of 32.
 * @param activation Applied to the accumulator before the requantization.
 * @param output_memory_config Memory config of all outputs. Defaults to the one of `x`.
 * @return `(out, scaler_x, scaler_w, scaler_out)`:
 *         - `out`: `[M, N]`, UINT8 tensor of TILE layout holding the sign-magnitude int8 bytes (bit 7 = sign, bits 0..6 =
 *           magnitude, so `0x85` is -5). With ReLU every byte is non-negative.
 *         - `scaler_x`, `scaler_w`: FLOAT32 `[1, 1]`, `2^kx` and `2^kw` (the quantization scalers of x and w).
 *         - `scaler_out`: FLOAT32 `[1, 1]`, `2^k_acc` of the requantization (`k_acc` is usually negative).
 *
 * Limits of this version: one core; FLOAT32 inputs only (an Int8 circular buffer cannot share a program with a bf16 one);
 * `K <= 1024` (the accumulator is converted to fp32 exactly only below 2^24); x, w, the accumulator and the outputs
 * must fit in the L1 of the core (the operation throws otherwise).
 */
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> execute_layer(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    Activation activation = Activation::ReLU,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);

}  // namespace zkml::prim
