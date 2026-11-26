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
 * them at HiFi4 into an exact int32 accumulator, the activation runs on it (on the SFPU, after each output block of the
 * matmul; it is not overlapped with the next block, see benchmarks/execute_layer), and the accumulator is requantized
 * per tensor to 8 bits, again with a power-of-two scaler, all in integer arithmetic.
 *
 * With `kx`, `kw` the exponents of x and w and `k_acc` the one of the requantization, `x @ w ~= out * 2^-(kx + kw +
 * k_acc)`.
 *
 * @param x `[M, K]`, FLOAT32 or BFLOAT16, TILE layout, interleaved, on device; finite values; `M`, `K` multiples of 32.
 * @param w `[K, N]`, same dtype as `x`, TILE layout, interleaved, on device; finite values; `N` a multiple of 32.
 * @param activation Applied to the accumulator before the requantization.
 * @param output_memory_config Memory config of all outputs. Defaults to the one of `x`.
 * @return `(out, acc, scaler_x, scaler_w, scaler_out)`:
 *         - `out`: `[M, N]`, UINT8 tensor of TILE layout holding the sign-magnitude int8 bytes (bit 7 = sign, bits 0..6
 * = magnitude, so `0x85` is -5). With ReLU every byte is non-negative.
 *         - `acc`: `[M, N]`, UINT32 holding the raw *sign-magnitude* bits of the FPU's int32 accumulator (bit 31 =
 * sign, bits 0..30 = magnitude; decode `sign ? -mag : mag`), the exact integer matmul `q_x @ q_w` *before* the
 * activation. The execution trace needs it, and recomputing it on the host costs M*K*N multiply-adds, which is what the
 * device was used to avoid. `x @ w ~= acc * 2^-(kx + kw)`.
 *         - `scaler_x`, `scaler_w`: FLOAT32 `[1, 1]`, `2^kx` and `2^kw` (the quantization scalers of x and w).
 *         - `scaler_out`: FLOAT32 `[1, 1]`, `2^k_acc` of the requantization (`k_acc` is usually negative).
 *
 * Limits of this version: one core; FLOAT32 or BFLOAT16 inputs, both the same (bf16 is quantized through UInt8 circular
 * buffers, which the FPU reads as Int8, because an Int8 buffer cannot share a program with a bf16 one; fp16 is not
 * supported); `K <= 66560` (K * 127 * 127 must stay below 2^30; in practice L1 limits K much earlier); x, w, the
 * accumulator and the outputs must fit in the L1 of the core (the operation throws otherwise).
 */
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> execute_layer(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    Activation activation = Activation::ReLU,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);

}  // namespace zkml::prim
