// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include "operations/pot_quantization/bit_width.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace zkml::prim {

/**
 * @brief Per-tensor power-of-two (PoT) symmetric quantization. The device computes the scaler.
 *
 * The formulas are:
 * - `Q = 2^(bit_width.bits() - 1) - 1`,
 * - `k = max{k : absmax(x) * 2^k <= Q}`,
 * - `q = clamp(round_half_even(x * 2^k), -Q, Q)`.
 *
 * The device finds `absmax(x)` and derives `k` from it, so the element with the largest magnitude
 * maps into `[-Q, Q]` without clipping. `k` is at most 127, so `2^k` stays a finite fp32.
 * An all-zero input gets `k = 127`.
 *
 * The operation launches one device program. The program reads the input twice: once to find `absmax(x)`
 * and once to quantize it (see `device/pot_quantize_per_tensor_device_operation.hpp`).
 *
 * @param input Tensor to quantize. It must be on device, in TILE layout, interleaved, FLOAT32 or
 *        BFLOAT16, and contain only finite values.
 * @param bit_width Number of bits used for the scaler.
 * @param output_dtype Data type of the quantized output tensor. INT32 works for any width. INT8 works only for
 *        `BitWidth::Bits8`.
 * @param output_memory_config Memory config of both outputs. Defaults to the memory config of the input.
 * @param noc_read_tile_batch_size Number of tiles the reader reads in parallel (async_read) before it waits on
 *        one async_read_barrier.
 *        - Must be at least 1. It is capped at the tile count of the input.
 *        - A smaller batch lets compute start sooner.
 *        - A larger batch keeps the reader ahead of compute.
 * @param cb_in_size_factor Size of the input CB in batches. 
 *        The CB holds `noc_read_tile_batch_size * cb_in_size_factor` tiles.
 *        - Must be at least 2, so the reader can fill one batch while compute drains the other.
 *        - A larger factor lets the reader run further ahead when reads are slow.
 *        - The CB must fit in L1, otherwise the operation throws.
 * @return `(quantized, scaler)`.
 *         - `quantized` has the shape of `input` and TILE layout.
 *         - `scaler` is a FLOAT32 tensor of shape `[1, 1]` holding `2^k`.
 *         Dequantize with `x ~= quantized / scaler`.
 */
std::tuple<ttnn::Tensor, ttnn::Tensor> pot_quantize_per_tensor(
    const ttnn::Tensor& input,
    BitWidth bit_width,
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INT32,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt,
    uint32_t noc_read_tile_batch_size = 16,
    uint32_t cb_in_size_factor = 2);

}  // namespace zkml::prim
