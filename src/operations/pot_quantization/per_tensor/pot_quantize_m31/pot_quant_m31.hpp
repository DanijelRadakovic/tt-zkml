// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>

#include "operations/pot_quantization/pot_range.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace zkml::prim {

/**
 * @brief Maps a PoT-quantized tensor into the Mersenne field M31 (p = 2^31 - 1), on the device.
 *
 * Every element `q` becomes its residue `q mod p`, as an unsigned integer in `[0, p)`:
 * `q >= 0` stays `q`, `q < 0` becomes `p + q`. The map is injective only if `Q <= (p - 1) / 2`, so the quantization
 * must use `QuantRange::M31` (at 32 bits `Q = 2^30 - 1`; 8 and 16 bits fit anyway). A verifier then reads a residue
 * `m` back as the signed number `m <= (p - 1) / 2 ? m : m - p`. With `QuantRange::Full` at 32 bits, `q = 2^31 - 1`
 * would map to `p` itself, which is not a canonical residue, and `-p`, `0`, `p` would all be the field element 0.
 *
 * @param quantized Output of `pot_quantize_per_tensor(..., QuantRange::M31)`: INT32 or INT8, on device, TILE layout, interleaved, every
 *        element in `[-Q, Q]` (not checked: the device cannot report an error, and the quantizer cannot produce
 *        anything else). INT8 is read as the two's-complement bytes the quantizer wrote.
 * @param output_memory_config Memory config of the output. Defaults to the memory config of the input.
 * @return UINT32 tensor of the shape of `quantized`, TILE layout, every element a canonical M31 value in `[0, p)`.
 *         These are the cells of a stwo trace column.
 */
ttnn::Tensor pot_quant_m31(
    const ttnn::Tensor& quantized, const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);

}  // namespace zkml::prim
