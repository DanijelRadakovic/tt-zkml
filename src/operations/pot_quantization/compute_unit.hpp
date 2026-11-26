// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace zkml::prim {

/// Which Tensix math unit multiplies the input by the scaler `2^k` in pass 2 of `pot_quantize_per_tensor`.
///
/// The FPU (matrix unit) multiplies a tile for less than the SFPU (vector unit), but it only keeps 19 bits of
/// an operand: its SrcA/SrcB registers hold at most a 19-bit float. The input times `2^k` is still exact for
/// a BFLOAT16 input (a 16-bit value times a power of two), but a FLOAT32 input would be truncated.
/// Only the multiply moves: the absmax of pass 1 stays on the SFPU, because the FPU has no abs and no min.
enum class ComputeUnit : uint32_t {
    /// The operation picks the faster unit: the FPU for BFLOAT16 input at 16 or 32 bits, the SFPU otherwise.
    Auto,
    /// The SFPU multiplies, for every input dtype.
    Sfpu,
    /// The FPU multiplies. The input must be BFLOAT16.
    Fpu,
};

}  // namespace zkml::prim
