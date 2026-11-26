// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// SFPU function that maps a signed quantized integer in DST to its residue modulo p = 2^31 - 1.
//
// For an Int32 tile `q in [-Q, Q]` with `Q <= 2^30 - 1`:   m = q < 0 ? q + p : q      (in [0, p))
// For a tile read from an Int8 tensor, DST holds the byte as 0..255 (two's complement): the values 128..255 stand for
// -128..-1, so   m = b > 127 ? b + (p - 256) : b.
// p + q is below 2^31, so the int32 add cannot wrap.

#pragma once

#include <cstdint>

#include "operations/pot_quantization/pot_range.hpp"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpi.h"
#endif

namespace zkml::sfpu {

#ifdef TRISC_MATH

template <bool FROM_INT8, int ITERATIONS = 8>
inline void to_m31_face() {
    constexpr int32_t P = static_cast<int32_t>(zkml::prim::m31_modulus);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        if constexpr (FROM_INT8) {
            v_if(v > 127) { v += P - 256; }
            v_endif;
        } else {
            v_if(v < 0) { v += P; }
            v_endif;
        }
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = v;
        sfpi::dst_reg++;
    }
}

/// Configures the SFPU. Call once, through MATH(...).
inline void to_m31_init() { ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>(); }

/// Converts DST tile `idst` in place. Call through MATH(...).
template <bool FROM_INT8>
inline void to_m31_tile(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(to_m31_face<FROM_INT8, 8>, idst, VectorMode::RC);
}

#endif  // TRISC_MATH

}  // namespace zkml::sfpu
