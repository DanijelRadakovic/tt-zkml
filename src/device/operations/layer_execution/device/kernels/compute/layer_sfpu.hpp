// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// SFPU step between the FPU matmul and the requantization of execute_layer: for each tile of the int32 accumulator the
// FPU leaves in DST (integer "32": sign-magnitude), apply the activation on the integers, convert to fp32 and fold the
// tile into the running absmax in LReg7 (the register the quantizer's absmax uses, see pot_scale_sfpu.hpp).
//
//   acc_to_float_absmax_tile<RELU>(idst)    DST tile idst, in place: int32 -> fp32 [after max(0, .)], absmax into LReg7
//
// ReLU on the sign-magnitude value is "if the sign bit is set, zero it" (-0 included); what is left is non-negative,
// where sign-magnitude and two's complement are the same bits. The conversion int32 -> fp32 is exact while |acc| < 2^24
// (the op limits K to 1024, so |acc| <= 1024 * 127 * 127 < 2^24). Running the activation on the accumulator before the
// requantize is exact for ReLU and makes the absmax (hence the scale) the one of the activated tensor.
//
// After absmax_init() (which also sets LReg7 to 0), call this for every accumulator tile; absmax_to_scaler() then turns
// LReg7 into the scaler 2^k of the requantization. Only the MATH thread runs SFPU code: call through MATH(...).

#pragma once

#include <cstdint>

#include "device/kernels/compute/pot_scale_sfpu.hpp"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpi.h"
#endif

namespace zkml::sfpu {

#ifdef TRISC_MATH

template <bool RELU, int ITERATIONS = 8>
inline void acc_to_float_absmax_face() {
    sfpi::vFloat acc = sfpi::l_reg[SCALE_LREG];
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt bits = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();  // raw bits: sign-magnitude
        if constexpr (RELU) {
            v_if(bits < 0) { bits = 0; }  // sign bit set: negative or -0
            v_endif;
        }
        const sfpi::vFloat f = sfpi::convert<sfpi::vFloat>(sfpi::vSMag(bits.get()), sfpi::RoundMode::Nearest);
        sfpi::dst_reg[0].mode<sfpi::DataLayout::F32>() = f;
        acc = max_nonneg(acc, sfpi::abs(f));
        sfpi::dst_reg++;
    }
    sfpi::l_reg[SCALE_LREG] = acc;
}

template <bool RELU>
inline void acc_to_float_absmax_tile(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(acc_to_float_absmax_face<RELU, 8>, idst, VectorMode::RC);
}

#endif  // TRISC_MATH

}  // namespace zkml::sfpu
