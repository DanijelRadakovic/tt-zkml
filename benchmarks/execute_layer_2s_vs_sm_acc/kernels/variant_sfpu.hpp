// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// BENCHMARK COPY of the first version of execute_layer's SFPU steps (acc_finalize_absmax_tile, int_to_float_tile: the
// accumulator converted to two's complement, float absmax, int32 -> fp32 requantization input) plus the intermediate
// variant that keeps the sign-magnitude accumulator but still goes through fp32 (acc_sm_absmax_float_tile,
// sm_to_float_tile). SFPU steps around the int32 accumulator of execute_layer.
//
//   acc_finalize_absmax_tile<RELU>(idst)   after the matmul: DST tile idst holds the FPU's int32 accumulator (integer
//   "32",
//                                          sign-magnitude). In place it becomes the two's complement int32 (what is
//                                          packed to L1 and returned as the `acc` tensor, before the activation), and
//                                          the absmax of the activated accumulator (|acc|, or max(0, acc) with ReLU) is
//                                          folded into LReg7.
//   int_to_float_tile<RELU>(idst)          before the requantization: a tile of two's complement int32 (read back from
//   L1)
//                                          becomes fp32, after max(0, .) with ReLU, in place. Exact while |acc| < 2^24.
//
// The activation runs on the integers: with ReLU the absmax and the requantization see max(0, acc). The int32 -> fp32
// conversion is exact while |acc| < 2^24; the op limits K to 1024, so |acc| <= 1024 * 127 * 127 < 2^24.
//
// After absmax_init() (which also sets LReg7 to 0), call acc_finalize_absmax_tile for every accumulator tile;
// absmax_to_scaler() then turns LReg7 into the scaler 2^k of the requantization. Only the MATH thread runs SFPU code:
// call through MATH(...).

#pragma once

#include <cstdint>

#include "device/kernels/compute/pot_scale_sfpu.hpp"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpi.h"
#endif

namespace zkml::sfpu {

#ifdef TRISC_MATH

template <bool RELU, int ITERATIONS = 8>
inline void acc_finalize_absmax_face() {
    sfpi::vFloat acc = sfpi::l_reg[SCALE_LREG];
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt bits = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();  // raw bits: sign-magnitude
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() =
            sfpi::convert<sfpi::vInt>(sfpi::vSMag(bits.get()));  // two's complement, for the acc output
        if constexpr (RELU) {
            v_if(bits < 0) { bits = 0; }  // sign bit set: negative or -0
            v_endif;
        }
        const sfpi::vFloat f = sfpi::convert<sfpi::vFloat>(sfpi::vSMag(bits.get()), sfpi::RoundMode::Nearest);
        acc = max_nonneg(acc, sfpi::abs(f));
        sfpi::dst_reg++;
    }
    sfpi::l_reg[SCALE_LREG] = acc;
}

template <bool RELU>
inline void acc_finalize_absmax_tile(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(acc_finalize_absmax_face<RELU, 8>, idst, VectorMode::RC);
}

template <bool RELU, int ITERATIONS = 8>
inline void int_to_float_face() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();  // two's complement
        if constexpr (RELU) {
            v_if(v < 0) { v = 0; }
            v_endif;
        }
        sfpi::dst_reg[0].mode<sfpi::DataLayout::F32>() = sfpi::convert<sfpi::vFloat>(v, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

template <bool RELU>
inline void int_to_float_tile(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(int_to_float_face<RELU, 8>, idst, VectorMode::RC);
}

// Variant "sm_float": the accumulator stays sign-magnitude (not modified), the absmax is still float.
template <bool RELU, int ITERATIONS = 8>
inline void acc_sm_absmax_float_face() {
    sfpi::vFloat acc = sfpi::l_reg[SCALE_LREG];
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt bits = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        if constexpr (RELU) {
            v_if(bits < 0) { bits = 0; }
            v_endif;
        }
        const sfpi::vFloat f = sfpi::convert<sfpi::vFloat>(sfpi::vSMag(bits.get()), sfpi::RoundMode::Nearest);
        acc = max_nonneg(acc, sfpi::abs(f));
        sfpi::dst_reg++;
    }
    sfpi::l_reg[SCALE_LREG] = acc;
}

template <bool RELU>
inline void acc_sm_absmax_float_tile(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(acc_sm_absmax_float_face<RELU, 8>, idst, VectorMode::RC);
}

// Sign-magnitude int32 (read back from L1) -> fp32, after max(0, .) with ReLU.
template <bool RELU, int ITERATIONS = 8>
inline void sm_to_float_face() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt bits = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        if constexpr (RELU) {
            v_if(bits < 0) { bits = 0; }
            v_endif;
        }
        sfpi::dst_reg[0].mode<sfpi::DataLayout::F32>() =
            sfpi::convert<sfpi::vFloat>(sfpi::vSMag(bits.get()), sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

template <bool RELU>
inline void sm_to_float_tile(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(sm_to_float_face<RELU, 8>, idst, VectorMode::RC);
}

// Variant "sm_int_sfpi": the first integer requantizer, SFPI code with branches (k in LReg7, i.e. without k_to_shift).
template <bool RELU, int ITERATIONS = 8>
inline void requant_sm_face() {
    const sfpi::vInt k = sfpi::vInt(sfpi::l_reg[SCALE_LREG]);
    // k < 0: shift right by s = -k with round half to even: (mag + (2^(s-1) - 1) + ((mag >> s) & 1)) >> s.
    sfpi::vInt add = 0, tie = 0;
    v_if(k < 0) {
        add = sfpi::vInt(sfpi::shft(sfpi::vUInt(1), sfpi::vInt(-1) - k)) - 1;  // 2^(s-1) - 1
        tie = 1;
    }
    v_endif;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vInt bits = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::vInt mag = bits & sfpi::vInt(MAG_MASK);
        const sfpi::vInt t = sfpi::vInt(sfpi::shft(sfpi::vUInt(mag.get()), k)) & tie;  // bit s of mag, the tie breaker
        sfpi::vInt q = sfpi::vInt(sfpi::shft(sfpi::vUInt((mag + add + t).get()), k));
        v_if(q != 0) {
            v_if(bits < 0) { q |= 0x80; }
            v_endif;
        }
        v_endif;
        if constexpr (RELU) {
            v_if(bits < 0) { q = 0; }
            v_endif;
        }
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = q;
        sfpi::dst_reg++;
    }
}

template <bool RELU>
inline void requant_sm_tile(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(requant_sm_face<RELU, 8>, idst, VectorMode::RC);
}

// The integer scaler without the K24 step: LReg7 = k for requant_sm_tile.
inline void acc_absmax_to_scaler_k(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(acc_absmax_reduce_to_k, idst, VectorMode::RC_custom);
    _llk_math_eltwise_unary_sfpu_params_(scaler_from_k_face, idst, VectorMode::RC);
}

inline void sfpi_requant_init() { ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>(); }

// The first integer absmax: SFPI code with branches (variants 3, 4).
template <bool RELU, int ITERATIONS = 8>
inline void acc_absmax_sfpi_face() {
    sfpi::vInt acc = sfpi::vInt(sfpi::l_reg[SCALE_LREG]);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vInt bits = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();  // raw sign-magnitude
        sfpi::vInt mag = bits & sfpi::vInt(MAG_MASK);
        if constexpr (RELU) {
            v_if(bits < 0) { mag = 0; }  // sign bit set: negative (or -0)
            v_endif;
        }
        v_if(mag > acc) { acc = mag; }
        v_endif;
        sfpi::dst_reg++;
    }
    sfpi::l_reg[SCALE_LREG] = sfpi::vFloat(acc.get());
}

template <bool RELU>
inline void acc_absmax_sfpi_tile(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(acc_absmax_sfpi_face<RELU, 8>, idst, VectorMode::RC);
}

#endif  // TRISC_MATH

}  // namespace zkml::sfpu
