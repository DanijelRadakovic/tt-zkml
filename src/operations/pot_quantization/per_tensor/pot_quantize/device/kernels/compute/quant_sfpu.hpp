// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// SFPU quantization of a PoT tensor with the scaler held in an SFPU register (SCALE_LREG = LReg7, see
// pot_scale_sfpu.hpp). absmax_to_scaler() leaves 2^k in every lane of LReg7, and these kernels multiply by it directly:
// no scaler tile in DST, no pack of the scaler to L1 and no copy of it back into DST for every block. Each kernel makes
// one sweep over a DST tile, in place: load x, multiply by 2^k, round half to even, convert to int32.
//
//   quant_small_init<INT8_OUT>(), quant_small_tile<INT8_OUT>(idst)    8 and 16 bits (|x * 2^k| < 2^22)
//       n = RNE(x * 2^k). Int32 output: n as a two's complement int32. Int8 output: (n + 128) ^ 0x80, the two's complement
//       byte of n in [0, 255], which the UInt8 packer writes unchanged (it saturates anything outside [0, 255]).
//       Hand-written SFPU instructions: SFPLOAD, SFPMAD, SFPNOP, SFPIADD [, SFPXOR], SFPSTORE per 4 rows x 8 lanes.
//       tt-metal's quant_tile / quant_int8_tile need 9 or 10 (a second SFPLOAD for the scale from DST, two SFPSWAPs for
//       a clamp, the replay), and for 16 bits the composed multiply, round, clamp and typecast ops make four passes.
//   quant32_init(), quant32_tile(idst)                              32 bits (|x * 2^k| < 2^31)
//       Same result, for values beyond the reach of the magic-number rounding. SFPI code, see below.
//
// Rounding for |v| < 2^22 (quant_small): m = v + 1.5 * 2^23 rounds to nearest even in the fp32 adder (the spacing of fp32
// around 1.5 * 2^23 is 1); bits(m) - bits(1.5 * 2^23) is the integer RNE(v). With the multiplication it is one SFPMAD,
// m = x * 2^k + 1.5 * 2^23.
//
// Rounding for 32 bits, with a = |v|: below 2^23 the same trick with 2^23 (bits(a + 2^23) - bits(2^23) = RNE(a), also when
// the sum carries into 2^24); from 2^23 on every fp32 is an integer, and the integer is the 24-bit significand shifted
// left by exponent - 23 (SFPEXMAN, SFPEXEXP, SFPSHFT). The sign is applied last.
//
// There is no clamp. k is the largest exponent with absmax * 2^k <= Q (see absmax_to_scaler), so |x * 2^k| <= Q for every
// finite x of the tensor and the clamp to [-Q, Q] of the composed ops never fires. An Inf or NaN input has no meaningful
// scaler anyway.
//
// LReg7 is read, never changed (the SFPI kernel writes it back unchanged at the end, in case the compiler used it). The
// code in between must not use it: tt-metal's quant kernels do not, but typecast, exp, recip, max/min and reduce do.
// quant_small writes LREG0, 2, 3 and 5. Call through MATH(...).

#pragma once

#include <cstdint>

#include "pot_scale_sfpu.hpp"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "sfpi.h"
#endif

namespace zkml::sfpu {

#ifdef TRISC_MATH

using namespace ckernel;

// The register the multiply reads: SCALE_LREG (pot_scale_sfpu.hpp) as the number the instructions take.
constexpr std::uint32_t SCALE_LREG_ID = p_sfpu::LREG7;
static_assert(static_cast<std::uint32_t>(SCALE_LREG) == 7, "SCALE_LREG must be LReg7");

constexpr std::uint32_t RNE_MAGIC_BITS = 0x4b400000u;  // 1.5 * 2^23 = 12582912.0f
constexpr std::uint32_t INT8_BYTE_SIGN = 0x80u;

template <bool INT8_OUT>
inline void quant_small_constants() {
    // ADDR_MOD_6: dest += 2 after each SFPSTORE, so the next of the 8 iterations of a face works on the next 4 rows.
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = sfpi::SFP_DESTREG_STRIDE},
    }
        .set(ADDR_MOD_6);
    ckernel::sfpu::_sfpu_load_imm32_(p_sfpu::LREG5, RNE_MAGIC_BITS);
    // n = bits(m) - bits(magic), plus 128 for the byte.
    ckernel::sfpu::_sfpu_load_imm32_(p_sfpu::LREG2, INT8_OUT ? (0u - RNE_MAGIC_BITS + 128u) : (0u - RNE_MAGIC_BITS));
    if constexpr (INT8_OUT) {
        ckernel::sfpu::_sfpu_load_imm32_(p_sfpu::LREG3, INT8_BYTE_SIGN);
    }
}

// One tile, 8 iterations of 4 rows x 8 lanes, in place. Called by the binary SFPU wrapper, which sets the DST base address
// of the tile; the offsets are relative to it.
template <bool INT8_OUT, int ITERATIONS = 8>
inline void quant_small_face(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr std::uint32_t dst_tile_size = 64;
    const std::uint32_t in_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;
    (void)dst_index_in1;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, in_off);
        TTI_SFPMAD(p_sfpu::LREG0, SCALE_LREG_ID, p_sfpu::LREG5, p_sfpu::LREG0, 0);  // m = x * 2^k + magic
        TTI_SFPNOP;  // SFPIADD does not wait for the SFPMAD result (without it LREG0 is still x: tested)
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        if constexpr (INT8_OUT) {
            TTI_SFPXOR(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);  // (n + 128) ^ 0x80
        }
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_6, out_off);
    }
}


// One face of a tile: 8 iterations of 4 rows x 8 lanes, SFPI code (the conditionals need the condition codes).
template <int ITERATIONS = 8>
inline void quant32_face() {
    const sfpi::vFloat scale = sfpi::l_reg[SCALE_LREG];
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat v = sfpi::vFloat(sfpi::dst_reg[0]) * scale;  // exact: 2^k
        const sfpi::vFloat a = sfpi::abs(v);
        sfpi::vInt q;
        v_if(a < 8388608.0f) {
            // RNE of a < 2^23 in the adder: 2^23 + RNE(a), also when it carries into 2^24.
            const sfpi::vFloat m = a + 8388608.0f;
            q = sfpi::vInt(m.get()) - 0x4B000000;  // the bits of m
        }
        v_else {
            // An integer already: significand (24 bits, unit bit included) << (exponent - 23), with exponent in [23, 30].
            const sfpi::vUInt significand(sfpi::exman(a, sfpi::MantissaMode::WithUnitBit).get());
            q = sfpi::vInt(sfpi::shft(significand, sfpi::exexp(a) - 23));
        }
        v_endif;
        v_if(v < 0.0f) { q = -q; }
        v_endif;
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = q;
        sfpi::dst_reg++;
    }
    sfpi::l_reg[SCALE_LREG] = scale;  // unchanged; restores it should the compiler have used LReg7 for a temporary
}

#endif  // TRISC_MATH

// ----- Compute API (call through MATH(...)) -----

#ifdef TRISC_MATH

// Once, after absmax_to_scaler() and before the first quant_small_tile().
template <bool INT8_OUT>
inline void quant_small_init() {
    ckernel::llk_math_eltwise_binary_sfpu_init<SfpuType::quant_int32>(quant_small_constants<INT8_OUT>);
}

template <bool INT8_OUT>
inline void quant_small_tile(uint32_t idst) {
    ::ckernel::_sfpu_binary_check_<DST_SYNC_MODE>(idst, idst, idst, VectorMode::RC);
    _llk_math_eltwise_binary_sfpu_params_(quant_small_face<INT8_OUT>, idst, idst, idst, VectorMode::RC);
}

// Once, after absmax_to_scaler() and before the first quant32_tile().
inline void quant32_init() { ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>(); }

inline void quant32_tile(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(quant32_face<8>, idst, VectorMode::RC);
}

#endif  // TRISC_MATH

}  // namespace zkml::sfpu
