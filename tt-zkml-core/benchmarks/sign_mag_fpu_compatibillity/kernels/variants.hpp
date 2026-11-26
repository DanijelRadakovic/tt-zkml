// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// SFPU quantize-to-int8 kernels with three byte encodings (see ../README.md). They are the replay-buffer kernel of
// tt-metal's quant_int8 (ckernel_sfpu_quant.h, tt-metal, Apache-2.0) with the clamp and the last instructions changed.
// Every variant computes n = clamp(RNE(x * scale)) in LREG0 (two's complement int32) and then turns n into its byte:
//
//   TWOS_OWN        n in [-128, 127]: (n + 128) ^ 0x80 = the two's complement byte                (1 instruction: XOR)
//                   a copy of the library kernel, to show that this file adds no overhead of its own
//   SM_INT8_PACK    n in [-127, 127]: SFPCAST to sign-magnitude, the Int8 packer takes sign + 7 magnitude bits   (1)
//   SM_UINT8_PACK   n in [-127, 127]: byte = |n| | ((n >>> 24) & 0x80), packed by the UInt8 packer     (4: abs, shift,
//   and, or)
//
// TWOS_LIB is the library function itself (quant_int8_tile), not defined here.

#pragma once

#include <cstdint>

namespace zkml_bench {

enum Variant : uint32_t { TWOS_LIB = 0, TWOS_OWN = 1, SM_INT8_PACK = 2, SM_UINT8_PACK = 3 };

}  // namespace zkml_bench

#ifdef TRISC_MATH

#include "ckernel_sfpu_quant.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace zkml_bench {

using namespace ckernel;
using namespace ckernel::sfpu;

// Like _rne_clamp_init_ in ckernel_sfpu_quant.h but with the lower clamp LO as a parameter. T = LREG2 holds the zero
// point on entry and -bits(t) (+128 with BIAS) on exit; LREG5, LREG12 and LREG13 are loaded as in the library.
template <int LO, int HI, bool BIAS>
inline void rne_clamp_init() {
    constexpr int IMM12_MASK = 0xfff;
    _sfpu_load_imm32_(p_sfpu::LREG5, RNE_MAGIC_FP32);
    _sfpu_load_imm32_(p_sfpu::LREG1, ZP_SPLIT_FP32);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LCONST_neg1, p_sfpu::LREG0, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_neg1, p_sfpu::LREG2, p_sfpu::LREG2, 0);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LCONST_neg1, p_sfpu::LREG5, p_sfpu::LREG2, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG5, 0);
    TTI_SFPIADD(LO & IMM12_MASK, p_sfpu::LREG2, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPCONFIG(0, p_sfpu::LREG12, 0);
    TTI_SFPIADD(HI & IMM12_MASK, p_sfpu::LREG2, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPCONFIG(0, p_sfpu::LREG13, 0);
    TTI_SFPIADD(
        0, p_sfpu::LCONST_0, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    if constexpr (BIAS) {
        TTI_SFPIADD(-LO, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    }
}

constexpr std::uint32_t body_len(std::uint32_t v) { return v == SM_UINT8_PACK ? 9 : 6; }

// Records the replay body. LREG0 = x on entry, LREG1 = scale.
template <std::uint32_t V>
inline void quant_init(const uint zero_point) {
    static_assert(V == TWOS_OWN || V == SM_INT8_PACK || V == SM_UINT8_PACK);
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    _quant_kernels_configure_dest_incr_addrmod_();
    if constexpr (V == TWOS_OWN) {
        rne_clamp_init<-128, 127, true>();
    } else {
        rne_clamp_init<-127, 127, false>();
    }
    if constexpr (V == TWOS_OWN || V == SM_UINT8_PACK) {
        _sfpu_load_imm32_(p_sfpu::LREG3, INT8_SIGN_MASK);  // 0x80
    }
    lltt::record<lltt::NoExec>(QUANT_REPLAY_SLOT, body_len(V));
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG0, 0);  // m = x * scale + RNE magic
    TTI_SFPNOP;
    TTI_SFPSWAP(0, p_sfpu::LREG12, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MAX_MIN);  // m = max(m, t + LO)
    TTI_SFPSWAP(0, p_sfpu::LREG13, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // m = min(m, t + HI)
    TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);  // n
    if constexpr (V == TWOS_OWN) {
        TTI_SFPXOR(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);  // (n + 128) ^ 0x80
    } else if constexpr (V == SM_INT8_PACK) {
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_INT32_TO_SM32);  // sign bit 31 + magnitude
    } else {
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPCAST_MOD1_INT32_ABS);  // LREG1 = |n|
        TTI_SFPSHFT((-24) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 1);              // LREG0 = n >>> 24 (0xFF if n < 0)
        TTI_SFPAND(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);                           // LREG0 &= 0x80
        TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);                            // LREG0 |= |n|
    }
}

// One call: one tile, 8 iterations of 4 rows x 8 lanes (as calculate_quant_int32_int8_pack).
template <std::uint32_t V, int ITERATIONS = 8>
inline void quant_tile_fn(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr std::uint32_t dst_tile_size = 64;
    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, in0_off);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, in1_off);
        lltt::replay(QUANT_REPLAY_SLOT, body_len(V));
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_6, out_off);  // raw 32 bits to DST
    }
}

}  // namespace zkml_bench

#endif  // TRISC_MATH
