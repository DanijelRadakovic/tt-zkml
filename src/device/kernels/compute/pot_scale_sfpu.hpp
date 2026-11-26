// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// SFPU helpers for per-tensor PoT quantization: running absmax over the input tiles and the
// power-of-two scaler 2^k computed from it.
//
// State lives in one SFPU register, SCALE_LREG, across calls:
//   absmax_init()            SCALE_LREG = 0 in every lane
//   absmax_tile(idst)        SCALE_LREG = max(SCALE_LREG, |tile|), per lane (32 lanes)
//   absmax_to_scaler(idst)   reduce the 32 lanes to the tensor absmax, turn it into 2^k and write
//                            2^k into every element of DST tile idst
//
// Between absmax_init() and absmax_to_scaler() nothing else may use the SFPU, or it may clobber
// SCALE_LREG. copy_tile() is safe: it runs on the FPU and never touches the SFPU registers.
//
// Only the MATH TRISC runs SFPU code; call the wrappers through MATH(...).

#pragma once

#include <cstdint>

#include "common/pot.hpp"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "sfpi.h"
#endif

namespace zkml::sfpu {

#ifdef TRISC_MATH

// LREG0-6 are scratch for SFPI code; LREG7 carries the running absmax and then the scaler.
constexpr sfpi::LRegs SCALE_LREG = sfpi::LRegs::LReg7;

// Max of two non-negative values. A float compare, not SFPSWAP: SFPSWAP's min/max direction is
// SFPCONFIG state that other SFPU ops (e.g. the SFPU reduce) flip.
sfpi_inline sfpi::vFloat max_nonneg(sfpi::vFloat a, sfpi::vFloat b) {
    v_if(b > a) { a = b; }
    v_endif;
    return a;
}

inline void absmax_reset() {
    sfpi::l_reg[SCALE_LREG] = sfpi::vFloat(0.0f);
    sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vFloat(0.0f);  // the running min of absmax_tile_body
}

// The absmax of one tile, hand-written and software pipelined (SFPLOADMACRO does not help here: SFPSWAP keeps the Simple sub-unit
// busy for 2 cycles, and a scheduled instruction landing on the cycle after it is lost). Rows (4 rows x 8 lanes each, 32 per tile) are
// handled in pairs: SFPSWAP with mod1 = 9 puts the max in VD and the min in VC (mod1 = 1 the other way round; the lane config
// EXCHANGE_SRCB_SRCC and ENABLE_DEST_INDEX must stay clear, see tt-isa-documentation SFPSWAP.md), and compares sign-magnitude. A swap
// orders the pair, the larger goes against the running max (SCALE_LREG), the smaller against the running min (LREG6, <= 0): |x| <=
// max(running max, -running min) needs no SFPABS. The final max of the two is taken in absmax_reduce_to_scaler. Per pair 3 swaps = 6
// Simple-unit cycles, and the loads of the next pair (into the other register pair) are issued in the gaps: 6 issue slots per 2 rows.
// The tile must be fp32 in DST; no other SFPU code may be issued while it runs.
inline void absmax_tile_body() {
    // an FPU write to Dst (the bf16 copy_tile) needs 3 unrelated instructions before a load reads it; the wrapper's STALLWAIT holds
    // back SFPU instructions only, so these are SFPNOPs issued after the stall is released
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    using namespace ckernel;
    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
    TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, 2);
#pragma GCC unroll 16
    for (int g = 0; g < 16; g++) {
        const int P = p_sfpu::LREG0 + 2 * (g & 1), Q = P + 1;
        const int P2 = p_sfpu::LREG0 + 2 * ((g + 1) & 1), Q2 = P2 + 1;
        TT_SFPSWAP(0, P, Q, sfpi::SFPSWAP_MOD1_VEC_MAX_MIN);  // Q = max, P = min of the pair
        if (g < 15) {
            TT_SFPLOAD(P2, InstrModLoadStore::FP32, ADDR_MOD_7, 4 * (g + 1));
        } else {
            TTI_SFPNOP;
        }
        TT_SFPSWAP(0, Q, p_sfpu::LREG7, sfpi::SFPSWAP_MOD1_VEC_MAX_MIN);  // running max
        if (g < 15) {
            TT_SFPLOAD(Q2, InstrModLoadStore::FP32, ADDR_MOD_7, 4 * (g + 1) + 2);
        } else {
            TTI_SFPNOP;
        }
        TT_SFPSWAP(0, P, p_sfpu::LREG6, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // running min
    }
    TTI_SFPNOP;
}

// Reduce the 32 lanes of SCALE_LREG to the tensor absmax, then replace it with the scaler 2^k:
//   Q = 2^MAG - 1 (MAG = pot_magnitude_bits(bits, range): 7, 15, 31 or 30),  k = the largest integer with absmax * 2^k
//   <= Q, capped at 127.
// Every lane ends up holding the same 2^k.
template <uint32_t MAG>
inline void absmax_reduce_to_scaler() {
    static_assert(MAG == 7 || MAG == 15 || MAG == 30 || MAG == 31, "magnitude bits must be 7, 15, 30 or 31");
    // fp32 cannot hold 2^MAG - 1 for MAG > 24 and rounds it to 2^MAG. Harmless: absmax * 2^k is a power-of-two
    // multiple of an fp32, so it is never strictly between 2^MAG - 1 and 2^MAG.
    constexpr float Q = static_cast<float>((1ull << MAG) - 1);

    // Lanes are 4 rows x 8 columns. Transposing 8 copies leaves row i of the accumulator in
    // register i (broadcast over its 4 rows), so the max of the first four is the column max.
    // absmax = max(running max, -running min)
    sfpi::vFloat a = max_nonneg(sfpi::l_reg[SCALE_LREG], -sfpi::vFloat(sfpi::l_reg[sfpi::LRegs::LReg6]));
    sfpi::vFloat b = a, c = a, d = a, e = a, f = a, g = a, h = a;
    sfpi::subvec_transp(a, b, c, d, e, f, g, h);
    a = max_nonneg(max_nonneg(a, b), max_nonneg(c, d));

    // Butterfly over the 8 columns: rotate by 4, 2, 1 and fold. The max lands in every lane.
    sfpi::vFloat r = sfpi::subvec_shflror1(a);
    r = sfpi::subvec_shflror1(r);
    r = sfpi::subvec_shflror1(r);
    r = sfpi::subvec_shflror1(r);
    a = max_nonneg(a, r);
    r = sfpi::subvec_shflror1(a);
    r = sfpi::subvec_shflror1(r);
    a = max_nonneg(a, r);
    r = sfpi::subvec_shflror1(a);
    a = max_nonneg(a, r);
    const sfpi::vFloat absmax = a;

    // absmax = m * 2^E with m in [1, 2), biased exponent Eb = E + 127. Q has exponent MAG - 1, so
    // the candidate is k = MAG - 1 - E, i.e. biased exponent 253 + MAG - Eb, and absmax * 2^k
    // lies in [2^(MAG-1), 2^MAG). If that is above Q, k - 1 is the answer.
    sfpi::vInt field = sfpi::vInt(253 + MAG) - sfpi::exexp(absmax, sfpi::ExponentMode::Biased);
    // Cap at k = 127 (biased 254) so 2^k stays a finite fp32. This also covers absmax = 0
    // (Eb = 0): every k works there and 2^127 is used.
    v_if(field > 254) { field = 254; }
    v_endif;
    sfpi::vFloat scaler = sfpi::setexp(sfpi::vFloat(1.0f), field);
    v_if(absmax * scaler > Q) { scaler = scaler * 0.5f; }
    v_endif;

    sfpi::l_reg[SCALE_LREG] = scaler;
}

// One face of the scaler tile: broadcast SCALE_LREG into 8 SFPU rows.
template <int ITERATIONS = 8>
inline void scaler_store_face() {
    const sfpi::vFloat scaler = sfpi::l_reg[SCALE_LREG];
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::dst_reg[0] = scaler;
        sfpi::dst_reg++;
    }
}

#endif  // TRISC_MATH

// ----- Compute API (call through MATH(...)) -----

#ifdef TRISC_MATH

// Configures the SFPU and zeroes the running absmax. Call once before the first absmax_tile().
inline void absmax_init() {
    ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>();
    _llk_math_eltwise_unary_sfpu_params_(absmax_reset, 0, VectorMode::RC_custom);
}

// Folds |tile| of DST tile idst into the running absmax. The tile must be fp32 in DST.
inline void absmax_tile(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(absmax_tile_body, idst, VectorMode::None);
}

// Turns the running absmax into the scaler 2^k (see absmax_reduce_to_scaler) and writes it into
// every element of DST tile idst, which quant_tile() reads as its scale operand.
template <uint32_t MAG>
inline void absmax_to_scaler(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(absmax_reduce_to_scaler<MAG>, idst, VectorMode::RC_custom);
    _llk_math_eltwise_unary_sfpu_params_(scaler_store_face<8>, idst, VectorMode::RC);
}

#endif  // TRISC_MATH

}  // namespace zkml::sfpu
