// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// SFPU steps around the int32 accumulator of execute_layer. Everything is integer, nothing converts to fp32 per
// element, and the accumulator stays in the FPU's own format, sign-magnitude (bit 31 = sign, bits 0..30 = magnitude),
// from the matmul to the `acc` output.
//
//   acc_absmax_tile<RELU>(idst)    after the matmul, per output block: folds the magnitude of the activated accumulator
//                                  (|acc|, or max(0, acc) with ReLU) into LReg7, as an integer max. The tile is not
//                                  modified, so the packer writes the raw sign-magnitude bits to L1 (the `acc` output).
//   acc_absmax_to_scaler(idst)     reduces LReg7 to the tensor absmax A, computes the requantization exponent
//                                  k = the largest integer with A * 2^k <= 127 and writes the fp32 scaler 2^k into
//                                  every element of DST tile idst. LReg7 then holds the integer k.
//   requant_tile<RELU>(idst)       the requantization of a tile of raw sign-magnitude accumulator (read back from L1)
//   with k in
//                                  LReg7: round half to even of acc * 2^k, as a sign-magnitude int8 byte, in place.
//
// Integer arithmetic: with t = bit length of A (32 - clz), 2^(t-1) <= A < 2^t, so k = 7 - t gives A * 2^k < 128. For k
// >= 0 that is already <= 127. For k < 0 A * 2^k is a fraction and may be in (127, 128): then k - 1. No bound on |acc|
// other than the 31 magnitude bits: the op limits K to 66560 (K * 127 * 127 < 2^30).
//
// After absmax_init() (LReg7 = 0) call acc_absmax_tile for every block; nothing else may use LReg7 until the last
// requant_tile. Only the MATH thread runs SFPU code: call through MATH(...).

#pragma once

#include <cstdint>

#include "device/kernels/compute/pot_scale_sfpu.hpp"
#include "device/kernels/compute/quant_sfpu.hpp"  // the ckernel headers and SCALE_LREG_ID

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpi.h"
#endif

namespace zkml::sfpu {

#ifdef TRISC_MATH
using namespace ckernel;
#endif

#ifdef TRISC_MATH

constexpr uint32_t MAG_MASK = 0x7FFFFFFF;

// The running max is kept in LReg7 and updated by SFPSWAP (mod1 = 9: VD = max, VC = min; the lane config
// EXCHANGE_SRCB_SRCC and ENABLE_DEST_INDEX must stay clear, see tt-isa-documentation SFPSWAP.md), which compares
// sign-magnitude, so with ReLU a negative accumulator never wins against the running max (>= 0): no masking at all, 2
// instructions per 4 rows x 8 lanes. Without ReLU the sign bit is cleared first (3).
template <bool RELU, int ITERATIONS = 8>
inline void acc_absmax_face() {
    if constexpr (!RELU) {
        ckernel::sfpu::_sfpu_load_imm32_(p_sfpu::LREG3, MAG_MASK);
    }
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(
            p_sfpu::LREG0,
            InstrModLoadStore::INT32_2S_COMP,
            ADDR_MOD_7,
            2 * d);  // raw sign-magnitude, the tile unchanged
        if constexpr (!RELU) {
            TTI_SFPAND(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);
        }
        TTI_SFPSWAP(0, p_sfpu::LREG0, SCALE_LREG_ID, sfpi::SFPSWAP_MOD1_VEC_MAX_MIN);
    }
}

template <bool RELU>
inline void acc_absmax_tile(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(acc_absmax_face<RELU, 8>, idst, VectorMode::RC);
}

sfpi_inline sfpi::vInt max_int(sfpi::vInt a, sfpi::vInt b) {
    v_if(b > a) { a = b; }
    v_endif;
    return a;
}

// LReg7 (per-lane integer absmax) -> the tensor absmax in every lane -> k (integer), left in LReg7.
inline void acc_absmax_reduce_to_k() {
    sfpi::vInt a = sfpi::vInt(sfpi::l_reg[SCALE_LREG]);
    sfpi::vInt b = a, c = a, d = a, e = a, f = a, g = a, h = a;
    sfpi::subvec_transp(a, b, c, d, e, f, g, h);
    a = max_int(max_int(a, b), max_int(c, d));
    sfpi::vInt r = sfpi::vInt(sfpi::subvec_shflror1(a));
    r = sfpi::vInt(sfpi::subvec_shflror1(r));
    r = sfpi::vInt(sfpi::subvec_shflror1(r));
    r = sfpi::vInt(sfpi::subvec_shflror1(r));
    a = max_int(a, r);
    r = sfpi::vInt(sfpi::subvec_shflror1(a));
    r = sfpi::vInt(sfpi::subvec_shflror1(r));
    a = max_int(a, r);
    r = sfpi::vInt(sfpi::subvec_shflror1(a));
    a = max_int(a, r);

    // k = 7 - bit_length(A); A = 0 -> 127 (every k works)
    sfpi::vInt t = sfpi::vInt(32) - sfpi::vInt(sfpi::lz(a));
    sfpi::vInt k = sfpi::vInt(7) - t;
    v_if(k < 0) {
        // A * 2^k is a fraction; it may exceed 127: A > 127 << -k
        sfpi::vInt limit = sfpi::vInt(sfpi::shft(sfpi::vUInt(127), sfpi::vInt(0) - k));
        v_if(a > limit) { k = k - 1; }
        v_endif;
    }
    v_endif;
    v_if(a == 0) { k = 127; }
    v_endif;
    sfpi::l_reg[SCALE_LREG] = sfpi::vFloat(k.get());
}

// One face of the scaler tile: 2^k from the integer k in LReg7, broadcast into 8 SFPU rows.
inline void scaler_from_k_face() {
    const sfpi::vInt k = sfpi::vInt(sfpi::l_reg[SCALE_LREG]);
    const sfpi::vFloat scaler = sfpi::setexp(sfpi::vFloat(1.0f), k + 127);
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        sfpi::dst_reg[0] = scaler;
        sfpi::dst_reg++;
    }
}

// LReg7 = k + 24: the shift amount of requant_tile, from the scaler 2^k that scaler_from_k_face has just written into
// the tile. Not from the k that reduce_to_k left in LReg7: the compiler allocates LReg7 for its own temporaries in the
// SFPI functions in between (observed: LReg7 was 0.5f when the layers were inlined into one kernel), so only a value
// read back from DEST and put into LReg7 by hand-written instructions, right before the sweep that needs it, is
// reliable.
inline void k_to_shift() {
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
    TTI_SFPNOP;
    TTI_SFPEXEXP(0, p_sfpu::LREG0, SCALE_LREG_ID, sfpi::SFPEXEXP_MOD1_DEBIAS);
    TTI_SFPNOP;
    TTI_SFPIADD(24, SCALE_LREG_ID, SCALE_LREG_ID, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
}

inline void acc_absmax_to_scaler(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(acc_absmax_reduce_to_k, idst, VectorMode::RC_custom);
    _llk_math_eltwise_unary_sfpu_params_(scaler_from_k_face, idst, VectorMode::RC);
    _llk_math_eltwise_unary_sfpu_params_(k_to_shift, idst, VectorMode::RC_custom);
}

// ---- the requantizer: hand-written SFPU instructions, 11 (ReLU) or 13 per 4 rows x 8 lanes, no branches ----
//
// With K24 = 24 + k in LReg7 (acc_absmax_to_scaler leaves it there) and mag the magnitude of the accumulator (< 2^30):
//   z = mag << K24            fixed point, q in bits 24..30 and the fraction in bits 0..23 (k >= -23, so K24 >= 1 and
//   the sign
//                             bit shifts out; k >= 0 shifts left by 24 or more: no fraction)
//   q = (z + 0x7FFFFF + ((z >> 24) & 1)) >> 24    round half to even
// The sign bit is shifted out, which is also why no mask is needed. Without ReLU the sign comes back as bit 7 of the
// byte, if q != 0 (no "negative zero"); with ReLU a negative accumulator is set to 0 first.
constexpr std::uint32_t REQUANT_SIGN_BYTE = 0x80u;
constexpr std::uint32_t REQUANT_ROUND = 0x7FFFFFu;

template <bool RELU>
inline void requant_constants() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = sfpi::SFP_DESTREG_STRIDE},
    }
        .set(ADDR_MOD_6);
    ckernel::sfpu::_sfpu_load_imm32_(p_sfpu::LREG4, REQUANT_SIGN_BYTE);
    ckernel::sfpu::_sfpu_load_imm32_(p_sfpu::LREG5, REQUANT_ROUND);
    ckernel::sfpu::_sfpu_load_imm32_(p_sfpu::LREG6, 1u);
}

template <bool RELU, int ITERATIONS = 8>
inline void requant_face(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr std::uint32_t dst_tile_size = 64;
    const std::uint32_t in_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;
    (void)dst_index_in1;
    constexpr std::uint32_t SHIFT_REG = 0;  // SFPSHFT: the amount is in LREG_C
    constexpr std::uint32_t SHIFT_IMM = 1;  // the amount is the immediate
    constexpr std::uint32_t SHIFT_IMM_FROM_C =
        1 | 4;  // immediate amount, the source is LREG_C instead of the destination
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_7, in_off);  // raw sign-magnitude
        if constexpr (RELU) {
            TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);  // negative (or -0)
            TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        } else {
            TTI_SFPSHFT(
                (-24) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG1, SHIFT_IMM_FROM_C);  // sign in bit 7 (and magnitude bits)
            TTI_SFPAND(0, p_sfpu::LREG4, p_sfpu::LREG1, 0);                      // LREG1 = sign << 7
        }
        TTI_SFPSHFT(0, SCALE_LREG_ID, p_sfpu::LREG0, SHIFT_REG);                     // z = mag << K24
        TTI_SFPSHFT((-24) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG2, SHIFT_IMM_FROM_C);  // floor(q)
        TTI_SFPAND(0, p_sfpu::LREG6, p_sfpu::LREG2, 0);                              // its lsb
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSHFT((-24) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, SHIFT_IMM);  // q
        if constexpr (!RELU) {
            TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_6, out_off);
    }
}

template <bool RELU>
inline void requant_init() {
    ckernel::llk_math_eltwise_binary_sfpu_init<SfpuType::quant_int32>(requant_constants<RELU>);
}

// Same arguments as requant_sm_tile; in place. After acc_absmax_to_scaler and requant_init.
template <bool RELU>
inline void requant_tile(uint32_t idst) {
    ::ckernel::_sfpu_binary_check_<DST_SYNC_MODE>(idst, idst, idst, VectorMode::RC);
    _llk_math_eltwise_binary_sfpu_params_(requant_face<RELU, 8>, idst, idst, idst, VectorMode::RC);
}

#endif  // TRISC_MATH

}  // namespace zkml::sfpu
