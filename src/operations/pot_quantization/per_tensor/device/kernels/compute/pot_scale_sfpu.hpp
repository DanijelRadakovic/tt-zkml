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

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
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

inline void absmax_reset() { sfpi::l_reg[SCALE_LREG] = sfpi::vFloat(0.0f); }

// One 16x16 face: 8 SFPU rows of 32 lanes.
template <int ITERATIONS = 8>
inline void absmax_accumulate_face() {
    sfpi::vFloat acc = sfpi::l_reg[SCALE_LREG];
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        acc = max_nonneg(acc, sfpi::abs(sfpi::vFloat(sfpi::dst_reg[0])));
        sfpi::dst_reg++;
    }
    sfpi::l_reg[SCALE_LREG] = acc;
}

// Reduce the 32 lanes of SCALE_LREG to the tensor absmax, then replace it with the scaler 2^k:
//   Q = 2^(BITS-1) - 1,  k = the largest integer with absmax * 2^k <= Q, capped at 127.
// Every lane ends up holding the same 2^k.
template <uint32_t BITS>
inline void absmax_reduce_to_scaler() {
    static_assert(BITS == 8 || BITS == 16 || BITS == 32, "bit width must be 8, 16 or 32");
    // fp32 cannot hold 2^31 - 1 and rounds it to 2^31. Harmless: absmax * 2^k is a power-of-two
    // multiple of an fp32, so it is never strictly between 2^31 - 1 and 2^31.
    constexpr float Q = static_cast<float>((1ull << (BITS - 1)) - 1);

    // Lanes are 4 rows x 8 columns. Transposing 8 copies leaves row i of the accumulator in
    // register i (broadcast over its 4 rows), so the max of the first four is the column max.
    sfpi::vFloat a = sfpi::l_reg[SCALE_LREG];
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

    // absmax = m * 2^E with m in [1, 2), biased exponent Eb = E + 127. Q has exponent BITS - 2, so
    // the candidate is k = BITS - 2 - E, i.e. biased exponent 252 + BITS - Eb, and absmax * 2^k
    // lies in [2^(BITS-2), 2^(BITS-1)). If that is above Q, k - 1 is the answer.
    sfpi::vInt field = sfpi::vInt(252 + BITS) - sfpi::exexp(absmax, sfpi::ExponentMode::Biased);
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
    _llk_math_eltwise_unary_sfpu_params_(absmax_accumulate_face<8>, idst, VectorMode::RC);
}

// Turns the running absmax into the scaler 2^k (see absmax_reduce_to_scaler) and writes it into
// every element of DST tile idst, which quant_tile() reads as its scale operand.
template <uint32_t BITS>
inline void absmax_to_scaler(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(absmax_reduce_to_scaler<BITS>, idst, VectorMode::RC_custom);
    _llk_math_eltwise_unary_sfpu_params_(scaler_store_face<8>, idst, VectorMode::RC);
}

#endif  // TRISC_MATH

}  // namespace zkml::sfpu
