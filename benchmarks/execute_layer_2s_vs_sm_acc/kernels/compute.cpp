// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"
#include "device/kernels/block_processing.hpp"
#include "device/operations/layer_execution/device/kernels/compute/layer_sfpu.hpp"  // variant 2: sm_int (production)
#include "variant_sfpu.hpp"                                                         // variants 0, 1
#include "device/kernels/compute/pot_scale_sfpu.hpp"
#include "device/kernels/compute/quant_sfpu.hpp"

// clang-format off
/**
 * @brief One quantized layer on one core: quantize x and W, matmul on the FPU, activation, requantize.
 *
 *     x (fp32) --quantize--> q_x (int8)  \\
 *                                          matmul (FPU, int32) -> relu -> requantize -> out (int8)
 *     W (fp32) --quantize--> q_w (int8)  /
 *
 * - Quantize (twice, the same code): pass 1 finds the absmax and the scaler `2^k` of the tensor, kept in LReg7 and packed to
 *   its scaler CB; pass 2 multiplies by it from LReg7, rounds half to even and packs the sign-magnitude byte
 *   (`quant_small_tile<true, true>`) through the UInt8 packer. Sign-magnitude is what the FPU's Int8 format expects.
 * - Matmul: `q_x` and `q_w` are read through a second CB index of the Int8 format over the same L1 bytes (the UInt8 CB is
 *   only the packer's view). Reconfigured to Int8, the FPU multiplies at HiFi4 into an int32 accumulator in DST, exact.
 *   Output blocks of up to `DST_TILES` tiles: the K loop, then an SFPU sweep (layer_sfpu.hpp) that turns the accumulator into
 *   two's complement and folds the absmax of the activated accumulator into LReg7, then the block is packed as Int32 to `c_6`.
 *   The packer drains block i while the FPU computes block i + 1. `c_6` is the raw accumulator, before the activation: the
 *   writer returns it as the `acc` tensor, which the host needs for the execution trace.
 * - Requantize: the absmax gives the scaler `2^k'` (`k'` = the largest with `absmax * 2^k' <= 127`, as the quantizer does), and
 *   pass 2 of the quantizer runs over `c_6`: int32 -> fp32 (after the activation), then the quantize sweep.
 *
 * | Circular buffer | Format                  | Contents                                                 |
 * |-----------------|-------------------------|----------------------------------------------------------|
 * | `c_0`, `c_1`    | Float32                 | x tiles (Mt x Kt), W tiles (Kt x Nt), resident           |
 * | `c_2`, `c_3`    | UInt8                   | q_x, q_w bytes (the packer's view), resident             |
 * | `c_4`, `c_5`    | Int8                    | the same bytes as `c_2`, `c_3`: the FPU's view           |
 * | `c_6`           | Int32                   | raw accumulator (Mt x Nt), resident; the `acc` output    |
 * | `c_16`          | UInt8                   | out tiles, a ring                                        |
 * | `c_17`..`c_19`  | Float32                 | scaler tiles of x, W and the requantization              |
 *
 * **Compile-time args**: `[0]` Mt, `[1]` Kt, `[2]` Nt, `[3]` out ring pages, `[4]` activation (0 none, 1 ReLU).
 * @pre  The program enables `fp32_dest_acc_en`, HiFi4. Float32 CBs use `UnpackToDestFp32`. Kt <= 32.
 * @note Between `absmax_to_scaler` and the last quantize sweep of a tensor nothing may use LReg7 (quant_sfpu.hpp).
 */
// clang-format on

// BENCHMARK COPY of execute_layer's compute kernel with the accumulator handling as a compile-time variant (arg [5]):
//   0 twos      the first version: sweep after the matmul turns the accumulator into two's complement (in place) and
//   finds the
//               float absmax; the requantization converts int32 -> fp32 and runs the fp32 quantizer.
//   1 sm_float  the accumulator stays sign-magnitude; float absmax and sm -> fp32 conversion (no two's complement
//   step). 2 sm_int    production: raw sign-magnitude accumulator, integer absmax, integer scaler, hand-written (TTI)
//   integer requantizer. 3 sm_int_sfpi  as 2 with the first integer absmax and requantizer: SFPI code with branches. 4
//   sm_int_sfpi_absmax  as 2 with the SFPI absmax only.
// Everything else (quantization of x and W, matmul, packing, dataflow) is identical.
constexpr uint32_t VARIANT = get_compile_time_arg_val(5);
constexpr uint32_t QMODE =
    get_compile_time_arg_val(6);  // experiments on the quantization: 1 no SFPU in pass 1, 2 no SFPU in pass 2
constexpr uint32_t QUANT_LM = get_compile_time_arg_val(7);  // 1: x / W pass 2 with SFPLOADMACRO (quant_sm8_lm), 0: pipelined TTI (quant_sm8)

constexpr uint32_t DST_TILES = 4;  // fp32 DST, half sync
constexpr uint32_t dst_scale = DST_TILES - 1;
constexpr uint32_t magnitude_bits = 7;  // Q = 127

constexpr uint32_t c_x = tt::CBIndex::c_0, c_w = tt::CBIndex::c_1;
constexpr uint32_t c_qx = tt::CBIndex::c_2,
                   c_qw = tt::CBIndex::c_3;  // Int8: written by the Int8 packer, read by the FPU
constexpr uint32_t c_acc = tt::CBIndex::c_6;
constexpr uint32_t c_out = tt::CBIndex::c_16;
constexpr uint32_t c_scale_x = tt::CBIndex::c_17, c_scale_w = tt::CBIndex::c_18, c_scale_out = tt::CBIndex::c_19;

// Pass 1 over `tiles` resident tiles of `cb_in`: absmax, the scaler 2^k in LReg7, and packed to `cb_scaler`.
template <uint32_t cb_in, uint32_t cb_scaler>
ALWI void scale_pass(uint32_t tiles) {
    CircularBuffer in(cb_in), scaler(cb_scaler);
    reconfig_data_format_srca(cb_in);
    MATH(zkml::sfpu::absmax_init());
    copy_init(cb_in);
    scaler.reserve_back(1);
    // One DST half per block, released without packing: the unpacker fills the other half while the SFPU folds this
    // one.
    for (uint32_t done = 0; done < tiles;) {
        const uint32_t n = tiles - done < DST_TILES ? tiles - done : DST_TILES;
        in.wait_front(done + n);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; i++) {
            if constexpr (QMODE != 3) {
                copy_tile(cb_in, done + i, i);
            }
        }
        for (uint32_t i = 0; i < n; i++) {
            if constexpr (QMODE != 1) {
                MATH(zkml::sfpu::absmax_tile(i));
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();
        done += n;
    }
    tile_regs_acquire();
    MATH(zkml::sfpu::absmax_to_scaler<magnitude_bits>(dst_scale));
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_scaler);
    pack_tile(dst_scale, cb_scaler);
    tile_regs_release();
    scaler.push_back(1);
}

// Pass 2: quantize the `tiles` resident tiles of `cb_in` with the scaler in LReg7 into sign-magnitude bytes in
// `cb_out`, a ring of `out_pages` pages (the whole tensor for q_x and q_w). With `int_in` the tiles are two's
// complement int32 (the accumulator): they are first converted to fp32, after max(0, .) when `relu`.
template <uint32_t cb_in, uint32_t cb_out, bool acc_in = false, bool relu = false>
ALWI void quant_pass(uint32_t tiles, uint32_t out_pages) {
    CircularBuffer in(cb_in), out(cb_out);
    reconfig_data_format_srca(cb_in);
    if constexpr (acc_in && (VARIANT == 2 || VARIANT == 4)) {
        MATH((zkml::sfpu::requant_init<relu>()));
    } else if constexpr (acc_in && VARIANT == 3) {
        MATH(zkml::sfpu::sfpi_requant_init());  // SFPU init only (absmax_init would zero LReg7)
    } else if constexpr (acc_in) {
        MATH((zkml::sfpu::quant_small_init<true, true>()));  // variants 0, 1: float requantizer, UInt8 packer
    } else if constexpr (QUANT_LM == 1) {
        MATH((zkml::sfpu::quant_sm8_lm_init()));
    } else {
        MATH((zkml::sfpu::quant_sm8_init()));
    }
    copy_init(cb_in);
    pack_reconfig_data_format(cb_out);
    zkml::common::RingCursor in_ring(tiles), out_ring(out_pages);
    zkml::common::DstBlockMediator blocks(DST_TILES, tiles, in_ring, out_ring);
    while (const uint32_t n = blocks.next()) {
        in.wait_front(blocks.done() + n);
        out.reserve_back(n);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; i++) {
            copy_tile(cb_in, blocks.done() + i, i);
        }
        for (uint32_t i = 0; i < n; i++) {
            if constexpr (acc_in && QMODE == 4) {
                // experiment: no requantization SFPU work at all (copy + pack only)
            } else if constexpr (acc_in && (VARIANT == 2 || VARIANT == 4)) {
                MATH((zkml::sfpu::requant_tile<relu>(i)));
            } else if constexpr (acc_in && VARIANT == 3) {
                MATH((zkml::sfpu::requant_sm_tile<relu>(i)));
            } else {
                if constexpr (acc_in && VARIANT == 0) {
                    MATH((zkml::sfpu::int_to_float_tile<relu>(i)));
                } else if constexpr (acc_in && VARIANT == 1) {
                    MATH((zkml::sfpu::sm_to_float_tile<relu>(i)));
                }
                if constexpr (acc_in) {
                    MATH((zkml::sfpu::quant_small_tile<true, true>(i)));
                } else if constexpr (QMODE != 2) {
                    if constexpr (QUANT_LM == 1) {
                        MATH((zkml::sfpu::quant_sm8_lm_tile(i)));
                    } else {
                        MATH((zkml::sfpu::quant_sm8_tile(i)));
                    }
                }
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; i++) {
            pack_tile(i, cb_out);
        }
        tile_regs_release();
        out.push_back(n);
        blocks.commit(n);
    }
}

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t out_pages = get_compile_time_arg_val(3);
    constexpr bool relu = get_compile_time_arg_val(4) == 1;
    static_assert(VARIANT >= 2 || Kt <= 32, "the float variants are exact only below 2^24");
    static_assert(Kt <= 2080, "K at most 66560");

    CircularBuffer qx(c_qx), qw(c_qw), acc(c_acc);

    compute_kernel_hw_startup(c_x, c_out);

    // ---- quantize x and W ----
    {
        DeviceZoneScopedN("QUANT_X");
        {
            DeviceZoneScopedN("QX_P1");
            scale_pass<c_x, c_scale_x>(Mt * Kt);
        }
        {
            DeviceZoneScopedN("QX_P2");
            quant_pass<c_x, c_qx>(Mt * Kt, Mt * Kt);
        }
    }
    {
        DeviceZoneScopedN("QUANT_W");
        scale_pass<c_w, c_scale_w>(Kt * Nt);
        quant_pass<c_w, c_qw>(Kt * Nt, Kt * Nt);
    }

    // ---- matmul (FPU, int8 -> int32); SFPU: two's complement and the absmax of the activated accumulator; pack int32
    // ----
    qx.wait_front(Mt * Kt);
    qw.wait_front(Kt * Nt);
    {
        DeviceZoneScopedN("MATMUL");
        reconfig_data_format<SrcOrder::Reverse>(c_qx, c_qw);  // in0 -> SrcB, in1 -> SrcA; turns the INT8 math on
        pack_reconfig_data_format(c_acc);
        MATH(zkml::sfpu::absmax_init());  // LReg7 = 0
        constexpr uint32_t out_tiles = Mt * Nt;
        for (uint32_t o0 = 0; o0 < out_tiles; o0 += DST_TILES) {
            const uint32_t n = out_tiles - o0 < DST_TILES ? out_tiles - o0 : DST_TILES;
            acc.reserve_back(n);
            tile_regs_acquire();
            matmul_init(c_qx, c_qw);
            for (uint32_t kt = 0; kt < Kt; kt++) {
                for (uint32_t i = 0; i < n; i++) {
                    const uint32_t mt = (o0 + i) / Nt, nt = (o0 + i) % Nt;
                    matmul_tiles(c_qx, c_qw, mt * Kt + kt, kt * Nt + nt, i);
                }
            }
            for (uint32_t i = 0; i < n; i++) {
                if constexpr (VARIANT == 0) {
                    MATH((zkml::sfpu::acc_finalize_absmax_tile<relu>(i)));
                } else if constexpr (VARIANT == 1) {
                    MATH((zkml::sfpu::acc_sm_absmax_float_tile<relu>(i)));
                } else if constexpr (VARIANT >= 3) {
                    MATH((zkml::sfpu::acc_absmax_sfpi_tile<relu>(i)));
                } else {
                    MATH((zkml::sfpu::acc_absmax_tile<relu>(i)));
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < n; i++) {
                pack_tile(i, c_acc);
            }
            tile_regs_release();
            acc.push_back(n);
        }
    }

    // ---- requantize: the scaler of the activated accumulator, then pass 2 of the quantizer over the accumulator ----
    {
        DeviceZoneScopedN("REQUANT");
        reconfig_data_format(c_acc, c_acc);  // the FPU is done: both sources back to the accumulator format
        CircularBuffer scaler(c_scale_out);
        scaler.reserve_back(1);
        tile_regs_acquire();
        if constexpr (VARIANT == 2 || VARIANT == 4) {
            MATH(zkml::sfpu::acc_absmax_to_scaler(dst_scale));
        } else if constexpr (VARIANT == 3) {
            MATH(zkml::sfpu::acc_absmax_to_scaler_k(dst_scale));
        } else {
            MATH(zkml::sfpu::absmax_to_scaler<magnitude_bits>(dst_scale));
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(c_scale_out);
        pack_tile(dst_scale, c_scale_out);
        tile_regs_release();
        scaler.push_back(1);
        quant_pass<c_acc, c_out, true, relu>(Mt * Nt, out_pages);
    }
}
