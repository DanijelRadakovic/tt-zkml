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
#include "layer_sfpu.hpp"
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
 *   (`quant_sm8_tile`: SFPCAST to sign-magnitude) through the Int8 packer, which writes sign + 7 magnitude bits: the FPU's Int8 format.
 * - Matmul: `q_x` and `q_w` are Int8 CBs. Reconfigured to Int8, the FPU multiplies at HiFi4 into an int32 accumulator in DST, exact.
 *   Output blocks of up to `DST_TILES` tiles: the K loop, then an SFPU sweep (layer_sfpu.hpp) that folds the integer absmax of the
 *   activated accumulator into LReg7 (the tile is untouched), then the block is packed as Int32 to `c_6`: the raw sign-magnitude bits.
 *   The packer drains block i while the FPU computes block i + 1. `c_6` is the raw accumulator, before the activation: the
 *   writer returns it as the `acc` tensor, which the host needs for the execution trace.
 * - Requantize: the absmax gives the scaler `2^k'` (`k'` = the largest with `absmax * 2^k' <= 127`, as the quantizer does), and
 *   the integer requantizer (round half to even, shift by k') runs over `c_6`. No fp32 anywhere.
 *
 * | Circular buffer | Format                  | Contents                                                 |
 * |-----------------|-------------------------|----------------------------------------------------------|
 * | `c_0`, `c_1`    | Float32                 | x tiles (Mt x Kt), W tiles (Kt x Nt), resident           |
 * | `c_2`, `c_3`    | Int8                    | q_x, q_w bytes, resident (Int8 packer in, FPU out)       |
 * | `c_6`           | Int32                   | raw accumulator (Mt x Nt), resident; the `acc` output    |
 * | `c_16`          | UInt8                   | out tiles, a ring                                        |
 * | `c_17`..`c_19`  | Float32                 | scaler tiles of x, W and the requantization              |
 *
 * **Compile-time args**: `[0]` Mt, `[1]` Kt, `[2]` Nt, `[3]` out ring pages, `[4]` activation (0 none, 1 ReLU), `[5]` 1 for fp32
 * input, 0 for bf16.
 * @pre  The program enables `fp32_dest_acc_en`, HiFi4. Float32 CBs use `UnpackToDestFp32`. Kt <= 2080.
 * @note Between `absmax_to_scaler` and the last quantize sweep of a tensor nothing may use LReg7 (quant_sfpu.hpp).
 */
// clang-format on

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
            copy_tile(cb_in, done + i, i);
        }
        for (uint32_t i = 0; i < n; i++) {
            MATH(zkml::sfpu::absmax_tile(i));
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
// `cb_out`, a ring of `out_pages` pages (the whole tensor for q_x and q_w). With `acc_in` the tiles are the raw
// sign-magnitude int32 accumulator and the integer requantizer runs instead (k in LReg7, max(0, .) when `relu`).
// `fp32_in`: the input is fp32 and cb_out an Int8 CB (SFPCAST + Int8 packer); otherwise (bf16) cb_out is UInt8.
template <uint32_t cb_in, uint32_t cb_out, bool acc_in = false, bool relu = false, bool fp32_in = true>
ALWI void quant_pass(uint32_t tiles, uint32_t out_pages) {
    CircularBuffer in(cb_in), out(cb_out);
    reconfig_data_format_srca(cb_in);
    if constexpr (acc_in) {
        MATH((zkml::sfpu::requant_init<relu>()));
    } else if constexpr (fp32_in) {
        MATH((zkml::sfpu::quant_sm8_init()));
    } else {
        MATH((zkml::sfpu::quant_smu8_init()));
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
            if constexpr (acc_in) {
                MATH((zkml::sfpu::requant_tile<relu>(i)));
            } else if constexpr (fp32_in) {
                MATH((zkml::sfpu::quant_sm8_tile(i)));
            } else {
                MATH((zkml::sfpu::quant_smu8_tile(i)));
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

// bf16 input: q_x and q_w live in UInt8 circular buffers (an Int8 one cannot share a program with the bf16 input buffer). The
// unpacker treats a UInt8 buffer as unsigned (SrcAUnsigned) and the data as plain bytes, but the quantizer wrote sign-magnitude
// bytes, which is what the FPU's Int8 format is. So srcA and srcB are configured for Int8 by hand, with the geometry of the
// buffers: the same registers `reconfig_data_format` writes for an Int8 buffer, which also enables the INT8 math.
ALWI void reconfig_int8_view(uint32_t cb_srca, uint32_t cb_srcb) {
    constexpr uint32_t i8 = static_cast<uint32_t>(DataFormat::Int8);
    UNPACK(({
        const uint32_t a = get_operand_id(cb_srca), b = get_operand_id(cb_srcb);
        _llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE, false>(
            i8, i8, get_local_cb_interface(a).fifo_page_size, get_operand_face_r_dim(a), get_operand_num_faces(a));
        _llk_unpack_reconfig_data_format_srcb_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE, false>(
            i8, i8, get_local_cb_interface(b).fifo_page_size, get_operand_face_r_dim(b), get_operand_num_faces(b));
    }));
    MATH((_llk_math_reconfig_data_format_<DST_ACCUM_MODE, false>(i8, i8)));
}

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t out_pages = get_compile_time_arg_val(3);
    constexpr bool relu = get_compile_time_arg_val(4) == 1;
    constexpr bool fp32_in = get_compile_time_arg_val(5) == 1;
    static_assert(Kt <= 2080, "K must be at most 66560: K * 127 * 127 must stay below 2^30 (31 magnitude bits)");

    CircularBuffer qx(c_qx), qw(c_qw), acc(c_acc);

    compute_kernel_hw_startup(c_x, c_out);

    // ---- quantize x and W ----
    {
        DeviceZoneScopedN("QUANT_X");
        scale_pass<c_x, c_scale_x>(Mt * Kt);
        quant_pass<c_x, c_qx, false, false, fp32_in>(Mt * Kt, Mt * Kt);
    }
    {
        DeviceZoneScopedN("QUANT_W");
        scale_pass<c_w, c_scale_w>(Kt * Nt);
        quant_pass<c_w, c_qw, false, false, fp32_in>(Kt * Nt, Kt * Nt);
    }

    // ---- matmul (FPU, int8 -> int32); SFPU: integer absmax of the activated accumulator; pack int32 ----
    qx.wait_front(Mt * Kt);
    qw.wait_front(Kt * Nt);
    {
        DeviceZoneScopedN("MATMUL");
        if constexpr (fp32_in) {
            reconfig_data_format<SrcOrder::Reverse>(c_qx, c_qw);  // in0 -> SrcB, in1 -> SrcA; turns the INT8 math on
        } else {
            reconfig_int8_view(c_qw, c_qx);  // srcA = q_w, srcB = q_x
        }
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
                MATH((zkml::sfpu::acc_absmax_tile<relu>(i)));
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
        MATH(zkml::sfpu::acc_absmax_to_scaler(dst_scale));
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(c_scale_out);
        pack_tile(dst_scale, c_scale_out);
        tile_regs_release();
        scaler.push_back(1);
        quant_pass<c_acc, c_out, true, relu>(Mt * Nt, out_pages);
    }
}
