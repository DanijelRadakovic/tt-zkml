// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// The stages of one quantized layer, shared by the compute kernels of execute_layer and execute_network (include after
// the kernel API headers: api/compute/*.h, api/dataflow/circular_buffer.h):
//
//   scale_pass<cb_in, cb_scaler>(tiles)               pass 1 of the quantizer over resident tiles: absmax, the scaler
//   2^k in LReg7,
//                                                     packed to cb_scaler
//   quant_pass<cb_in, cb_out, ...>(tiles, pages)      pass 2: quantize with the scaler in LReg7 into sign-magnitude
//   bytes, or (acc_in)
//                                                     requantize the raw accumulator
//   matmul_stage<cb_a, cb_b, cb_acc, relu, ..>()      the FPU int8 matmul into the int32 accumulator, with the integer
//   absmax of the
//                                                     activated accumulator folded into LReg7
//   pack_requant_scaler<cb_scaler>()                  the scaler 2^k' of the requantization, packed (not pushed) to
//   cb_scaler
//
// Between scale_pass and the last quant_pass of a tensor, and between matmul_stage and the end of the requantization,
// nothing may use LReg7 (the stage functions of one layer follow each other without anything in between).

#pragma once

#include <cstdint>

#include "device/kernels/block_processing.hpp"
#include "device/kernels/compute/layer_sfpu.hpp"
#include "device/kernels/compute/pot_scale_sfpu.hpp"
#include "device/kernels/compute/quant_sfpu.hpp"

// Profiler-only fine zones (what each of the three TRISCs does inside a stage). Off unless the program is built with ZKML_FINE_ZONES (the program
// factory sets it when the environment variable ZKML_FINE_ZONES is set): they fill the profiler buffer of a RISC after about 6 layers.
#if defined(ZKML_FINE_ZONES)
#define ZKML_FZONE(name) DeviceZoneScopedN(name)
#else
#define ZKML_FZONE(name)
#endif

namespace zkml::stages {

constexpr uint32_t DST_TILES = 4;  // fp32 DST, half sync
constexpr uint32_t dst_scale = DST_TILES - 1;
constexpr uint32_t magnitude_bits = 7;  // Q = 127

// Pass 1 over `tiles` resident tiles of `cb_in`: absmax, the scaler 2^k in LReg7, and packed to `cb_scaler`. The `_rt` functions
// take the circular buffers (and sizes) as run-time values, so that one copy of the code serves every layer of execute_network; the
// functions without the suffix are the same with the buffers as template arguments (execute_layer): once inlined with constants they
// are the same code.
ALWI void scale_pass_rt(uint32_t cb_in, uint32_t cb_scaler, uint32_t tiles) {
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
        {
            ZKML_FZONE("SP_COPY");  // profiler-only fine zones: what each of the three TRISCs does in this block
            for (uint32_t i = 0; i < n; i++) {
                copy_tile(cb_in, done + i, i);
            }
        }
        {
            ZKML_FZONE("SP_SFPU");
            for (uint32_t i = 0; i < n; i++) {
                MATH(zkml::sfpu::absmax_tile(i));
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();
        done += n;
    }
    {
        ZKML_FZONE("SP_SCALER");
        tile_regs_acquire();
        MATH(zkml::sfpu::absmax_to_scaler<magnitude_bits>(dst_scale));
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_scaler);
        pack_tile(dst_scale, cb_scaler);
        tile_regs_release();
        scaler.push_back(1);
    }
}
template <uint32_t cb_in, uint32_t cb_scaler>
ALWI void scale_pass(uint32_t tiles) {
    scale_pass_rt(cb_in, cb_scaler, tiles);
}

// A circular buffer that holds the quantized bytes is a UInt8 buffer when an Int8 one cannot share the program with the
// input buffer (bf16 input), or when the program uses UInt8 buffers throughout (execute_network). The packer writes a
// UInt8 buffer as plain bytes and saturates outside [0, 255], but the quantizer produces sign-magnitude int32, which
// the Int8 packer turns into sign + 7 magnitude bits (the FPU's Int8 format). So the packer is configured for Int8 by
// hand, with the geometry of the buffer: the registers `pack_reconfig_data_format` writes for an Int8 buffer.
ALWI void pack_reconfig_int8_view(uint32_t cb_out) {
    constexpr uint32_t i8 = static_cast<uint32_t>(DataFormat::Int8);
    PACK(({
        const uint32_t id = get_output_id(cb_out);
        _llk_pack_reconfig_data_format_<DST_ACCUM_MODE>(
            i8,
            i8,
            get_local_cb_interface(id).fifo_page_size,
            get_output_tile_c_dim(id),
            get_output_num_faces(id),
            false);
    }));
}

// Pass 2: quantize the `tiles` resident tiles of `cb_in` with the scaler in LReg7 into sign-magnitude bytes in
// `cb_out`, a ring of `out_pages` pages (the whole tensor for the quantized tensors). With `acc_in` the tiles are the
// raw sign-magnitude int32 accumulator and the integer requantizer runs instead (k in LReg7, max(0, .) when `relu`);
// its bytes go through the UInt8 packer. `int8_pack_view`: cb_out is a UInt8 buffer written with the Int8 packer (see
// pack_reconfig_int8_view); false: cb_out is an Int8 buffer.
template <bool acc_in = false, bool relu = false, bool int8_pack_view = false, bool acc_float = false>
ALWI void quant_pass_rt(uint32_t cb_in, uint32_t cb_out, uint32_t tiles, uint32_t out_pages) {
    CircularBuffer in(cb_in), out(cb_out);
    reconfig_data_format_srca(cb_in);
    if constexpr (acc_in && acc_float) {
        MATH((zkml::sfpu::requant_lm_init()));
    } else if constexpr (acc_in) {
        MATH((zkml::sfpu::requant_init<relu>()));
    } else {
        MATH((zkml::sfpu::quant_sm8_lm_init()));
    }
    copy_init(cb_in);
    pack_reconfig_data_format(cb_out);
    if constexpr ((!acc_in || acc_float) && int8_pack_view) {
        pack_reconfig_int8_view(cb_out);
    }
    if constexpr (acc_in && acc_float && relu) {
        pack_relu_config(ReluConfig::zero());  // the packer zeroes negative (sign-magnitude sign bit) values: the ReLU is free
    }
    zkml::common::RingCursor in_ring(tiles), out_ring(out_pages);
    zkml::common::DstBlockMediator blocks(DST_TILES, tiles, in_ring, out_ring);
    while (const uint32_t n = blocks.next()) {
        in.wait_front(blocks.done() + n);
        out.reserve_back(n);
        tile_regs_acquire();
        {
            ZKML_FZONE("QP_COPY");
            for (uint32_t i = 0; i < n; i++) {
                copy_tile(cb_in, blocks.done() + i, i);
            }
        }
        {
            ZKML_FZONE("QP_SFPU");
            for (uint32_t i = 0; i < n; i++) {
                if constexpr (acc_in && acc_float) {
                    MATH((zkml::sfpu::requant_lm_tile(i)));
                } else if constexpr (acc_in) {
                    MATH((zkml::sfpu::requant_tile<relu>(i)));
                } else {
                    MATH((zkml::sfpu::quant_sm8_lm_tile(i)));
                }
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        {
            ZKML_FZONE("QP_PACK");
            for (uint32_t i = 0; i < n; i++) {
                pack_tile(i, cb_out);
            }
        }
        tile_regs_release();
        out.push_back(n);
        blocks.commit(n);
    }
    if constexpr (acc_in && acc_float && relu) {
        pack_relu_config(ReluConfig::none());
    }
}
template <uint32_t cb_in, uint32_t cb_out, bool acc_in = false, bool relu = false, bool int8_pack_view = false>
ALWI void quant_pass(uint32_t tiles, uint32_t out_pages) {
    quant_pass_rt<acc_in, relu, int8_pack_view>(cb_in, cb_out, tiles, out_pages);
}

// The unpacker treats a UInt8 buffer as unsigned (SrcAUnsigned) and the data as plain bytes, but the quantizer wrote
// sign-magnitude bytes, which is what the FPU's Int8 format is. So srcA and srcB are configured for Int8 by hand, with
// the geometry of the buffers: the same registers `reconfig_data_format` writes for an Int8 buffer, which also enables
// the INT8 math.
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

// The int8 matmul: x [Mt, Kt] times w [Kt, Nt] tiles, both resident quantized buffers, into the int32 accumulator
// `cb_acc` (blocks of up to DST_TILES output tiles; pushed to cb_acc as they are packed). The SFPU sweep after each
// block folds the integer absmax of the activated accumulator into LReg7 (the tile itself is untouched: the raw
// sign-magnitude bits are packed). `int8_view`: the buffers are UInt8 buffers, so the source registers are configured
// for Int8 by hand; false: Int8 buffers.
template <bool relu, bool int8_view>
ALWI void matmul_stage_rt(uint32_t cb_a, uint32_t cb_b, uint32_t cb_acc, uint32_t Mt, uint32_t Kt, uint32_t Nt) {
    CircularBuffer a(cb_a), b(cb_b), acc(cb_acc);
    a.wait_front(Mt * Kt);
    b.wait_front(Kt * Nt);
    if constexpr (int8_view) {
        reconfig_int8_view(cb_b, cb_a);  // srcA = b (w), srcB = a (x)
    } else {
        reconfig_data_format<SrcOrder::Reverse>(cb_a, cb_b);  // in0 -> SrcB, in1 -> SrcA; turns the INT8 math on
    }
    pack_reconfig_data_format(cb_acc);
    MATH(zkml::sfpu::absmax_init());  // LReg7 = 0
    matmul_init(cb_a, cb_b);          // once per layer: the absmax sweep between the blocks does not touch the matmul configuration
    const uint32_t out_tiles = Mt * Nt;
    for (uint32_t o0 = 0; o0 < out_tiles; o0 += DST_TILES) {
        const uint32_t n = out_tiles - o0 < DST_TILES ? out_tiles - o0 : DST_TILES;
        acc.reserve_back(n);
        tile_regs_acquire();
        // (mt, nt) of the first output tile of the block: one division per block, not per tile
        const uint32_t mt0 = o0 / Nt, nt0 = o0 - mt0 * Nt;
        {
            ZKML_FZONE("MM_FPU");
            for (uint32_t kt = 0; kt < Kt; kt++) {
                uint32_t mt = mt0, nt = nt0;
                for (uint32_t i = 0; i < n; i++) {
                    matmul_tiles(cb_a, cb_b, mt * Kt + kt, kt * Nt + nt, i);
                    if (++nt == Nt) {
                        nt = 0;
                        mt++;
                    }
                }
            }
        }
        {
            ZKML_FZONE("MM_SWEEP");
            for (uint32_t i = 0; i < n; i++) {
                MATH((zkml::sfpu::acc_absmax_tile<relu>(i)));
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        {
            ZKML_FZONE("MM_PACK");
            for (uint32_t i = 0; i < n; i++) {
                pack_tile(i, cb_acc);
            }
        }
        tile_regs_release();
        acc.push_back(n);
    }
}
template <
    uint32_t cb_a,
    uint32_t cb_b,
    uint32_t cb_acc,
    bool relu,
    bool int8_view,
    uint32_t Mt,
    uint32_t Kt,
    uint32_t Nt>
ALWI void matmul_stage() {
    matmul_stage_rt<relu, int8_view>(cb_a, cb_b, cb_acc, Mt, Kt, Nt);
}

// The scaler 2^k' of the requantization from the integer absmax in LReg7 (left there as k' + 24 for the requantizer),
// packed to `cb_scaler`; the caller pushes it. The accumulator buffer is the one the matmul packed into: both source
// registers go back to its format first (the FPU is done).
template <bool acc_float = false>
ALWI void pack_requant_scaler_rt(uint32_t cb_acc, uint32_t cb_scaler) {
    reconfig_data_format(cb_acc, cb_acc);
    CircularBuffer scaler(cb_scaler);
    ZKML_FZONE("RQ_SCALER");
    scaler.reserve_back(1);
    tile_regs_acquire();
    if constexpr (acc_float) {
        MATH(zkml::sfpu::acc_absmax_to_scaler_float(dst_scale));  // LReg7 = 2^k for the float requantizer
    } else {
        MATH(zkml::sfpu::acc_absmax_to_scaler(dst_scale));  // LReg7 = k + 24 for the integer requantizer
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_scaler);
    pack_tile(dst_scale, cb_scaler);
    tile_regs_release();
}
template <uint32_t cb_acc, uint32_t cb_scaler>
ALWI void pack_requant_scaler() {
    pack_requant_scaler_rt(cb_acc, cb_scaler);
}

}  // namespace zkml::stages
