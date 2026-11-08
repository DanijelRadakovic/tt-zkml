// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Compute kernel for the L1-fit benchmark: the real pot_quantize kernel (src/operations/
// pot_quantization/per_tensor/device/kernels/compute/quantize.cpp) minus debug prints, plus
//   - RESIDENT (compile-time arg 4): the whole tensor sits in c_0 (c_0 has num_tiles pages). Pass 1
//     waits for tiles cumulatively and does NOT pop, so pass 2 re-reads the same pages from L1
//     (copy_tile(cb, i, ...) addresses read_ptr + i, nothing has to be reset). The reader sends
//     the tensor once. With RESIDENT = 0 it is the streaming kernel: pass 1 pops, the reader sends
//     the tensor twice.
//   - profiler zones: PASS1, SCALER, PASS2 (all three TRISCs), and WAIT_C0 around every wait_front
//     on c_0 (UNPACK TRISC only, where wait_front actually blocks). WAIT_C0 is the time compute was
//     starved by the reader; if it is ~0 the reader is not the bottleneck.
// Only the Int32 output path is built; the pass logic is otherwise identical to the real kernel.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/quantization.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"
#include "pot_scale_sfpu.hpp"

namespace {

inline uint32_t next_chunk(uint32_t chunk, uint32_t remaining, uint32_t ring_pos, uint32_t ring_pages) {
    uint32_t n = chunk < remaining ? chunk : remaining;
    const uint32_t to_ring_end = ring_pages - ring_pos;
    return n < to_ring_end ? n : to_ring_end;
}

inline uint32_t advance_ring(uint32_t ring_pos, uint32_t n, uint32_t ring_pages) {
    ring_pos += n;
    return ring_pos == ring_pages ? 0 : ring_pos;
}

// wait_front on c_0, timed on the UNPACK TRISC (the only one that blocks).
inline void wait_c0(CircularBuffer& cb, uint32_t n) {
#ifdef TRISC_UNPACK
    DeviceZoneScopedN("WAIT_C0");
#endif
    cb.wait_front(n);
}

}  // namespace

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t bit_width = get_compile_time_arg_val(1);
    static_assert(bit_width == 8 || bit_width == 16 || bit_width == 32, "bit_width must be 8, 16 or 32");
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(2);
    constexpr uint32_t DST_TILES = get_compile_time_arg_val(3);
    constexpr bool RESIDENT = get_compile_time_arg_val(4) == 1;

    // Streaming: c_0 is a ring of two reader blocks. Resident: c_0 holds the whole tensor, so the
    // "ring" is the tensor and a chunk never has to be cut at a ring end.
    const uint32_t in_ring_pages = RESIDENT ? num_tiles : 2 * BLOCK_SIZE;
    constexpr uint32_t absmax_chunk = DST_TILES;
    constexpr uint32_t quant_chunk = DST_TILES - 1;
    constexpr uint32_t out_ring_pages = 2 * quant_chunk;

    constexpr uint32_t cb_in_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_scale_id = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_id = tt::CBIndex::c_16;
    constexpr uint32_t cb_scale_out_id = tt::CBIndex::c_17;
    constexpr uint32_t zero_point_bits = 0;

    constexpr float q_max = static_cast<float>((1ull << (bit_width - 1)) - 1);
    constexpr uint32_t q_max_bits = __builtin_bit_cast(uint32_t, q_max);
    constexpr uint32_t q_min_bits = __builtin_bit_cast(uint32_t, -q_max);
    constexpr uint32_t dst_scale = DST_TILES - 1;

    CircularBuffer cb_in(cb_in_id);
    CircularBuffer cb_scale(cb_scale_id);
    CircularBuffer cb_out(cb_out_id);
    CircularBuffer cb_scale_out(cb_scale_out_id);

    compute_kernel_hw_startup(cb_in_id, cb_out_id);

    // ---- Pass 1: absmax ----
    uint32_t ring_pos = 0;
    {
        DeviceZoneScopedN("PASS1");
        MATH(zkml::sfpu::absmax_init());
        copy_init(cb_in_id);
        cb_scale.reserve_back(1);
        cb_scale_out.reserve_back(1);
        tile_regs_acquire();
        for (uint32_t done = 0; done < num_tiles;) {
            const uint32_t n = next_chunk(absmax_chunk, num_tiles - done, ring_pos, in_ring_pages);
            if constexpr (RESIDENT) {
                wait_c0(cb_in, done + n);  // cumulative: nothing is popped, tiles stay in L1
                for (uint32_t i = 0; i < n; i++) {
                    copy_tile(cb_in_id, done + i, i);  // page done + i from the (unmoved) read pointer
                }
            } else {
                wait_c0(cb_in, n);
                for (uint32_t i = 0; i < n; i++) {
                    copy_tile(cb_in_id, i, i);
                }
            }
            for (uint32_t i = 0; i < n; i++) {
                MATH(zkml::sfpu::absmax_tile(i));
            }
            if constexpr (!RESIDENT) {
                cb_in.pop_front(n);
                ring_pos = advance_ring(ring_pos, n, in_ring_pages);
            }
            done += n;
        }
    }

    // ---- Scaler ----
    {
        DeviceZoneScopedN("SCALER");
        MATH(zkml::sfpu::absmax_to_scaler<bit_width>(dst_scale));
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_scale_id);
        pack_tile(dst_scale, cb_scale_id);
        pack_tile(dst_scale, cb_scale_out_id);
        tile_regs_release();
        cb_scale.push_back(1);
        cb_scale_out.push_back(1);
        pack_reconfig_data_format(cb_out_id);
    }

    // ---- Pass 2: quantize ----
    {
        DeviceZoneScopedN("PASS2");
        if constexpr (bit_width == 8) {
            quant_tile_init(zero_point_bits);
        }
        cb_scale.wait_front(1);
        uint32_t out_pos = 0;
        for (uint32_t done = 0; done < num_tiles;) {
            uint32_t n = next_chunk(quant_chunk, num_tiles - done, ring_pos, in_ring_pages);
            n = next_chunk(n, n, out_pos, out_ring_pages);
            if constexpr (RESIDENT) {
                wait_c0(cb_in, n);  // already in L1: returns at once, timed to prove it
            } else {
                wait_c0(cb_in, n);
            }
            cb_out.reserve_back(n);

            tile_regs_acquire();
            copy_init(cb_in_id);
            for (uint32_t i = 0; i < n; i++) {
                copy_tile(cb_in_id, i, i);
            }
            reconfig_data_format_srca(cb_in_id, cb_scale_id);
            copy_init(cb_scale_id);
            copy_tile(cb_scale_id, 0, dst_scale);
            if constexpr (bit_width == 8) {
                for (uint32_t i = 0; i < n; i++) {
                    quant_tile(i, dst_scale, i);
                }
            } else {
                mul_binary_tile_init();
                for (uint32_t i = 0; i < n; i++) {
                    mul_binary_tile(i, dst_scale, i);
                }
                rounding_op_tile_init();
                for (uint32_t i = 0; i < n; i++) {
                    round_tile(i, 0);
                }
                clamp_tile_init();
                for (uint32_t i = 0; i < n; i++) {
                    clamp_tile(i, q_min_bits, q_max_bits);
                }
                typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Int32)>();
                for (uint32_t i = 0; i < n; i++) {
                    typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Int32)>(i);
                }
            }
            reconfig_data_format_srca(cb_scale_id, cb_in_id);
            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t i = 0; i < n; i++) {
                pack_tile(i, cb_out_id);
            }
            tile_regs_release();
            cb_out.push_back(n);
            cb_in.pop_front(n);  // resident: pass 2 is the first pop, the tiles leave L1 here
            done += n;
            ring_pos = advance_ring(ring_pos, n, in_ring_pages);
            out_pos = advance_ring(out_pos, n, out_ring_pages);
        }
        cb_scale.pop_front(1);
    }
}
