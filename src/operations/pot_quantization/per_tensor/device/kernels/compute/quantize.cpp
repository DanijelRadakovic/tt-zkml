// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

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
#include "common/kernel_debug.hpp"
#include "pot_scale_sfpu.hpp"

// clang-format off
/**
 * @brief Per-tensor power-of-two (PoT) symmetric quantization. The device computes the scaler.
 *
 * The formulas are:
 * - `Q = 2^(bit_width.bits() - 1) - 1`,
 * - `k = max{k : absmax(x) * 2^k <= Q}`,
 * - `q = clamp(round_half_even(x * 2^k), -Q, Q)`.
 *
 * It reads the input twice, in chunks sized to the DST tiles. A chunk is the next run of tiles in
 * `c_0`: at most the chunk size, and never across the end of the tensor or of the `c_0` ring,
 * because `copy_tile(cb, i, ...)` addresses page `read_ptr + i` without wrapping. The ring has
 * `IN_RING_PAGES` pages: `noc_read_tile_batch_size * cb_in_size_factor`, a whole number of reader
 * batches. A chunk can be longer than the ring (a tiny batch), then the ring cut shortens it.
 *
 * Pass 1, absmax: chunks of up to `DST_TILES` tiles are copied into DST slots 0.. and folded into a
 * running per-lane absmax that stays in an SFPU register (see pot_scale_sfpu.hpp). After the last
 * tile the 32 lanes are reduced and turned into the scaler `2^k`, still in the SFPU register.
 *
 * The scaler is then broadcast into a DST tile and packed to `c_1` (reused by pass 2) and to
 * `c_17` (returned to the host).
 *
 * Pass 2, quantize: chunks of up to `DST_TILES - 1` input tiles go into DST slots 0.., and the scaler
 * into the last slot, copied once per chunk instead of once per tile. Each SFPU op (and its init)
 * then runs over the whole chunk, and the chunk is packed with one MATH -> PACK handshake.
 * - 8 bits: `quant_tile` / `quant_int8_tile`. Their clamp is [-128, 127], not [-127, 127], but
 *   `|x * 2^k| <= 127` by the choice of k, so -128 is never produced.
 * - 16 and 32 bits: `quant_tile` clamps to the int8 range and its RNE trick only holds for
 *   `|v| < 2^22`, so these widths compose SFPU ops instead: multiply by `2^k` (exact), round half
 *   to even, clamp to [-Q, Q] and convert fp32 -> int32 (exact, the value is already an integer).
 *
 * | Circular buffer | Direction          | Format                          | Contents                       |
 * |-----------------|--------------------|---------------------------------|--------------------------------|
 * | `c_0`           | consumes           | Float32 or Float16_b            | Input tiles, twice             |
 * | `c_1`           | produces, consumes | Float32                         | Scaler tile (`2^k`), held      |
 * | `c_16`          | produces           | Int32, or UInt8 for Int8 output | Quantized tiles, 2 chunks      |
 * | `c_17`          | produces           | Float32                         | Scaler tile for the host       |
 *
 * **Compile-time args**
 *
 * | Index | Name          | Values     | Description                                                         |
 * |-------|---------------|------------|---------------------------------------------------------------------|
 * | 0     | `output_int8` | 0          | Int32 output, two's complement                                      |
 * |       |               | 1          | Int8 output, two's complement bytes packed through the UInt8 packer |
 * | 1     | `bit_width`   | 8, 16, 32  | Quantized width; Int8 output requires 8                             |
 * | 2     | `IN_RING_PAGES` | >= 2     | Pages of `c_0`: reader batch * `cb_in_size_factor`                  |
 * | 3     | `DST_TILES`   | 2 .. 4     | DST tiles used per section; the host passes `dst_tiles` (4)         |
 *
 * **Runtime args**
 *
 * | Index | Name        | Type     | Description                       |
 * |-------|-------------|----------|-----------------------------------|
 * | 0     | `num_tiles` | uint32_t | Number of tiles in the input      |
 *
 * @pre  The program enables `fp32_dest_acc_en`. Float32 CBs use `UnpackToDestFp32`.
 * @pre  `c_16` has `2 * (DST_TILES - 1)` pages.
 * @note Chunking does not overlap copy_tile with the SFPU. With `UnpackToDestFp32` every copy_tile
 *       waits until the SFPU is idle before the unpacker may write DST
 *       (`math_unpack_to_dest_math_ready`), whichever slot it targets; with Float16_b input the copy
 *       already hides behind the SFPU. Pass 1 is bound by the SFPU absmax (~210 cycles per tile).
 *       Pass 2 gains from the shared scaler copy and from fewer inits and handshakes.
 * @pre  The input is finite. An Inf or NaN absmax gives a meaningless scaler.
 * @note Rounding is round-half-to-even. A ZK circuit that checks this result must use the same rule.
 * @see ../dataflow/reader.cpp, ../dataflow/writer.cpp, pot_scale_sfpu.hpp
 */
// clang-format on

namespace {

/// Tiles in the next chunk of `c_0`: at most `chunk`, and neither past the end of the tensor
/// (`remaining`) nor across the end of the ring (`ring_pos` of `ring_pages`).
inline uint32_t next_chunk(uint32_t chunk, uint32_t remaining, uint32_t ring_pos, uint32_t ring_pages) {
    uint32_t n = chunk < remaining ? chunk : remaining;
    const uint32_t to_ring_end = ring_pages - ring_pos;
    return n < to_ring_end ? n : to_ring_end;
}

/// Advances a position in the `c_0` ring by `n` pages (n never crosses the end, see next_chunk).
inline uint32_t advance_ring(uint32_t ring_pos, uint32_t n, uint32_t ring_pages) {
    ring_pos += n;
    return ring_pos == ring_pages ? 0 : ring_pos;
}

}  // namespace

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr bool is_output_int8 = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t bit_width = get_compile_time_arg_val(1);
    constexpr uint32_t in_ring_pages = get_compile_time_arg_val(2);  // c_0 pages, checked on the host
    constexpr uint32_t DST_TILES = get_compile_time_arg_val(3);

    static_assert(bit_width == 8 || bit_width == 16 || bit_width == 32, "bit_width must be 8, 16 or 32");
    static_assert(!is_output_int8 || bit_width == 8, "Int8 output requires bit_width 8");
    static_assert(
        DST_TILES <= ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>(),
        "DST_TILES exceeds one DST section");
    
    // Pass 1 (absmax) uses every DST slot for input.
    constexpr uint32_t absmax_dst_slots = DST_TILES;
    // Pass 2 (quantize) keeps the last DST slot for the scaler.
    constexpr uint32_t quant_dst_slots = DST_TILES - 1;

    // next_chunk() shortens a chunk that is longer than what is left in the tensor or the c_0 ring.
    constexpr uint32_t out_ring_pages = 2 * quant_dst_slots;  // c_16 holds two chunks (see the host)

    constexpr uint32_t cb_in_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_scale_id = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_id = tt::CBIndex::c_16;
    constexpr uint32_t cb_scale_out_id = tt::CBIndex::c_17;
    constexpr uint32_t zero_point_bits = 0;  // fp32 bits of 0.0f: symmetric quantization

    // fp32 bit patterns of -Q and Q for the 16/32-bit clamp. 2^31 - 1 rounds to 2^31 in fp32; the
    // fp32 -> int32 typecast saturates 2^31 to INT32_MAX, and |x * 2^k| <= Q keeps it unreachable.
    constexpr float q_max = static_cast<float>((1ull << (bit_width - 1)) - 1);
    constexpr uint32_t q_max_bits = __builtin_bit_cast(uint32_t, q_max);
    constexpr uint32_t q_min_bits = __builtin_bit_cast(uint32_t, -q_max);

    // DST slots: input tiles in 0 .. chunk - 1, the scaler in the last slot
    constexpr uint32_t dst_scale = DST_TILES - 1;

    CircularBuffer cb_in(cb_in_id);
    CircularBuffer cb_scale(cb_scale_id);
    CircularBuffer cb_out(cb_out_id);
    CircularBuffer cb_scale_out(cb_scale_out_id);

    compute_kernel_hw_startup(cb_in_id, cb_out_id);

    // ---- Pass 1: absmax over the whole tensor, kept in an SFPU register ----
    // The whole pass and the scaler below share one DST section: nothing is packed until the
    // scaler, so one MATH -> PACK handshake replaces one per input tile. Each chunk overwrites
    // slots 0 .. n - 1; the SFPU stall in absmax_tile orders it after the copies.
    MATH(zkml::sfpu::absmax_init());
    copy_init(cb_in_id);
    cb_scale.reserve_back(1);
    cb_scale_out.reserve_back(1);
    uint32_t ring_pos = 0;  // position of the c_0 read pointer in its ring, shared by both passes
    tile_regs_acquire();
    for (uint32_t done = 0; done < num_tiles;) {
        const uint32_t n = next_chunk(absmax_dst_slots, num_tiles - done, ring_pos, in_ring_pages);
        cb_in.wait_front(n);
        for (uint32_t i = 0; i < n; i++) {
            copy_tile(cb_in_id, i, i);
        }
        for (uint32_t i = 0; i < n; i++) {
            MATH(zkml::sfpu::absmax_tile(i));
        }
        cb_in.pop_front(n);
        done += n;
        ring_pos = advance_ring(ring_pos, n, in_ring_pages);
    }

    // ---- Scaler: SFPU register -> DST -> c_1 (for pass 2) and c_17 (for the host) ----
    // TODO(perf): quant_tile() and mul_binary_tile() read the scale from DST, so the scaler that is
    // already in an SFPU register is broadcast into a DST tile, packed to c_1 and copied back into
    // DST for every input tile. A quant variant that takes the scale from an LREG would remove the
    // broadcast, the pack and the per-tile copy.
    MATH(zkml::sfpu::absmax_to_scaler<bit_width>(dst_scale));
    DPRINT_MATH("Scaler in DST {}:\n", dst_scale);
    zkml::debug::print_dest_tile_rows(dst_scale, DataFormat::Float32, 0, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_scale_id);
    pack_tile(dst_scale, cb_scale_id);
    pack_tile(dst_scale, cb_scale_out_id);
    tile_regs_release();
    cb_scale.push_back(1);
    cb_scale_out.push_back(1);
    pack_reconfig_data_format(cb_out_id);

    // ---- Pass 2: quantize ----
    // TODO(perf): pass 2 reads the whole input from DRAM a second time. Keeping the tiles in L1
    // (when they fit) or fusing the absmax into a previous op would remove the re-read.
    if constexpr (bit_width == 8) {
        // Records the replay body and loads LREG constants, so it must come after pass 1.
        if constexpr (is_output_int8) {
            quant_int8_tile_init(zero_point_bits);
        } else {
            quant_tile_init(zero_point_bits);
        }
    }

    cb_scale.wait_front(1);
    uint32_t out_pos = 0;  // position of the c_16 write pointer in its ring
    for (uint32_t done = 0; done < num_tiles;) {
        // A reserve/push must be contiguous in the CB: cut the chunk at the end of both rings.
        uint32_t n = next_chunk(quant_dst_slots, num_tiles - done, ring_pos, in_ring_pages);
        n = next_chunk(n, n, out_pos, out_ring_pages);
        cb_in.wait_front(n);
        cb_out.reserve_back(n);

        tile_regs_acquire();
        copy_init(cb_in_id);
        for (uint32_t i = 0; i < n; i++) {
            DPRINT_UNPACK("Tile={}\n", done + i);
            UNPACK(zkml::debug::print_tile_rows(cb_in_id, i));
            copy_tile(cb_in_id, i, i);
        }
        // The input may be Float16_b while the scaler is always Float32.
        reconfig_data_format_srca(cb_in_id, cb_scale_id);
        copy_init(cb_scale_id);
        copy_tile(cb_scale_id, 0, dst_scale);  // once per chunk
        if constexpr (bit_width == 8) {
            for (uint32_t i = 0; i < n; i++) {
                if constexpr (is_output_int8) {
                    // DST holds each result as its two's-complement byte: negative q prints as 256 + q.
                    quant_int8_tile(i, dst_scale, i);
                } else {
                    quant_tile(i, dst_scale, i);
                }
            }
        } else {
            // TODO(perf): four SFPU passes. A fused 16/32-bit quant SFPU function (multiply, RNE,
            // clamp, convert in one sweep over DST) would replace them. Each init runs once per chunk.
            mul_binary_tile_init();
            for (uint32_t i = 0; i < n; i++) {
                mul_binary_tile(i, dst_scale, i);  // x * 2^k, exact for a normal 2^k
            }
            rounding_op_tile_init();
            for (uint32_t i = 0; i < n; i++) {
                round_tile(i, 0);  // round half to even
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
        for (uint32_t i = 0; i < n; i++) {
            DPRINT_MATH("Contents of DST {} after quantization:\n", i);
            zkml::debug::print_dest_tile_rows(i, DataFormat::Int32);
        }
        reconfig_data_format_srca(cb_scale_id, cb_in_id);
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < n; i++) {
            pack_tile(i, cb_out_id);
        }
        tile_regs_release();
        for (uint32_t i = 0; i < n; i++) {
            DPRINT_PACK("Tile={}\n", done + i);
            PACK(zkml::debug::print_tile_rows(cb_out_id, i));
        }
        cb_out.push_back(n);
        cb_in.pop_front(n);
        done += n;
        ring_pos = advance_ring(ring_pos, n, in_ring_pages);
        out_pos = advance_ring(out_pos, n, out_ring_pages);
    }

    cb_scale.pop_front(1);
}
