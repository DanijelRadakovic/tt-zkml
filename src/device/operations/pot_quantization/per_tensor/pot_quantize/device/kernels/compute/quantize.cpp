// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"
#include "device/kernels/block_processing.hpp"
#include "device/kernels/compute/kernel_debug.hpp"
#include "common/pot.hpp"
#include "device/kernels/compute/pot_scale_sfpu.hpp"
#include "device/kernels/compute/quant_sfpu.hpp"

// clang-format off
/**
 * @brief Per-tensor power-of-two (PoT) symmetric quantization. The device computes the scaler.
 *
 * The formulas are:
 * - `Q = 2^m - 1` with `m` = 7, 15, 31 for 8, 16, 32 bits, or 30 for 32 bits with `m31_range` (see
 *   common/pot.hpp),
 * - `k = max{k : absmax(x) * 2^k <= Q}`,
 * - `q = clamp(round_half_even(x * 2^k), -Q, Q)` (the clamp never fires).
 *
 * It reads the input twice, in blocks sized to the DST tiles. A block is the next run of tiles in
 * `c_0`: at most the block size, and never across the end of the tensor or of the `c_0` ring,
 * because `copy_tile(cb, i, ...)` addresses page `read_ptr + i` without wrapping. The ring has
 * `IN_RING_PAGES` pages: `noc_read_tile_batch_size * cb_in_size_factor`, a whole number of reader
 * batches. A block can be longer than the ring (a tiny batch), then the ring cut shortens it.
 *
 * Pass 1, absmax: blocks of up to `DST_TILES` tiles are copied into DST slots 0.. and folded into a
 * running per-lane absmax that stays in an SFPU register (see pot_scale_sfpu.hpp). After the last
 * tile the 32 lanes are reduced and turned into the scaler `2^k`, still in the SFPU register.
 *
 * The scaler is then broadcast into a DST tile and packed to `c_17` (returned to the host). It also stays in LReg7, and
 * pass 2 uses it from there (quant_sfpu.hpp): no scaler CB, no DST slot for it and no copy of it for every block.
 *
 * Pass 2, quantize: blocks of up to `DST_TILES` input tiles are copied into DST slots 0..; one fused SFPU sweep per tile
 * then multiplies by the scaler in LReg7, rounds half to even and converts to int32 (`quant_small_tile` for 8 and 16
 * bits, `quant32_tile` for 32 bits). The block is packed with one MATH -> PACK handshake.
 *
 * The sweeps have no clamp: `|x * 2^k| <= Q` by the choice of k, so the clamp to [-Q, Q] never fires.
 * Rounding is half to even; Int8 output is the two's complement byte (negative q as 256 + q).
 *
 * | Circular buffer | Direction          | Format                          | Contents                       |
 * |-----------------|--------------------|---------------------------------|--------------------------------|
 * | `c_0`           | consumes           | Float32 or Float16_b            | Input tiles, twice             |
 * | `c_16`          | produces           | Int32, or UInt8 for Int8 output | Quantized tiles, 2 blocks      |
 * | `c_17`          | produces           | Float32                         | Scaler tile for the host       |
 *
 * **Compile-time args**
 *
 * | Index | Name            | Values     | Description                                                         |
 * |-------|-----------------|------------|---------------------------------------------------------------------|
 * | 0     | `output_kind`   | 0          | Int32 output, two's complement                                      |
 * |       |                 | 1          | Int8 output, two's complement bytes packed through the UInt8 packer |
 * |       |                 | 2          | UInt8 output, sign-magnitude bytes (bit 7 = sign): what the FPU reads as Int8 |
 * | 1     | `bit_width`     | 8, 16, 32  | Quantized width; Int8 output requires 8                             |
 * | 2     | `IN_RING_PAGES` | >= 2       | Pages of `c_0`: reader batch * `cb_in_size_factor`                  |
 * | 3     | `DST_TILES`     | 2 .. 4     | DST tiles used per section; the host passes `dst_tiles` (4)         |
 * | 4     | `m31_range`     | 0          | `Q = 2^(bit_width - 1) - 1`                                         |
 * |       |                 | 1          | 32 bits capped at `Q = 2^30 - 1` (values map injectively into M31)  |
 *
 * **Runtime args**
 *
 * | Index | Name        | Type     | Description                       |
 * |-------|-------------|----------|-----------------------------------|
 * | 0     | `num_tiles` | uint32_t | Number of tiles in the input      |
 *
 * @pre  The program enables `fp32_dest_acc_en`. Float32 CBs use `UnpackToDestFp32`.
 * @pre  `c_16` has `2 * DST_TILES` pages.
 * @note Nothing between `absmax_to_scaler` and the last quantize sweep may use LReg7 (see
 *       quant_sfpu.hpp): the copies are done by the unpacker and the packer, which do not touch the SFPU registers.
 * @note Working in blocks does not overlap copy_tile with the SFPU. With `UnpackToDestFp32` every copy_tile
 *       waits until the SFPU is idle before the unpacker may write DST
 *       (`math_unpack_to_dest_math_ready`), whichever slot it targets; with Float16_b input the copy
 *       already hides behind the SFPU. Pass 1 is bound by the SFPU absmax (~210 cycles per tile).
 * @pre  The input is finite. An Inf or NaN absmax gives a meaningless scaler.
 * @note Rounding is round-half-to-even. A ZK circuit that checks this result must use the same rule.
 * @see ../dataflow/reader.cpp, ../dataflow/writer.cpp, pot_scale_sfpu.hpp, device/kernels/block_processing.hpp
 */
// clang-format on

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t output_kind = get_compile_time_arg_val(0);  // 0 Int32, 1 Int8 (two's complement), 2 UInt8 (sign-magnitude)
    constexpr bool is_output_int8 = output_kind != 0;
    constexpr bool sign_mag = output_kind == 2;
    constexpr uint32_t bit_width = get_compile_time_arg_val(1);
    constexpr uint32_t in_ring_pages = get_compile_time_arg_val(2);  // c_0 pages, checked on the host
    constexpr uint32_t DST_TILES = get_compile_time_arg_val(3);

    static_assert(bit_width == 8 || bit_width == 16 || bit_width == 32, "bit_width must be 8, 16 or 32");
    static_assert(!is_output_int8 || bit_width == 8, "Int8 output requires bit_width 8");
    static_assert(
        DST_TILES <= ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>(),
        "DST_TILES exceeds one DST section");

    // Both passes use every DST slot for input: the scaler of pass 2 is in LReg7.
    constexpr uint32_t absmax_dst_slots = DST_TILES;
    constexpr zkml::prim::QuantRange range =
        get_compile_time_arg_val(4) == 1 ? zkml::prim::QuantRange::M31 : zkml::prim::QuantRange::Full;
    constexpr uint32_t magnitude_bits = zkml::prim::pot_magnitude_bits(bit_width, range);
    constexpr uint32_t quant_dst_slots = DST_TILES;
    constexpr uint32_t out_ring_pages = 2 * quant_dst_slots;  // c_16 holds two blocks (see the host)

    constexpr uint32_t cb_in_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_out_id = tt::CBIndex::c_16;
    constexpr uint32_t cb_scale_out_id = tt::CBIndex::c_17;

    // DST slots: input tiles in 0 .. block - 1, the scaler in the last slot
    constexpr uint32_t dst_scale = DST_TILES - 1;

    CircularBuffer cb_in(cb_in_id);
    CircularBuffer cb_out(cb_out_id);
    CircularBuffer cb_scale_out(cb_scale_out_id);

    compute_kernel_hw_startup(cb_in_id, cb_out_id);

    // ---- Pass 1: absmax over the whole tensor, kept in an SFPU register ----
    // The whole pass and the scaler below share one DST section: nothing is packed until the
    // scaler, so one MATH -> PACK handshake replaces one per input tile. Each block overwrites
    // slots 0 .. n - 1; the SFPU stall in absmax_tile orders it after the copies.
    MATH(zkml::sfpu::absmax_init());
    copy_init(cb_in_id);
    cb_scale_out.reserve_back(1);
    zkml::common::RingCursor in_ring(in_ring_pages);  // c_0 read position, shared by both passes
    zkml::common::DstBlockMediator absmax_blocks(absmax_dst_slots, num_tiles, in_ring);
    tile_regs_acquire();
    {
        DeviceZoneScopedN("PASS1");  // profiler zone (no code without TT_METAL_DEVICE_PROFILER)
        while (const uint32_t n = absmax_blocks.next()) {
            cb_in.wait_front(n);
            for (uint32_t i = 0; i < n; i++) {
                copy_tile(cb_in_id, i, i);
            }
            for (uint32_t i = 0; i < n; i++) {
                MATH(zkml::sfpu::absmax_tile(i));
            }
            cb_in.pop_front(n);
            absmax_blocks.commit(n);
        }
    }

    // ---- Scaler: SFPU register -> DST -> c_17 (for the host) ----
    // The scaler also stays in LReg7, which is all pass 2 needs (quant_sfpu.hpp).
    {
        DeviceZoneScopedN("SCALER");
        MATH(zkml::sfpu::absmax_to_scaler<magnitude_bits>(dst_scale));
        DEVICE_PRINT_MATH("Scaler in DST {}:\n", dst_scale);
        zkml::debug::print_dest_tile_rows(dst_scale, DataFormat::Float32, 0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_scale_out_id);
        pack_tile(dst_scale, cb_scale_out_id);
        tile_regs_release();
        cb_scale_out.push_back(1);
        pack_reconfig_data_format(cb_out_id);
    }

    // ---- Pass 2: quantize ----
    // Loads the LREG constants of the sweeps (not LReg7: it holds the scaler), so it must come after pass 1.
    if constexpr (bit_width == 32) {
        MATH(zkml::sfpu::quant32_init());
    } else {
        MATH((zkml::sfpu::quant_small_init<is_output_int8, sign_mag>()));
    }

    zkml::common::RingCursor out_ring(out_ring_pages);  // c_16 write position
    // A reserve/push must be contiguous in the CB: the mediator cuts the block at the end of both rings.
    zkml::common::DstBlockMediator quant_blocks(quant_dst_slots, num_tiles, in_ring, out_ring);
    DeviceZoneScopedN("PASS2");
    while (const uint32_t n = quant_blocks.next()) {
        cb_in.wait_front(n);
        cb_out.reserve_back(n);

        tile_regs_acquire();
        copy_init(cb_in_id);
        for (uint32_t i = 0; i < n; i++) {
            DEVICE_PRINT_UNPACK("Tile={}\n", quant_blocks.done() + i);
            UNPACK(zkml::debug::print_tile_rows(cb_in_id, i));
            copy_tile(cb_in_id, i, i);
        }
        // One SFPU sweep per tile: x * 2^k from LReg7, round half to even, convert to int32 (Int8 output: the two's
        // complement byte, negative q as 256 + q).
        for (uint32_t i = 0; i < n; i++) {
            if constexpr (bit_width == 32) {
                MATH(zkml::sfpu::quant32_tile(i));
            } else {
                MATH((zkml::sfpu::quant_small_tile<is_output_int8, sign_mag>(i)));
            }
        }
        for (uint32_t i = 0; i < n; i++) {
            DEVICE_PRINT_MATH("Contents of DST {} after quantization:\n", i);
            zkml::debug::print_dest_tile_rows(i, DataFormat::Int32);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < n; i++) {
            pack_tile(i, cb_out_id);
        }
        tile_regs_release();
        for (uint32_t i = 0; i < n; i++) {
            DEVICE_PRINT_PACK("Tile={}\n", quant_blocks.done() + i);
            PACK(zkml::debug::print_tile_rows(cb_out_id, i));
        }
        cb_out.push_back(n);
        cb_in.pop_front(n);
        quant_blocks.commit(n);
    }
}
