// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "m31_sfpu.hpp"

// clang-format off
/**
 * @brief Compute kernel of pot_quant_m31: q (Int32, or UInt8 bytes) -> its residue modulo 2^31 - 1 (UInt32).
 *
 * Blocks of `BLOCK` tiles: copy into DST, convert every tile on the SFPU (m31_sfpu.hpp), pack. `c_0` and `c_16` hold
 * two blocks each and every block starts at a multiple of `BLOCK`, so a block never wraps a ring (unlike the
 * quantize kernel, whose rings have different sizes).
 *
 * | Circular buffer | Direction | Format                | Contents                |
 * |-----------------|-----------|-----------------------|-------------------------|
 * | `c_0`           | consumes  | Int32, or UInt8       | Quantized tiles         |
 * | `c_16`          | produces  | UInt32                | M31 residues            |
 *
 * **Compile-time args**: `[0]` input_int8 (0 Int32, 1 Int8), `[1]` BLOCK (tiles per block, at most the DST tiles).
 * **Runtime args**: `[0]` num_tiles.
 * @pre The program enables `fp32_dest_acc_en`; Int32 input uses `UnpackToDestFp32` for `c_0`.
 */
// clang-format on
void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr bool input_int8 = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t BLOCK = get_compile_time_arg_val(1);
    static_assert(
        BLOCK <= ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>(),
        "BLOCK exceeds one DST section");

    constexpr uint32_t cb_in_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_out_id = tt::CBIndex::c_16;
    CircularBuffer cb_in(cb_in_id);
    CircularBuffer cb_out(cb_out_id);

    compute_kernel_hw_startup(cb_in_id, cb_out_id);
    copy_init(cb_in_id);
    MATH(zkml::sfpu::to_m31_init());

    for (uint32_t done = 0; done < num_tiles; done += BLOCK) {
        const uint32_t n = num_tiles - done < BLOCK ? num_tiles - done : BLOCK;
        cb_in.wait_front(n);
        cb_out.reserve_back(n);

        tile_regs_acquire();
        for (uint32_t i = 0; i < n; i++) {
            copy_tile(cb_in_id, i, i);
        }
        for (uint32_t i = 0; i < n; i++) {
            MATH(zkml::sfpu::to_m31_tile<input_int8>(i));
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < n; i++) {
            pack_tile(i, cb_out_id);
        }
        tile_regs_release();

        cb_out.push_back(n);
        cb_in.pop_front(n);
    }
}
