// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

/**
 * @brief Reader of pot_quant_m31: streams the quantized tiles from DRAM into `c_0` in batches.
 *
 * It reserves `BATCH_SIZE` tiles (one compute block), issues their reads back to back, waits once for all of them and
 * pushes the batch. One barrier per batch instead of one per tile (a barrier is a round trip to DRAM, about the time of
 * the whole conversion of a tile). The compute kernel consumes blocks that start at multiples of `BATCH_SIZE` and `c_0`
 * holds two blocks, so a batch never wraps the ring; only the last one can be shorter.
 *
 * **Compile-time args**: `[0]` `BATCH_SIZE` (the compute block size), then `TensorAccessorArgs` of the input.
 * **Runtime args**: `[0]` input DRAM address, `[1]` number of tiles.
 * @pre  `c_0` holds `2 * BATCH_SIZE` pages.
 * @see ../compute/to_m31.cpp, writer.cpp
 */
void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t BATCH_SIZE = get_compile_time_arg_val(0);
    constexpr auto in_args = TensorAccessorArgs<1>();
    const auto in = TensorAccessor(in_args, in_addr);

    Noc noc;
    CircularBuffer cb_in(tt::CBIndex::c_0);
    const uint32_t tile_bytes = cb_in.get_tile_size();
    for (uint32_t done = 0; done < num_tiles; done += BATCH_SIZE) {
        const uint32_t left = num_tiles - done;
        const uint32_t n = left < BATCH_SIZE ? left : BATCH_SIZE;
        cb_in.reserve_back(n);
        for (uint32_t i = 0; i < n; i++) {
            noc.async_read(in, cb_in, tile_bytes, {.page_id = done + i}, {.offset_bytes = i * tile_bytes});
        }
        noc.async_read_barrier();
        cb_in.push_back(n);
    }
}
