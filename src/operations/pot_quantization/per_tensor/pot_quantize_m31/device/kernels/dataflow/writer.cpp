// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

/**
 * @brief Writer of pot_quant_m31: drains `c_16` (UInt32 tiles) to the output tensor in batches.
 *
 * It waits for `BATCH_SIZE` tiles (one compute block), issues their writes back to back, waits once until they have left
 * L1 (`async_writes_flushed`) and pops the batch. One wait per batch instead of one write barrier per tile. The compute
 * kernel starts every block at a multiple of `BATCH_SIZE` and `c_16` holds two blocks, so a batch never wraps the ring;
 * only the last one can be shorter. The only full `async_write_barrier` (the data has arrived in DRAM) is the last statement.
 *
 * **Compile-time args**: `[0]` `BATCH_SIZE` (the compute block size), then `TensorAccessorArgs` of the output.
 * **Runtime args**: `[0]` output DRAM address, `[1]` number of tiles.
 * @pre  `c_16` holds `2 * BATCH_SIZE` pages.
 * @see reader.cpp, ../compute/to_m31.cpp
 */
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t BATCH_SIZE = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();
    const auto out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer cb_out(tt::CBIndex::c_16);
    const uint32_t tile_bytes = cb_out.get_tile_size();
    for (uint32_t done = 0; done < num_tiles; done += BATCH_SIZE) {
        const uint32_t left = num_tiles - done;
        const uint32_t n = left < BATCH_SIZE ? left : BATCH_SIZE;
        cb_out.wait_front(n);
        for (uint32_t i = 0; i < n; i++) {
            noc.async_write(cb_out, out, tile_bytes, {.offset_bytes = i * tile_bytes}, {.page_id = done + i});
        }
        noc.async_writes_flushed();  // the data has left L1: the pages can be reused
        cb_out.pop_front(n);
    }
    noc.async_write_barrier();  // everything has arrived in DRAM
}
