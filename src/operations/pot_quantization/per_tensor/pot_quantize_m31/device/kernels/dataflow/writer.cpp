// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

/**
 * @brief Writer of pot_quant_m31: drains `c_16` (UInt32 tiles) to the output tensor, one tile at a time.
 *
 * **Compile-time args**: `TensorAccessorArgs` of the output from index 0.
 * **Runtime args**: `[0]` output DRAM address, `[1]` number of tiles.
 * @see reader.cpp, ../compute/to_m31.cpp
 */
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer cb_out(tt::CBIndex::c_16);
    const uint32_t tile_bytes = cb_out.get_tile_size();
    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_out.wait_front(1);
        noc.async_write(cb_out, out, tile_bytes, {}, {.page_id = t});
        noc.async_write_barrier();
        cb_out.pop_front(1);
    }
}
