// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

/**
 * @brief Reader of pot_quant_m31: streams the quantized tiles, one at a time, from DRAM into `c_0`.
 *
 * **Compile-time args**: `TensorAccessorArgs` of the input from index 0.
 * **Runtime args**: `[0]` input DRAM address, `[1]` number of tiles.
 * @see ../compute/to_m31.cpp, writer.cpp
 */
void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(in_args, in_addr);

    Noc noc;
    CircularBuffer cb_in(tt::CBIndex::c_0);
    const uint32_t tile_bytes = cb_in.get_tile_size();
    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_in.reserve_back(1);
        noc.async_read(in, cb_in, tile_bytes, {.page_id = t}, {});
        noc.async_read_barrier();
        cb_in.push_back(1);
    }
}
