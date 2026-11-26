// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Writes the output tiles (c_16) to DRAM.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t tiles = get_arg_val<uint32_t>(1);
    constexpr auto args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(args, addr);
    Noc noc;
    CircularBuffer cb(tt::CBIndex::c_16);
    const uint32_t bytes = cb.get_tile_size();
    cb.wait_front(tiles);
    for (uint32_t t = 0; t < tiles; t++) {
        noc.async_write(cb, out, bytes, {.offset_bytes = t * bytes}, {.page_id = t});
    }
    noc.async_write_barrier();
    cb.pop_front(tiles);
}
