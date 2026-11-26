// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    constexpr auto args = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(args, addr);
    Noc noc;
    CircularBuffer cb(tt::CBIndex::c_0);
    const uint32_t bytes = cb.get_tile_size();
    cb.reserve_back(num_tiles);
    for (uint32_t t = 0; t < num_tiles; t++) {
        noc.async_read(in, cb, bytes, {.page_id = t}, {.offset_bytes = t * bytes});
    }
    noc.async_read_barrier();
    cb.push_back(num_tiles);
}
