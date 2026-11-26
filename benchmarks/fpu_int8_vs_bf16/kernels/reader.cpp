// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Copies the A tiles into c_0 and the B tiles into c_1 once; the compute kernel then reuses them from L1.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t addr_a = get_arg_val<uint32_t>(0);
    const uint32_t addr_b = get_arg_val<uint32_t>(1);
    const uint32_t tiles_a = get_arg_val<uint32_t>(2);
    const uint32_t tiles_b = get_arg_val<uint32_t>(3);
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, addr_a);
    const auto b = TensorAccessor(b_args, addr_b);
    Noc noc;
    CircularBuffer cb_a(tt::CBIndex::c_0), cb_b(tt::CBIndex::c_1);
    const uint32_t bytes_a = cb_a.get_tile_size(), bytes_b = cb_b.get_tile_size();
    cb_a.reserve_back(tiles_a);
    for (uint32_t t = 0; t < tiles_a; t++) {
        noc.async_read(a, cb_a, bytes_a, {.page_id = t}, {.offset_bytes = t * bytes_a});
    }
    cb_b.reserve_back(tiles_b);
    for (uint32_t t = 0; t < tiles_b; t++) {
        noc.async_read(b, cb_b, bytes_b, {.page_id = t}, {.offset_bytes = t * bytes_b});
    }
    noc.async_read_barrier();
    cb_a.push_back(tiles_a);
    cb_b.push_back(tiles_b);
}
