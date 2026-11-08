// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Writer for the L1-fit benchmark: the scaler tile from c_17, then every quantized tile from c_16,
// one page at a time. The tile loop is one "WRITE_ALL" profiler zone.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t scale_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto scale_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);
    const auto scale = TensorAccessor(scale_args, scale_addr);

    Noc noc;
    CircularBuffer cb_out(tt::CBIndex::c_16);
    CircularBuffer cb_scale_out(tt::CBIndex::c_17);

    cb_scale_out.wait_front(1);
    noc.async_write(cb_scale_out, scale, cb_scale_out.get_tile_size(), {}, {.page_id = 0});
    noc.async_write_barrier();
    cb_scale_out.pop_front(1);

    DeviceZoneScopedN("WRITE_ALL");
    const uint32_t tile_bytes = cb_out.get_tile_size();
    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_out.wait_front(1);
        noc.async_write(cb_out, out, tile_bytes, {}, {.page_id = t});
        noc.async_write_barrier();
        cb_out.pop_front(1);
    }
}
