// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Copies the input tiles (fp32 or bf16) into c_0 and the scaler tile into c_1 once; the compute kernel reuses them from L1.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t addr_in = get_arg_val<uint32_t>(0);
    const uint32_t addr_scale = get_arg_val<uint32_t>(1);
    const uint32_t tiles = get_arg_val<uint32_t>(2);
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto scale_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto in = TensorAccessor(in_args, addr_in);
    const auto scale = TensorAccessor(scale_args, addr_scale);
    Noc noc;
    CircularBuffer cb_in(tt::CBIndex::c_0), cb_scale(tt::CBIndex::c_1);
    const uint32_t in_bytes = cb_in.get_tile_size(), scale_bytes = cb_scale.get_tile_size();
    cb_in.reserve_back(tiles);
    for (uint32_t t = 0; t < tiles; t++) {
        noc.async_read(in, cb_in, in_bytes, {.page_id = t}, {.offset_bytes = t * in_bytes});
    }
    cb_scale.reserve_back(1);
    noc.async_read(scale, cb_scale, scale_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_in.push_back(tiles);
    cb_scale.push_back(1);
}
