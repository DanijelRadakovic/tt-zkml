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
    constexpr auto args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(args, addr);
    Noc noc;
    CircularBuffer cb(tt::CBIndex::c_16);
    cb.wait_front(1);
    noc.async_write(cb, out, cb.get_tile_size(), {}, {.page_id = 0});
    noc.async_write_barrier();
    cb.pop_front(1);
}
