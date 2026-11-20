// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// clang-format off
/**
 * @brief Reader of execute_layer: reads the tiles of x into `c_0` and then the tiles of W into `c_1`, once.
 *
 * Both CBs hold the whole tensor (they are resident: the compute kernel reads every tile twice and never pops). The tiles
 * are read in batches of `BATCH_SIZE` behind one barrier and pushed after every batch, so compute starts quantizing x
 * while W is still being read.
 *
 * **Compile-time args**: `[0]` `BATCH_SIZE`, then `TensorAccessorArgs` of x, then of W.
 * **Runtime args**: `[0]` x address, `[1]` W address, `[2]` tiles of x, `[3]` tiles of W.
 */
// clang-format on
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t x_tiles = get_arg_val<uint32_t>(2);
    const uint32_t w_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t BATCH_SIZE = get_compile_time_arg_val(0);
    constexpr auto x_args = TensorAccessorArgs<1>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, x_addr);
    const auto w = TensorAccessor(w_args, w_addr);

    Noc noc;
    CircularBuffer cb_x(tt::CBIndex::c_0), cb_w(tt::CBIndex::c_1);
    const uint32_t x_bytes = cb_x.get_tile_size(), w_bytes = cb_w.get_tile_size();

    for (uint32_t done = 0; done < x_tiles; done += BATCH_SIZE) {
        const uint32_t n = x_tiles - done < BATCH_SIZE ? x_tiles - done : BATCH_SIZE;
        cb_x.reserve_back(n);
        for (uint32_t i = 0; i < n; i++) {
            noc.async_read(x, cb_x, x_bytes, {.page_id = done + i}, {.offset_bytes = i * x_bytes});
        }
        noc.async_read_barrier();
        cb_x.push_back(n);
    }
    for (uint32_t done = 0; done < w_tiles; done += BATCH_SIZE) {
        const uint32_t n = w_tiles - done < BATCH_SIZE ? w_tiles - done : BATCH_SIZE;
        cb_w.reserve_back(n);
        for (uint32_t i = 0; i < n; i++) {
            noc.async_read(w, cb_w, w_bytes, {.page_id = done + i}, {.offset_bytes = i * w_bytes});
        }
        noc.async_read_barrier();
        cb_w.push_back(n);
    }
}
