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
 * @brief Reader kernel for per-tensor PoT quantization: streams the input tiles from DRAM twice.
 *
 * The compute kernel makes two passes over the tensor: the first finds the absmax and the scaler
 * `2^k`, the second quantizes with it. Both passes consume the same tiles in the same order, so the
 * reader sends the whole tensor twice.
 *
 * | Circular buffer | Direction | Format                | Pages | Contents                           |
 * |-----------------|-----------|-----------------------|-------|------------------------------------|
 * | `c_0`           | produces  | Float32 or Float16_b  | 2     | Input tiles in page order, twice   |
 *
 * **Compile-time args**
 *
 * | Index | Name      | Description                              |
 * |-------|-----------|------------------------------------------|
 * | 0..   | `in_args` | `TensorAccessorArgs` of the input buffer |
 *
 * **Runtime args**
 *
 * | Index | Name        | Type     | Description                                         |
 * |-------|-------------|----------|-----------------------------------------------------|
 * | 0     | `in_addr`   | uint32_t | DRAM base address of the input tensor (TILE layout) |
 * | 1     | `num_tiles` | uint32_t | Number of input tiles                               |
 *
 * @pre  The input tensor is interleaved, in TILE layout, and holds at least `num_tiles` pages.
 * @post `c_0` has received `2 * num_tiles` tiles.
 * @see ../compute/quantize.cpp, writer.cpp
 */
// clang-format on
void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in_id = tt::CBIndex::c_0;
    constexpr uint32_t num_passes = 2;  // absmax, then quantize

    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(in_args, in_addr);

    Noc noc;
    CircularBuffer cb_in(cb_in_id);

    const uint32_t tile_bytes = cb_in.get_tile_size();
    for (uint32_t pass = 0; pass < num_passes; pass++) {
        for (uint32_t t = 0; t < num_tiles; t++) {
            cb_in.reserve_back(1);
            noc.async_read(in, cb_in, tile_bytes, {.page_id = t}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_in.push_back(1);
        }
    }
}
