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
 * @brief Writer kernel for per-tensor PoT quantization: drains quantized tiles from `c_16` to DRAM.
 *
 * | Circular buffer | Direction | Format                          | Contents        |
 * |-----------------|-----------|---------------------------------|-----------------|
 * | `c_16`          | consumes  | Int32, or UInt8 for Int8 output | Quantized tiles |
 *
 * **Compile-time args**
 *
 * | Index | Name       | Description                               |
 * |-------|------------|-------------------------------------------|
 * | 0..   | `out_args` | `TensorAccessorArgs` of the output buffer |
 *
 * **Runtime args**
 *
 * | Index | Name        | Type     | Description                                          |
 * |-------|-------------|----------|------------------------------------------------------|
 * | 0     | `out_addr`  | uint32_t | DRAM base address of the output tensor (TILE layout) |
 * | 1     | `num_tiles` | uint32_t | Number of tiles to write                             |
 *
 * @pre  The output tensor is interleaved, in TILE layout, and holds at least `num_tiles` pages.
 * @post Tile `t` of `c_16` has been written to output page `t`, for `t` in `[0, num_tiles)`.
 * @see reader.cpp, ../compute/quantize.cpp
 */
// clang-format on
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out_id = tt::CBIndex::c_16;

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer cb_out(cb_out_id);

    const uint32_t tile_bytes = cb_out.get_tile_size();
    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_out.wait_front(1);
        noc.async_write(cb_out, out, tile_bytes, {}, {.page_id = t});
        noc.async_write_barrier();
        cb_out.pop_front(1);
    }
}
