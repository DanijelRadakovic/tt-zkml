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
 * @brief Reader kernel for per-tensor PoT quantization: streams input tiles from DRAM and builds the
 *        scale tile from a host-supplied value.
 *
 * The host chooses the number of fractional bits `k` and passes the reciprocal scale `2^k` as an fp32
 * bit pattern. The kernel writes that value into every element of one Float32 tile in `c_1`, because
 * `quant_tile` reads the scale element-wise from DST. It pushes that tile once, and the compute kernel
 * reuses it for every input tile. It then reads the input tiles one at a time from DRAM into `c_0`.
 *
 * | Circular buffer | Direction | Format                | Pages | Contents                          |
 * |-----------------|-----------|-----------------------|-------|-----------------------------------|
 * | `c_0`           | produces  | Float32 or Float16_b  | 2     | Input tiles, in page order        |
 * | `c_1`           | produces  | Float32               | 1     | Scale tile, every element = `2^k` |
 *
 * **Compile-time args**
 *
 * | Index | Name      | Description                              |
 * |-------|-----------|------------------------------------------|
 * | 0..   | `in_args` | `TensorAccessorArgs` of the input buffer |
 *
 * **Runtime args**
 *
 * | Index | Name         | Type     | Description                                         |
 * |-------|--------------|----------|-----------------------------------------------------|
 * | 0     | `in_addr`    | uint32_t | DRAM base address of the input tensor (TILE layout) |
 * | 1     | `num_tiles`  | uint32_t | Number of input tiles to read                       |
 * | 2     | `scale_bits` | uint32_t | fp32 bit pattern of the reciprocal scale `2^k`      |
 *
 * @pre  The input tensor is interleaved, in TILE layout, and holds at least `num_tiles` pages.
 * @post `c_0` has received `num_tiles` tiles and `c_1` exactly one tile.
 * @note `2^k` must be a normal fp32 value (`-126 <= k <= 127`) for the multiply to be exact.
 *       The host checks this.
 * @see ../compute/quantize.cpp, writer.cpp
 */
// clang-format on
void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t scale_bits = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_scale_id = tt::CBIndex::c_1;

    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(in_args, in_addr);

    Noc noc;
    CircularBuffer cb_in(cb_in_id);
    CircularBuffer cb_scale(cb_scale_id);
    
    // Scale tile: quant_tile reads the scale element-wise from DST, so fill all 32x32 elements.
    cb_scale.reserve_back(1);
    auto* scale_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_scale.get_write_ptr());
    const uint32_t scale_elems = cb_scale.get_tile_size() / sizeof(uint32_t);
    for (uint32_t i = 0; i < scale_elems; i++) {
        scale_ptr[i] = scale_bits;
    }
    cb_scale.push_back(1);

    const uint32_t tile_bytes = cb_in.get_tile_size();
    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_in.reserve_back(1);
        noc.async_read(in, cb_in, tile_bytes, {.page_id = t}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in.push_back(1);
    }
}
