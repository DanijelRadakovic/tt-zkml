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
 * @brief Writer kernel for per-tensor PoT quantization: writes the scaler tile from `c_17`, then
 *        drains the quantized tiles from `c_16` to DRAM.
 *
 * The compute kernel produces the scaler before any quantized tile, so it is written first.
 *
 * The tiles are written in batches: wait for `BATCH_SIZE` tiles, issue their writes back to back, wait once until the
 * writes have left L1 (`async_writes_flushed`), then pop the batch. One wait per batch instead of one write barrier per
 * tile; the writes of a batch overlap each other. A batch never crosses the end of the `c_16` ring (`copy` addresses page
 * `read_ptr + i` without wrapping, and the compute kernel cuts its blocks there too) and is cut at the end of the tensor, so
 * it is shorter only there. The only full barrier (`async_write_barrier`, the data has arrived in DRAM) is the last
 * statement: the kernel must not end before the data is written. With `BATCH_SIZE` the size of a compute block, the
 * writer drains one block while compute fills the other half of the ring.
 *
 * | Circular buffer | Direction | Format                          | Contents                  |
 * |-----------------|-----------|---------------------------------|---------------------------|
 * | `c_16`          | consumes  | Int32, or UInt8 for Int8 output | Quantized tiles           |
 * | `c_17`          | consumes  | Float32                         | Scaler tile, all `2^k`    |
 *
 * **Compile-time args**
 *
 * | Index | Name         | Description                                                                         |
 * |-------|--------------|-------------------------------------------------------------------------------------|
 * | 0     | `BATCH_SIZE` | Tiles per batch, 1 .. `RING_PAGES` (the host passes the compute block size)         |
 * | 1     | `RING_PAGES` | Pages of `c_16`                                                                     |
 * | 2..   | `out_args`   | `TensorAccessorArgs` of the output buffer                                           |
 * | ..    | `scale_args` | `TensorAccessorArgs` of the scaler buffer                                           |
 *
 * **Runtime args**
 *
 * | Index | Name         | Type     | Description                                          |
 * |-------|--------------|----------|------------------------------------------------------|
 * | 0     | `out_addr`   | uint32_t | DRAM base address of the output tensor (TILE layout) |
 * | 1     | `scale_addr` | uint32_t | DRAM base address of the scaler tensor (one tile)    |
 * | 2     | `num_tiles`  | uint32_t | Number of quantized tiles to write                   |
 *
 * @pre  The output tensor is interleaved, in TILE layout, and holds at least `num_tiles` pages.
 * @pre  Compute pushes its blocks without crossing the end of the `c_16` ring, and `c_16` holds at least `2 * BATCH_SIZE`
 *       pages, so a batch is always completed without the writer having to free space first.
 * @post Scaler page 0 holds the scaler tile. Tile `t` of `c_16` has been written to output page
 *       `t`, for `t` in `[0, num_tiles)`, and the writes have completed.
 * @see reader.cpp, ../compute/quantize.cpp
 */
// clang-format on
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t scale_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t BATCH_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t RING_PAGES = get_compile_time_arg_val(1);
    static_assert(BATCH_SIZE >= 1 && BATCH_SIZE <= RING_PAGES / 2, "the batch must be at most half of the c_16 ring");

    constexpr uint32_t cb_out_id = tt::CBIndex::c_16;
    constexpr uint32_t cb_scale_out_id = tt::CBIndex::c_17;

    constexpr auto out_args = TensorAccessorArgs<2>();
    constexpr auto scale_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);
    const auto scale = TensorAccessor(scale_args, scale_addr);

    Noc noc;
    CircularBuffer cb_out(cb_out_id);
    CircularBuffer cb_scale_out(cb_scale_out_id);

    cb_scale_out.wait_front(1);
    noc.async_write(cb_scale_out, scale, cb_scale_out.get_tile_size(), {}, {.page_id = 0});
    noc.async_writes_flushed();
    cb_scale_out.pop_front(1);

    const uint32_t tile_bytes = cb_out.get_tile_size();
    uint32_t pos = 0;  // read position in the c_16 ring, in pages
    for (uint32_t done = 0; done < num_tiles;) {
        const uint32_t left = num_tiles - done;
        uint32_t n = left < BATCH_SIZE ? left : BATCH_SIZE;
        const uint32_t room = RING_PAGES - pos;  // the longest run that does not wrap
        n = n < room ? n : room;
        cb_out.wait_front(n);
        for (uint32_t i = 0; i < n; i++) {
            noc.async_write(cb_out, out, tile_bytes, {.offset_bytes = i * tile_bytes}, {.page_id = done + i});
        }
        noc.async_writes_flushed();  // the data has left L1: the pages can be reused
        cb_out.pop_front(n);
        done += n;
        pos += n;
        if (pos == RING_PAGES) {
            pos = 0;
        }
    }
    noc.async_write_barrier();  // everything has arrived in DRAM
}
