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
 * The reads are batched: `BATCH_SIZE` reads are issued in parallel (async) and then waited for with one barrier.
 * The two passes form a single stream of `2 * num_tiles` reads (page = read index mod `num_tiles`), which is cut
 * into batches. Only the very last batch can be shorter than `BATCH_SIZE`, and only when `BATCH_SIZE` does not
 * divide `2 * num_tiles` (for example, with a prime tile count).
 * The input buffer holds `BATCH_SIZE * cb_in_size_factor` tiles. With `cb_in_size_factor >= 2` there are enough
 * slots for the compute kernel to process one batch while the reader fetches the next one.
 *
 * | Circular buffer | Direction | Format                | Pages            | Contents                         |
 * |-----------------|-----------|-----------------------|------------------|----------------------------------|
 * | `c_0`           | produces  | Float32 or Float16_b  | `BATCH_SIZE * F` | Input tiles in page order, twice |
 *
 * **Compile-time args**
 *
 * | Index | Name         | Description                                                                      |
 * |-------|--------------|----------------------------------------------------------------------------------|
 * | 0     | `BATCH_SIZE` | Tiles per NoC batch, `1 <= BATCH_SIZE <= num_tiles` (`noc_read_tile_batch_size`) |
 * | 1..   | `in_args`    | `TensorAccessorArgs` of the input buffer                                         |
 *
 * **Runtime args**
 *
 * | Index | Name        | Type     | Description                                         |
 * |-------|-------------|----------|-----------------------------------------------------|
 * | 0     | `in_addr`   | uint32_t | DRAM base address of the input tensor (TILE layout) |
 * | 1     | `num_tiles` | uint32_t | Number of input tiles                               |
 *
 * @pre  The input tensor is interleaved, in TILE layout, and holds at least `num_tiles` pages.
 * @pre  `c_0` has `BATCH_SIZE * cb_in_size_factor` pages, `cb_in_size_factor >= 2`, and `BATCH_SIZE <= num_tiles`.
 * @post `c_0` has received `2 * num_tiles` tiles.
 * @see ../compute/quantize.cpp, writer.cpp
 */
// clang-format on
void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t BATCH_SIZE = get_compile_time_arg_val(0);  // 1 .. num_tiles, set on the host
    constexpr uint32_t num_passes = 2;  // absmax, then quantize

    constexpr auto in_args = TensorAccessorArgs<1>();
    const auto in = TensorAccessor(in_args, in_addr);

    Noc noc;
    CircularBuffer cb_in(tt::CBIndex::c_0);

    const uint32_t tile_bytes = cb_in.get_tile_size();
    const uint32_t total_reads = num_passes * num_tiles;
    uint32_t page = 0;  // next input page; wraps to 0 where pass 2 starts
    for (uint32_t done = 0; done < total_reads; done += BATCH_SIZE) {
        const uint32_t remaining = total_reads - done;
        const uint32_t n = remaining < BATCH_SIZE ? remaining : BATCH_SIZE;  // short only at the end
        cb_in.reserve_back(n);
        for (uint32_t i = 0; i < n; i++) {
            noc.async_read(in, cb_in, tile_bytes, {.page_id = page}, {.offset_bytes = i * tile_bytes});
            if (++page == num_tiles) {
                page = 0;
            }
        }
        noc.async_read_barrier();
        cb_in.push_back(n);
    }
}
