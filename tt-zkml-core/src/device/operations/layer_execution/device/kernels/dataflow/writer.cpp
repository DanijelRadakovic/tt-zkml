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
 * @brief Writer of execute_layer: in the order compute produces them, the scalers of x and W, the accumulator tiles (`c_6`, the
 *        `acc` tensor), the scaler of the requantization, and the output tiles of `c_16`.
 *
 * The accumulator is resident in L1 (compute reads it again for the requantization and never pops it), so the writer only
 * waits for the tiles to arrive and writes them from where they are, in batches. The output tiles are written in batches too
 * (see pot_quantize's writer): wait for `BATCH_SIZE` tiles, issue the writes, one `async_writes_flushed`, pop. A batch never
 * crosses the end of the `c_16` ring. The only full `async_write_barrier` is the last statement.
 *
 * **Compile-time args**: `[0]` `BATCH_SIZE`, `[1]` `RING_PAGES` (pages of `c_16`), then `TensorAccessorArgs` of the output, of
 * the accumulator and of the three scaler tensors.
 * **Runtime args**: `[0]` output address, `[1]` accumulator address, `[2..4]` scaler addresses (x, W, out), `[5]` number of
 * output tiles (the accumulator has as many).
 */
// clang-format on
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t acc_addr = get_arg_val<uint32_t>(1);
    const uint32_t sx_addr = get_arg_val<uint32_t>(2);
    const uint32_t sw_addr = get_arg_val<uint32_t>(3);
    const uint32_t so_addr = get_arg_val<uint32_t>(4);
    const uint32_t num_tiles = get_arg_val<uint32_t>(5);

    constexpr uint32_t BATCH_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t RING_PAGES = get_compile_time_arg_val(1);
    static_assert(BATCH_SIZE >= 1 && BATCH_SIZE <= RING_PAGES / 2, "the batch must be at most half of the c_16 ring");

    constexpr auto out_args = TensorAccessorArgs<2>();
    constexpr auto acc_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto sx_args = TensorAccessorArgs<acc_args.next_compile_time_args_offset()>();
    constexpr auto sw_args = TensorAccessorArgs<sx_args.next_compile_time_args_offset()>();
    constexpr auto so_args = TensorAccessorArgs<sw_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);
    const auto acc = TensorAccessor(acc_args, acc_addr);
    const auto sx = TensorAccessor(sx_args, sx_addr);
    const auto sw = TensorAccessor(sw_args, sw_addr);
    const auto so = TensorAccessor(so_args, so_addr);

    Noc noc;
    CircularBuffer cb_out(tt::CBIndex::c_16), cb_acc(tt::CBIndex::c_6);
    CircularBuffer cb_sx(tt::CBIndex::c_17), cb_sw(tt::CBIndex::c_18), cb_so(tt::CBIndex::c_19);

    cb_sx.wait_front(1);
    noc.async_write(cb_sx, sx, cb_sx.get_tile_size(), {}, {.page_id = 0});
    noc.async_writes_flushed();
    cb_sx.pop_front(1);
    cb_sw.wait_front(1);
    noc.async_write(cb_sw, sw, cb_sw.get_tile_size(), {}, {.page_id = 0});
    noc.async_writes_flushed();
    cb_sw.pop_front(1);
    // The accumulator tiles, as the matmul blocks finish. They stay in L1 (compute reads them again).
    const uint32_t acc_bytes = cb_acc.get_tile_size();
    for (uint32_t done = 0; done < num_tiles; done += BATCH_SIZE) {
        const uint32_t n = num_tiles - done < BATCH_SIZE ? num_tiles - done : BATCH_SIZE;
        cb_acc.wait_front(done + n);
        for (uint32_t i = 0; i < n; i++) {
            noc.async_write(cb_acc, acc, acc_bytes, {.offset_bytes = (done + i) * acc_bytes}, {.page_id = done + i});
        }
        noc.async_writes_flushed();
    }

    cb_so.wait_front(1);
    noc.async_write(cb_so, so, cb_so.get_tile_size(), {}, {.page_id = 0});
    noc.async_writes_flushed();
    cb_so.pop_front(1);

    const uint32_t tile_bytes = cb_out.get_tile_size();
    uint32_t pos = 0;
    for (uint32_t done = 0; done < num_tiles;) {
        uint32_t n = num_tiles - done < BATCH_SIZE ? num_tiles - done : BATCH_SIZE;
        n = n < RING_PAGES - pos ? n : RING_PAGES - pos;
        cb_out.wait_front(n);
        for (uint32_t i = 0; i < n; i++) {
            noc.async_write(cb_out, out, tile_bytes, {.offset_bytes = i * tile_bytes}, {.page_id = done + i});
        }
        noc.async_writes_flushed();
        cb_out.pop_front(n);
        done += n;
        pos += n;
        if (pos == RING_PAGES) {
            pos = 0;
        }
    }
    noc.async_write_barrier();
}
