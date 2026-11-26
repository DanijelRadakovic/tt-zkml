// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/tensor/noc_traits.h"
#include "device/operations/network_execution/device/network_layout.hpp"

// clang-format off
/**
 * @brief Writer of execute_network. Per layer, in order: the scaler of the weights, (last layer only) the output tiles, then once
 *        compute has requantized the layer, its accumulator and the scaler of the requantization. The scaler of x comes first.
 *
 * The outputs are packed (see execute_network_device_operation.hpp): the scalers are the tiles of one tensor, the accumulators of all layers
 * are the column tiles of one tensor, so the host allocates three output tensors however many layers there are.
 *
 * The accumulator buffer `c_3` is reused for every layer, and compute reads it (by index, without popping) for the
 * requantization, so the writer must not pop it before compute is done. Compute pushes the scaler of the requantization only
 * after the requantization sweep, so "the scaler of layer l has arrived" means "the accumulator of layer l is free to be
 * written and popped". The accumulator is padded to its capacity `CAP_ACC` like the other reused buffers: the writer waits for
 * and pops `CAP_ACC` tiles. The output tiles of the last layer are streamed through the ring `c_16` in batches (see
 * pot_quantize's writer); the only full `async_write_barrier` is the last statement.
 *
 * **Compile-time args**: `[0]` `BATCH_SIZE`, `[1]` `RING_PAGES`, `[2]` number of layers, `[3]` `CAP_ACC`, then `TensorAccessorArgs` of out, of
 * the scalers and of the accumulators.
 * **Runtime args**: `[0]` out address, `[1]` scalers address, `[2]` acc address, `[3]` tiles of out, `[4]` Mt, `[5]` column tiles of the packed
 * accumulators, then per layer `(Nt, first column tile of its accumulator)`.
 */
// clang-format on

namespace net = zkml::network;

constexpr uint32_t BATCH_SIZE = get_compile_time_arg_val(0);
constexpr uint32_t RING_PAGES = get_compile_time_arg_val(1);
constexpr uint32_t LAYERS = get_compile_time_arg_val(2);
constexpr uint32_t CAP_ACC = get_compile_time_arg_val(3);
static_assert(BATCH_SIZE >= 1 && BATCH_SIZE <= RING_PAGES / 2, "the batch must be at most half of the c_16 ring");

// The scaler tile at the front of `cb` (already waited for) to tile `page` of the scalers tensor, then pop it.
template <typename Accessor>
inline void write_scaler_from(Noc& noc, CircularBuffer& cb, const Accessor& tensor, uint32_t page) {
    noc.async_write(cb, tensor, cb.get_tile_size(), {}, {.page_id = page});
    noc.async_writes_flushed();
    cb.pop_front(1);
}

template <typename Accessor>
inline void write_scaler(Noc& noc, CircularBuffer& cb, const Accessor& tensor, uint32_t page) {
    cb.wait_front(1);
    write_scaler_from(noc, cb, tensor, page);
}

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t sc_addr = get_arg_val<uint32_t>(1);
    const uint32_t acc_addr = get_arg_val<uint32_t>(2);
    const uint32_t out_tiles = get_arg_val<uint32_t>(3);
    const uint32_t Mt = get_arg_val<uint32_t>(4);
    const uint32_t total_nt = get_arg_val<uint32_t>(5);

    constexpr auto out_args = TensorAccessorArgs<4>();
    constexpr auto sc_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto acc_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);
    const auto scalers = TensorAccessor(sc_args, sc_addr);
    const auto acc = TensorAccessor(acc_args, acc_addr);

    Noc noc;
    CircularBuffer cb_acc(net::c_acc), cb_sx(net::c_scale_x), cb_sw(net::c_scale_w), cb_so(net::c_scale_out), cb_out(net::c_out);
    // tiles of the scalers tensor: scaler_x, then scaler_w of every layer, then scaler_out of every layer
    write_scaler(noc, cb_sx, scalers, 0);

    for (uint32_t l = 0; l < LAYERS; l++) {
        const uint32_t nt = get_arg_val<uint32_t>(6 + 2 * l);
        const uint32_t nt_off = get_arg_val<uint32_t>(7 + 2 * l);
        const uint32_t acc_tiles = Mt * nt;

        write_scaler(noc, cb_sw, scalers, 1 + l);

        if (l + 1 == LAYERS) {
            // The output of the last layer, as the requantization sweep produces it.
            const uint32_t tile_bytes = cb_out.get_tile_size();
            uint32_t pos = 0;
            for (uint32_t done = 0; done < out_tiles;) {
                uint32_t n = out_tiles - done < BATCH_SIZE ? out_tiles - done : BATCH_SIZE;
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
        }

        // The layer is done (the scaler of the requantization is pushed last): the accumulator is free to be written and popped.
        {
            DeviceZoneScopedN("WAIT_DONE");  // compute still working on the layer (until the requantization pushed its scaler)
            cb_so.wait_front(1);
            cb_acc.wait_front(CAP_ACC);
        }
        DeviceZoneScopedN("WRITE");  // the accumulator and the scaler of the layer to DRAM
        const uint32_t acc_bytes = cb_acc.get_tile_size();
        // tile (mt, c) of the layer is page mt * total_nt + nt_off + c of the packed tensor
        uint32_t mt = 0, c = 0;
        for (uint32_t done = 0; done < acc_tiles; done += BATCH_SIZE) {
            const uint32_t n = acc_tiles - done < BATCH_SIZE ? acc_tiles - done : BATCH_SIZE;
            for (uint32_t i = 0; i < n; i++) {
                noc.async_write(
                    cb_acc, acc, acc_bytes, {.offset_bytes = (done + i) * acc_bytes}, {.page_id = mt * total_nt + nt_off + c});
                if (++c == nt) {
                    c = 0;
                    mt++;
                }
            }
            noc.async_writes_flushed();
        }
        cb_acc.pop_front(CAP_ACC);
        write_scaler_from(noc, cb_so, scalers, 1 + LAYERS + l);
    }
    noc.async_write_barrier();
}
