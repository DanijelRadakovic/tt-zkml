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
 * @brief Reader of execute_network: the tiles of x into `c_0` (resident, layer 0 only), then, layer by layer, the tiles of the
 *        weights of that layer into `c_1`.
 *
 * `c_1` holds one layer at a time. A layer's tiles are read in batches of `BATCH_SIZE` behind one barrier and pushed after every
 * batch, so compute starts quantizing while the rest is read; the tiles of the buffer the layer does not use are pushed as
 * padding, so that every layer pushes exactly `CAP_W` tiles and the pointers are back at the start for the next one. The reader
 * runs ahead: it waits (in `reserve_back`) only until compute has popped the previous layer's weights, which it does as soon as
 * the weights are quantized, so the next layer's weights are read while the previous layer's matmul and requantization run.
 *
 * **Compile-time args**: `[0]` `BATCH_SIZE`, `[1]` number of layers, `[2]` `CAP_W` (tiles of `c_1`), then `TensorAccessorArgs` of x and
 * one set for all the weight tensors (they have the same dtype and memory config).
 * **Runtime args**: `[0]` x address, `[1]` tiles of x, then per layer `(weight address, tiles of the weight)`.
 */
// clang-format on

constexpr uint32_t BATCH_SIZE = get_compile_time_arg_val(0);
constexpr uint32_t LAYERS = get_compile_time_arg_val(1);
constexpr uint32_t CAP_W = get_compile_time_arg_val(2);

template <typename Accessor>
void read_layer_weights(Noc& noc, CircularBuffer& cb_w, const Accessor& w, uint32_t w_tiles) {
    const uint32_t w_bytes = cb_w.get_tile_size();
    for (uint32_t done = 0; done < w_tiles; done += BATCH_SIZE) {
        const uint32_t n = w_tiles - done < BATCH_SIZE ? w_tiles - done : BATCH_SIZE;
        cb_w.reserve_back(n);
        for (uint32_t i = 0; i < n; i++) {
            noc.async_read(w, cb_w, w_bytes, {.page_id = done + i}, {.offset_bytes = i * w_bytes});
        }
        noc.async_read_barrier();
        cb_w.push_back(n);
    }
    if (w_tiles < CAP_W) {  // padding: no data, only the counters and the pointer
        cb_w.reserve_back(CAP_W - w_tiles);
        cb_w.push_back(CAP_W - w_tiles);
    }
}

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_tiles = get_arg_val<uint32_t>(1);

    constexpr auto x_args = TensorAccessorArgs<3>();
    const auto x = TensorAccessor(x_args, x_addr);

    Noc noc;
    CircularBuffer cb_x(zkml::network::c_x), cb_w(zkml::network::c_w);
    const uint32_t x_bytes = cb_x.get_tile_size();

    for (uint32_t done = 0; done < x_tiles; done += BATCH_SIZE) {
        const uint32_t n = x_tiles - done < BATCH_SIZE ? x_tiles - done : BATCH_SIZE;
        cb_x.reserve_back(n);
        for (uint32_t i = 0; i < n; i++) {
            noc.async_read(x, cb_x, x_bytes, {.page_id = done + i}, {.offset_bytes = i * x_bytes});
        }
        noc.async_read_barrier();
        cb_x.push_back(n);
    }
    // the accessor arguments are the same for every weight tensor: one set, one loop over the layers
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    for (uint32_t l = 0; l < LAYERS; l++) {
        const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(2 + 2 * l));
        DeviceZoneScopedN("READ_W");  // reading the weights of layer l, including the wait for space in c_1
        read_layer_weights(noc, cb_w, w, get_arg_val<uint32_t>(3 + 2 * l));
    }
}
