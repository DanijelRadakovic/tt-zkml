// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Circular-buffer indices of execute_network, shared by the program factory and the three kernels (plain constants
// only: this header is included on the host and on the device).
//
//   c_0   x             input tiles (fp32 or bf16), layer 0 only
//   c_1   w             weight tiles of the layer being run (fp32 or bf16), reused for every layer
//   c_2   q_w           quantized weights (UInt8, read by the FPU as Int8), reused for every layer
//   c_3   acc           raw int32 accumulator (sign-magnitude), reused for every layer
//   c_4 .. c_4+L-1     the quantized activations: act_l is the input of layer l (act_0 = quantized x, act_{l+1} =
//   requantized output of
//                       layer l); one buffer per layer boundary, never reused
//   c_16  out           the output tiles of the last layer (a ring drained by the writer)
//   c_17  scaler_x      c_18 scaler_w     c_19 scaler_out
//
// c_1, c_2 and c_3 hold one layer at a time and are reused: every layer pushes and pops the full capacity (the tiles it
// does not use are padding), so that the read and write pointers are back at the start of the buffer before the next
// layer. The tiles of a layer are read by index, which assumes they are contiguous.

#pragma once

#include <cstdint>

namespace zkml::network {

constexpr uint32_t c_x = 0;
constexpr uint32_t c_w = 1;
constexpr uint32_t c_qw = 2;
constexpr uint32_t c_acc = 3;
constexpr uint32_t c_act0 = 4;
constexpr uint32_t c_out = 16;
constexpr uint32_t c_scale_x = 17;
constexpr uint32_t c_scale_w = 18;
constexpr uint32_t c_scale_out = 19;

/// The layers of one network. Every layer is compiled into the kernel (its sizes are template arguments), and the
/// kernel text of the three compute RISC-Vs must fit the 70 KB kernel config buffer of a core together with the reader
/// and the writer: 8 layers were 1.8 KB too large. The activation buffers (c_4 and up) would allow 12.
constexpr uint32_t max_layers = 6;

constexpr uint32_t act_cb(uint32_t layer) { return c_act0 + layer; }

/// A fp32 DST (fp32_dest_acc_en) with half sync holds 4 tiles.
constexpr uint32_t dst_tiles = 4;
/// Pages of the scaler buffers: a layer's scaler is pushed before the writer has popped the previous one.
constexpr uint32_t scaler_pages = 2;
/// Largest K in tiles: K * 127 * 127 stays below 2^30 (31 magnitude bits of the accumulator).
constexpr uint32_t max_k_tiles = 2080;

}  // namespace zkml::network
