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
//   c_4   act_a         the quantized activations, ping-pong: layer l reads act_a (even l) or act_b (odd l), which holds the
//   c_5   act_b         requantized output of layer l - 1 (layer 0: the quantized x), and writes the other one
//   c_16  out           the output tiles of the last layer (a ring drained by the writer)
//   c_17  scaler_x      c_18 scaler_w     c_19 scaler_out
//
// Every buffer except c_0 and c_16 is reused: every layer pushes and pops the full capacity (the tiles it does not use are
// padding), so that the read and write pointers are back at the start of the buffer before the next layer. The tiles of a layer
// are read by index, which assumes they are contiguous.

#pragma once

#include <cstdint>

namespace zkml::network {

constexpr uint32_t c_x = 0;
constexpr uint32_t c_w = 1;
constexpr uint32_t c_qw = 2;
constexpr uint32_t c_acc = 3;
constexpr uint32_t c_act_a = 4;
constexpr uint32_t c_act_b = 5;
constexpr uint32_t c_out = 16;
constexpr uint32_t c_scale_x = 17;
constexpr uint32_t c_scale_w = 18;
constexpr uint32_t c_scale_out = 19;

/// The layers of one network. The layer sizes are run-time arguments of one compute kernel that loops over the layers, so the
/// kernel text does not depend on the number of layers. The limit is the runtime arguments of the writer (3 + 4 per layer, at most
/// 256 words) and the L1 of the buffers, which do not depend on the number of layers.
constexpr uint32_t max_layers = 32;

/// A fp32 DST (fp32_dest_acc_en) with half sync holds 4 tiles.
constexpr uint32_t dst_tiles = 4;
/// Pages of the scaler buffers: a layer's scaler is pushed before the writer has popped the previous one.
constexpr uint32_t scaler_pages = 2;
/// Largest K in tiles: K * 127 * 127 stays below 2^30 (31 magnitude bits of the accumulator).
constexpr uint32_t max_k_tiles = 2080;

}  // namespace zkml::network
