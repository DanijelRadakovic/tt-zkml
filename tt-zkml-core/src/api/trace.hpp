// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

namespace zkml {

/// The execution trace of one forward pass of a quantized network, copied to the host. Everything the constraints need to check the
/// execution is here as plain integers and floats.
struct Trace {
    uint32_t batch = 0;        ///< M, the rows of the input
    uint32_t in_features = 0;  ///< K_0
    std::vector<uint32_t> widths;  ///< N_l, the output features of every layer

    /// `[batch, N_last]` row-major: the sign-magnitude int8 bytes of the last layer (bit 7 = sign, bits 0..6 = magnitude).
    std::vector<uint8_t> output;

    float scaler_x = 0.f;               ///< 2^kx, the quantization scaler of the input of layer 0
    std::vector<float> scaler_w;        ///< 2^kw_l, the quantization scaler of the weights of layer l
    std::vector<float> scaler_out;      ///< 2^k_l of the requantization of layer l; its output is the quantized input of layer l + 1

    /// Per layer, `[batch, N_l]` row-major: the raw sign-magnitude bits of the int32 accumulator `q_x @ q_w`, before the activation
    /// (bit 31 = sign, bits 0..30 = magnitude).
    std::vector<std::vector<uint32_t>> acc;

    uint32_t layers() const { return static_cast<uint32_t>(widths.size()); }
};

}  // namespace zkml
