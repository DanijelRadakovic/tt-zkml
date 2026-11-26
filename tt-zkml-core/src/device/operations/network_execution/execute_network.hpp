// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <vector>

#include "device/operations/layer_execution/execute_layer.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace zkml::prim {

/// The outputs of `execute_network` for a network of L layers. Three device tensors however many layers there are (the host pays about
/// 3.6 us per allocated output tensor), and host helpers that read one value out of them.
struct NetworkResult {
    /// `[M, N_last]`, UINT8: the sign-magnitude int8 bytes of the last layer (bit 7 = sign, bits 0..6 = magnitude).
    ttnn::Tensor out;
    /// FLOAT32 `[32 * S, 32]`, S = 1 + 2L tiles, tile i holding the scaler `2^k` in every element. Tile order: `scaler_x`, then
    /// `scaler_w` of layer 0 .. L-1, then `scaler_out` of layer 0 .. L-1.
    ttnn::Tensor scalers;
    /// UINT32 `[M, N_0 + ... + N_{L-1}]`: the raw *sign-magnitude* bits of the int32 accumulator of every layer, `q_x_l @ q_w_l`, before the
    /// activation (bit 31 = sign, bits 0..30 = magnitude), side by side: layer l is the columns `[N_0 + .. + N_{l-1}, N_0 + .. + N_l)`.
    ttnn::Tensor acc;
    /// `N_l`, the output features of every layer.
    std::vector<uint32_t> widths;

    uint32_t layers() const { return static_cast<uint32_t>(widths.size()); }

    /// `2^kx`, the quantization scaler of the input of layer 0 (reads the device).
    float scaler_x() const;
    /// `2^kw_l`, the quantization scaler of the weights of layer l.
    float scaler_w(uint32_t l) const;
    /// `2^k_l` of the requantization of layer l (usually negative k). The requantized output of layer l is the quantized input of layer l + 1.
    float scaler_out(uint32_t l) const;
    /// The accumulator of layer l, `[M, N_l]` row-major, raw sign-magnitude bits (as in `acc`).
    std::vector<uint32_t> acc_raw(uint32_t l) const;
    /// The accumulator of layer l as signed integers.
    std::vector<int32_t> acc_values(uint32_t l) const;
};

/**
 * @brief A network of quantized layers on one core, in one program: `y = layer_{L-1}(... layer_1(layer_0(x)))`.
 *
 * Layer `l` is what `execute_layer` computes: `out_l = requantize(activation(q_x_l @ q_w_l))`. Only the input `x` of
 * layer 0 and the weights are quantized (per tensor, to 8 bits, with a power-of-two scaler, `Q = 127`, round half to
 * even). The requantized output of layer l is the quantized input of layer l + 1: it stays in L1 as int8 between the
 * layers and is never dequantized. All layers run in one compute kernel; the weights are streamed one layer at a time,
 * so the next layer's weights are read while the previous layer computes. The quantized activations ping-pong between two L1 buffers, so the L1 use does not grow with L.
 *
 * @param x `[M, K_0]`, FLOAT32 or BFLOAT16, TILE layout, interleaved, on device; `M`, `K_0` multiples of 32.
 * @param weights L tensors, `weights[l]` is `[K_l, N_l]` with `K_{l+1} == N_l`, same dtype as `x`, TILE layout,
 * interleaved, on device; `N_l` a multiple of 32; `K_l <= 66560`. L is at most 6 (the kernel text of the layers must
 * fit the kernel config buffer).
 * @param activations one per layer, applied to the integer accumulator before the requantization.
 * @param output_memory_config Memory config of all outputs. Defaults to the one of `x`.
 *
 * Limits of this version: one core; `x`, one layer of weights, one accumulator, the int8 activations of all layers and
 * the output must fit in L1 (the operation throws otherwise).
 */
NetworkResult execute_network(
    const ttnn::Tensor& x,
    const std::vector<ttnn::Tensor>& weights,
    const std::vector<Activation>& activations,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);

/// As above with one activation for the network: applied to every layer except the last, which has none (the convention
/// of ttml's `MultiLayerPerceptron`).
NetworkResult execute_network(
    const ttnn::Tensor& x,
    const std::vector<ttnn::Tensor>& weights,
    Activation activation = Activation::ReLU,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);

}  // namespace zkml::prim
