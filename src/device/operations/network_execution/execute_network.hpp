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

/// The outputs of `execute_network` for a network of L layers.
struct NetworkResult {
    /// `[M, N_last]`, UINT8: the sign-magnitude int8 bytes of the last layer (bit 7 = sign, bits 0..6 = magnitude).
    ttnn::Tensor out;
    /// FLOAT32 `[1, 1]`: `2^kx`, the quantization scaler of the input of layer 0.
    ttnn::Tensor scaler_x;
    /// L tensors `[M, N_l]`, UINT32: the raw *sign-magnitude* bits of the int32 accumulator of layer l, `q_x_l @
    /// q_w_l`, before the activation (bit 31 = sign, bits 0..30 = magnitude).
    std::vector<ttnn::Tensor> acc;
    /// L tensors FLOAT32 `[1, 1]`: `2^kw_l`, the quantization scaler of the weights of layer l.
    std::vector<ttnn::Tensor> scaler_w;
    /// L tensors FLOAT32 `[1, 1]`: `2^k_l` of the requantization of layer l (usually negative k). The requantized
    /// output of layer l is the quantized input of layer l + 1.
    std::vector<ttnn::Tensor> scaler_out;
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
