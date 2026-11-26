// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include "execute_network.hpp"

#include <algorithm>
#include <numeric>

#include "device/execute_network_device_operation.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"

namespace zkml::prim {

namespace {
const std::vector<ttnn::Tensor> kNoScalers;
}

namespace {
// Tile i of the scalers tensor holds the scaler in every element: the first element of the tile, row-major [32 * S, 32] -> index i * 32 * 32.
float scaler_tile(const ttnn::Tensor& scalers, uint32_t tile) {
    return scalers.to_vector<float>().at(static_cast<size_t>(tile) * 32 * 32);
}
}  // namespace

float NetworkResult::scaler_x() const { return scaler_tile(scalers, 0); }
float NetworkResult::scaler_w(uint32_t l) const {
    if (pre_quantized()) {
        return given_scaler_w.at(l).to_vector<float>().at(0);
    }
    return scaler_tile(scalers, 1 + l);
}
float NetworkResult::scaler_out(uint32_t l) const {
    return scaler_tile(scalers, (pre_quantized() ? 1 : 1 + layers()) + l);
}
std::vector<uint32_t> NetworkResult::acc_raw(uint32_t l) const {
    const auto all = acc.to_vector<uint32_t>();
    const size_t total = std::accumulate(widths.begin(), widths.end(), size_t{0});
    const size_t off = std::accumulate(widths.begin(), widths.begin() + l, size_t{0});
    const size_t m = all.size() / total;
    std::vector<uint32_t> v(m * widths.at(l));
    for (size_t i = 0; i < m; i++) {
        std::copy_n(&all[i * total + off], widths[l], &v[i * widths[l]]);
    }
    return v;
}
std::vector<int32_t> NetworkResult::acc_values(uint32_t l) const {
    const auto raw = acc_raw(l);
    std::vector<int32_t> v(raw.size());
    for (size_t i = 0; i < raw.size(); i++) {
        v[i] = (raw[i] >> 31) ? -static_cast<int32_t>(raw[i] & 0x7FFFFFFFu) : static_cast<int32_t>(raw[i]);
    }
    return v;
}

NetworkResult execute_network(
    const ttnn::Tensor& x,
    const std::vector<ttnn::Tensor>& weights,
    const std::vector<Activation>& activations,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config) {
    using Op = zkml::operations::network_execution::ExecuteNetworkOperation;
    // Checked here: the output specs are computed before the device operation validates its arguments.
    TT_FATAL(!weights.empty(), "execute_network: at least one layer is needed");
    TT_FATAL(
        activations.size() == weights.size(),
        "execute_network: {} activations for {} layers",
        activations.size(),
        weights.size());
    const auto attributes = Op::operation_attributes_t{
        .activations = activations,
        .output_memory_config = output_memory_config.value_or(x.memory_config()),
    };
    auto outputs = ttnn::device_operation::launch<Op>(attributes, Op::tensor_args_t{x, weights, kNoScalers});

    NetworkResult result;
    result.out = outputs[Op::out_index()];
    result.scalers = outputs[Op::scalers_index()];
    result.acc = outputs[Op::acc_index()];
    for (const auto& w : weights) {
        result.widths.push_back(w.logical_shape()[1]);
    }
    return result;
}

NetworkResult execute_network(
    const ttnn::Tensor& x,
    const std::vector<ttnn::Tensor>& weights,
    Activation activation,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config) {
    std::vector<Activation> activations(weights.size(), activation);
    if (!activations.empty()) {
        activations.back() = Activation::None;
    }
    return execute_network(x, weights, activations, output_memory_config);
}

NetworkResult execute_network_quantized(
    const ttnn::Tensor& x,
    const std::vector<ttnn::Tensor>& q_weights,
    const std::vector<ttnn::Tensor>& weight_scalers,
    const std::vector<Activation>& activations,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config) {
    using Op = zkml::operations::network_execution::ExecuteNetworkOperation;
    TT_FATAL(!q_weights.empty(), "execute_network_quantized: at least one layer is needed");
    TT_FATAL(
        activations.size() == q_weights.size() && weight_scalers.size() == q_weights.size(),
        "execute_network_quantized: {} weights, {} scalers, {} activations",
        q_weights.size(),
        weight_scalers.size(),
        activations.size());
    const auto attributes = Op::operation_attributes_t{
        .activations = activations,
        .output_memory_config = output_memory_config.value_or(x.memory_config()),
        .pre_quantized = true,
    };
    auto outputs = ttnn::device_operation::launch<Op>(attributes, Op::tensor_args_t{x, q_weights, weight_scalers});

    NetworkResult result;
    result.out = outputs[Op::out_index()];
    result.scalers = outputs[Op::scalers_index()];
    result.acc = outputs[Op::acc_index()];
    for (const auto& w : q_weights) {
        result.widths.push_back(w.logical_shape()[1]);
    }
    result.given_scaler_w = weight_scalers;
    return result;
}

NetworkResult execute_network_quantized(
    const ttnn::Tensor& x,
    const std::vector<ttnn::Tensor>& q_weights,
    const std::vector<ttnn::Tensor>& weight_scalers,
    Activation activation,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config) {
    std::vector<Activation> activations(q_weights.size(), activation);
    if (!activations.empty()) {
        activations.back() = Activation::None;
    }
    return execute_network_quantized(x, q_weights, weight_scalers, activations, output_memory_config);
}

}  // namespace zkml::prim
