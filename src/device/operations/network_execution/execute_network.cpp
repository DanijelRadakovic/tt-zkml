// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include "execute_network.hpp"

#include "device/execute_network_device_operation.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"

namespace zkml::prim {

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
    auto outputs = ttnn::device_operation::launch<Op>(attributes, Op::tensor_args_t{x, weights});

    const uint32_t layers = static_cast<uint32_t>(weights.size());
    NetworkResult result;
    result.out = outputs[Op::out_index()];
    result.scaler_x = outputs[Op::scaler_x_index()];
    for (uint32_t l = 0; l < layers; l++) {
        result.acc.push_back(outputs[Op::acc_index(l)]);
        result.scaler_w.push_back(outputs[Op::scaler_w_index(layers, l)]);
        result.scaler_out.push_back(outputs[Op::scaler_out_index(layers, l)]);
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

}  // namespace zkml::prim
