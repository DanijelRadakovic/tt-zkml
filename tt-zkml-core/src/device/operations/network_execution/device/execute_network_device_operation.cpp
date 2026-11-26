// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include "execute_network_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "network_plan.hpp"

using tt::tt_metal::DataType;

namespace zkml::operations::network_execution {

ExecuteNetworkOperation::program_factory_t ExecuteNetworkOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return SingleCore{};
}

void ExecuteNetworkOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(!attributes.output_memory_config.is_sharded(), "execute_network: sharded output is not supported");
    // Checks the tensors, the dimensions of the chain, the activations and the number of layers.
    (void)NetworkPlan::make(tensor_args.x, tensor_args.weights, attributes.activations);
}

ExecuteNetworkOperation::spec_return_value_t ExecuteNetworkOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& weights = tensor_args.weights;
    const uint32_t layers = static_cast<uint32_t>(weights.size());
    const uint32_t m = tensor_args.x.logical_shape()[0];
    const tt::tt_metal::PageConfig tile_page(ttnn::Layout::TILE);
    auto spec = [&](ttnn::Shape shape, DataType type) {
        return tt::tt_metal::TensorSpec(
            std::move(shape), tt::tt_metal::TensorLayout(type, tile_page, attributes.output_memory_config));
    };
    uint32_t total_n = 0;
    for (const auto& w : weights) {
        total_n += w.logical_shape()[1];
    }
    spec_return_value_t specs;
    specs.reserve(3);
    specs.push_back(spec(ttnn::Shape({m, weights.back().logical_shape()[1]}), DataType::UINT8));  // out
    specs.push_back(spec(
        ttnn::Shape({tt::constants::TILE_HEIGHT * scaler_tiles(layers), tt::constants::TILE_WIDTH}),
        DataType::FLOAT32));                                                                       // scalers
    specs.push_back(spec(ttnn::Shape({m, total_n}), DataType::UINT32));                           // acc
    return specs;
}

ExecuteNetworkOperation::tensor_return_value_t ExecuteNetworkOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto* device = tensor_args.x.device();
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(attributes, tensor_args)) {
        outputs.push_back(ttnn::create_device_tensor(spec, device));
    }
    return outputs;
}

}  // namespace zkml::operations::network_execution
