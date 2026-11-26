// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include "execute_layer_device_operation.hpp"

#include <tt-metalium/constants.hpp>

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using namespace tt::constants;

namespace zkml::operations::layer_execution {

ExecuteLayerOperation::program_factory_t ExecuteLayerOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return SingleCore{};
}

void ExecuteLayerOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.x;
    const auto& w = tensor_args.w;
    for (const auto* t : {&x, &w}) {
        TT_FATAL(t->storage_type() == ttnn::StorageType::DEVICE, "execute_layer: inputs must be on device");
        TT_FATAL(t->buffer() != nullptr, "execute_layer: inputs must be allocated on device");
        TT_FATAL(t->layout() == Layout::TILE, "execute_layer: inputs must be in TILE layout");
        TT_FATAL(
            t->dtype() == DataType::FLOAT32 || t->dtype() == DataType::BFLOAT16,
            "execute_layer: inputs must be FLOAT32 or BFLOAT16, got {}",
            t->dtype());
        TT_FATAL(!t->memory_config().is_sharded(), "execute_layer: sharded inputs are not supported");
        TT_FATAL(t->logical_shape().rank() == 2, "execute_layer: inputs must be 2-D");
    }
    TT_FATAL(
        x.dtype() == w.dtype(), "execute_layer: x and w must have the same dtype, got {} and {}", x.dtype(), w.dtype());
    TT_FATAL(!attributes.output_memory_config.is_sharded(), "execute_layer: sharded output is not supported");
    const auto m = x.logical_shape()[0], k = x.logical_shape()[1];
    const auto k2 = w.logical_shape()[0], n = w.logical_shape()[1];
    TT_FATAL(k == k2, "execute_layer: x is [{}, {}] but w is [{}, {}]: the inner dimensions differ", m, k, k2, n);
    TT_FATAL(
        m % TILE_HEIGHT == 0 && k % TILE_WIDTH == 0 && n % TILE_WIDTH == 0,
        "execute_layer: M, K and N must be multiples of 32, got M={} K={} N={}",
        m,
        k,
        n);
    TT_FATAL(
        k / TILE_WIDTH <= max_k_tiles,
        "execute_layer: K must be at most {} (K * 127 * 127 must stay below 2^30), got {}",
        max_k_tiles * TILE_WIDTH,
        k);
    TT_FATAL(
        attributes.activation == zkml::prim::Activation::None || attributes.activation == zkml::prim::Activation::ReLU,
        "execute_layer: unknown activation");
}

ExecuteLayerOperation::spec_return_value_t ExecuteLayerOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto m = tensor_args.x.logical_shape()[0];
    const auto n = tensor_args.w.logical_shape()[1];
    const tt::tt_metal::PageConfig tile_page(Layout::TILE);
    auto scaler = [&] {
        // One tile holding 2^k in every element; the logical shape exposes one of them.
        return tt::tt_metal::TensorSpec(
            ttnn::Shape({1, 1}),
            tt::tt_metal::TensorLayout(DataType::FLOAT32, tile_page, attributes.output_memory_config));
    };
    return {
        tt::tt_metal::TensorSpec(
            ttnn::Shape({m, n}),
            tt::tt_metal::TensorLayout(DataType::UINT8, tile_page, attributes.output_memory_config)),
        tt::tt_metal::TensorSpec(
            ttnn::Shape({m, n}),
            tt::tt_metal::TensorLayout(DataType::UINT32, tile_page, attributes.output_memory_config)),
        scaler(),
        scaler(),
        scaler()};
}

ExecuteLayerOperation::tensor_return_value_t ExecuteLayerOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto [out, acc, sx, sw, so] = compute_output_specs(attributes, tensor_args);
    auto* device = tensor_args.x.device();
    return {
        ttnn::create_device_tensor(out, device),
        ttnn::create_device_tensor(acc, device),
        ttnn::create_device_tensor(sx, device),
        ttnn::create_device_tensor(sw, device),
        ttnn::create_device_tensor(so, device)};
}

}  // namespace zkml::operations::layer_execution
