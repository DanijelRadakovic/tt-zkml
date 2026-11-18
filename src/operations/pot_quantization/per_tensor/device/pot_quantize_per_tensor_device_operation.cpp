// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include "pot_quantize_per_tensor_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt_stl/reflection.hpp>

namespace zkml::operations::pot_quantization {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;

PotQuantizePerTensorOperation::program_factory_t PotQuantizePerTensorOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return SingleCore{};
}

void PotQuantizePerTensorOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    TT_FATAL(input.storage_type() == ttnn::StorageType::DEVICE, "pot_quantize: input must be on device");
    TT_FATAL(input.buffer() != nullptr, "pot_quantize: input must be allocated on device");
    TT_FATAL(input.layout() == Layout::TILE, "pot_quantize: input must be in TILE layout");
    TT_FATAL(
        input.dtype() == DataType::FLOAT32 || input.dtype() == DataType::BFLOAT16,
        "pot_quantize: input dtype must be FLOAT32 or BFLOAT16, got {}",
        input.dtype());
    TT_FATAL(!input.memory_config().is_sharded(), "pot_quantize: sharded input is not supported");
    TT_FATAL(
        !attributes.output_memory_config.is_sharded(), "pot_quantize: sharded output is not supported");
    TT_FATAL(
        attributes.bit_width == 8 || attributes.bit_width == 16 || attributes.bit_width == 32,
        "pot_quantize: bit_width must be 8, 16 or 32, got {}",
        attributes.bit_width);
    TT_FATAL(
        attributes.output_dtype == DataType::INT32 || attributes.output_dtype == DataType::INT8,
        "pot_quantize: output dtype must be INT32 or INT8, got {}",
        attributes.output_dtype);
    TT_FATAL(
        attributes.output_dtype != DataType::INT8 || attributes.bit_width == 8,
        "pot_quantize: INT8 output requires bit_width 8, got {}",
        attributes.bit_width);
}

PotQuantizePerTensorOperation::spec_return_value_t PotQuantizePerTensorOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const tt::tt_metal::PageConfig tile_page(Layout::TILE);
    return {
        tt::tt_metal::TensorSpec(
            input.logical_shape(),
            tt::tt_metal::TensorLayout(attributes.output_dtype, tile_page, attributes.output_memory_config)),
        // One tile holding 2^k in every element; the logical shape exposes one of them.
        tt::tt_metal::TensorSpec(
            ttnn::Shape({1, 1}),
            tt::tt_metal::TensorLayout(DataType::FLOAT32, tile_page, attributes.output_memory_config)),
    };
}

PotQuantizePerTensorOperation::tensor_return_value_t PotQuantizePerTensorOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto [quantized_spec, scaler_spec] = compute_output_specs(attributes, tensor_args);
    auto* device = tensor_args.input.device();
    return {ttnn::create_device_tensor(quantized_spec, device), ttnn::create_device_tensor(scaler_spec, device)};
}

ttsl::hash::hash_t PotQuantizePerTensorOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return ttsl::hash::hash_objects_with_default_seed(
        tensor_args.input.tensor_spec(),
        attributes.bit_width,
        attributes.output_dtype,
        attributes.output_memory_config);
}

}  // namespace zkml::operations::pot_quantization
