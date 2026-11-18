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
        attributes.output_dtype == DataType::INT32 || attributes.output_dtype == DataType::INT8,
        "pot_quantize: output dtype must be INT32 or INT8, got {}",
        attributes.output_dtype);
    // 2^k must be a normal fp32 so it reaches the device unchanged. 
    // Tensix flushes denormals (k <= -127) to 0, which would quantize every element to 0, and 2^128 and up overflow to Inf.
    // See how Tensix handles special values because it's not fully IEEE-754 compliant:
    // https://github.com/tenstorrent/tt-metal/blob/v0.81.0/tech_reports/Handling_Special_Value/special_values.md
    TT_FATAL(
        attributes.k >= -126 && attributes.k <= 127, "pot_quantize: k must be in [-126, 127], got {}", attributes.k);
}

PotQuantizePerTensorOperation::spec_return_value_t PotQuantizePerTensorOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return tt::tt_metal::TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(
            attributes.output_dtype, tt::tt_metal::PageConfig(Layout::TILE), attributes.output_memory_config));
}

PotQuantizePerTensorOperation::tensor_return_value_t PotQuantizePerTensorOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.input.device());
}

ttsl::hash::hash_t PotQuantizePerTensorOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // Seed only: input spec, output dtype, output memory config.
    // Remove k from the seed, override_runtime_arguments rewrites it as runtime arg.
    return ttsl::hash::hash_objects_with_default_seed(
        tensor_args.input.tensor_spec(), attributes.output_dtype, attributes.output_memory_config);
}

}  // namespace zkml::operations::pot_quantization
