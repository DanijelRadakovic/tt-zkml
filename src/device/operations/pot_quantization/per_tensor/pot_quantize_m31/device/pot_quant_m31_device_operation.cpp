// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include "pot_quant_m31_device_operation.hpp"

#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace zkml::operations::pot_quantization {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;

PotQuantM31Operation::program_factory_t PotQuantM31Operation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return SingleCore{};
}

void PotQuantM31Operation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    TT_FATAL(input.storage_type() == ttnn::StorageType::DEVICE, "pot_quant_m31: input must be on device");
    TT_FATAL(input.buffer() != nullptr, "pot_quant_m31: input must be allocated on device");
    TT_FATAL(input.layout() == Layout::TILE, "pot_quant_m31: input must be in TILE layout");
    TT_FATAL(
        input.dtype() == DataType::INT32 || input.dtype() == DataType::INT8,
        "pot_quant_m31: input dtype must be INT32 or INT8, got {}",
        input.dtype());
    TT_FATAL(!input.memory_config().is_sharded(), "pot_quant_m31: sharded input is not supported");
    TT_FATAL(!attributes.output_memory_config.is_sharded(), "pot_quant_m31: sharded output is not supported");
}

PotQuantM31Operation::spec_return_value_t PotQuantM31Operation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return tt::tt_metal::TensorSpec(
        tensor_args.input.logical_shape(),
        tt::tt_metal::TensorLayout(
            DataType::UINT32, tt::tt_metal::PageConfig(Layout::TILE), attributes.output_memory_config));
}

PotQuantM31Operation::tensor_return_value_t PotQuantM31Operation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.input.device());
}

}  // namespace zkml::operations::pot_quantization
