// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include "pot_quantize_per_tensor_device_operation.hpp"

#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

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
    // An enum class can still hold any number through a cast.
    const uint32_t width = attributes.bit_width.bits();
    TT_FATAL(
        width == 8 || width == 16 || width == 32, "pot_quantize: bit_width must be 8, 16 or 32 bits, got {}", width);
    TT_FATAL(
        attributes.output_dtype == DataType::INT32 || attributes.output_dtype == DataType::INT8,
        "pot_quantize: output dtype must be INT32 or INT8, got {}",
        attributes.output_dtype);
    TT_FATAL(
        attributes.output_dtype != DataType::INT8 || attributes.bit_width == zkml::prim::BitWidth::Bits8,
        "pot_quantize: INT8 output requires bit_width 8, got {}",
        width);
    TT_FATAL(
        attributes.noc_read_tile_batch_size >= 1,
        "pot_quantize: noc_read_tile_batch_size must be at least 1, got {}",
        attributes.noc_read_tile_batch_size);
    TT_FATAL(
        attributes.cb_in_size_factor >= 2,
        "pot_quantize: cb_in_size_factor must be at least 2 (one batch being read, one being consumed), got {}",
        attributes.cb_in_size_factor);
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

}  // namespace zkml::operations::pot_quantization
