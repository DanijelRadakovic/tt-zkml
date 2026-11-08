// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include "pot_quantize_per_tensor.hpp"

#include "device/pot_quantize_per_tensor_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace zkml::prim {

std::tuple<ttnn::Tensor, ttnn::Tensor> pot_quantize_per_tensor(
    const ttnn::Tensor& input,
    BitWidth bit_width,
    tt::tt_metal::DataType output_dtype,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config,
    uint32_t noc_read_tile_batch_size,
    uint32_t cb_in_size_factor) {
    using Op = zkml::operations::pot_quantization::PotQuantizePerTensorOperation;
    const auto attributes = Op::operation_attributes_t{
        .bit_width = bit_width,
        .output_dtype = output_dtype,
        .output_memory_config = output_memory_config.value_or(input.memory_config()),
        .noc_read_tile_batch_size = noc_read_tile_batch_size,
        .cb_in_size_factor = cb_in_size_factor,
    };
    return ttnn::device_operation::launch<Op>(attributes, Op::tensor_args_t{input});
}

}  // namespace zkml::prim
