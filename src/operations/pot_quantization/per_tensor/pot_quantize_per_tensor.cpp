// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include "pot_quantize_per_tensor.hpp"

#include "device/pot_quantize_per_tensor_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace zkml::prim {

ttnn::Tensor pot_quantize_per_tensor(
    const ttnn::Tensor& input,
    int32_t k,
    tt::tt_metal::DataType output_dtype,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config) {
    using Op = zkml::operations::pot_quantization::PotQuantizePerTensorOperation;
    // k is not part of the program cache key (see compute_program_hash): calls that differ only in
    // k reuse one cached program, and override_runtime_arguments sets the new scale.
    const auto attributes = Op::operation_attributes_t{
        .k = k,
        .output_dtype = output_dtype,
        .output_memory_config = output_memory_config.value_or(input.memory_config()),
    };
    return ttnn::device_operation::launch<Op>(attributes, Op::tensor_args_t{input});
}

}  // namespace zkml::prim
