// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include "pot_quant_m31.hpp"

#include "device/pot_quant_m31_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace zkml::prim {

ttnn::Tensor pot_quant_m31(
    const ttnn::Tensor& quantized, const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config) {
    using Op = zkml::operations::pot_quantization::PotQuantM31Operation;
    const auto attributes = Op::operation_attributes_t{
        .output_memory_config = output_memory_config.value_or(quantized.memory_config()),
    };
    return ttnn::device_operation::launch<Op>(attributes, Op::tensor_args_t{quantized});
}

}  // namespace zkml::prim
