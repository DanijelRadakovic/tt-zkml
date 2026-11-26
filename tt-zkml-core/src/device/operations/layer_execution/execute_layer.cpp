// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include "execute_layer.hpp"

#include "device/execute_layer_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace zkml::prim {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> execute_layer(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    Activation activation,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config) {
    using Op = zkml::operations::layer_execution::ExecuteLayerOperation;
    const auto attributes = Op::operation_attributes_t{
        .activation = activation,
        .output_memory_config = output_memory_config.value_or(x.memory_config()),
    };
    return ttnn::device_operation::launch<Op>(attributes, Op::tensor_args_t{x, w});
}

}  // namespace zkml::prim
