// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// One quantized layer on one core (see execute_layer.hpp): quantize x and W, FPU int8 matmul, activation, requantize.

#pragma once

#include <cstdint>
#include <tuple>
#include <variant>

#include <tt-metalium/host_api.hpp>

#include "operations/layer_execution/execute_layer.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace zkml::operations::layer_execution {

struct ExecuteLayerOperation {
    struct operation_attributes_t {
        zkml::prim::Activation activation;
        tt::tt_metal::MemoryConfig output_memory_config;
    };

    struct tensor_args_t {
        const ttnn::Tensor& x;
        const ttnn::Tensor& w;
    };

    // (out, scaler_x, scaler_w, scaler_out)
    using spec_return_value_t =
        std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor>;

    struct SingleCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_id;
            tt::tt_metal::KernelHandle writer_id;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& outputs);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& outputs);
    };

    using program_factory_t = std::variant<SingleCore>;

    /// A fp32 DST (fp32_dest_acc_en) with half sync holds 4 tiles.
    static constexpr uint32_t dst_tiles = 4;
    /// Largest K in tiles: |acc| <= K * 127 * 127 stays below 2^24, so the int32 -> fp32 conversion is exact.
    static constexpr uint32_t max_k_tiles = 32;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace zkml::operations::layer_execution
