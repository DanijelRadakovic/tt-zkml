// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// A network of quantized layers on one core (see execute_network.hpp): one program, one compute kernel that loops over
// the layers, one reader, one writer.

#pragma once

#include <cstdint>
#include <variant>
#include <vector>

#include <tt-metalium/host_api.hpp>

#include "device/operations/layer_execution/execute_layer.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace zkml::operations::network_execution {

struct ExecuteNetworkOperation {
    struct operation_attributes_t {
        std::vector<zkml::prim::Activation> activations;
        tt::tt_metal::MemoryConfig output_memory_config;
    };

    struct tensor_args_t {
        const ttnn::Tensor& x;
        const std::vector<ttnn::Tensor>& weights;
    };

    // Flat, for L layers: [out, scaler_x, acc_0 .. acc_{L-1}, scaler_w_0 .. scaler_w_{L-1}, scaler_out_0 ..
    // scaler_out_{L-1}]
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<ttnn::Tensor>;

    /// Positions in the flat output vector for `layers` layers.
    static constexpr uint32_t out_index() { return 0; }
    static constexpr uint32_t scaler_x_index() { return 1; }
    static constexpr uint32_t acc_index(uint32_t l) { return 2 + l; }
    static uint32_t scaler_w_index(uint32_t layers, uint32_t l) { return 2 + layers + l; }
    static uint32_t scaler_out_index(uint32_t layers, uint32_t l) { return 2 + 2 * layers + l; }

    struct SingleCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_id;
            tt::tt_metal::KernelHandle writer_id;
            uint32_t layers;
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

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace zkml::operations::network_execution
