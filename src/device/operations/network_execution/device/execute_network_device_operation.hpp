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
        /// The weights are already quantized (UINT8 tensors of sign-magnitude bytes plus their scalers): the quantization of the weights is skipped.
        bool pre_quantized = false;
    };

    struct tensor_args_t {
        const ttnn::Tensor& x;
        const std::vector<ttnn::Tensor>& weights;
        /// With `pre_quantized`: the FLOAT32 [1, 1] scaler of every weight tensor; empty otherwise.
        const std::vector<ttnn::Tensor>& weight_scalers;
    };

    // Three outputs, however many layers: allocating an output tensor costs the host about 3.6 us, so the intermediate results are packed.
    //   [0] out      UINT8   [M, N_last]      the quantized output of the last layer
    //   [1] scalers  FLOAT32 [32 * S, 32]     S tiles, tile i holds the scaler 2^k in every element: scaler_x, then (not pre-quantized)
    //                                         scaler_w of every layer, then scaler_out of every layer
    //   [2] acc      UINT32  [M, sum N_l]     the raw sign-magnitude accumulators of the layers side by side: layer l is the columns
    //                                         [sum_{j<l} N_j, sum_{j<=l} N_j)
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<ttnn::Tensor>;

    static constexpr uint32_t out_index() { return 0; }
    static constexpr uint32_t scalers_index() { return 1; }
    static constexpr uint32_t acc_index() { return 2; }
    /// Tiles of the scalers tensor.
    static constexpr uint32_t scaler_x_tile() { return 0; }
    static constexpr uint32_t scaler_w_tile(uint32_t l) { return 1 + l; }  // not with pre-quantized weights
    static constexpr uint32_t scaler_out_tile(uint32_t layers, uint32_t l, bool pre_quantized) {
        return (pre_quantized ? 1 : 1 + layers) + l;
    }
    static constexpr uint32_t scaler_tiles(uint32_t layers, bool pre_quantized) { return (pre_quantized ? 1 : 1 + layers) + layers; }

    struct SingleCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_id;
            tt::tt_metal::KernelHandle writer_id;
            uint32_t layers;
            bool pre_quantized;
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
