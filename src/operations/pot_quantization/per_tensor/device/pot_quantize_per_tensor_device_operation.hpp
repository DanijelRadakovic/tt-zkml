// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Per-tensor power-of-two (PoT) symmetric quantization with a device-computed scaler:
//   Q = 2^(bit_width - 1) - 1
//   k = max{k : absmax(x) * 2^k <= Q}   (capped at 127)
//   q = clamp(round_half_even(x * 2^k), -Q, Q)
//
// The device finds absmax(x), derives the scaler 2^k from it and quantizes, all in one program.
// The outputs are the quantized tensor and the scaler 2^k (so x ~= q / 2^k).
//
// Implementation detail of the op: callers use ../pot_quantize_per_tensor.hpp.

#pragma once

#include <cstdint>
#include <tuple>
#include <variant>

#include <tt-metalium/host_api.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace zkml::operations::pot_quantization {

struct PotQuantizePerTensorOperation {
    struct operation_attributes_t {
        uint32_t bit_width;  // 8, 16 or 32: Q = 2^(bit_width - 1) - 1
        tt::tt_metal::DataType output_dtype;
        tt::tt_metal::MemoryConfig output_memory_config;
    };

    struct tensor_args_t {
        const ttnn::Tensor& input;
    };

    // (quantized, scaler). The scaler is a FLOAT32 [1, 1] tensor holding 2^k.
    using spec_return_value_t = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::tuple<ttnn::Tensor, ttnn::Tensor>;

    // Single core, two passes over every tile of the input.
    struct SingleCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_id;
            tt::tt_metal::KernelHandle writer_id;
            tt::tt_metal::KernelHandle compute_id;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        // Cache miss: build the program (CBs, kernels, runtime args).
        static cached_program_t create(
            const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& outputs);

        // Cache hit: the program is reused, only the buffer addresses are patched.
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

    // bit_width and output_dtype are compile-time args of the compute kernel, so both are in the key.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace zkml::operations::pot_quantization
