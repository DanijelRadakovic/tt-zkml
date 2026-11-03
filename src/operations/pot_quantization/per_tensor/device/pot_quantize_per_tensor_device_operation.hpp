// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Per-tensor power-of-two (PoT) symmetric quantization:
//   q = clamp(round_half_even(x * 2^k), int8 range)
//
// The host chooses k (the number of fractional bits, scale = 2^-k) and passes it in; the device
// only applies it. The output is Int32 (values in int8 range) or Int8.
//
// Implementation detail of the op: callers use ../pot_quantize_per_tensor.hpp.

#pragma once

#include <cstdint>
#include <variant>

#include <tt-metalium/host_api.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace zkml::operations::pot_quantization {

struct PotQuantizePerTensorOperation {
    struct operation_attributes_t {
        int32_t k;  // fractional bits: scale = 2^-k, the kernel multiplies by 2^k
        tt::tt_metal::DataType output_dtype;
        tt::tt_metal::MemoryConfig output_memory_config;
    };

    struct tensor_args_t {
        const ttnn::Tensor& input;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;

    // Single core, loops over every tile of the input.
    struct SingleCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_id;
            tt::tt_metal::KernelHandle writer_id;
            tt::tt_metal::KernelHandle compute_id;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        // Cache miss: build the program (CBs, kernels, runtime args).
        static cached_program_t create(
            const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output);

        // Cache hit: the program is reused, only the buffer addresses (and scale) are patched.
        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<SingleCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Leaves out k: it only reaches the device as a runtime arg, which
    // override_runtime_arguments patches, so one cached program serves every k.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace zkml::operations::pot_quantization
