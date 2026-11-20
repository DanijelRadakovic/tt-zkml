// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Quantized tensor -> M31 residues. See pot_quant_m31.hpp.
//
// One program on one core streams the tiles: reader (DRAM -> c_0), compute (c_0 -> SFPU -> c_16), writer.
// The SFPU adds p to every negative element (see kernels/compute/m31_sfpu.hpp).

#pragma once

#include <cstdint>
#include <variant>

#include <tt-metalium/host_api.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace zkml::operations::pot_quantization {

struct PotQuantM31Operation {
    struct operation_attributes_t {
        tt::tt_metal::MemoryConfig output_memory_config;
    };

    struct tensor_args_t {
        const ttnn::Tensor& input;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;

    struct SingleCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_id;
            tt::tt_metal::KernelHandle writer_id;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<SingleCore>;

    /// Tiles per compute block. Both rings hold two blocks, and every block starts at a multiple of this, so a block
    /// never wraps around a ring.
    static constexpr uint32_t block_tiles = 4;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace zkml::operations::pot_quantization
