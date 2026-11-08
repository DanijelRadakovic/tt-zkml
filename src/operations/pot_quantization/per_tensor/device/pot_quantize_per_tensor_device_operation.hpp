// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Per-tensor power-of-two (PoT) symmetric quantization. The device computes the scaler.
//
// The formulas are:
//  - `Q = 2^(bit_width.bits() - 1) - 1`,
//  - `k = max{k : absmax(x) * 2^k <= Q}`,
//  - `q = clamp(round_half_even(x * 2^k), -Q, Q)`.
//
// One program does everything: it finds absmax(x), derives the scaler 2^k from it and quantizes.
// The outputs are the quantized tensor and the scaler 2^k, so it can Dequantize with `x ~= q / 2^k`.

#pragma once

#include <cstdint>
#include <tuple>
#include <variant>

#include <tt-metalium/host_api.hpp>

#include "operations/pot_quantization/bit_width.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace zkml::operations::pot_quantization {

struct PotQuantizePerTensorOperation {
    struct operation_attributes_t {
        // Width of the quantized integers.
        zkml::prim::BitWidth bit_width;
        tt::tt_metal::DataType output_dtype;
        tt::tt_metal::MemoryConfig output_memory_config;
        // Number of tiles the reader reads in parallel (async_read) before it waits on one
        // async_read_barrier. At least 1. Capped at the tile count of the input.
        // The last batch of the read stream is shorter when the batch size does not divide the
        // stream (2 * tile count). That is fine.
        uint32_t noc_read_tile_batch_size;
        // The input CB (c_0) holds noc_read_tile_batch_size * cb_in_size_factor tiles. At least 2.
        // - With 2, the reader fills one batch while compute drains the other.
        // - A larger factor lets the reader run further ahead when a read is slow.
        // - If noc_read_tile_batch_size * cb_in_size_factor tiles do not fit, the operation throws.
        uint32_t cb_in_size_factor;
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

    // A fp32 DST (fp32_dest_acc_en) with half sync holds 4 tiles.
    // The kernel static_asserts this value against the hardware limit.
    static constexpr uint32_t dst_tiles = 4;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace zkml::operations::pot_quantization
