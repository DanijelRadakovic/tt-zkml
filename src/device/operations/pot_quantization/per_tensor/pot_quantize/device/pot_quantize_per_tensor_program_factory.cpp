// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "common/kernel_path.hpp"
#include "common/l1_check.hpp"
#include "pot_quantize_per_tensor_device_operation.hpp"
#include "ttnn/operations/cb_utils.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;
using zkml::common::kernel_include_dir;
using zkml::common::kernel_path;

namespace zkml::operations::pot_quantization {

PotQuantizePerTensorOperation::SingleCore::cached_program_t PotQuantizePerTensorOperation::SingleCore::create(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& outputs) {
    const auto& input = tensor_args.input;
    auto& [quantized, scaler] = outputs;

    const uint32_t num_tiles = input.physical_volume() / TILE_HW;
    const bool is_output_int8 = attributes.output_dtype == DataType::INT8 || attributes.output_dtype == DataType::UINT8;
    // 0: INT32, 1: INT8 (two's complement bytes), 2: UINT8 (sign-magnitude bytes: the format the FPU reads as Int8, see the kernel)
    const uint32_t output_kind = attributes.output_dtype == DataType::INT32 ? 0u : (attributes.output_dtype == DataType::INT8 ? 1u : 2u);

    Program program{};
    const CoreCoord core{0, 0};

    // Int8 output goes through the UInt8 packer path; cb_dataformat_for picks that.
    const DataFormat in_format = cb_dataformat_for(input.dtype());
    const DataFormat scale_format = DataFormat::Float32;
    const DataFormat out_format = cb_dataformat_for(attributes.output_dtype);

    // The size of the input circular buffer (c_0) is calculated as: factor * batch.
    // The batch is capped at the tile count, so a tiny tensor does not allocate a ring it can never fill.
    // Every NoC read operation reads its tile independently (async) of the other operations, which means
    // that num_tiles does not have to be divisible by the batch size. Only the last batch of read operations
    // might perform fewer reads than the batch size, meaning that not all CB slots would be updated, which is fine.
    const uint32_t batch_tiles = std::min(attributes.noc_read_tile_batch_size, num_tiles);
    const uint32_t in_cb_pages = batch_tiles * attributes.cb_in_size_factor;
    // Pass 2 keeps the scaler in an SFPU register (LReg7): no scaler CB and no DST slot for it, so a block is dst_tiles
    // tiles. c_16 holds two blocks so packing overlaps the writer.
    const uint32_t block_tiles = dst_tiles;
    const uint32_t out_cb_pages = 2 * block_tiles;

    const uint32_t in_tile_bytes = tt::tile_size(in_format);
    const uint32_t out_tile_bytes = tt::tile_size(out_format);
    const uint32_t scale_tile_bytes = tt::tile_size(scale_format);

    constexpr uint32_t cb_in = CBIndex::c_0;
    constexpr uint32_t cb_out = CBIndex::c_16;
    constexpr uint32_t cb_scale_out = CBIndex::c_17;

    // In the order the program creates them (the check repeats tt-metal's placement).
    const std::vector<zkml::common::CbSpec> cbs = {
        {cb_in, in_cb_pages, in_tile_bytes},
        {cb_out, out_cb_pages, out_tile_bytes},
        {cb_scale_out, 1, scale_tile_bytes},
    };
    zkml::common::check_if_cbs_can_fit_in_l1(
        input.device(), cbs, "Lower noc_read_tile_batch_size or cb_in_size_factor.");

    log_debug(
        tt::LogOp,
        "pot_quantize: {} tiles, reader batch {} tiles, c_0 {} pages ({} B)",
        num_tiles,
        batch_tiles,
        in_cb_pages,
        in_cb_pages * in_tile_bytes);

    create_cb(cb_in, program, core, in_tile_bytes, in_cb_pages, in_format);      // factor reader batches
    create_cb(cb_out, program, core, out_tile_bytes, out_cb_pages, out_format);  // two compute blocks
    create_cb(cb_scale_out, program, core, scale_tile_bytes, 1, scale_format);   // scaler for the host

    // Reader: compile-time args = batch size, then TensorAccessorArgs for input
    std::vector<uint32_t> reader_ct_args = {batch_tiles};
    TensorAccessorArgs(*input.buffer()).append_to(reader_ct_args);
    const KernelHandle reader_id = CreateKernel(
        program, kernel_path(__FILE__, "kernels/dataflow/reader.cpp"), core, ReaderDataMovementConfig(reader_ct_args));

    // Writer: compile-time args = batch (one compute block), c_16 ring size, then TensorAccessorArgs for the quantized
    // output and for the scaler
    std::vector<uint32_t> writer_ct_args = {block_tiles, out_cb_pages};
    TensorAccessorArgs(*quantized.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*scaler.buffer()).append_to(writer_ct_args);
    const KernelHandle writer_id = CreateKernel(
        program, kernel_path(__FILE__, "kernels/dataflow/writer.cpp"), core, WriterDataMovementConfig(writer_ct_args));

    // Compute: fp32 DST is required (fp32 absmax and scaler, Int32 SFPU result).
    // UnpackToDestFp32 keeps Float32 operands at full precision on their way into DST.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (in_format == DataFormat::Float32) {
        unpack_to_dest_mode[cb_in] = UnpackToDestMode::UnpackToDestFp32;
    }

    const KernelHandle compute_id = CreateKernel(
        program,
        kernel_path(__FILE__, "kernels/compute/quantize.cpp"),
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
            .math_approx_mode = false,
            // output kind, bit_width, c_0 ring size in tiles, DST tiles per section, M31 range
            .compile_args =
                {output_kind,
                 attributes.bit_width.bits(),
                 in_cb_pages,
                 dst_tiles,
                 attributes.range == zkml::prim::QuantRange::M31 ? 1u : 0u},
            // Lets the kernel include shared headers from src/, e.g. device/kernels/compute/kernel_debug.hpp
            .compiler_include_paths = {kernel_include_dir()},
        });

    SetRuntimeArgs(program, reader_id, core, {input.buffer()->address(), num_tiles});
    SetRuntimeArgs(program, writer_id, core, {quantized.buffer()->address(), scaler.buffer()->address(), num_tiles});
    SetRuntimeArgs(program, compute_id, core, {num_tiles});

    return {std::move(program), {reader_id, writer_id, compute_id}};
}

void PotQuantizePerTensorOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    const CoreCoord core{0, 0};
    auto& [quantized, scaler] = outputs;

    // Shapes, dtypes and bit_width are part of the cache key, so num_tiles is unchanged on a hit.
    // Only the buffer addresses move.
    auto& reader_args = GetRuntimeArgs(program, shared.reader_id, core);
    reader_args[0] = tensor_args.input.buffer()->address();

    auto& writer_args = GetRuntimeArgs(program, shared.writer_id, core);
    writer_args[0] = quantized.buffer()->address();
    writer_args[1] = scaler.buffer()->address();
}

}  // namespace zkml::operations::pot_quantization
