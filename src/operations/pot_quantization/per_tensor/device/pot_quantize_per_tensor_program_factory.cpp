// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "common/kernel_path.hpp"
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
    const bool output_int8 = attributes.output_dtype == DataType::INT8;

    Program program{};
    const CoreCoord core{0, 0};

    // Int8 output goes through the UInt8 packer path; cb_dataformat_for picks that.
    const DataFormat in_format = cb_dataformat_for(input.dtype());
    const DataFormat scale_format = DataFormat::Float32;
    const DataFormat out_format = cb_dataformat_for(attributes.output_dtype);

    constexpr uint32_t cb_in = CBIndex::c_0;
    constexpr uint32_t cb_scale = CBIndex::c_1;
    constexpr uint32_t cb_out = CBIndex::c_16;
    constexpr uint32_t cb_scale_out = CBIndex::c_17;
    create_cb(cb_in, program, core, tile_size(in_format), 2, in_format);              // double-buffered input
    create_cb(cb_scale, program, core, tile_size(scale_format), 1, scale_format);     // packed once, held by pass 2
    create_cb(cb_out, program, core, tile_size(out_format), 2, out_format);           // double-buffered output
    create_cb(cb_scale_out, program, core, tile_size(scale_format), 1, scale_format);  // scaler for the host

    // Reader: compile-time args = TensorAccessorArgs for input
    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*input.buffer()).append_to(reader_ct_args);
    const KernelHandle reader_id = CreateKernel(
        program,
        kernel_path(__FILE__, "kernels/dataflow/reader.cpp"),
        core,
        ReaderDataMovementConfig(reader_ct_args));

    // Writer: compile-time args = TensorAccessorArgs for the quantized output, then for the scaler
    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(*quantized.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*scaler.buffer()).append_to(writer_ct_args);
    const KernelHandle writer_id = CreateKernel(
        program,
        kernel_path(__FILE__, "kernels/dataflow/writer.cpp"),
        core,
        WriterDataMovementConfig(writer_ct_args));

    // Compute: fp32 DST is required (fp32 absmax and scaler, Int32 SFPU result).
    // UnpackToDestFp32 keeps Float32 operands at full precision on their way into DST.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (in_format == DataFormat::Float32) {
        unpack_to_dest_mode[cb_in] = UnpackToDestMode::UnpackToDestFp32;
    }
    unpack_to_dest_mode[cb_scale] = UnpackToDestMode::UnpackToDestFp32;

    const KernelHandle compute_id = CreateKernel(
        program,
        kernel_path(__FILE__, "kernels/compute/quantize.cpp"),
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
            .math_approx_mode = false,
            .compile_args = {output_int8 ? 1u : 0u, attributes.bit_width},
            // Lets the kernel include shared headers from src/, e.g. common/kernel_debug.hpp
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

    // Shapes, dtypes and bit_width are part of the cache key, so num_tiles is unchanged on a hit;
    // only the buffer addresses move.
    auto& reader_args = GetRuntimeArgs(program, shared.reader_id, core);
    reader_args[0] = tensor_args.input.buffer()->address();

    auto& writer_args = GetRuntimeArgs(program, shared.writer_id, core);
    writer_args[0] = quantized.buffer()->address();
    writer_args[1] = scaler.buffer()->address();
}

}  // namespace zkml::operations::pot_quantization
