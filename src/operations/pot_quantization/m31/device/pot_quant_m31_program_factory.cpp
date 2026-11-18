// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "common/kernel_path.hpp"
#include "common/l1_check.hpp"
#include "pot_quant_m31_device_operation.hpp"
#include "ttnn/operations/cb_utils.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;
using zkml::common::kernel_include_dir;
using zkml::common::kernel_path;

namespace zkml::operations::pot_quantization {

PotQuantM31Operation::SingleCore::cached_program_t PotQuantM31Operation::SingleCore::create(
    const operation_attributes_t&, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const uint32_t num_tiles = input.physical_volume() / TILE_HW;
    const bool input_int8 = input.dtype() == DataType::INT8;

    Program program{};
    const CoreCoord core{0, 0};

    const DataFormat in_format = cb_dataformat_for(input.dtype());
    const DataFormat out_format = cb_dataformat_for(DataType::UINT32);
    const uint32_t in_tile_bytes = tt::tile_size(in_format);
    const uint32_t out_tile_bytes = tt::tile_size(out_format);
    constexpr uint32_t ring_pages = 2 * block_tiles;

    constexpr uint32_t cb_in = CBIndex::c_0;
    constexpr uint32_t cb_out = CBIndex::c_16;
    zkml::common::check_if_cbs_can_fit_in_l1(
        input.device(), {{{cb_in, ring_pages, in_tile_bytes}, {cb_out, ring_pages, out_tile_bytes}}});
    create_cb(cb_in, program, core, in_tile_bytes, ring_pages, in_format);
    create_cb(cb_out, program, core, out_tile_bytes, ring_pages, out_format);

    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*input.buffer()).append_to(reader_ct_args);
    const KernelHandle reader_id = CreateKernel(
        program, kernel_path(__FILE__, "kernels/dataflow/reader.cpp"), core, ReaderDataMovementConfig(reader_ct_args));

    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args);
    const KernelHandle writer_id = CreateKernel(
        program, kernel_path(__FILE__, "kernels/dataflow/writer.cpp"), core, WriterDataMovementConfig(writer_ct_args));

    // The unpacker must put an Int32 operand into DST without passing it through SrcA (19 bits).
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (!input_int8) {
        unpack_to_dest_mode[cb_in] = UnpackToDestMode::UnpackToDestFp32;
    }
    const KernelHandle compute_id = CreateKernel(
        program,
        kernel_path(__FILE__, "kernels/compute/to_m31.cpp"),
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
            .math_approx_mode = false,
            .compile_args = {input_int8 ? 1u : 0u, block_tiles},
            .compiler_include_paths = {kernel_include_dir()},
        });
    SetRuntimeArgs(program, compute_id, core, {num_tiles});

    SetRuntimeArgs(program, reader_id, core, {input.buffer()->address(), num_tiles});
    SetRuntimeArgs(program, writer_id, core, {output.buffer()->address(), num_tiles});
    return {std::move(program), {reader_id, writer_id}};
}

void PotQuantM31Operation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    const CoreCoord core{0, 0};
    GetRuntimeArgs(program, shared.reader_id, core)[0] = tensor_args.input.buffer()->address();
    GetRuntimeArgs(program, shared.writer_id, core)[0] = output.buffer()->address();
}

}  // namespace zkml::operations::pot_quantization
