// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "common/kernel_path.hpp"
#include "common/l1_check.hpp"
#include "execute_layer_device_operation.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;
using zkml::common::kernel_include_dir;
using zkml::common::kernel_path;

namespace zkml::operations::layer_execution {

ExecuteLayerOperation::SingleCore::cached_program_t ExecuteLayerOperation::SingleCore::create(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& outputs) {
    const auto& x = tensor_args.x;
    const auto& w = tensor_args.w;
    auto& [out, scaler_x, scaler_w, scaler_out] = outputs;

    const uint32_t Mt = x.logical_shape()[0] / TILE_HEIGHT;
    const uint32_t Kt = x.logical_shape()[1] / TILE_WIDTH;
    const uint32_t Nt = w.logical_shape()[1] / TILE_WIDTH;

    Program program{};
    const CoreCoord core{0, 0};

    const uint32_t f32_bytes = tt::tile_size(DataFormat::Float32);
    const uint32_t u8_bytes = tt::tile_size(DataFormat::UInt8);
    TT_FATAL(
        u8_bytes == tt::tile_size(DataFormat::Int8), "execute_layer: UInt8 and Int8 tiles must have the same size");

    constexpr uint32_t c_x = CBIndex::c_0, c_w = CBIndex::c_1;
    constexpr uint32_t c_qx = CBIndex::c_2, c_qw = CBIndex::c_3;
    constexpr uint32_t c_qx_i8 = CBIndex::c_4, c_qw_i8 = CBIndex::c_5;
    constexpr uint32_t c_acc = CBIndex::c_6;
    constexpr uint32_t c_out = CBIndex::c_16;
    constexpr uint32_t c_scale_x = CBIndex::c_17, c_scale_w = CBIndex::c_18, c_scale_out = CBIndex::c_19;

    // c_16 holds two compute blocks so packing overlaps the writer.
    const uint32_t out_cb_pages = 2 * dst_tiles;
    const uint32_t out_tiles = Mt * Nt;

    // In the order the program creates them (the check repeats tt-metal's placement). c_4 / c_5 share the memory of c_2 / c_3.
    const std::vector<zkml::common::CbSpec> cbs = {
        {c_x, Mt * Kt, f32_bytes},
        {c_w, Kt * Nt, f32_bytes},
        {c_qx, Mt * Kt, u8_bytes},
        {c_qw, Kt * Nt, u8_bytes},
        {c_acc, Mt * Nt, f32_bytes},
        {c_out, out_cb_pages, u8_bytes},
        {c_scale_x, 1, f32_bytes},
        {c_scale_w, 1, f32_bytes},
        {c_scale_out, 1, f32_bytes},
    };
    zkml::common::check_if_cbs_can_fit_in_l1(
        x.device(), cbs, "x, w, the accumulator and the outputs must fit in the L1 of one core: use smaller tensors.");

    auto create_cb = [&](uint32_t index, uint32_t pages, uint32_t page_bytes, DataFormat format) {
        CreateCircularBuffer(
            program, core, CircularBufferConfig(pages * page_bytes, {{index, format}}).set_page_size(index, page_bytes));
    };
    // The same bytes under two formats: the packer writes through the UInt8 CB, the FPU reads through the Int8 CB.
    auto create_aliased_cb = [&](uint32_t u8_index, uint32_t i8_index, uint32_t pages) {
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(
                pages * u8_bytes, {{u8_index, DataFormat::UInt8}, {i8_index, DataFormat::Int8}})
                .set_page_size(u8_index, u8_bytes)
                .set_page_size(i8_index, u8_bytes));
    };
    create_cb(c_x, Mt * Kt, f32_bytes, DataFormat::Float32);
    create_cb(c_w, Kt * Nt, f32_bytes, DataFormat::Float32);
    create_aliased_cb(c_qx, c_qx_i8, Mt * Kt);
    create_aliased_cb(c_qw, c_qw_i8, Kt * Nt);
    create_cb(c_acc, Mt * Nt, f32_bytes, DataFormat::Float32);
    create_cb(c_out, out_cb_pages, u8_bytes, DataFormat::UInt8);
    create_cb(c_scale_x, 1, f32_bytes, DataFormat::Float32);
    create_cb(c_scale_w, 1, f32_bytes, DataFormat::Float32);
    create_cb(c_scale_out, 1, f32_bytes, DataFormat::Float32);

    constexpr uint32_t reader_batch = 16;
    std::vector<uint32_t> reader_ct_args = {reader_batch};
    TensorAccessorArgs(*x.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*w.buffer()).append_to(reader_ct_args);
    const KernelHandle reader_id = CreateKernel(
        program, kernel_path(__FILE__, "kernels/dataflow/reader.cpp"), core, ReaderDataMovementConfig(reader_ct_args));

    std::vector<uint32_t> writer_ct_args = {dst_tiles, out_cb_pages};
    TensorAccessorArgs(*out.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*scaler_x.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*scaler_w.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*scaler_out.buffer()).append_to(writer_ct_args);
    const KernelHandle writer_id = CreateKernel(
        program, kernel_path(__FILE__, "kernels/dataflow/writer.cpp"), core, WriterDataMovementConfig(writer_ct_args));

    // fp32 DST: the int32 accumulator of the FPU and the fp32 SFPU work. Float32 CBs go to DST at full precision.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t cb : {c_x, c_w, c_acc}) {
        unpack_to_dest_mode[cb] = UnpackToDestMode::UnpackToDestFp32;
    }
    CreateKernel(
        program,
        kernel_path(__FILE__, "kernels/compute/execute_layer.cpp"),
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,  // the int8 FPU matmul is exact only with all four phases
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
            .math_approx_mode = false,
            // Mt, Kt, Nt, c_16 pages, activation
            .compile_args = {Mt, Kt, Nt, out_cb_pages, static_cast<uint32_t>(attributes.activation)},
            .compiler_include_paths = {kernel_include_dir()},
        });

    SetRuntimeArgs(program, reader_id, core, {x.buffer()->address(), w.buffer()->address(), Mt * Kt, Kt * Nt});
    SetRuntimeArgs(
        program,
        writer_id,
        core,
        {out.buffer()->address(),
         scaler_x.buffer()->address(),
         scaler_w.buffer()->address(),
         scaler_out.buffer()->address(),
         out_tiles});

    return {std::move(program), {reader_id, writer_id}};
}

void ExecuteLayerOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    const CoreCoord core{0, 0};
    auto& [out, scaler_x, scaler_w, scaler_out] = outputs;

    // Shapes and dtypes are part of the cache key, so the tile counts are unchanged on a hit; only the addresses move.
    auto& reader_args = GetRuntimeArgs(program, shared.reader_id, core);
    reader_args[0] = tensor_args.x.buffer()->address();
    reader_args[1] = tensor_args.w.buffer()->address();

    auto& writer_args = GetRuntimeArgs(program, shared.writer_id, core);
    writer_args[0] = out.buffer()->address();
    writer_args[1] = scaler_x.buffer()->address();
    writer_args[2] = scaler_w.buffer()->address();
    writer_args[3] = scaler_out.buffer()->address();
}

}  // namespace zkml::operations::layer_execution
