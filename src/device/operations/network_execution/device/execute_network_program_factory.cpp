// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "common/kernel_path.hpp"
#include "common/l1_check.hpp"
#include "execute_network_device_operation.hpp"
#include "network_plan.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;
using zkml::common::kernel_include_dir;
using zkml::common::kernel_path;
namespace n = zkml::network;

namespace zkml::operations::network_execution {

ExecuteNetworkOperation::SingleCore::cached_program_t ExecuteNetworkOperation::SingleCore::create(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& outputs) {
    const auto& x = tensor_args.x;
    const auto& weights = tensor_args.weights;
    const NetworkPlan plan = NetworkPlan::make(x, weights, attributes.activations, attributes.pre_quantized, tensor_args.weight_scalers);
    const bool pre_q = plan.pre_quantized;
    const uint32_t L = plan.layers;

    zkml::common::check_if_cbs_can_fit_in_l1(
        x.device(),
        plan.cbs,
        "execute_network: x, one layer of weights, the accumulator, the quantized activations of every layer and the "
        "output "
        "must fit in the L1 of one core: use smaller tensors or fewer layers.");

    Program program{};
    const CoreCoord core{0, 0};

    const DataFormat in_format = plan.fp32_in ? DataFormat::Float32 : DataFormat::Float16_b;
    // Every quantized buffer is a UInt8 buffer (exempt from the JIT's exponent-family check, so also usable next to
    // bf16); the packer and the FPU's source registers are configured for the Int8 format by hand in the compute
    // kernel.
    auto create_cb = [&](uint32_t index, uint32_t pages, DataFormat format) {
        const uint32_t page_bytes = tt::tile_size(format);
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(pages * page_bytes, {{index, format}}).set_page_size(index, page_bytes));
    };
    create_cb(n::c_x, plan.x_tiles(), in_format);
    if (!pre_q) {
        create_cb(n::c_w, plan.cap_w, in_format);
    }
    create_cb(n::c_qw, plan.cap_qw, DataFormat::UInt8);
    create_cb(n::c_acc, plan.cap_acc, DataFormat::Int32);
    create_cb(n::c_act_a, plan.cap_act, DataFormat::UInt8);
    create_cb(n::c_act_b, plan.cap_act, DataFormat::UInt8);
    create_cb(n::c_out, plan.out_ring_pages, DataFormat::UInt8);
    create_cb(n::c_scale_x, n::scaler_pages, DataFormat::Float32);
    if (!pre_q) {
        create_cb(n::c_scale_w, n::scaler_pages, DataFormat::Float32);
    }
    create_cb(n::c_scale_out, n::scaler_pages, DataFormat::Float32);

    // ---- reader: x, then the weights of layer 0, 1, ... into c_1 ----
    constexpr uint32_t reader_batch = 16;
    // The weights go to c_1 and are quantized by compute, or (pre-quantized) straight into the quantized-weight buffer c_2.
    std::vector<uint32_t> reader_ct_args = {reader_batch, L, plan.cap_w, pre_q ? n::c_qw : n::c_w};
    // The accessor arguments of interleaved tensors of one dtype and memory config are the same for all of them, so one set serves
    // every weight and the reader can loop over the layers.
    TensorAccessorArgs(*x.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*weights.front().buffer()).append_to(reader_ct_args);
    auto reader_config = ReaderDataMovementConfig(reader_ct_args);
    reader_config.compiler_include_paths = {kernel_include_dir()};  // network_layout.hpp
    const KernelHandle reader_id =
        CreateKernel(program, kernel_path(__FILE__, "kernels/dataflow/reader.cpp"), core, reader_config);

    // ---- writer: scalers, accumulators and the output ----
    std::vector<uint32_t> writer_ct_args = {n::dst_tiles, plan.out_ring_pages, L, plan.cap_acc, pre_q ? 1u : 0u};
    TensorAccessorArgs(*outputs[ExecuteNetworkOperation::out_index()].buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*outputs[ExecuteNetworkOperation::scalers_index()].buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*outputs[ExecuteNetworkOperation::acc_index()].buffer()).append_to(writer_ct_args);
    auto writer_config = WriterDataMovementConfig(writer_ct_args);
    writer_config.compiler_include_paths = {kernel_include_dir()};
    const KernelHandle writer_id =
        CreateKernel(program, kernel_path(__FILE__, "kernels/dataflow/writer.cpp"), core, writer_config);

    // ---- compute ----
    // fp32 DST: the int32 accumulator of the FPU and the fp32 SFPU work. Float32 buffers go to DST at full precision.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[n::c_acc] = UnpackToDestMode::UnpackToDestFp32;
    if (plan.fp32_in) {
        unpack_to_dest_mode[n::c_x] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[n::c_w] = UnpackToDestMode::UnpackToDestFp32;  // (no c_w in a program with pre-quantized weights)
    }
    // Mt, fp32 input, output ring pages, capacities of the weight / quantized-weight / accumulator / activation buffers (tiles). The
    // layers are run-time arguments (below), so the kernel is the same for every number of layers.
    std::vector<uint32_t> compute_ct_args = {
        plan.Mt, plan.fp32_in ? 1u : 0u, plan.out_ring_pages, plan.cap_w, plan.cap_qw, plan.cap_acc, plan.cap_act, pre_q ? 1u : 0u};
    const KernelHandle compute_id = CreateKernel(
        program,
        kernel_path(__FILE__, "kernels/compute/execute_network.cpp"),
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,  // the int8 FPU matmul is exact only with all four phases
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
            .math_approx_mode = false,
            .compile_args = compute_ct_args,
            .defines = std::getenv("ZKML_FINE_ZONES") ? std::map<std::string, std::string>{{"ZKML_FINE_ZONES", "1"}} : std::map<std::string, std::string>{},
            .compiler_include_paths = {kernel_include_dir()},
        });

    // ---- runtime args ----
    // compute: [0] number of layers, then per layer (Kt, Nt, relu). They follow from the shapes and the activations, which are part of
    // the cache key, so they are not touched on a cache hit.
    std::vector<uint32_t> compute_args = {L};
    for (uint32_t l = 0; l < L; l++) {
        compute_args.push_back(plan.Kt[l]);
        compute_args.push_back(plan.Nt[l]);
        compute_args.push_back(plan.relu[l]);
    }
    SetRuntimeArgs(program, compute_id, core, compute_args);
    // reader: [0] x address, [1] x tiles, then per layer (weight address, weight tiles)
    std::vector<uint32_t> reader_args = {x.buffer()->address(), plan.x_tiles()};
    for (uint32_t l = 0; l < L; l++) {
        reader_args.push_back(weights[l].buffer()->address());
        reader_args.push_back(plan.w_tiles(l));
    }
    SetRuntimeArgs(program, reader_id, core, reader_args);
    // writer: [0] out address, [1] scalers address, [2] acc address, [3] tiles of out, [4] Mt, [5] column tiles of the packed acc, then per
    // layer (Nt, first column tile of its accumulator)
    std::vector<uint32_t> writer_args = {
        outputs[ExecuteNetworkOperation::out_index()].buffer()->address(),
        outputs[ExecuteNetworkOperation::scalers_index()].buffer()->address(),
        outputs[ExecuteNetworkOperation::acc_index()].buffer()->address(),
        plan.out_tiles(L - 1),
        plan.Mt,
        plan.total_nt};
    for (uint32_t l = 0; l < L; l++) {
        writer_args.push_back(plan.Nt[l]);
        writer_args.push_back(plan.nt_off[l]);
    }
    SetRuntimeArgs(program, writer_id, core, writer_args);

    return {std::move(program), {reader_id, writer_id, L, pre_q}};
}

void ExecuteNetworkOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    const uint32_t L = shared.layers;
    const CoreCoord core{0, 0};

    // Shapes and dtypes are part of the cache key, so the tile counts are unchanged on a hit; only the addresses move.
    auto& reader_args = GetRuntimeArgs(program, shared.reader_id, core);
    reader_args[0] = tensor_args.x.buffer()->address();
    for (uint32_t l = 0; l < L; l++) {
        reader_args[2 + 2 * l] = tensor_args.weights[l].buffer()->address();
    }
    auto& writer_args = GetRuntimeArgs(program, shared.writer_id, core);
    writer_args[0] = outputs[ExecuteNetworkOperation::out_index()].buffer()->address();
    writer_args[1] = outputs[ExecuteNetworkOperation::scalers_index()].buffer()->address();
    writer_args[2] = outputs[ExecuteNetworkOperation::acc_index()].buffer()->address();
}

}  // namespace zkml::operations::network_execution
