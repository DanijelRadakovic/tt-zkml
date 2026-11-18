// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Benchmark: does keeping the whole tensor in L1 between the two passes of pot_quantize help?
//
//   stream    the real op's dataflow: the reader sends the tensor twice, c_0 is a ring of 2 blocks
//   resident  the reader sends the tensor once, c_0 holds all tiles, pass 2 re-reads them from L1
//
// Usage: zkml_quant_l1_bench <stream|resident> <f32|bf16> <bits 8|16|32> <num_tiles> <block> <iters>
//
// The program runs <iters> times back to back on core (0,0). Timing comes from the device profiler
// (run with TT_METAL_DEVICE_PROFILER=1; see README.md): the kernels carry DeviceZoneScopedN zones
// and tools/.../analyze.py reads them from profile_log_device.csv. The line printed here is only a
// correctness check (both modes must print the same checksum) and the host wall time.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "common/kernel_path.hpp"

using namespace tt::tt_metal;

int main(int argc, char** argv) {
    if (argc != 7) {
        fmt::print(stderr, "usage: {} <stream|resident> <f32|bf16> <bits 8|16|32> <num_tiles> <block> <iters>\n", argv[0]);
        return 2;
    }
    const bool resident = std::string(argv[1]) == "resident";
    const bool bf16 = std::string(argv[2]) == "bf16";
    const uint32_t bits = std::stoul(argv[3]);
    const uint32_t num_tiles = std::stoul(argv[4]);
    const uint32_t block = std::min<uint32_t>(std::stoul(argv[5]), num_tiles);
    const uint32_t iters = std::stoul(argv[6]);
    constexpr uint32_t dst_tiles = 4;  // same as PotQuantizePerTensorOperation::dst_tiles

    SetRootDir(ZKML_TT_METAL_HOME);
    auto dev = distributed::MeshDevice::create_unit_mesh(0);

    const tt::DataFormat in_format = bf16 ? tt::DataFormat::Float16_b : tt::DataFormat::Float32;
    const uint32_t in_tile_bytes = tt::tile_size(in_format);
    const uint32_t out_tile_bytes = tt::tile_size(tt::DataFormat::Int32);
    const uint32_t scale_tile_bytes = tt::tile_size(tt::DataFormat::Float32);

    auto make_buffer = [&](uint32_t pages, uint32_t page_bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = static_cast<uint64_t>(pages) * page_bytes},
            distributed::DeviceLocalBufferConfig{.page_size = page_bytes, .buffer_type = BufferType::DRAM},
            dev.get());
    };
    auto in_buf = make_buffer(num_tiles, in_tile_bytes);
    auto out_buf = make_buffer(num_tiles, out_tile_bytes);
    auto scale_buf = make_buffer(1, scale_tile_bytes);

    // Finite values in [-8, 8); the content does not change the timing, only k.
    {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
        std::vector<uint32_t> words(static_cast<size_t>(num_tiles) * in_tile_bytes / 4);
        for (size_t i = 0; i < words.size(); i++) {
            if (bf16) {  // two bf16 per word: the upper 16 bits of an fp32
                uint32_t a, b;
                float fa = dist(rng), fb = dist(rng);
                std::memcpy(&a, &fa, 4);
                std::memcpy(&b, &fb, 4);
                words[i] = (a >> 16) | (b & 0xFFFF0000u);
            } else {
                float f = dist(rng);
                std::memcpy(&words[i], &f, 4);
            }
        }
        distributed::EnqueueWriteMeshBuffer(dev->mesh_command_queue(), in_buf, words, true);
    }

    Program program = CreateProgram();
    const CoreCoord core{0, 0};
    auto cb = [&](uint32_t index, uint32_t pages, uint32_t page_bytes, tt::DataFormat fmt) {
        CreateCircularBuffer(
            program, core, CircularBufferConfig(pages * page_bytes, {{index, fmt}}).set_page_size(index, page_bytes));
    };
    // c_0: the whole tensor when resident, otherwise two reader blocks
    cb(0, resident ? num_tiles : 2 * block, in_tile_bytes, in_format);
    cb(1, 1, scale_tile_bytes, tt::DataFormat::Float32);
    cb(16, 2 * (dst_tiles - 1), out_tile_bytes, tt::DataFormat::Int32);
    cb(17, 1, scale_tile_bytes, tt::DataFormat::Float32);

    std::vector<uint32_t> reader_ct = {block, resident ? 1u : 2u};
    TensorAccessorArgs(*in_buf->get_backing_buffer()).append_to(reader_ct);
    const auto reader = CreateKernel(
        program, zkml::common::kernel_path(__FILE__, "kernels/reader.cpp"), core, ReaderDataMovementConfig(reader_ct));

    std::vector<uint32_t> writer_ct;
    TensorAccessorArgs(*out_buf->get_backing_buffer()).append_to(writer_ct);
    TensorAccessorArgs(*scale_buf->get_backing_buffer()).append_to(writer_ct);
    const auto writer = CreateKernel(
        program, zkml::common::kernel_path(__FILE__, "kernels/writer.cpp"), core, WriterDataMovementConfig(writer_ct));

    std::vector<UnpackToDestMode> unpack(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (!bf16) {
        unpack[0] = UnpackToDestMode::UnpackToDestFp32;
    }
    unpack[1] = UnpackToDestMode::UnpackToDestFp32;
    // pot_scale_sfpu.hpp lives with the real kernel; the bench compute kernel includes it from there.
    const std::string sfpu_dir = zkml::common::kernel_path(
        __FILE__, "../../src/operations/pot_quantization/per_tensor/device/kernels/compute");
    const auto compute = CreateKernel(
        program,
        zkml::common::kernel_path(__FILE__, "kernels/compute.cpp"),
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = std::move(unpack),
            .math_approx_mode = false,
            // [0] unused (output_int8), bit_width, BLOCK_SIZE, DST_TILES, RESIDENT
            .compile_args = {0u, bits, block, dst_tiles, resident ? 1u : 0u},
            .compiler_include_paths = {sfpu_dir}});

    SetRuntimeArgs(program, reader, core, {static_cast<uint32_t>(in_buf->address()), num_tiles});
    SetRuntimeArgs(
        program, writer, core,
        {static_cast<uint32_t>(out_buf->address()), static_cast<uint32_t>(scale_buf->address()), num_tiles});
    SetRuntimeArgs(program, compute, core, {num_tiles});

    auto& cq = dev->mesh_command_queue();
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(program));

    // Run 0 compiles and warms the caches; the profiler log keeps every run, analyze.py drops run 0.
    double host_ms = 0;
    for (uint32_t it = 0; it < iters; it++) {
        const auto t0 = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        const auto t1 = std::chrono::steady_clock::now();
        if (it > 0) {
            host_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
    }
    ReadMeshDeviceProfilerResults(*dev);  // writes profile_log_device.csv

    std::vector<uint32_t> out, scale;
    distributed::EnqueueReadMeshBuffer(cq, out, out_buf, true);
    distributed::EnqueueReadMeshBuffer(cq, scale, scale_buf, true);
    uint64_t checksum = 1469598103934665603ull;  // FNV-1a over the output words
    for (uint32_t w : out) {
        checksum = (checksum ^ w) * 1099511628211ull;
    }
    float scaler;
    std::memcpy(&scaler, &scale[0], 4);
    fmt::print(
        "config mode {} in {} bits {} tiles {} block {} | scaler {} checksum {:016x} | host_ms_per_run {:.3f}\n",
        resident ? "resident" : "stream", bf16 ? "bf16" : "f32", bits, num_tiles, block, scaler, checksum,
        host_ms / std::max(1u, iters - 1));

    in_buf.reset();
    out_buf.reset();
    scale_buf.reset();
    dev->close();
    return 0;
}
