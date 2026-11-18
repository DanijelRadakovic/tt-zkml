// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Probe for the FPU primitives behind the SFPU-vs-FPU benchmark (see ../README.md, "What the FPU can and
// cannot do"). Runs ONE kernel mode on core (0,0) over `tiles` bf16 tiles that stay in L1 and prints what the
// device computed next to the host answer. With TT_METAL_DEVICE_PROFILER=1 the kernel zones LOOP / LOOP2 give
// cycles (zones.py).
//
// Usage: zkml_fpu_probe <mode> <tiles> <neg|pos|mixed> [reps] [bf16|u16]
//   mode 0    FPU max pool over columns, scaler tile = +1     (result: per-column max of x)
//   mode 1    the same with scaler tile = -1                    (shows the pool ignores the scaler sign)
//   mode 2    FPU max pool over the whole tile (REDUCE_SCALAR)
//   mode 100  SFPU absmax with copy_tile (what pass 1 does)    cycles only
//   mode 101  FPU column max pool                                cycles only
//   mode 102  FPU scalar max pool                                cycles only
//   mode 103  FPU mul_tiles by a scaler tile (pass 2 multiply)  cycles only
//   mode 104  copy_tile + scaler copy + SFPU mul_binary_tile    cycles only
//   mode 105  copy_tile alone                                    cycles only
//   mode 106  SFPU absmax alone (no copy)                        cycles only
// reps repeats the pass over the tiles; cycles / (tiles * reps) is the cost per tile.

#include <cstdint>
#include <cstring>
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
    const uint32_t mode = std::stoul(argv[1]);
    const uint32_t num_tiles = std::stoul(argv[2]);
    const std::string pattern = argv[3]; const uint32_t REPS_ = argc > 4 ? std::stoul(argv[4]) : 1; const std::string ifmt = argc > 5 ? argv[5] : "bf16";  // neg | pos | mixed
    SetRootDir(ZKML_TT_METAL_HOME);
    auto dev = distributed::MeshDevice::create_unit_mesh(0);
    const uint32_t in_bytes = 2048, out_bytes = 4096;
    auto mk = [&](uint32_t pages, uint32_t pb) {
        return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = uint64_t(pages) * pb},
            distributed::DeviceLocalBufferConfig{.page_size = pb, .buffer_type = BufferType::DRAM}, dev.get());
    };
    auto in_buf = mk(num_tiles, in_bytes);
    auto out_buf = mk(1, out_bytes);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(pattern == "neg" ? -8.f : pattern == "pos" ? 1.f : -8.f,
                                            pattern == "neg" ? -1.f : 8.f);
    std::vector<float> vals(size_t(num_tiles) * 1024);
    std::vector<uint32_t> words(size_t(num_tiles) * 512);
    for (size_t i = 0; i < vals.size(); i++) {
        float f = d(rng);
        uint32_t u; std::memcpy(&u, &f, 4); u &= 0xFFFF0000u; std::memcpy(&f, &u, 4);  // truncate to bf16
        vals[i] = f;
    }
    for (size_t i = 0; i < words.size(); i++) {
        uint32_t a, b; std::memcpy(&a, &vals[2*i], 4); std::memcpy(&b, &vals[2*i+1], 4);
        words[i] = (a >> 16) | (b & 0xFFFF0000u);  // element 2i low half, 2i+1 high half
    }
    // face layout: tile elements are stored face by face; compute the host column max over tile-row-major
    // positions using the face layout (16x16 faces: f0 top-left, f1 top-right, f2 bottom-left, f3 bottom-right)
    distributed::EnqueueWriteMeshBuffer(dev->mesh_command_queue(), in_buf, words, true);
    Program program = CreateProgram();
    const CoreCoord core{0, 0};
    auto cb = [&](uint32_t idx, uint32_t pages, uint32_t pb, tt::DataFormat f) {
        CreateCircularBuffer(program, core, CircularBufferConfig(pages * pb, {{idx, f}}).set_page_size(idx, pb));
    };
    cb(0, num_tiles, in_bytes, ifmt == "u16" ? tt::DataFormat::UInt16 : ifmt == "i16" ? tt::DataFormat::Int16 : tt::DataFormat::Float16_b);
    cb(1, 1, 4096, tt::DataFormat::Float32);
    cb(2, 1, in_bytes, tt::DataFormat::Float16_b);
    cb(16, 1, out_bytes, tt::DataFormat::Float32);
    std::vector<uint32_t> rct; TensorAccessorArgs(*in_buf->get_backing_buffer()).append_to(rct);
    auto reader = CreateKernel(program, zkml::common::kernel_path(__FILE__, "kernels/reader.cpp"), core, ReaderDataMovementConfig(rct));
    std::vector<uint32_t> wct; TensorAccessorArgs(*out_buf->get_backing_buffer()).append_to(wct);
    auto writer = CreateKernel(program, zkml::common::kernel_path(__FILE__, "kernels/writer.cpp"), core, WriterDataMovementConfig(wct));
    auto compute = CreateKernel(program, zkml::common::kernel_path(__FILE__, "kernels/compute.cpp"), core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false,
                      .compile_args = {mode, REPS_}, .compiler_include_paths = {zkml::common::kernel_path(__FILE__, "../../../src/operations/pot_quantization/per_tensor/device/kernels/compute"), zkml::common::kernel_include_dir()}});
    SetRuntimeArgs(program, reader, core, {uint32_t(in_buf->address()), num_tiles});
    SetRuntimeArgs(program, writer, core, {uint32_t(out_buf->address())});
    SetRuntimeArgs(program, compute, core, {num_tiles});
    auto& cq = dev->mesh_command_queue();
    distributed::MeshWorkload wl; wl.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, wl, false); distributed::Finish(cq);
    ReadMeshDeviceProfilerResults(*dev);
    std::vector<uint32_t> out; distributed::EnqueueReadMeshBuffer(cq, out, out_buf, true);
    // host reference: max over tiles of each (row, col) position within the 32x32 tile... tile storage is
    // face-ordered but we only compare per physical element index inside the face-ordered tile.
    std::vector<float> o(1024); std::memcpy(o.data(), out.data(), 4096);
    // reference per element index e in [0,1024): max over tiles
    std::vector<float> mx(1024, -1e30f), mn(1024, 1e30f);
    for (uint32_t t = 0; t < num_tiles; t++) for (int e = 0; e < 1024; e++) {
        mx[e] = std::max(mx[e], vals[t*1024+e]); mn[e] = std::min(mn[e], vals[t*1024+e]);
    }
    float gmax = -1e30f, gmin = 1e30f; for (float v : vals) { gmax = std::max(gmax, v); gmin = std::min(gmin, v); }
    fmt::print("mode {} tiles {} pattern {} | global max {} min {}\n", mode, num_tiles, pattern, gmax, gmin);
    int nz = 0; for (int e = 0; e < 1024; e++) if (o[e] != 0.f) nz++;
    fmt::print("nonzero elements in result tile: {}\n", nz);
    for (int f = 0; f < 4; f++) {
        fmt::print("face {} row0:", f);
        for (int c = 0; c < 16; c++) fmt::print(" {:.4g}", o[f*256 + c]);
        fmt::print("\n");
    }
    // which face-element positions are nonzero
    fmt::print("raw words face0 row0:"); for (int c = 0; c < 16; c++) fmt::print(" {:08x}", out[c]); fmt::print("\n");
    fmt::print("raw words face1 row0:"); for (int c = 0; c < 16; c++) fmt::print(" {:08x}", out[256+c]); fmt::print("\n");
    fmt::print("nonzero idx (first 40):");
    int shown = 0; for (int e = 0; e < 1024 && shown < 40; e++) if (o[e] != 0.f) { fmt::print(" {}", e); shown++; }
    fmt::print("\n");
    // candidates: per-column max over the tile rows (face-ordered: element e -> face e/256, row (e%256)/16, col e%16)
    std::vector<float> colmax(32, -1e30f), colmin(32, 1e30f);
    for (uint32_t t = 0; t < num_tiles; t++) for (int e = 0; e < 1024; e++) {
        int f = e / 256, r = (e % 256) / 16, c = e % 16;
        int col = (f % 2) * 16 + c;
        colmax[col] = std::max(colmax[col], vals[t*1024+e]); colmin[col] = std::min(colmin[col], vals[t*1024+e]);
    }
    fmt::print("host colmax[0..15]:"); for (int c = 0; c < 16; c++) fmt::print(" {:.4g}", colmax[c]); fmt::print("\n");
    fmt::print("host colmax[16..31]:"); for (int c = 16; c < 32; c++) fmt::print(" {:.4g}", colmax[c]); fmt::print("\n");
    fmt::print("host -colmin[0..15]:"); for (int c = 0; c < 16; c++) fmt::print(" {:.4g}", -colmin[c]); fmt::print("\n");
    in_buf.reset(); out_buf.reset(); dev->close();
    return 0;
}
