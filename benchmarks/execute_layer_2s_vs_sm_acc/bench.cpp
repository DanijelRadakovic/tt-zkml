// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Two's-complement accumulator against raw sign-magnitude accumulator in execute_layer (see README.md).
//
//   zkml_execute_layer_2s_vs_sm_acc_bench check   <Mt> <Kt> <Nt> <relu|none>                 outputs of the variants
//   agree zkml_execute_layer_2s_vs_sm_acc_bench profile <variant> <Mt> <Kt> <Nt> <relu|none> <iters>  with
//   TT_METAL_DEVICE_PROFILER=1 zkml_execute_layer_2s_vs_sm_acc_bench wall    <variant> <Mt> <Kt> <Nt> <relu|none>
//   <iters>  device time per run, enqueue + finish
// variant: twos | sm_float | sm_int | sm_int_sfpi. The program is the one of the op (same CBs, reader, writer) with its
// own compute kernel.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "common/kernel_path.hpp"

using namespace tt;
using namespace tt::tt_metal;
using Clock = std::chrono::steady_clock;
namespace dist = tt::tt_metal::distributed;

namespace {

constexpr uint32_t kDstTiles = 4;
const char* kNames[] = {"twos", "sm_float", "sm_int", "sm_int_sfpi", "sm_int_sfpi_absmax"};
constexpr int kVariants = 5;

uint32_t qmode() {
    const char* e = std::getenv("ZKML_QMODE");
    return e ? static_cast<uint32_t>(std::atoi(e)) : 0;
}

uint32_t qlm() {
    const char* e = std::getenv("ZKML_QLM");  // 1: pass 2 of the x / W quantization with SFPLOADMACRO
    return e ? static_cast<uint32_t>(std::atoi(e)) : 0;
}

struct Dims {
    uint32_t Mt, Kt, Nt;
    bool relu;
};

struct Buffers {
    std::shared_ptr<dist::MeshBuffer> x, w, out, acc, sx, sw, so;
};

std::shared_ptr<dist::MeshBuffer> make(dist::MeshDevice& dev, uint32_t pages, uint32_t page_bytes) {
    return dist::MeshBuffer::create(
        dist::ReplicatedBufferConfig{.size = uint64_t(pages) * page_bytes},
        dist::DeviceLocalBufferConfig{.page_size = page_bytes, .buffer_type = BufferType::DRAM},
        &dev);
}

Buffers make_buffers(dist::MeshDevice& dev, const Dims& s) {
    const uint32_t f32 = tt::tile_size(DataFormat::Float32), u8 = tt::tile_size(DataFormat::UInt8);
    Buffers b;
    b.x = make(dev, s.Mt * s.Kt, f32);
    b.w = make(dev, s.Kt * s.Nt, f32);
    b.out = make(dev, s.Mt * s.Nt, u8);
    b.acc = make(dev, s.Mt * s.Nt, tt::tile_size(DataFormat::Int32));
    b.sx = make(dev, 1, f32);
    b.sw = make(dev, 1, f32);
    b.so = make(dev, 1, f32);
    // Random inputs in the tile order the reader expects: the values are what matters here, not their layout.
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-2.f, 2.f);
    auto fill = [&](std::shared_ptr<dist::MeshBuffer>& buf, size_t tiles) {
        std::vector<uint32_t> words(tiles * 1024);
        for (auto& wd : words) {
            const float f = d(rng);
            std::memcpy(&wd, &f, 4);
        }
        dist::EnqueueWriteMeshBuffer(dev.mesh_command_queue(), buf, words, true);
    };
    fill(b.x, size_t(s.Mt) * s.Kt);
    fill(b.w, size_t(s.Kt) * s.Nt);
    return b;
}

dist::MeshWorkload build(dist::MeshDevice& dev, const Buffers& b, const Dims& s, uint32_t variant) {
    Program program = CreateProgram();
    const CoreCoord core{0, 0};
    const uint32_t f32 = tt::tile_size(DataFormat::Float32), u8 = tt::tile_size(DataFormat::UInt8),
                   i32 = tt::tile_size(DataFormat::Int32);
    const uint32_t out_cb_pages = 2 * kDstTiles, out_tiles = s.Mt * s.Nt;
    auto cb = [&](uint32_t index, uint32_t pages, uint32_t bytes, DataFormat f) {
        CreateCircularBuffer(
            program, core, CircularBufferConfig(pages * bytes, {{index, f}}).set_page_size(index, bytes));
    };
    cb(0, s.Mt * s.Kt, f32, DataFormat::Float32);
    cb(1, s.Kt * s.Nt, f32, DataFormat::Float32);
    cb(2, s.Mt * s.Kt, u8, DataFormat::Int8);
    cb(3, s.Kt * s.Nt, u8, DataFormat::Int8);
    cb(6, out_tiles, i32, DataFormat::Int32);
    cb(16, out_cb_pages, u8, DataFormat::UInt8);
    cb(17, 1, f32, DataFormat::Float32);
    cb(18, 1, f32, DataFormat::Float32);
    cb(19, 1, f32, DataFormat::Float32);

    // The reader and writer of the op.
    const std::string df =
        zkml::common::kernel_path(__FILE__, "../../src/device/operations/layer_execution/device/kernels/dataflow/");
    std::vector<uint32_t> rct = {16};
    TensorAccessorArgs(*b.x->get_backing_buffer()).append_to(rct);
    TensorAccessorArgs(*b.w->get_backing_buffer()).append_to(rct);
    auto reader = CreateKernel(program, df + "reader.cpp", core, ReaderDataMovementConfig(rct));
    std::vector<uint32_t> wct = {kDstTiles, out_cb_pages};
    for (auto* m : {&b.out, &b.acc, &b.sx, &b.sw, &b.so}) {
        TensorAccessorArgs(*(*m)->get_backing_buffer()).append_to(wct);
    }
    auto writer = CreateKernel(program, df + "writer.cpp", core, WriterDataMovementConfig(wct));

    std::vector<UnpackToDestMode> mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t i : {0u, 1u, 6u}) {
        mode[i] = UnpackToDestMode::UnpackToDestFp32;
    }
    CreateKernel(
        program,
        zkml::common::kernel_path(__FILE__, "kernels/compute.cpp"),
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = mode,
            .math_approx_mode = false,
            .compile_args = {s.Mt, s.Kt, s.Nt, out_cb_pages, s.relu ? 1u : 0u, variant, qmode(), qlm()},
            .compiler_include_paths = {
                zkml::common::kernel_include_dir(), zkml::common::kernel_path(__FILE__, "kernels")}});
    SetRuntimeArgs(
        program, reader, core, {uint32_t(b.x->address()), uint32_t(b.w->address()), s.Mt * s.Kt, s.Kt * s.Nt});
    SetRuntimeArgs(
        program,
        writer,
        core,
        {uint32_t(b.out->address()),
         uint32_t(b.acc->address()),
         uint32_t(b.sx->address()),
         uint32_t(b.sw->address()),
         uint32_t(b.so->address()),
         out_tiles});
    dist::MeshWorkload wl;
    wl.add_program(dist::MeshCoordinateRange(dev.shape()), std::move(program));
    return wl;
}

struct Result {
    std::vector<uint8_t> out;
    std::vector<int64_t> acc;  // decoded: two's complement for variant 0, sign-magnitude otherwise
    uint32_t so;
};

Result read_result(dist::MeshDevice& dev, Buffers b, uint32_t variant) {
    auto& cq = dev.mesh_command_queue();
    std::vector<uint32_t> out, acc, so;
    dist::EnqueueReadMeshBuffer(cq, out, b.out, true);
    dist::EnqueueReadMeshBuffer(cq, acc, b.acc, true);
    dist::EnqueueReadMeshBuffer(cq, so, b.so, true);
    Result r;
    r.out.resize(out.size() * 4);
    std::memcpy(r.out.data(), out.data(), r.out.size());
    for (uint32_t v : acc) {
        r.acc.push_back(
            variant == 0 ? static_cast<int64_t>(static_cast<int32_t>(v))
                         : ((v >> 31) ? -static_cast<int64_t>(v & 0x7FFFFFFF) : static_cast<int64_t>(v)));
    }
    r.so = so[0];
    return r;
}

int parse_variant(const std::string& n) {
    for (int i = 0; i < kVariants; i++) {
        if (n == kNames[i]) {
            return i;
        }
    }
    std::fprintf(stderr, "unknown variant %s\n", n.c_str());
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: see the top of bench.cpp\n");
        return 2;
    }
    const std::string cmd = argv[1];
    int a = 2;
    int variant = -1;
    if (cmd != "check") {
        variant = parse_variant(argv[a++]);
    }
    Dims s{
        static_cast<uint32_t>(std::stoul(argv[a])),
        static_cast<uint32_t>(std::stoul(argv[a + 1])),
        static_cast<uint32_t>(std::stoul(argv[a + 2])),
        std::string(argv[a + 3]) == "relu"};
    const int iters = cmd == "check" ? 1 : std::atoi(argv[a + 4]);

    SetRootDir(ZKML_TT_METAL_HOME);
    auto dev = dist::MeshDevice::create_unit_mesh(0);
    const Buffers b = make_buffers(*dev, s);
    auto& cq = dev->mesh_command_queue();

    if (cmd == "check") {
        Result res[kVariants];
        for (uint32_t v = 0; v < kVariants; v++) {
            auto wl = build(*dev, b, s, v);
            dist::EnqueueMeshWorkload(cq, wl, false);
            dist::Finish(cq);
            res[v] = read_result(*dev, b, v);
        }
        int bad = 0;
        for (int v = 1; v < kVariants; v++) {
            size_t bo = 0, ba = 0;
            for (size_t i = 0; i < res[0].out.size(); i++) {
                bo += res[v].out[i] != res[0].out[i];
            }
            for (size_t i = 0; i < res[0].acc.size(); i++) {
                ba += res[v].acc[i] != res[0].acc[i];
            }
            const bool so = res[v].so == res[0].so;
            std::printf(
                "%-8s against twos: %zu/%zu out bytes differ, %zu/%zu acc differ, scaler_out %s\n",
                kNames[v],
                bo,
                res[0].out.size(),
                ba,
                res[0].acc.size(),
                so ? "same" : "DIFFERENT");
            bad += (bo || ba || !so);
        }
        dev->close();
        return bad ? 1 : 0;
    }

    auto wl = build(*dev, b, s, variant);
    dist::EnqueueMeshWorkload(cq, wl, false);  // JIT compile, cold caches
    dist::Finish(cq);
    if (cmd == "profile") {
        for (int i = 0; i < iters; i++) {
            dist::EnqueueMeshWorkload(cq, wl, false);
            dist::Finish(cq);
        }
        ReadMeshDeviceProfilerResults(*dev);
        std::printf("profile %s Mt %u Kt %u Nt %u iters %d\n", kNames[variant], s.Mt, s.Kt, s.Nt, iters);
    } else {
        std::vector<double> runs;
        for (int rep = 0; rep < 5; rep++) {
            const auto t0 = Clock::now();
            for (int i = 0; i < iters; i++) {
                dist::EnqueueMeshWorkload(cq, wl, false);
            }
            dist::Finish(cq);
            runs.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / iters);
        }
        std::sort(runs.begin(), runs.end());
        std::printf(
            "wall %-8s Mt %u Kt %u Nt %u %s: %.1f us per run (median of 5 x %d runs)\n",
            kNames[variant],
            s.Mt,
            s.Kt,
            s.Nt,
            s.relu ? "relu" : "none",
            runs[2],
            iters);
    }
    dev->close();
    return 0;
}
