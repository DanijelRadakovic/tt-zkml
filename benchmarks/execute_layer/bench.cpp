// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Device cycles and wall time of execute_layer (see README.md).
//
//   zkml_execute_layer_bench profile <Mt> <Kt> <Nt> <relu|none> <iters>   run `iters` times; run with
//                                                                       TT_METAL_DEVICE_PROFILER=1 and analyze.py
//   zkml_execute_layer_bench wall    <Mt> <Kt> <Nt> <relu|none> <iters>   host time per call, program cache warm
// Mt, Kt, Nt are the sizes in tiles of 32: x is (32 Mt) x (32 Kt), w is (32 Kt) x (32 Nt).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

#include "device/operations/layer_execution/execute_layer.hpp"

using namespace ttnn;
using tt::tt_metal::DataType;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: %s profile|wall <Mt> <Kt> <Nt> <relu|none> <iters>\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const uint32_t Mt = std::stoul(argv[2]), Kt = std::stoul(argv[3]), Nt = std::stoul(argv[4]);
    const auto act = std::string(argv[5]) == "relu" ? zkml::prim::Activation::ReLU : zkml::prim::Activation::None;
    const int iters = std::atoi(argv[6]);
    const uint32_t M = 32 * Mt, K = 32 * Kt, N = 32 * Nt;

    tt::tt_metal::SetRootDir(ZKML_TT_METAL_HOME);
    auto dev = open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-2.f, 2.f);
    std::vector<float> x(static_cast<size_t>(M) * K), w(static_cast<size_t>(K) * N);
    for (auto& v : x) {
        v = d(rng);
    }
    for (auto& v : w) {
        v = d(rng);
    }
    auto spec = [&](uint32_t h, uint32_t wd) {
        return tt::tt_metal::TensorSpec(
            Shape({h, wd}),
            tt::tt_metal::TensorLayout(DataType::FLOAT32, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
    };
    Tensor tx = Tensor::from_vector(x, spec(M, K), dev.get());
    Tensor tw = Tensor::from_vector(w, spec(K, N), dev.get());
    auto call = [&] { return std::get<0>(zkml::prim::execute_layer(tx, tw, act)); };
    auto sync = [&] { tt::tt_metal::distributed::Finish(dev->mesh_command_queue()); };

    (void)call();  // compile, cache
    sync();
    if (mode == "profile") {
        for (int i = 0; i < iters; i++) {
            (void)call();
            sync();
        }
        tt::tt_metal::ReadMeshDeviceProfilerResults(*dev);
        std::printf("profile Mt %u Kt %u Nt %u %s iters %d\n", Mt, Kt, Nt, argv[5], iters);
    } else {
        std::vector<double> runs;
        for (int rep = 0; rep < 5; rep++) {
            const auto t0 = Clock::now();
            for (int i = 0; i < iters; i++) {
                (void)call();
            }
            sync();
            runs.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / iters);
        }
        std::sort(runs.begin(), runs.end());
        std::printf(
            "wall Mt %u Kt %u Nt %u %s: %.1f us per call (median of 5 x %d calls)\n",
            Mt,
            Kt,
            Nt,
            argv[5],
            runs[2],
            iters);
    }
    dev->close();
    return 0;
}
