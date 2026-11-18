// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Host program that runs zkml::prim::pot_quantize_per_tensor on the device and checks it against a
// CPU reference (same cases as the CTest test). Use it with tt-metal's debug tools through the run
// targets in CMakeLists.txt, e.g. `cmake --build .build/default --target run_zkml_pot_quant_per_tensor_dprint`.
//
// Computes:
//   q = clamp(round_half_even(x * 2^k), -128, 127)
// Covers FLOAT32/BFLOAT16 input, INT32/INT8 output, ties (RNE) and clamping.
//
// TT_METAL_RUNTIME_ROOT defaults to the TT_METAL_HOME used at configure time; set it to override.

#include <tt-metalium/host_api.hpp>
#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include "operations/pot_quantization/per_tensor/pot_quantize_per_tensor.hpp"

using namespace ttnn;
using tt::tt_metal::DataType;

static int run(MeshDevice* dev, DataType in_dt, DataType out_dt, int k) {
    const uint32_t H = 64, W = 96;
    std::vector<float> x(H * W);
    for (size_t i = 0; i < x.size(); i++) {
        // bf16-exact grid with ties and clamping; fp32 gets an extra non-grid offset
        x[i] = (float((int)(i % 512) - 256)) / 32.0f;
        if (in_dt == DataType::FLOAT32) {
            x[i] += (i % 7) * 1e-3f;
        }
    }
    tt::tt_metal::TensorSpec spec(
        Shape({H, W}), 
        tt::tt_metal::TensorLayout(
            in_dt, 
            tt::tt_metal::PageConfig(Layout::TILE), 
            MemoryConfig{}
        )
    );
    Tensor in = Tensor::from_vector(x, spec, dev);
    Tensor out = zkml::prim::pot_quantize_per_tensor(in, k, out_dt);
    std::vector<int32_t> q;
    if (out_dt == DataType::INT8) {
        auto v = out.to_vector<int8_t>();
        q.assign(v.begin(), v.end());
    } else {
        q = out.to_vector<int32_t>();
    }
    size_t bad = 0;
    for (size_t i = 0; i < x.size(); i++) {
        int32_t ref = std::clamp<int32_t>((int32_t)std::nearbyint(std::ldexp(x[i], k)), -128, 127);
        if (q[i] != ref) {
            if (bad < 5) {
                std::printf("  mismatch i=%zu x=%f got=%d ref=%d\n", i, x[i], q[i], ref);
            }
            bad++;
        }
    }
    std::printf(
        "in=%s out=%s k=%d: %zu/%zu mismatches\n",
        in_dt == DataType::FLOAT32 ? "f32" : "bf16",
        out_dt == DataType::INT8 ? "int8" : "int32",
        k,
        bad,
        x.size());
    return bad != 0;
}

int main() {
    // TT_METAL_RUNTIME_ROOT, if set, takes precedence over this.
    tt::tt_metal::SetRootDir(ZKML_TT_METAL_HOME);

    auto dev = open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);
    int fails = 0;
    // Pass 0 builds each program (cache miss -> create()).
    // Pass 1 repeats every case with new tensors addresses (cache hit -> override_runtime_arguments()).
    for (int pass = 0; pass < 2; pass++) {
        std::printf("pass %d (%s)\n", pass, pass == 0 ? "program cache miss" : "program cache hit");
        for (auto in_dt : {DataType::FLOAT32, DataType::BFLOAT16}) {
            for (auto out_dt : {DataType::INT32, DataType::INT8}) {
                for (int k : {2, 4}) {
                    fails += run(dev.get(), in_dt, out_dt, k);
                }
            }
        }

        const size_t entries = dev->num_program_cache_entries();
        std::printf("program cache entries: %zu\n", entries);
        if (entries != 4) {
            std::printf("  expected 4 program cache entries\n");
            fails++;
        }
    }
    dev->close();
    std::printf(fails ? "FAIL\n" : "PASS\n");
    return fails;
}
