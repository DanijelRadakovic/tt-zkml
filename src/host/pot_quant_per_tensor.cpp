// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Host program that runs zkml::prim::pot_quantize_per_tensor on the device and checks it against a
// CPU reference. Handy with tt-metal's debug tools (see "Debugging" in README.md).
//
// Computes, with the scaler derived on the device from the data:
//   Q = 2^(bits - 1) - 1,  k = max{k <= 127 : absmax(x) * 2^k <= Q},
//   q = clamp(round_half_even(x * 2^k), -Q, Q)
//
// Usage: zkml_pot_quant_per_tensor [bit_width ...]     bit widths 8, 16, 32 (default: all three)
//
// TT_METAL_RUNTIME_ROOT defaults to the TT_METAL_HOME used at configure time; set it to override.

#include <tt-metalium/host_api.hpp>
#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "common/pot_reference.hpp"
#include "operations/pot_quantization/per_tensor/pot_quantize_per_tensor.hpp"

using namespace ttnn;
using tt::tt_metal::DataType;

static int run(MeshDevice* dev, DataType in_dt, DataType out_dt, uint32_t bits) {
    const uint32_t H = 64, W = 96;
    std::vector<float> data(H * W);
    for (size_t i = 0; i < data.size(); i++) {
        // bf16-exact grid with ties; fp32 gets an extra non-grid offset
        data[i] = (float((int)(i % 512) - 256)) / 32.0f;
        if (in_dt == DataType::FLOAT32) {
            data[i] += (i % 7) * 1e-3f;
        }
    }
    tt::tt_metal::TensorSpec spec(
        Shape({H, W}), tt::tt_metal::TensorLayout(in_dt, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
    Tensor in = Tensor::from_vector(data, spec, dev);
    const std::vector<float> x = in.to_vector<float>();  // what the device sees (bf16 rounds)

    auto [out, scaler] = zkml::prim::pot_quantize_per_tensor(in, bits, out_dt);
    std::vector<int32_t> q;
    if (out_dt == DataType::INT8) {
        auto v = out.to_vector<int8_t>();
        q.assign(v.begin(), v.end());
    } else {
        q = out.to_vector<int32_t>();
    }
    const float got_scaler = scaler.to_vector<float>().at(0);

    const float absmax = zkml::reference::absmax(x);
    const int32_t k = zkml::reference::pot_k(absmax, bits);
    const std::vector<int32_t> ref = zkml::reference::pot_quantize(x, k, bits);

    size_t bad = 0;
    for (size_t i = 0; i < x.size(); i++) {
        if (q[i] != ref[i]) {
            if (bad < 5) {
                std::printf("  mismatch i=%zu x=%f got=%d ref=%d\n", i, x[i], q[i], ref[i]);
            }
            bad++;
        }
    }
    const bool scaler_ok = got_scaler == std::ldexp(1.0f, k);
    std::printf(
        "in=%s out=%s bits=%u: absmax=%g, device scaler=%g (k=%g), reference k=%d%s, %zu/%zu mismatches\n",
        in_dt == DataType::FLOAT32 ? "f32" : "bf16",
        out_dt == DataType::INT8 ? "int8" : "int32",
        bits,
        absmax,
        got_scaler,
        std::log2(got_scaler),
        k,
        scaler_ok ? "" : " (SCALER MISMATCH)",
        bad,
        x.size());
    return bad != 0 || !scaler_ok;
}

int main(int argc, char** argv) {
    std::vector<uint32_t> widths;
    for (int i = 1; i < argc; i++) {
        const auto b = static_cast<uint32_t>(std::strtoul(argv[i], nullptr, 10));
        if (b != 8 && b != 16 && b != 32) {
            std::fprintf(stderr, "usage: %s [bit_width ...]   (bit widths: 8, 16, 32)\n", argv[0]);
            return 2;
        }
        widths.push_back(b);
    }
    if (widths.empty()) {
        widths = {8, 16, 32};
    }

    // TT_METAL_RUNTIME_ROOT, if set, takes precedence over this.
    tt::tt_metal::SetRootDir(ZKML_TT_METAL_HOME);

    auto dev = open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);
    int fails = 0;
    for (uint32_t bits : widths) {
        for (auto in_dt : {DataType::FLOAT32, DataType::BFLOAT16}) {
            fails += run(dev.get(), in_dt, DataType::INT32, bits);
            if (bits == 8) {
                fails += run(dev.get(), in_dt, DataType::INT8, bits);
            }
        }
    }
    dev->close();
    std::printf(fails ? "FAIL\n" : "PASS\n");
    return fails;
}
