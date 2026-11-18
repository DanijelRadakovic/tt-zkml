// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// On-device test for zkml::prim::pot_quantize_per_tensor against a CPU reference:
//   Q = 2^(bits - 1) - 1,  k = max{k <= 127 : absmax(x) * 2^k <= Q},
//   q = clamp(round_half_even(x * 2^k), -Q, Q)
// Checks both outputs: the scaler 2^k the device derived from the data, and q.
//
// Covers:
//   - FLOAT32/BFLOAT16 input, bit widths 8/16/32, INT32 output (all widths) and INT8 (8 bits)
//   - data at several magnitudes, so k is positive, negative, and large enough that
//     |x * 2^k| >= 2^23 (32 bits), plus ties (RNE) on the grid values
//   - absmax set by a positive or a negative element
//   - absmax exactly Q (k = 0) and just above it (k = -1)
//   - an all-zero tensor (k = 127, q = 0)
//   - the absmax in every one of the 1024 positions of a tile, and in every tile of a multi-tile
//     tensor, which exercises the SFPU per-lane accumulation and the 32-lane reduction
//   - the program cache: a second pass over every case must hit cached programs
//
// TT_METAL_RUNTIME_ROOT defaults to the TT_METAL_HOME used at configure time; set it to override.

#include <tt-metalium/host_api.hpp>
#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "common/pot_reference.hpp"
#include "operations/pot_quantization/per_tensor/pot_quantize_per_tensor.hpp"

using namespace ttnn;
using tt::tt_metal::DataType;

namespace {

const char* name_of(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32: return "f32";
        case DataType::BFLOAT16: return "bf16";
        case DataType::INT32: return "int32";
        case DataType::INT8: return "int8";
        default: return "?";
    }
}

struct Result {
    bool ok;
    int32_t k;
    std::vector<int32_t> q;
};

// Runs one case. `verbose` prints a line even when the case passes.
Result run(
    MeshDevice* dev,
    const std::string& label,
    uint32_t H,
    uint32_t W,
    const std::vector<float>& data,
    DataType in_dt,
    DataType out_dt,
    uint32_t bits,
    bool verbose = true) {
    tt::tt_metal::TensorSpec spec(
        Shape({H, W}), tt::tt_metal::TensorLayout(in_dt, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
    Tensor in = Tensor::from_vector(data, spec, dev);
    // The reference uses the values the device actually sees (BFLOAT16 rounds the input).
    const std::vector<float> x = in.to_vector<float>();

    auto [out, scaler] = zkml::prim::pot_quantize_per_tensor(in, bits, out_dt);

    std::vector<int32_t> q;
    if (out_dt == DataType::INT8) {
        auto v = out.to_vector<int8_t>();
        q.assign(v.begin(), v.end());
    } else {
        q = out.to_vector<int32_t>();
    }
    const float got_scaler = scaler.to_vector<float>().at(0);

    const int32_t k = zkml::reference::pot_k(zkml::reference::absmax(x), bits);
    const float ref_scaler = std::ldexp(1.0f, k);
    const std::vector<int32_t> ref = zkml::reference::pot_quantize(x, k, bits);

    bool ok = true;
    if (got_scaler != ref_scaler) {
        std::printf(
            "  [%s] scaler mismatch: got %g (k=%g) expected %g (k=%d), absmax=%g\n",
            label.c_str(),
            got_scaler,
            std::log2(got_scaler),
            ref_scaler,
            k,
            zkml::reference::absmax(x));
        ok = false;
    }
    size_t bad = 0;
    for (size_t i = 0; i < x.size(); i++) {
        if (q[i] != ref[i]) {
            if (bad < 5) {
                std::printf(
                    "  [%s] mismatch i=%zu x=%.9g got=%d ref=%d\n", label.c_str(), i, x[i], q[i], ref[i]);
            }
            bad++;
        }
    }
    ok = ok && bad == 0;
    if (verbose || !ok) {
        std::printf(
            "%-4s %-32s in=%-4s out=%-5s bits=%-2u k=%-4d %zu/%zu mismatches\n",
            ok ? "ok" : "FAIL",
            label.c_str(),
            name_of(in_dt),
            name_of(out_dt),
            bits,
            k,
            bad,
            x.size());
    }
    return {ok, k, std::move(q)};
}

// bf16-exact grid in [-8, 8) with ties at small k; FLOAT32 adds a non-grid offset.
std::vector<float> grid(uint32_t n, DataType in_dt, float multiplier) {
    std::vector<float> x(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = static_cast<float>(static_cast<int>(i % 512) - 256) / 32.0f;
        if (in_dt == DataType::FLOAT32) {
            x[i] += static_cast<float>(i % 7) * 1e-3f;
        }
        x[i] *= multiplier;
    }
    return x;
}

// Small background with one large element at `pos`.
std::vector<float> spike(uint32_t n, uint32_t pos, float value) {
    std::vector<float> x(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 1e-3f;
    }
    x[pos] = value;
    return x;
}

struct Config {
    DataType in_dt;
    DataType out_dt;
    uint32_t bits;
};

std::vector<Config> all_configs() {
    std::vector<Config> configs;
    for (auto in_dt : {DataType::FLOAT32, DataType::BFLOAT16}) {
        configs.push_back({in_dt, DataType::INT32, 8});
        configs.push_back({in_dt, DataType::INT8, 8});
        configs.push_back({in_dt, DataType::INT32, 16});
        configs.push_back({in_dt, DataType::INT32, 32});
    }
    return configs;
}

// Every case except the position sweeps. Returns the quantized outputs, in order, so a second
// pass can check that the cached programs give the same results.
int run_matrix(MeshDevice* dev, std::vector<std::vector<int32_t>>& outputs) {
    constexpr uint32_t H = 64, W = 96;
    int fails = 0;
    auto record = [&](Result r) {
        fails += !r.ok;
        outputs.push_back(std::move(r.q));
    };

    for (const auto& c : all_configs()) {
        // 1: absmax 8 from a negative element. -1: absmax 8 from a positive one.
        // 16: absmax 128, above Q for 8 bits. 1e-3: large k. 3e4: negative k.
        for (float m : {1.0f, -1.0f, 16.0f, 1e-3f, 3e4f}) {
            char label[64];
            std::snprintf(label, sizeof label, "grid x %g", m);
            record(run(dev, label, H, W, grid(H * W, c.in_dt, m), c.in_dt, c.out_dt, c.bits));
        }

        // absmax exactly Q gives k = 0; Q + 0.5 needs k = -1 (the device's k - 1 correction).
        // BFLOAT16 rounds 32767 up to 32768, so its 16-bit "Q" case already lands on k = -1.
        // Q = 2^31 - 1 is not an fp32, so the 32-bit case uses the fp32 just below it; no fp32 lies
        // between 2^31 - 1 and 2^31, so the 32-bit correction never fires and has no Q + 0.5 case.
        const float q = c.bits == 32 ? 2147483520.0f : static_cast<float>(zkml::reference::pot_qmax(c.bits));
        record(run(dev, "absmax = Q", H, W, spike(H * W, 1234, q), c.in_dt, c.out_dt, c.bits));
        if (c.bits != 32) {
            record(run(dev, "absmax = Q + 0.5", H, W, spike(H * W, 1234, q + 0.5f), c.in_dt, c.out_dt, c.bits));
        }

        record(run(dev, "all zeros", H, W, std::vector<float>(H * W, 0.0f), c.in_dt, c.out_dt, c.bits));
    }
    return fails;
}

// The absmax at every position of one tile, then in every tile of a multi-tile tensor. The
// background alone would give a much larger k, so a lane the reduction missed shows up as a
// scaler mismatch.
int run_position_sweeps(MeshDevice* dev) {
    int fails = 0;
    size_t cases = 0;

    constexpr uint32_t T = 32;
    for (uint32_t pos = 0; pos < T * T; pos++) {
        const float value = (pos % 2) ? -5.0f : 5.0f;
        char label[64];
        std::snprintf(label, sizeof label, "1-tile spike at %u", pos);
        fails += !run(dev, label, T, T, spike(T * T, pos, value), DataType::FLOAT32, DataType::INT32, 8, false).ok;
        cases++;
    }

    // 3 x 2 tiles. Row-major positions of the first and last element of each tile.
    constexpr uint32_t H = 96, W = 64;
    for (uint32_t tr = 0; tr < H / T; tr++) {
        for (uint32_t tc = 0; tc < W / T; tc++) {
            for (uint32_t corner : {0u, T * T - 1}) {
                const uint32_t r = tr * T + corner / T;
                const uint32_t col = tc * T + corner % T;
                char label[64];
                std::snprintf(label, sizeof label, "6-tile spike at (%u, %u)", r, col);
                fails += !run(dev, label, H, W, spike(H * W, r * W + col, -5.0f), DataType::FLOAT32,
                              DataType::INT32, 8, false).ok;
                cases++;
            }
        }
    }
    std::printf("%-4s position sweeps: %zu cases, %d failed\n", fails ? "FAIL" : "ok", cases, fails);
    return fails;
}

}  // namespace

int main() {
    // TT_METAL_RUNTIME_ROOT, if set, takes precedence over this.
    tt::tt_metal::SetRootDir(ZKML_TT_METAL_HOME);

    auto dev = open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);
    int fails = 0;

    // Pass 0 builds each program (cache miss -> create()). Pass 1 repeats every case with new
    // tensor addresses (cache hit -> override_runtime_arguments()) and must add no cache entry.
    std::vector<std::vector<int32_t>> outputs[2];
    size_t entries[2] = {0, 0};
    for (int pass = 0; pass < 2; pass++) {
        std::printf("pass %d (%s)\n", pass, pass == 0 ? "program cache miss" : "program cache hit");
        fails += run_matrix(dev.get(), outputs[pass]);
        entries[pass] = dev->num_program_cache_entries();
        std::printf("program cache entries: %zu\n", entries[pass]);
    }
    if (entries[1] != entries[0]) {
        std::printf("FAIL the second pass added %zu program cache entries\n", entries[1] - entries[0]);
        fails++;
    }
    if (outputs[0] != outputs[1]) {
        std::printf("FAIL the cached programs gave different results\n");
        fails++;
    }

    fails += run_position_sweeps(dev.get());

    dev->close();
    std::printf(fails ? "FAIL\n" : "PASS\n");
    return fails;
}
