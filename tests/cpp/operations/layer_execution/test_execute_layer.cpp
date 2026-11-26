// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// execute_layer against a host model of the whole pipeline: per-tensor PoT quantization of x and W to 8 bits, the exact
// integer matmul, the activation on the accumulator, and the per-tensor requantization to 8 bits. The scalers and every
// output byte must match exactly.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

#include "common/pot_reference.hpp"
#include "device/operations/layer_execution/execute_layer.hpp"

using namespace ttnn;
using tt::tt_metal::DataType;
using zkml::prim::Activation;

namespace {

struct Model {
    float scaler_x, scaler_w, scaler_out;
    std::vector<uint8_t> out;  // sign-magnitude bytes
    std::vector<int32_t> acc;  // after the activation
};

Model host_model(
    const std::vector<float>& x, const std::vector<float>& w, uint32_t M, uint32_t K, uint32_t N, Activation act) {
    const int32_t kx = zkml::reference::pot_k(zkml::reference::absmax(x), 8);
    const int32_t kw = zkml::reference::pot_k(zkml::reference::absmax(w), 8);
    const auto qx = zkml::reference::pot_quantize(x, kx, 8);
    const auto qw = zkml::reference::pot_quantize(w, kw, 8);
    Model m;
    m.scaler_x = std::ldexp(1.0f, kx);
    m.scaler_w = std::ldexp(1.0f, kw);
    m.acc.assign(static_cast<size_t>(M) * N, 0);
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            int64_t sum = 0;
            for (uint32_t k = 0; k < K; k++) {
                sum += static_cast<int64_t>(qx[i * K + k]) * qw[k * N + j];
            }
            if (act == Activation::ReLU) {
                sum = std::max<int64_t>(sum, 0);
            }
            m.acc[i * N + j] = static_cast<int32_t>(sum);
        }
    }
    std::vector<float> acc_f(m.acc.begin(), m.acc.end());
    const int32_t ko = zkml::reference::pot_k(zkml::reference::absmax(acc_f), 8);
    m.scaler_out = std::ldexp(1.0f, ko);
    const auto q = zkml::reference::pot_quantize(acc_f, ko, 8);
    m.out.resize(q.size());
    for (size_t i = 0; i < q.size(); i++) {
        m.out[i] = q[i] < 0 ? static_cast<uint8_t>(0x80 | (-q[i])) : static_cast<uint8_t>(q[i]);
    }
    return m;
}

std::vector<float> uniform(size_t n, uint32_t seed, float lo, float hi) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& e : v) {
        e = d(rng);
    }
    return v;
}

class ExecuteLayer : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tt::tt_metal::SetRootDir(ZKML_TT_METAL_HOME);
        device_ = open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);
    }
    static void TearDownTestSuite() {
        device_->close();
        device_.reset();
    }
    static MeshDevice* dev() { return device_.get(); }
    static inline std::shared_ptr<MeshDevice> device_;

    // Runs one case and compares with the host model. Returns the number of wrong output bytes.
    static size_t run(
        const std::string& label,
        const std::vector<float>& x,
        const std::vector<float>& w,
        uint32_t M,
        uint32_t K,
        uint32_t N,
        Activation act,
        bool verbose = true) {
        auto spec = [&](uint32_t h, uint32_t wd) {
            return tt::tt_metal::TensorSpec(
                Shape({h, wd}),
                tt::tt_metal::TensorLayout(DataType::FLOAT32, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
        };
        Tensor tx = Tensor::from_vector(x, spec(M, K), dev());
        Tensor tw = Tensor::from_vector(w, spec(K, N), dev());
        auto [out, sx, sw, so] = zkml::prim::execute_layer(tx, tw, act);
        const auto got = out.to_vector<uint8_t>();
        const float gsx = sx.to_vector<float>().at(0), gsw = sw.to_vector<float>().at(0),
                    gso = so.to_vector<float>().at(0);
        const Model m = host_model(x, w, M, K, N, act);

        EXPECT_EQ(gsx, m.scaler_x) << label << ": scaler of x";
        EXPECT_EQ(gsw, m.scaler_w) << label << ": scaler of w";
        EXPECT_EQ(gso, m.scaler_out) << label << ": scaler of the requantization";
        EXPECT_EQ(got.size(), m.out.size()) << label;
        size_t bad = 0;
        for (size_t i = 0; i < std::min(got.size(), m.out.size()); i++) {
            if (got[i] != m.out[i]) {
                if (bad < 5) {
                    std::printf(
                        "  [%s] i=%zu got=0x%02x want=0x%02x (acc %d)\n", label.c_str(), i, got[i], m.out[i], m.acc[i]);
                }
                bad++;
            }
        }
        EXPECT_EQ(bad, 0u) << label;
        if (verbose || bad) {
            std::printf(
                "%-4s %-28s M=%u K=%u N=%u act=%s scalers %g %g %g, %zu/%zu bytes wrong\n",
                bad ? "FAIL" : "ok",
                label.c_str(),
                M,
                K,
                N,
                act == Activation::ReLU ? "relu" : "none",
                gsx,
                gsw,
                gso,
                bad,
                got.size());
        }
        return bad;
    }
};

}  // namespace

TEST_F(ExecuteLayer, SingleTile) {
    for (Activation act : {Activation::None, Activation::ReLU}) {
        run("one tile", uniform(32 * 32, 1, -2.f, 2.f), uniform(32 * 32, 2, -2.f, 2.f), 32, 32, 32, act);
    }
}

TEST_F(ExecuteLayer, Shapes) {
    struct Shape3 {
        uint32_t Mt, Kt, Nt;
    };
    for (const Shape3 s : {Shape3{1, 2, 1}, {2, 3, 2}, {2, 4, 3}, {3, 2, 5}, {1, 1, 4}, {5, 1, 1}, {2, 8, 2}}) {
        const uint32_t M = 32 * s.Mt, K = 32 * s.Kt, N = 32 * s.Nt;
        for (Activation act : {Activation::None, Activation::ReLU}) {
            run("shape",
                uniform(static_cast<size_t>(M) * K, 10 + s.Mt, -3.f, 3.f),
                uniform(static_cast<size_t>(K) * N, 20 + s.Nt, -1.f, 1.f),
                M,
                K,
                N,
                act);
        }
    }
}

TEST_F(ExecuteLayer, LongestK) {
    // K = 1024: |acc| can reach 1024 * 127 * 127 < 2^24, the limit of the exact int32 -> fp32 conversion.
    const uint32_t M = 64, K = 1024, N = 64;
    for (Activation act : {Activation::None, Activation::ReLU}) {
        run("K = 1024",
            uniform(static_cast<size_t>(M) * K, 31, -1.f, 1.f),
            uniform(static_cast<size_t>(K) * N, 32, -1.f, 1.f),
            M,
            K,
            N,
            act);
    }
    // The worst case of the accumulator: every product is 127 * 127 and all add up.
    std::vector<float> x(static_cast<size_t>(M) * K, 1.0f), w(static_cast<size_t>(K) * N, 1.0f);
    run("all ones, K = 1024", x, w, M, K, N, Activation::ReLU);
}

TEST_F(ExecuteLayer, AllNegativeReLU) {
    // Every accumulator entry is <= 0: the ReLU zeroes them all, absmax = 0, k = 127, and the output is all zero.
    const uint32_t M = 32, K = 64, N = 32;
    std::vector<float> x(static_cast<size_t>(M) * K, 1.0f), w(static_cast<size_t>(K) * N, -1.0f);
    run("relu of negatives", x, w, M, K, N, Activation::ReLU);
    run("negatives, no activation", x, w, M, K, N, Activation::None);
}

TEST_F(ExecuteLayer, ProgramCache) {
    const uint32_t M = 64, K = 96, N = 32;
    const auto x = uniform(static_cast<size_t>(M) * K, 41, -2.f, 2.f),
               w = uniform(static_cast<size_t>(K) * N, 42, -2.f, 2.f);
    run("first call", x, w, M, K, N, Activation::ReLU, false);
    const size_t entries = dev()->num_program_cache_entries();
    run("cache hit, new buffers", x, w, M, K, N, Activation::ReLU, false);
    EXPECT_EQ(dev()->num_program_cache_entries(), entries);
    run("other activation: a new program", x, w, M, K, N, Activation::None, false);
    EXPECT_EQ(dev()->num_program_cache_entries(), entries + 1);
}
