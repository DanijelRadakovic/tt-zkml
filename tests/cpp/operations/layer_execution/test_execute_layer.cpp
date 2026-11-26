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
#include <cstring>
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
    std::vector<uint8_t> out;      // sign-magnitude bytes
    std::vector<int32_t> acc_raw;  // the matmul, before the activation (the `acc` output)
    std::vector<int32_t> acc;      // after the activation
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
    m.acc_raw.assign(static_cast<size_t>(M) * N, 0);
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            int64_t sum = 0;
            for (uint32_t k = 0; k < K; k++) {
                sum += static_cast<int64_t>(qx[i * K + k]) * qw[k * N + j];
            }
            m.acc_raw[i * N + j] = static_cast<int32_t>(sum);
            if (act == Activation::ReLU) {
                sum = std::max<int64_t>(sum, 0);
            }
            m.acc[i * N + j] = static_cast<int32_t>(sum);
        }
    }
    // The requantization in integers: k = the largest with A * 2^k <= 127, round half to even.
    int64_t A = 0;
    for (int32_t v : m.acc) {
        A = std::max<int64_t>(A, std::llabs(v));
    }
    int32_t ko = 127;
    if (A != 0) {
        int t = 0;
        while ((A >> t) != 0) {
            t++;
        }
        ko = 7 - t;
        if (ko < 0 && A > (int64_t{127} << -ko)) {
            ko--;
        }
    }
    m.scaler_out = std::ldexp(1.0f, ko);
    m.out.resize(m.acc.size());
    for (size_t i = 0; i < m.acc.size(); i++) {
        const int64_t mag = std::llabs(m.acc[i]);
        int64_t q;
        if (ko >= 0) {
            q = ko > 40 ? 0 : mag << ko;
        } else {
            const int s = -ko;
            const int64_t half = int64_t{1} << (s - 1);
            q = (mag + half - 1 + ((mag >> s) & 1)) >> s;
        }
        m.out[i] = (m.acc[i] < 0 && q != 0) ? static_cast<uint8_t>(0x80 | q) : static_cast<uint8_t>(q);
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

// float -> bf16 (round to nearest even) -> float: the values the device sees when the input tensor is bf16.
std::vector<float> round_bf16(std::vector<float> v) {
    for (auto& f : v) {
        uint32_t b;
        std::memcpy(&b, &f, 4);
        b += 0x7FFF + ((b >> 16) & 1);
        b &= 0xFFFF0000u;
        std::memcpy(&f, &b, 4);
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
    // The dtype of x and w in `run`; with BFLOAT16 the inputs are first rounded to bf16 (the host model sees the same values).
    static inline DataType dtype_ = DataType::FLOAT32;

    // Runs one case and compares with the host model. Returns the number of wrong output bytes.
    static size_t run(
        const std::string& label,
        const std::vector<float>& x_in,
        const std::vector<float>& w_in,
        uint32_t M,
        uint32_t K,
        uint32_t N,
        Activation act,
        bool verbose = true) {
        const bool bf16 = dtype_ == DataType::BFLOAT16;
        const std::vector<float> x = bf16 ? round_bf16(x_in) : x_in, w = bf16 ? round_bf16(w_in) : w_in;
        auto spec = [&](uint32_t h, uint32_t wd) {
            return tt::tt_metal::TensorSpec(
                Shape({h, wd}),
                tt::tt_metal::TensorLayout(dtype_, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
        };
        Tensor tx = Tensor::from_vector(x, spec(M, K), dev());
        Tensor tw = Tensor::from_vector(w, spec(K, N), dev());
        auto [out, acc, sx, sw, so] = zkml::prim::execute_layer(tx, tw, act);
        const auto got = out.to_vector<uint8_t>();
        const auto got_acc = acc.to_vector<uint32_t>();
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
        // The accumulator is returned before the activation: negative entries stay negative with ReLU.
        EXPECT_EQ(got_acc.size(), m.acc_raw.size()) << label;
        size_t bad_acc = 0;
        for (size_t i = 0; i < std::min(got_acc.size(), m.acc_raw.size()); i++) {
            const int32_t got_v =
                (got_acc[i] >> 31) ? -static_cast<int32_t>(got_acc[i] & 0x7FFFFFFF) : static_cast<int32_t>(got_acc[i]);
            if (got_v != m.acc_raw[i]) {
                if (bad_acc < 5) {
                    std::printf("  [%s] acc i=%zu got=%d want=%d\n", label.c_str(), i, got_acc[i], m.acc_raw[i]);
                }
                bad_acc++;
            }
        }
        EXPECT_EQ(bad_acc, 0u) << label << ": acc";
        bad += bad_acc;
        if (verbose || bad) {
            std::printf(
                "%-4s %-28s M=%u K=%u N=%u act=%s scalers %g %g %g, %zu/%zu entries wrong (out bytes + acc)\n",
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
                2 * got.size());
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
    // K = 1024: |acc| can reach 2^24 and more.
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

TEST_F(ExecuteLayer, VeryLongK) {
    // K = 2048: all-ones gives acc = 2048 * 127 * 127 = 33M > 2^24, which fp32 could not hold exactly.
    const uint32_t M = 32, K = 2048, N = 32;
    std::vector<float> x(static_cast<size_t>(M) * K, 1.0f), w(static_cast<size_t>(K) * N, 1.0f);
    run("all ones, K = 2048", x, w, M, K, N, Activation::ReLU);
    for (Activation act : {Activation::None, Activation::ReLU}) {
        run("K = 2048",
            uniform(static_cast<size_t>(M) * K, 51, -1.f, 1.f),
            uniform(static_cast<size_t>(K) * N, 52, -1.f, 1.f),
            M,
            K,
            N,
            act);
    }
    // An odd accumulator above 2^24: the integer requantization keeps the low bits.
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = (i % 7 == 0) ? 0.99f : 1.0f;
    }
    run("odd acc, K = 2048", x, w, M, K, N, Activation::None);
}

TEST_F(ExecuteLayer, QuantizationTies) {
    // x * 2^5 = m / 2: every half-integer of [-127, 127] (ties round to even), tiny negatives (round to 0, never "-0"),
    // and the absmax 127 / 32 itself, for both x and w.
    const uint32_t M = 32, K = 64, N = 32;
    auto make = [](size_t n, int salt) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; i++) {
            const int m = static_cast<int>((i * 37 + salt) % 509) - 254;
            v[i] = static_cast<float>(m) / 64.0f;
        }
        v[0] = 127.0f / 32.0f;
        v[1] = -1.0f / 1024.0f;
        v[2] = -1.0f / 128.0f;
        return v;
    };
    for (Activation act : {Activation::None, Activation::ReLU}) {
        run("ties", make(static_cast<size_t>(M) * K, 3), make(static_cast<size_t>(K) * N, 11), M, K, N, act);
    }
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

// bf16 inputs: quantized through UInt8 circular buffers (an Int8 one cannot share a program with a bf16 one).
class ExecuteLayerBf16 : public ExecuteLayer {
protected:
    void SetUp() override { dtype_ = DataType::BFLOAT16; }
    void TearDown() override { dtype_ = DataType::FLOAT32; }
};

TEST_F(ExecuteLayerBf16, SingleTile) {
    const uint32_t M = 32, K = 32, N = 32;
    for (Activation act : {Activation::None, Activation::ReLU}) {
        run("bf16 one tile",
            uniform(static_cast<size_t>(M) * K, 61, -2.f, 2.f),
            uniform(static_cast<size_t>(K) * N, 62, -2.f, 2.f),
            M,
            K,
            N,
            act);
    }
}

TEST_F(ExecuteLayerBf16, Shapes) {
    struct S {
        uint32_t M, K, N;
    };
    for (const S s : {S{32, 64, 32}, S{64, 96, 64}, S{96, 64, 160}, S{160, 32, 32}, S{64, 256, 64}}) {
        for (Activation act : {Activation::None, Activation::ReLU}) {
            run("bf16 shape",
                uniform(static_cast<size_t>(s.M) * s.K, 63, -3.f, 3.f),
                uniform(static_cast<size_t>(s.K) * s.N, 64, -1.f, 1.f),
                s.M,
                s.K,
                s.N,
                act);
        }
    }
}

TEST_F(ExecuteLayerBf16, QuantizationTies) {
    // The same exact half-integers and tiny negatives as the fp32 test; all of them are exact in bf16 (m / 64 with |m| <= 254).
    const uint32_t M = 32, K = 64, N = 32;
    auto make = [](size_t n, int salt) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; i++) {
            v[i] = static_cast<float>(static_cast<int>((i * 37 + salt) % 509) - 254) / 64.0f;
        }
        v[0] = 127.0f / 32.0f;
        v[1] = -1.0f / 1024.0f;
        v[2] = -1.0f / 128.0f;
        return v;
    };
    for (Activation act : {Activation::None, Activation::ReLU}) {
        run("bf16 ties", make(static_cast<size_t>(M) * K, 3), make(static_cast<size_t>(K) * N, 11), M, K, N, act);
    }
}

TEST_F(ExecuteLayerBf16, VeryLongK) {
    const uint32_t M = 32, K = 2048, N = 32;
    std::vector<float> x(static_cast<size_t>(M) * K, 1.0f), w(static_cast<size_t>(K) * N, 1.0f);
    run("bf16 all ones, K = 2048", x, w, M, K, N, Activation::ReLU);
}

TEST_F(ExecuteLayerBf16, MixedDtypeRejected) {
    const uint32_t M = 32, K = 32, N = 32;
    auto spec = [&](uint32_t h, uint32_t wd, DataType t) {
        return tt::tt_metal::TensorSpec(
            Shape({h, wd}), tt::tt_metal::TensorLayout(t, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
    };
    Tensor tx = Tensor::from_vector(std::vector<float>(M * K, 1.0f), spec(M, K, DataType::BFLOAT16), dev());
    Tensor tw = Tensor::from_vector(std::vector<float>(K * N, 1.0f), spec(K, N, DataType::FLOAT32), dev());
    EXPECT_ANY_THROW(zkml::prim::execute_layer(tx, tw, Activation::ReLU));
}
