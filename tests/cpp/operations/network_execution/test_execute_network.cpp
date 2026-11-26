// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// execute_network against a host model of the chain: the input x and every weight tensor are quantized per tensor to 8
// bits, each layer is an exact integer matmul, the activation on the accumulator and the per-tensor requantization to 8
// bits, and the requantized output of a layer is the quantized input of the next one. Every scaler, every accumulator
// and the final output bytes must match exactly.

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
#include "device/operations/network_execution/execute_network.hpp"

using namespace ttnn;
using tt::tt_metal::DataType;
using zkml::prim::Activation;

namespace {

struct LayerModel {
    float scaler_w, scaler_out;
    std::vector<int32_t> acc_raw;  // the matmul, before the activation
    std::vector<uint8_t> out;      // sign-magnitude bytes
};
struct NetModel {
    float scaler_x;
    std::vector<LayerModel> layers;
};

int32_t decode_byte(uint8_t b) { return (b & 0x80) ? -static_cast<int32_t>(b & 0x7F) : static_cast<int32_t>(b); }

NetModel host_model(
    const std::vector<float>& x,
    const std::vector<std::vector<float>>& ws,
    uint32_t M,
    uint32_t K0,
    const std::vector<uint32_t>& Ns,
    const std::vector<Activation>& acts) {
    NetModel net;
    const int32_t kx = zkml::reference::pot_k(zkml::reference::absmax(x), 8);
    net.scaler_x = std::ldexp(1.0f, kx);
    std::vector<int32_t> qx = zkml::reference::pot_quantize(x, kx, 8);
    uint32_t K = K0;
    for (size_t l = 0; l < ws.size(); l++) {
        const uint32_t N = Ns[l];
        LayerModel lm;
        const int32_t kw = zkml::reference::pot_k(zkml::reference::absmax(ws[l]), 8);
        lm.scaler_w = std::ldexp(1.0f, kw);
        const auto qw = zkml::reference::pot_quantize(ws[l], kw, 8);
        std::vector<int32_t> acc(static_cast<size_t>(M) * N);
        lm.acc_raw.assign(acc.size(), 0);
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t j = 0; j < N; j++) {
                int64_t sum = 0;
                for (uint32_t k = 0; k < K; k++) {
                    sum += static_cast<int64_t>(qx[i * K + k]) * qw[k * N + j];
                }
                lm.acc_raw[i * N + j] = static_cast<int32_t>(sum);
                acc[i * N + j] = static_cast<int32_t>(acts[l] == Activation::ReLU ? std::max<int64_t>(sum, 0) : sum);
            }
        }
        // The requantization in integers: k = the largest with A * 2^k <= 127, round half to even.
        int64_t A = 0;
        for (int32_t v : acc) {
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
        lm.scaler_out = std::ldexp(1.0f, ko);
        lm.out.resize(acc.size());
        for (size_t i = 0; i < acc.size(); i++) {
            const int64_t mag = std::llabs(acc[i]);
            int64_t q;
            if (ko >= 0) {
                q = ko > 40 ? 0 : mag << ko;
            } else {
                const int s = -ko;
                q = (mag + (int64_t{1} << (s - 1)) - 1 + ((mag >> s) & 1)) >> s;
            }
            lm.out[i] = (acc[i] < 0 && q != 0) ? static_cast<uint8_t>(0x80 | q) : static_cast<uint8_t>(q);
        }
        // The requantized output is the quantized input of the next layer.
        qx.resize(lm.out.size());
        for (size_t i = 0; i < lm.out.size(); i++) {
            qx[i] = decode_byte(lm.out[i]);
        }
        net.layers.push_back(std::move(lm));
        K = N;
    }
    return net;
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

class ExecuteNetwork : public ::testing::Test {
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
    static inline DataType dtype_ = DataType::FLOAT32;

    static Tensor to_device(const std::vector<float>& v, uint32_t h, uint32_t w) {
        const std::vector<float> data = dtype_ == DataType::BFLOAT16 ? round_bf16(v) : v;
        return Tensor::from_vector(
            data,
            tt::tt_metal::TensorSpec(
                Shape({h, w}),
                tt::tt_metal::TensorLayout(dtype_, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{})),
            dev());
    }

    // Weights [K_l, N_l] for the chain K0 -> Ns[0] -> Ns[1] ...; random data, one seed per layer.
    static std::vector<std::vector<float>> make_weights(uint32_t K0, const std::vector<uint32_t>& Ns, uint32_t seed) {
        std::vector<std::vector<float>> ws;
        uint32_t K = K0;
        for (size_t l = 0; l < Ns.size(); l++) {
            ws.push_back(uniform(static_cast<size_t>(K) * Ns[l], seed + 7 * static_cast<uint32_t>(l), -1.f, 1.f));
            K = Ns[l];
        }
        return ws;
    }

    // Runs the network and compares everything with the host model. Returns the number of differences.
    static size_t run(
        const std::string& label,
        const std::vector<float>& x_in,
        const std::vector<std::vector<float>>& ws_in,
        uint32_t M,
        uint32_t K0,
        const std::vector<uint32_t>& Ns,
        const std::vector<Activation>& acts) {
        const bool bf16 = dtype_ == DataType::BFLOAT16;
        const std::vector<float> x = bf16 ? round_bf16(x_in) : x_in;
        std::vector<std::vector<float>> ws;
        for (const auto& w : ws_in) {
            ws.push_back(bf16 ? round_bf16(w) : w);
        }
        Tensor tx = to_device(x, M, K0);
        std::vector<Tensor> tws;
        uint32_t K = K0;
        for (size_t l = 0; l < ws.size(); l++) {
            tws.push_back(to_device(ws[l], K, Ns[l]));
            K = Ns[l];
        }
        const auto r = zkml::prim::execute_network(tx, tws, acts);
        const NetModel m = host_model(x, ws, M, K0, Ns, acts);

        size_t bad = 0;
        auto expect_scaler = [&](const Tensor& t, float want, const std::string& what) {
            const float got = t.to_vector<float>().at(0);
            if (got != want) {
                std::printf("  [%s] %s: got %g want %g\n", label.c_str(), what.c_str(), got, want);
                bad++;
            }
        };
        expect_scaler(r.scaler_x, m.scaler_x, "scaler_x");
        EXPECT_EQ(r.acc.size(), ws.size());
        for (size_t l = 0; l < ws.size(); l++) {
            expect_scaler(r.scaler_w[l], m.layers[l].scaler_w, "scaler_w " + std::to_string(l));
            expect_scaler(r.scaler_out[l], m.layers[l].scaler_out, "scaler_out " + std::to_string(l));
            const auto got_acc = r.acc[l].to_vector<uint32_t>();
            size_t acc_bad = 0;
            for (size_t i = 0; i < got_acc.size() && i < m.layers[l].acc_raw.size(); i++) {
                const int32_t v = (got_acc[i] >> 31) ? -static_cast<int32_t>(got_acc[i] & 0x7FFFFFFF)
                                                     : static_cast<int32_t>(got_acc[i]);
                if (v != m.layers[l].acc_raw[i]) {
                    if (acc_bad < 3) {
                        std::printf(
                            "  [%s] layer %zu acc i=%zu got=%d want=%d\n",
                            label.c_str(),
                            l,
                            i,
                            v,
                            m.layers[l].acc_raw[i]);
                    }
                    acc_bad++;
                }
            }
            if (got_acc.size() != m.layers[l].acc_raw.size()) {
                acc_bad++;
            }
            bad += acc_bad;
        }
        const auto got = r.out.to_vector<uint8_t>();
        const auto& want = m.layers.back().out;
        size_t out_bad = got.size() != want.size() ? 1 : 0;
        for (size_t i = 0; i < std::min(got.size(), want.size()); i++) {
            if (got[i] != want[i]) {
                if (out_bad < 3) {
                    std::printf("  [%s] out i=%zu got=0x%02x want=0x%02x\n", label.c_str(), i, got[i], want[i]);
                }
                out_bad++;
            }
        }
        bad += out_bad;
        std::printf(
            "%-4s %-34s M=%u K0=%u layers=%zu: %zu differences\n",
            bad == 0 ? "ok" : "FAIL",
            label.c_str(),
            M,
            K0,
            ws.size(),
            bad);
        EXPECT_EQ(bad, 0u) << label;
        return bad;
    }
};

class ExecuteNetworkBf16 : public ExecuteNetwork {
protected:
    void SetUp() override { dtype_ = DataType::BFLOAT16; }
    void TearDown() override { dtype_ = DataType::FLOAT32; }
};

}  // namespace

TEST_F(ExecuteNetwork, OneLayerEqualsExecuteLayer) {
    const uint32_t M = 64, K = 64, N = 96;
    const auto x = uniform(static_cast<size_t>(M) * K, 1, -2.f, 2.f);
    const auto ws = make_weights(K, {N}, 100);
    run("one layer, relu", x, ws, M, K, {N}, {Activation::ReLU});
    // The same layer through execute_layer: identical bytes and scalers.
    Tensor tx = to_device(x, M, K), tw = to_device(ws[0], K, N);
    auto [out, acc, sx, sw, so] = zkml::prim::execute_layer(tx, tw, Activation::ReLU);
    const auto net = zkml::prim::execute_network(tx, {tw}, std::vector<Activation>{Activation::ReLU});
    EXPECT_EQ(net.out.to_vector<uint8_t>(), out.to_vector<uint8_t>());
    EXPECT_EQ(net.acc[0].to_vector<uint32_t>(), acc.to_vector<uint32_t>());
    EXPECT_EQ(net.scaler_x.to_vector<float>().at(0), sx.to_vector<float>().at(0));
    EXPECT_EQ(net.scaler_w[0].to_vector<float>().at(0), sw.to_vector<float>().at(0));
    EXPECT_EQ(net.scaler_out[0].to_vector<float>().at(0), so.to_vector<float>().at(0));
}

TEST_F(ExecuteNetwork, OneLayerNoRelu) {
    for (uint32_t d : {32u, 64u}) {
        const auto x = uniform(static_cast<size_t>(d) * d, 1, -2.f, 2.f);
        run("one layer, none", x, make_weights(d, {d}, 100), d, d, {d}, {Activation::None});
    }
}

TEST_F(ExecuteNetwork, TwoLayers) {
    const uint32_t M = 32, K = 32;
    const std::vector<uint32_t> Ns = {32, 32};
    for (auto acts :
         {std::vector<Activation>{Activation::ReLU, Activation::None},
          std::vector<Activation>{Activation::None, Activation::None},
          std::vector<Activation>{Activation::ReLU, Activation::ReLU}}) {
        run("2 layers 32", uniform(M * K, 2, -2.f, 2.f), make_weights(K, Ns, 200), M, K, Ns, acts);
    }
}

TEST_F(ExecuteNetwork, DifferentWidths) {
    // The widths change from layer to layer: the reused buffers (weights, accumulator) are sized for the largest layer.
    const uint32_t M = 64, K = 64;
    const std::vector<uint32_t> Ns = {160, 32, 96, 64};
    run("4 layers, widths 160 32 96 64",
        uniform(static_cast<size_t>(M) * K, 3, -2.f, 2.f),
        make_weights(K, Ns, 300),
        M,
        K,
        Ns,
        {Activation::ReLU, Activation::ReLU, Activation::None, Activation::ReLU});
}

TEST_F(ExecuteNetwork, Wide) {
    // The widest layer first and last, larger batch.
    const uint32_t M = 128, K = 128;
    const std::vector<uint32_t> Ns = {256, 64, 256};
    run("3 layers 256 64 256",
        uniform(static_cast<size_t>(M) * K, 4, -3.f, 3.f),
        make_weights(K, Ns, 400),
        M,
        K,
        Ns,
        {Activation::ReLU, Activation::ReLU, Activation::None});
}

TEST_F(ExecuteNetwork, SingleActivationOverload) {
    // One activation for the network: every layer but the last.
    const uint32_t M = 32, K = 64;
    const std::vector<uint32_t> Ns = {64, 64, 32};
    const auto x = uniform(static_cast<size_t>(M) * K, 5, -2.f, 2.f);
    const auto ws = make_weights(K, Ns, 500);
    Tensor tx = to_device(x, M, K);
    std::vector<Tensor> tws;
    uint32_t k = K;
    for (size_t l = 0; l < ws.size(); l++) {
        tws.push_back(to_device(ws[l], k, Ns[l]));
        k = Ns[l];
    }
    const auto a = zkml::prim::execute_network(tx, tws, Activation::ReLU);
    const auto b = zkml::prim::execute_network(tx, tws, {Activation::ReLU, Activation::ReLU, Activation::None});
    EXPECT_EQ(a.out.to_vector<uint8_t>(), b.out.to_vector<uint8_t>());
    run("single activation", x, ws, M, K, Ns, {Activation::ReLU, Activation::ReLU, Activation::None});
}

TEST_F(ExecuteNetwork, ManyLayers) {
    const uint32_t M = 32, K = 32;
    const std::vector<uint32_t> Ns(6, 32);
    run("6 layers",
        uniform(M * K, 6, -2.f, 2.f),
        make_weights(K, Ns, 600),
        M,
        K,
        Ns,
        std::vector<Activation>(6, Activation::ReLU));
}

TEST_F(ExecuteNetwork, ProgramCache) {
    const uint32_t M = 32, K = 64;
    const std::vector<uint32_t> Ns = {64, 32};
    const std::vector<Activation> acts = {Activation::ReLU, Activation::None};
    run("first call", uniform(M * K, 7, -2.f, 2.f), make_weights(K, Ns, 700), M, K, Ns, acts);
    const size_t entries = dev()->num_program_cache_entries();
    run("cache hit, new data", uniform(M * K, 8, -1.f, 1.f), make_weights(K, Ns, 800), M, K, Ns, acts);
    EXPECT_EQ(dev()->num_program_cache_entries(), entries);
}

TEST_F(ExecuteNetwork, Rejected) {
    const uint32_t M = 32, K = 32;
    Tensor tx = to_device(uniform(M * K, 9, -1.f, 1.f), M, K);
    Tensor w_ok = to_device(uniform(K * 32, 10, -1.f, 1.f), K, 32);
    Tensor w_bad = to_device(uniform(64 * 32, 11, -1.f, 1.f), 64, 32);  // K = 64 but the previous layer has 32 outputs
    EXPECT_ANY_THROW(zkml::prim::execute_network(tx, {w_ok, w_bad}, Activation::ReLU));
    EXPECT_ANY_THROW(zkml::prim::execute_network(tx, {}, Activation::ReLU));
    EXPECT_ANY_THROW(
        zkml::prim::execute_network(tx, {w_ok}, std::vector<Activation>{Activation::ReLU, Activation::None}));
}

TEST_F(ExecuteNetworkBf16, ThreeLayers) {
    const uint32_t M = 64, K = 64;
    const std::vector<uint32_t> Ns = {128, 64, 32};
    run("bf16, 3 layers",
        uniform(static_cast<size_t>(M) * K, 12, -2.f, 2.f),
        make_weights(K, Ns, 900),
        M,
        K,
        Ns,
        {Activation::ReLU, Activation::ReLU, Activation::None});
}
