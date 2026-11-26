// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// The C ABI (include/tt_zkml/tt_zkml.h): versions, errors, the host function, and a forward pass checked against a host reference.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "tt_zkml/tt_zkml.h"

namespace {

std::vector<float> uniform(size_t n, uint32_t seed, float lo, float hi) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& e : v) {
        e = d(rng);
    }
    return v;
}

// One layer of the quantized network on the host (the definition the device implements), no activation.
struct HostLayer {
    float sx, sw, so;
    std::vector<int32_t> acc;
    std::vector<uint8_t> out;
};

HostLayer host_layer(const std::vector<float>& x, const std::vector<float>& w, size_t M, size_t K, size_t N) {
    auto scale = [](const std::vector<float>& v) {
        float m = 0.f;
        for (float e : v) {
            m = std::max(m, std::fabs(e));
        }
        int32_t k = 0;
        EXPECT_EQ(ttzk_pot_k(m, 8, &k), TTZK_OK);
        return std::ldexp(1.0f, k);
    };
    HostLayer r;
    r.sx = scale(x);
    r.sw = scale(w);
    std::vector<int32_t> qx(M * K), qw(K * N);
    for (size_t i = 0; i < qx.size(); i++) {
        qx[i] = static_cast<int32_t>(std::nearbyintf(x[i] * r.sx));
    }
    for (size_t i = 0; i < qw.size(); i++) {
        qw[i] = static_cast<int32_t>(std::nearbyintf(w[i] * r.sw));
    }
    r.acc.assign(M * N, 0);
    int32_t amax = 0;
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            int64_t s = 0;
            for (size_t k = 0; k < K; k++) {
                s += static_cast<int64_t>(qx[i * K + k]) * qw[k * N + j];
            }
            r.acc[i * N + j] = static_cast<int32_t>(s);
            amax = std::max(amax, std::abs(r.acc[i * N + j]));
        }
    }
    // the requantization: k = the largest with amax * 2^k <= 127, round half to even, in integers
    int32_t ko = 127;
    if (amax != 0) {
        int t = 0;
        while ((static_cast<int64_t>(amax) >> t) != 0) {
            t++;
        }
        ko = 7 - t;
        if (ko < 0 && static_cast<int64_t>(amax) > (int64_t{127} << -ko)) {
            ko--;
        }
    }
    r.so = std::ldexp(1.0f, ko);
    r.out.resize(M * N);
    for (size_t i = 0; i < M * N; i++) {
        const int32_t a = r.acc[i];
        const int64_t mag = a < 0 ? -static_cast<int64_t>(a) : a;
        int64_t q;
        if (ko >= 0) {
            q = ko > 40 ? 0 : mag << ko;
        } else {
            const int sh = -ko;
            q = (mag + (int64_t{1} << (sh - 1)) - 1 + ((mag >> sh) & 1)) >> sh;
        }
        r.out[i] = static_cast<uint8_t>((a < 0 && q != 0) ? (0x80 | q) : q);
    }
    return r;
}

}  // namespace

TEST(Abi, Versions) {
    EXPECT_EQ(ttzk_abi_version(), TTZK_ABI_VERSION);
    EXPECT_GT(std::strlen(ttzk_version()), 0u);
}

TEST(Abi, PotK) {
    int32_t k = 0;
    ASSERT_EQ(ttzk_pot_k(1.0f, 8, &k), TTZK_OK);
    EXPECT_EQ(k, 6);
    ASSERT_EQ(ttzk_pot_k(127.0f, 8, &k), TTZK_OK);
    EXPECT_EQ(k, 0);
    ASSERT_EQ(ttzk_pot_k(128.0f, 8, &k), TTZK_OK);
    EXPECT_EQ(k, -1);
    ASSERT_EQ(ttzk_pot_k(0.0f, 8, &k), TTZK_OK);
    EXPECT_EQ(k, 127);
    ASSERT_EQ(ttzk_pot_k(1.0f, 16, &k), TTZK_OK);
    EXPECT_EQ(k, 14);
}

TEST(Abi, Errors) {
    int32_t k = 0;
    EXPECT_EQ(ttzk_pot_k(1.0f, 12, &k), TTZK_ERR_INVALID_ARGUMENT);
    EXPECT_GT(std::strlen(ttzk_last_error()), 0u);
    EXPECT_EQ(ttzk_pot_k(1.0f, 8, nullptr), TTZK_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(ttzk_pot_k(-1.0f, 8, &k), TTZK_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(ttzk_num_devices(nullptr), TTZK_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(ttzk_device_open(0, nullptr), TTZK_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(ttzk_model_create(nullptr, nullptr, 0, TTZK_FLOAT32, nullptr), TTZK_ERR_INVALID_ARGUMENT);
    ttzk_device_close(nullptr);  // allowed
    ttzk_model_destroy(nullptr);
    ttzk_trace_destroy(nullptr);
    ASSERT_EQ(ttzk_pot_k(1.0f, 8, &k), TTZK_OK);  // a success clears the error
    EXPECT_STREQ(ttzk_last_error(), "");
}

TEST(Abi, DeviceCount) {
    uint32_t n = 99;
    ASSERT_EQ(ttzk_num_devices(&n), TTZK_OK) << ttzk_last_error();
    EXPECT_LT(n, 99u);
}

// The whole path with a device: open, create a model, load parameters, forward, read the trace, release. One layer, compared with the host
// reference bit for bit; then two layers for the shapes and for rejected arguments.
TEST(Abi, ForwardOnDevice) {
    const uint32_t M = 32, K = 64, N = 32;
    ttzk_device* dev = nullptr;
    ASSERT_EQ(ttzk_device_open(0, &dev), TTZK_OK) << ttzk_last_error();

    {  // one layer, fp32, no activation
        const ttzk_layer layer = {K, N, 0};
        ttzk_model* model = nullptr;
        ASSERT_EQ(ttzk_model_create(dev, &layer, 1, TTZK_FLOAT32, &model), TTZK_OK) << ttzk_last_error();
        const auto x = uniform(static_cast<size_t>(M) * K, 1, -2.f, 2.f);
        const auto w = uniform(static_cast<size_t>(K) * N, 2, -1.f, 1.f);
        ttzk_trace* trace = nullptr;
        EXPECT_EQ(ttzk_model_forward(model, x.data(), x.size(), M, &trace), TTZK_ERR_INVALID_ARGUMENT);  // no parameters yet
        EXPECT_EQ(ttzk_model_load_parameters(model, w.data(), w.size() - 1), TTZK_ERR_INVALID_ARGUMENT);  // wrong count
        ASSERT_EQ(ttzk_model_load_parameters(model, w.data(), w.size()), TTZK_OK) << ttzk_last_error();
        EXPECT_EQ(ttzk_model_forward(model, x.data(), x.size(), 16, &trace), TTZK_ERR_INVALID_ARGUMENT);  // batch not a multiple of 32
        ASSERT_EQ(ttzk_model_forward(model, x.data(), x.size(), M, &trace), TTZK_OK) << ttzk_last_error();

        ttzk_trace_info info{};
        ASSERT_EQ(ttzk_trace_get_info(trace, &info), TTZK_OK);
        EXPECT_EQ(info.batch, M);
        EXPECT_EQ(info.in_features, K);
        EXPECT_EQ(info.layers, 1u);
        ttzk_trace_layer l0{};
        ASSERT_EQ(ttzk_trace_get_layer(trace, 0, &l0), TTZK_OK);
        EXPECT_EQ(ttzk_trace_get_layer(trace, 1, &l0), TTZK_ERR_INVALID_ARGUMENT);
        ASSERT_EQ(ttzk_trace_get_layer(trace, 0, &l0), TTZK_OK);
        const uint8_t* out = nullptr;
        size_t out_len = 0;
        ASSERT_EQ(ttzk_trace_get_output(trace, &out, &out_len), TTZK_OK);

        const HostLayer ref = host_layer(x, w, M, K, N);
        EXPECT_EQ(info.scaler_x, ref.sx);
        EXPECT_EQ(l0.scaler_w, ref.sw);
        EXPECT_EQ(l0.scaler_out, ref.so);
        ASSERT_EQ(l0.acc_len, ref.acc.size());
        size_t bad = 0;
        for (size_t i = 0; i < ref.acc.size(); i++) {
            const uint32_t raw = l0.acc[i];
            const int32_t v = (raw >> 31) ? -static_cast<int32_t>(raw & 0x7FFFFFFFu) : static_cast<int32_t>(raw);
            bad += v != ref.acc[i];
        }
        EXPECT_EQ(bad, 0u) << "accumulator differences";
        ASSERT_EQ(out_len, ref.out.size());
        EXPECT_EQ(std::memcmp(out, ref.out.data(), out_len), 0);
        ttzk_trace_destroy(trace);
        ttzk_model_destroy(model);
    }
    {  // two layers, bf16: the shapes of the trace
        const ttzk_layer layers[2] = {{64, 128, 1}, {128, 32, 0}};
        ttzk_model* model = nullptr;
        ASSERT_EQ(ttzk_model_create(dev, layers, 2, TTZK_BFLOAT16, &model), TTZK_OK) << ttzk_last_error();
        const ttzk_layer bad_chain[2] = {{64, 128, 1}, {64, 32, 0}};
        ttzk_model* none = nullptr;
        EXPECT_EQ(ttzk_model_create(dev, bad_chain, 2, TTZK_BFLOAT16, &none), TTZK_ERR_INVALID_ARGUMENT);
        EXPECT_EQ(none, nullptr);
        const auto w = uniform(64 * 128 + 128 * 32, 3, -1.f, 1.f);
        ASSERT_EQ(ttzk_model_load_parameters(model, w.data(), w.size()), TTZK_OK) << ttzk_last_error();
        const auto x = uniform(64 * 64, 4, -2.f, 2.f);
        ttzk_trace* trace = nullptr;
        ASSERT_EQ(ttzk_model_forward(model, x.data(), x.size(), 64, &trace), TTZK_OK) << ttzk_last_error();
        ttzk_trace_layer a{}, b{};
        ASSERT_EQ(ttzk_trace_get_layer(trace, 0, &a), TTZK_OK);
        ASSERT_EQ(ttzk_trace_get_layer(trace, 1, &b), TTZK_OK);
        EXPECT_EQ(a.out_features, 128u);
        EXPECT_EQ(b.out_features, 32u);
        EXPECT_EQ(a.acc_len, 64u * 128u);
        EXPECT_EQ(b.acc_len, 64u * 32u);
        const uint8_t* out = nullptr;
        size_t out_len = 0;
        ASSERT_EQ(ttzk_trace_get_output(trace, &out, &out_len), TTZK_OK);
        EXPECT_EQ(out_len, 64u * 32u);
        ttzk_trace_destroy(trace);
        ttzk_model_destroy(model);
    }
    ttzk_device_close(dev);
}
