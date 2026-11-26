// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// The public C++ API (src/api): zkml::Model::forward must give the trace of zkml::prim::execute_network on the same tensors, and reject what it
// cannot run.

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "api/tt_zkml.hpp"

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

// One device for all the tests of the file (opening it is slow), on the thread that runs them.
class ModelApi : public ::testing::Test {
protected:
    static void SetUpTestSuite() { device_ = std::make_unique<zkml::Device>(0); }
    static void TearDownTestSuite() { device_.reset(); }
    static inline std::unique_ptr<zkml::Device> device_;
};

}  // namespace

TEST_F(ModelApi, DeviceCount) { EXPECT_GE(zkml::Device::count(), 1u); }

TEST_F(ModelApi, ForwardEqualsExecuteNetwork) {
    const uint32_t M = 64;
    const std::vector<zkml::LayerSpec> layers = {{64, 128, true}, {128, 64, true}, {64, 32, false}};
    for (auto dtype : {zkml::DataType::Float32, zkml::DataType::BFloat16}) {
        zkml::Model model(*device_, layers, dtype);
        std::vector<std::vector<float>> ws;
        for (size_t l = 0; l < layers.size(); l++) {
            ws.push_back(uniform(static_cast<size_t>(layers[l].in_features) * layers[l].out_features, 10 + l, -1.f, 1.f));
        }
        const auto x = uniform(static_cast<size_t>(M) * 64, 1, -2.f, 2.f);
        model.load_parameters(ws);
        const zkml::Trace t = model.forward(x, M);

        // the same through the operation directly
        const auto tt_dtype = dtype == zkml::DataType::Float32 ? tt::tt_metal::DataType::FLOAT32 : tt::tt_metal::DataType::BFLOAT16;
        auto make = [&](const std::vector<float>& v, uint32_t h, uint32_t w) {
            return ttnn::Tensor::from_vector(
                v,
                tt::tt_metal::TensorSpec(
                    ttnn::Shape({h, w}), tt::tt_metal::TensorLayout(tt_dtype, tt::tt_metal::PageConfig(ttnn::Layout::TILE), ttnn::MemoryConfig{})),
                device_->mesh().get());
        };
        std::vector<ttnn::Tensor> tws;
        for (size_t l = 0; l < layers.size(); l++) {
            tws.push_back(make(ws[l], layers[l].in_features, layers[l].out_features));
        }
        const auto r = zkml::prim::execute_network(
            make(x, M, 64), tws, std::vector<zkml::prim::Activation>{zkml::prim::Activation::ReLU, zkml::prim::Activation::ReLU, zkml::prim::Activation::None});

        EXPECT_EQ(t.batch, M);
        EXPECT_EQ(t.layers(), 3u);
        EXPECT_EQ(t.output, r.out.to_vector<uint8_t>());
        EXPECT_EQ(t.output.size(), static_cast<size_t>(M) * 32);
        EXPECT_EQ(t.scaler_x, r.scaler_x());
        for (uint32_t l = 0; l < 3; l++) {
            EXPECT_EQ(t.scaler_w[l], r.scaler_w(l));
            EXPECT_EQ(t.scaler_out[l], r.scaler_out(l));
            EXPECT_EQ(t.acc[l], r.acc_raw(l));
            EXPECT_EQ(t.acc[l].size(), static_cast<size_t>(M) * layers[l].out_features);
        }
    }
}

TEST_F(ModelApi, Rejected) {
    EXPECT_THROW(zkml::Model(*device_, {}), std::invalid_argument);
    EXPECT_THROW(zkml::Model(*device_, {{33, 64, false}}), std::invalid_argument);           // not a multiple of 32
    EXPECT_THROW(zkml::Model(*device_, {{32, 64, false}, {32, 32, false}}), std::invalid_argument);  // 64 outputs into 32 inputs
    zkml::Model model(*device_, {{32, 32, false}});
    EXPECT_THROW(model.forward(std::vector<float>(32 * 32), 32), std::logic_error);  // no parameters loaded
    EXPECT_THROW(model.load_parameters({}), std::invalid_argument);
    EXPECT_THROW(model.load_parameters({std::vector<float>(10)}), std::invalid_argument);
    model.load_parameters({uniform(32 * 32, 1, -1.f, 1.f)});
    EXPECT_THROW(model.forward(std::vector<float>(32 * 32), 16), std::invalid_argument);  // batch not a multiple of 32
    EXPECT_THROW(model.forward(std::vector<float>(10), 32), std::invalid_argument);       // wrong size
    EXPECT_NO_THROW(model.forward(uniform(32 * 32, 2, -1.f, 1.f), 32));
}
