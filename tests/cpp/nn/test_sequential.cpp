// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// zkml::nn::Sequential against a direct call of zkml::prim::execute_network with the same layers: model.eval(x) must be
// that call, byte for byte, and the model rejects what it cannot run.

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

#include "nn/nn.hpp"

using namespace ttnn;
using tt::tt_metal::DataType;

namespace {

std::vector<float> uniform(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> v(n);
    for (auto& e : v) {
        e = d(rng);
    }
    return v;
}

class NnSequential : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tt::tt_metal::SetRootDir(ZKML_TT_METAL_HOME);
        device_ = open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);
    }
    static void TearDownTestSuite() {
        device_->close();
        device_.reset();
    }
    static Tensor to_device(uint32_t h, uint32_t w, uint32_t seed) {
        return Tensor::from_vector(
            uniform(static_cast<size_t>(h) * w, seed),
            tt::tt_metal::TensorSpec(
                Shape({h, w}),
                tt::tt_metal::TensorLayout(DataType::FLOAT32, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{})),
            device_.get());
    }
    static inline std::shared_ptr<MeshDevice> device_;
};

void expect_same(const zkml::prim::NetworkResult& a, const zkml::prim::NetworkResult& b) {
    EXPECT_EQ(a.out.to_vector<uint8_t>(), b.out.to_vector<uint8_t>());
    EXPECT_EQ(a.scaler_x.to_vector<float>(), b.scaler_x.to_vector<float>());
    ASSERT_EQ(a.acc.size(), b.acc.size());
    for (size_t l = 0; l < a.acc.size(); l++) {
        EXPECT_EQ(a.acc[l].to_vector<uint32_t>(), b.acc[l].to_vector<uint32_t>()) << "layer " << l;
        EXPECT_EQ(a.scaler_w[l].to_vector<float>(), b.scaler_w[l].to_vector<float>()) << "layer " << l;
        EXPECT_EQ(a.scaler_out[l].to_vector<float>(), b.scaler_out[l].to_vector<float>()) << "layer " << l;
    }
}

}  // namespace

TEST_F(NnSequential, MlpEqualsExecuteNetwork) {
    using namespace zkml::nn;
    zkml::nn::Sequential model(Linear(64, 128), ReLU{}, Linear(128, 64), ReLU{}, Linear(64, 32));
    EXPECT_EQ(model.num_parameters(), 3u);
    const std::vector<Tensor> ws = {to_device(64, 128, 1), to_device(128, 64, 2), to_device(64, 32, 3)};
    model.load_parameters(ws);
    const Tensor x = to_device(32, 64, 4);

    const auto got = model.eval(x);
    const auto want = zkml::prim::execute_network(
        x,
        ws,
        std::vector<zkml::prim::Activation>{
            zkml::prim::Activation::ReLU, zkml::prim::Activation::ReLU, zkml::prim::Activation::None});
    expect_same(got, want);
}

TEST_F(NnSequential, SingleLinearAndTrailingReLU) {
    using namespace zkml::nn;
    zkml::nn::Sequential model(Linear(32, 64), ReLU{});
    const std::vector<Tensor> ws = {to_device(32, 64, 5)};
    model.load_parameters(ws);
    const Tensor x = to_device(32, 32, 6);
    expect_same(
        model.eval(x),
        zkml::prim::execute_network(x, ws, std::vector<zkml::prim::Activation>{zkml::prim::Activation::ReLU}));
}

TEST_F(NnSequential, TheModelIsReusableWithNewWeightsAndInputs) {
    using namespace zkml::nn;
    zkml::nn::Sequential model(Linear(32, 32), Linear(32, 32));
    for (uint32_t round = 0; round < 2; round++) {
        const std::vector<Tensor> ws = {to_device(32, 32, 10 + round), to_device(32, 32, 20 + round)};
        model.load_parameters(ws);
        const Tensor x = to_device(32, 32, 30 + round);
        expect_same(
            model.eval(x),
            zkml::prim::execute_network(
                x,
                ws,
                std::vector<zkml::prim::Activation>{zkml::prim::Activation::None, zkml::prim::Activation::None}));
    }
}

TEST_F(NnSequential, Rejected) {
    using namespace zkml::nn;
    zkml::nn::Sequential model(Linear(32, 64), ReLU{}, Linear(64, 32));
    const Tensor x = to_device(32, 32, 40);
    EXPECT_ANY_THROW(model.eval(x));                                   // no parameters yet
    EXPECT_ANY_THROW(model.load_parameters({to_device(32, 64, 41)}));  // one weight for two Linears
    EXPECT_ANY_THROW(
        model.load_parameters({to_device(32, 64, 41), to_device(32, 32, 42)}));  // [32, 32] where [64, 32] is expected
    model.load_parameters({to_device(32, 64, 41), to_device(64, 32, 42)});
    EXPECT_ANY_THROW(model.eval(to_device(32, 64, 43)));                     // 64 features where 32 are expected
    EXPECT_ANY_THROW(zkml::nn::Sequential(Linear(32, 64), Linear(32, 32)));  // 64 outputs into 32 inputs
    EXPECT_ANY_THROW(zkml::nn::Sequential(Linear(32, 64), ReLU{}, ReLU{}));
}
