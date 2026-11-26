// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include "api/model.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>
#include <tt-umd/pcie/pci_device.hpp>

#include "device/operations/network_execution/execute_network.hpp"

namespace zkml {

namespace {

tt::tt_metal::DataType to_tt(DataType t) { return t == DataType::Float32 ? tt::tt_metal::DataType::FLOAT32 : tt::tt_metal::DataType::BFLOAT16; }

tt::tt_metal::TensorSpec matrix_spec(uint32_t rows, uint32_t cols, DataType t) {
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({rows, cols}),
        tt::tt_metal::TensorLayout(to_tt(t), tt::tt_metal::PageConfig(ttnn::Layout::TILE), ttnn::MemoryConfig{}));
}

}  // namespace

Device::Device(uint32_t id) {
    // Where tt-metal finds its runtime files (kernels' headers, firmware): the environment variable if set, else the tt-metal this library was built against.
    const char* root = std::getenv("TT_METAL_RUNTIME_ROOT");
    if (root == nullptr || *root == '\0') {
        tt::tt_metal::SetRootDir(ZKML_TT_METAL_HOME);
    }
    mesh_ = ttnn::open_mesh_device(static_cast<int>(id), DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);
}

Device::~Device() {
    if (mesh_) {
        mesh_->close();
        mesh_.reset();
    }
}

uint32_t Device::count() { return static_cast<uint32_t>(tt::umd::PCIDevice::enumerate_devices().size()); }

Model::Model(Device& device, std::vector<LayerSpec> layers, DataType dtype) :
    mesh_(device.mesh()), layers_(std::move(layers)), dtype_(dtype) {
    if (layers_.empty()) {
        throw std::invalid_argument("a model needs at least one layer");
    }
    for (size_t l = 0; l < layers_.size(); l++) {
        const auto& s = layers_[l];
        if (s.in_features == 0 || s.out_features == 0 || s.in_features % 32 != 0 || s.out_features % 32 != 0) {
            throw std::invalid_argument("layer " + std::to_string(l) + ": the features must be non-zero multiples of 32");
        }
        if (l > 0 && layers_[l - 1].out_features != s.in_features) {
            throw std::invalid_argument("layer " + std::to_string(l) + ": in_features does not match out_features of the previous layer");
        }
    }
}

Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

void Model::load_parameters(const std::vector<std::vector<float>>& weights) {
    if (weights.size() != layers_.size()) {
        throw std::invalid_argument("load_parameters: " + std::to_string(weights.size()) + " weight tensors for " + std::to_string(layers_.size()) + " layers");
    }
    std::vector<ttnn::Tensor> loaded;
    for (size_t l = 0; l < weights.size(); l++) {
        const auto& s = layers_[l];
        if (weights[l].size() != static_cast<size_t>(s.in_features) * s.out_features) {
            throw std::invalid_argument("load_parameters: layer " + std::to_string(l) + " expects " + std::to_string(static_cast<size_t>(s.in_features) * s.out_features) + " values");
        }
        loaded.push_back(ttnn::Tensor::from_vector(weights[l], matrix_spec(s.in_features, s.out_features, dtype_), mesh_.get()));
    }
    weights_ = std::move(loaded);
}

Trace Model::forward(const std::vector<float>& x, uint32_t batch) const {
    if (weights_.empty()) {
        throw std::logic_error("forward: call load_parameters first");
    }
    if (batch == 0 || batch % 32 != 0) {
        throw std::invalid_argument("forward: batch must be a non-zero multiple of 32");
    }
    const uint32_t in = layers_.front().in_features;
    if (x.size() != static_cast<size_t>(batch) * in) {
        throw std::invalid_argument("forward: x must have batch * in_features = " + std::to_string(static_cast<size_t>(batch) * in) + " values");
    }
    ttnn::Tensor tx = ttnn::Tensor::from_vector(x, matrix_spec(batch, in, dtype_), mesh_.get());
    std::vector<zkml::prim::Activation> acts;
    for (const auto& s : layers_) {
        acts.push_back(s.relu ? zkml::prim::Activation::ReLU : zkml::prim::Activation::None);
    }
    const auto r = zkml::prim::execute_network(tx, weights_, acts);

    Trace t;
    t.batch = batch;
    t.in_features = in;
    t.widths = r.widths;
    t.output = r.out.to_vector<uint8_t>();
    t.scaler_x = r.scaler_x();
    for (uint32_t l = 0; l < r.layers(); l++) {
        t.scaler_w.push_back(r.scaler_w(l));
        t.scaler_out.push_back(r.scaler_out(l));
        t.acc.push_back(r.acc_raw(l));
    }
    return t;
}

}  // namespace zkml
