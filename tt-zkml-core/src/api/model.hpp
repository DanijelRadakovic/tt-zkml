// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "api/trace.hpp"

namespace ttnn {
class Tensor;
}
namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace zkml {

/// A Tenstorrent device. Opening it creates tt-metal's context, which is thread-affine: create, use and destroy a Device (and the Models on
/// it) on one thread, and destroy them before the process exits.
class Device {
public:
    /// Opens device `id`.
    explicit Device(uint32_t id = 0);
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    /// Number of Tenstorrent devices on the PCIe bus (does not open one).
    static uint32_t count();

    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh() const { return mesh_; }

private:
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_;
};

/// The type of the floats on the device (the input and the weights; the arithmetic is int8 either way).
enum class DataType { Float32, BFloat16 };

/// One fully connected layer `y = x @ W`, `W` of shape `[in_features, out_features]`, with an optional ReLU on the accumulator.
struct LayerSpec {
    uint32_t in_features = 0;
    uint32_t out_features = 0;
    bool relu = false;
};

/// A chain of quantized fully connected layers executed as one device program (`zkml::prim::execute_network`).
class Model {
public:
    /// The layers must chain (`in_features` of a layer is `out_features` of the previous one); all sizes are multiples of 32.
    Model(Device& device, std::vector<LayerSpec> layers, DataType dtype = DataType::BFloat16);
    ~Model();
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;

    const std::vector<LayerSpec>& layers() const { return layers_; }
    DataType dtype() const { return dtype_; }

    /// Copies the weights to the device: `weights[l]` is `[in_features, out_features]` row-major, for every layer.
    void load_parameters(const std::vector<std::vector<float>>& weights);

    /// Runs the network on `x` (`[batch, in_features]` row-major, `batch` a multiple of 32) and returns the trace.
    Trace forward(const std::vector<float>& x, uint32_t batch) const;

private:
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_;
    std::vector<LayerSpec> layers_;
    DataType dtype_;
    std::vector<ttnn::Tensor> weights_;
};

}  // namespace zkml
