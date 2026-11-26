// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include "device/operations/network_execution/execute_network.hpp"
#include "nn/layers.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace zkml::nn {

/**
 * @brief A chain of layers, executed as one device program (`zkml::prim::execute_network`).
 *
 *     zkml::nn::Sequential model(zkml::nn::Linear(64, 128), zkml::nn::ReLU{}, zkml::nn::Linear(128, 32));
 *     model.load_parameters({w0, w1});         // device tensors [64, 128] and [128, 32]
 *     auto result = model.eval(x);             // x: [M, 64] on device; result.out: the int8 bytes of the last layer
 *
 * The layers are a `std::tuple`, so the model is fixed at compile time (like a `torch.nn.Sequential` that is never
 * modified, or a luminal `Module` tuple). The weights are one device tensor per `Linear`, in order. `eval` lowers the
 * whole chain to one call of `execute_network`: a `Linear` is a layer, a `ReLU` after it is the activation of that
 * layer. A `ReLU` must follow a `Linear`, and the sizes of the Linears must chain (`in_features` of a layer ==
 * `out_features` of the previous one).
 */
template <typename... Layers>
class Sequential {
    static_assert(sizeof...(Layers) >= 1, "a Sequential needs at least one layer");
    static_assert(
        (... && (std::is_same_v<Layers, Linear> || std::is_same_v<Layers, ReLU>)),
        "a Sequential can hold Linear and ReLU layers");
    static_assert(
        std::is_same_v<std::tuple_element_t<0, std::tuple<Layers...>>, Linear>,
        "a Sequential must start with a Linear layer");

public:
    explicit Sequential(Layers... layers) : layers_(std::move(layers)...) {
        std::apply([this](const auto&... layer) { (lower(layer), ...); }, layers_);
        for (size_t l = 1; l < linears_.size(); l++) {
            TT_FATAL(
                linears_[l].in_features() == linears_[l - 1].out_features(),
                "Sequential: Linear {} takes {} features but the previous Linear gives {}",
                l,
                linears_[l].in_features(),
                linears_[l - 1].out_features());
        }
    }

    /// Number of Linear layers, which is the number of weight tensors `load_parameters` takes.
    size_t num_parameters() const { return linears_.size(); }

    /// The weights, one device tensor `[in_features, out_features]` per Linear, in order (same dtype as the inputs of
    /// `eval`).
    void load_parameters(const std::vector<ttnn::Tensor>& weights) {
        TT_FATAL(
            weights.size() == linears_.size(),
            "load_parameters: {} weight tensors for {} Linear layers",
            weights.size(),
            linears_.size());
        for (size_t l = 0; l < weights.size(); l++) {
            const auto shape = weights[l].logical_shape();
            TT_FATAL(
                shape.rank() == 2 && shape[0] == linears_[l].in_features() && shape[1] == linears_[l].out_features(),
                "load_parameters: weight {} is {} but Linear {} is [{}, {}]",
                l,
                shape,
                l,
                linears_[l].in_features(),
                linears_[l].out_features());
        }
        weights_ = weights;
    }

    /// Runs the network on `x` `[M, in_features]`. The result has the quantized output of the last layer and, for
    /// checking and proving, the accumulator and the scalers of every layer.
    zkml::prim::NetworkResult eval(const ttnn::Tensor& x) const {
        TT_FATAL(!weights_.empty(), "eval: call load_parameters first");
        const auto shape = x.logical_shape();
        TT_FATAL(
            shape.rank() == 2 && shape[1] == linears_.front().in_features(),
            "eval: the input is {} but the first Linear takes {} features",
            shape,
            linears_.front().in_features());
        return zkml::prim::execute_network(x, weights_, activations_);
    }

private:
    void lower(const Linear& linear) {
        linears_.push_back(linear);
        activations_.push_back(zkml::prim::Activation::None);
    }
    void lower(const ReLU&) {
        TT_FATAL(
            activations_.back() == zkml::prim::Activation::None,
            "Sequential: two activations after the same Linear (ReLU after ReLU)");
        activations_.back() = zkml::prim::Activation::ReLU;
    }

    std::tuple<Layers...> layers_;
    std::vector<Linear> linears_;
    std::vector<zkml::prim::Activation> activations_;  // one per Linear
    std::vector<ttnn::Tensor> weights_;
};

template <typename... Layers>
Sequential(Layers...) -> Sequential<Layers...>;

}  // namespace zkml::nn
