// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace zkml::nn {

/// A fully connected layer `y = x @ W`, `W` of shape `[in_features, out_features]` (no bias: the quantized layer has
/// none). The weights are not part of the layer: they are given to the model with `load_parameters`.
class Linear {
public:
    Linear(uint32_t in_features, uint32_t out_features) : in_(in_features), out_(out_features) {}

    uint32_t in_features() const { return in_; }
    uint32_t out_features() const { return out_; }

private:
    uint32_t in_;
    uint32_t out_;
};

/// `max(0, x)`. In a `Sequential` it is fused into the Linear before it: it is applied to the integer accumulator,
/// before the requantization.
struct ReLU {};

}  // namespace zkml::nn
