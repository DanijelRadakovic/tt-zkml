// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// The plan of one execute_network call on one core: the dimensions of every layer, the capacities of the reused
// buffers, and the circular buffers with their sizes (so that the L1 budget can be checked before a program is built).

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

#include "common/l1_check.hpp"
#include "device/operations/layer_execution/execute_layer.hpp"
#include "device/operations/network_execution/device/network_layout.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace zkml::operations::network_execution {

struct NetworkPlan {
    uint32_t layers = 0;
    uint32_t Mt = 0;                        // tiles of the batch dimension
    bool fp32_in = true;                    // fp32 or bf16 inputs
    std::vector<uint32_t> Kt, Nt;           // per layer: tiles of the input and of the output features
    std::vector<uint32_t> relu;             // per layer: 1 with ReLU
    uint32_t cap_w = 0;                     // tiles of the weight buffer (the largest layer)
    uint32_t cap_qw = 0;                    // tiles of the quantized-weight buffer
    uint32_t cap_acc = 0;                   // tiles of the accumulator buffer
    uint32_t cap_act = 0;                   // tiles of each of the two activation buffers (the largest layer input)
    uint32_t out_ring_pages = 0;            // pages of the output ring
    std::vector<zkml::common::CbSpec> cbs;  // in the order the program creates them

    uint32_t x_tiles() const { return Mt * Kt.front(); }
    uint32_t w_tiles(uint32_t l) const { return Kt[l] * Nt[l]; }
    uint32_t out_tiles(uint32_t l) const { return Mt * Nt[l]; }

    /// Checks the tensors and the activations (TT_FATAL on a violation) and plans the buffers.
    static NetworkPlan make(
        const ttnn::Tensor& x,
        const std::vector<ttnn::Tensor>& weights,
        const std::vector<zkml::prim::Activation>& activations) {
        namespace n = zkml::network;
        using tt::tt_metal::DataType;
        TT_FATAL(!weights.empty(), "execute_network: at least one weight tensor is needed");
        TT_FATAL(
            weights.size() <= n::max_layers,
            "execute_network: at most {} layers, got {}",
            n::max_layers,
            weights.size());
        TT_FATAL(
            activations.size() == weights.size(),
            "execute_network: {} activations for {} layers",
            activations.size(),
            weights.size());
        auto check_tensor = [&](const ttnn::Tensor& t, const char* what) {
            TT_FATAL(t.storage_type() == ttnn::StorageType::DEVICE, "execute_network: {} must be on device", what);
            TT_FATAL(t.buffer() != nullptr, "execute_network: {} must be allocated on device", what);
            TT_FATAL(t.layout() == ttnn::Layout::TILE, "execute_network: {} must be in TILE layout", what);
            TT_FATAL(
                t.dtype() == DataType::FLOAT32 || t.dtype() == DataType::BFLOAT16,
                "execute_network: {} must be FLOAT32 or BFLOAT16, got {}",
                what,
                t.dtype());
            TT_FATAL(!t.memory_config().is_sharded(), "execute_network: sharded {} is not supported", what);
            TT_FATAL(t.logical_shape().rank() == 2, "execute_network: {} must be 2-D", what);
            TT_FATAL(t.dtype() == x.dtype(), "execute_network: {} must have the dtype of x ({})", what, x.dtype());
        };
        check_tensor(x, "x");
        for (const auto& w : weights) {
            check_tensor(w, "a weight");
        }

        NetworkPlan p;
        p.layers = static_cast<uint32_t>(weights.size());
        p.fp32_in = x.dtype() == DataType::FLOAT32;
        const uint32_t m = x.logical_shape()[0];
        uint32_t k = x.logical_shape()[1];
        TT_FATAL(
            m % tt::constants::TILE_HEIGHT == 0 && k % tt::constants::TILE_WIDTH == 0,
            "execute_network: M and K must be multiples of 32, got M={} K={}",
            m,
            k);
        p.Mt = m / tt::constants::TILE_HEIGHT;
        for (uint32_t l = 0; l < p.layers; l++) {
            const uint32_t kw = weights[l].logical_shape()[0], nw = weights[l].logical_shape()[1];
            TT_FATAL(
                kw == k, "execute_network: layer {} has weight [{}, {}] but its input has {} features", l, kw, nw, k);
            TT_FATAL(
                nw % tt::constants::TILE_WIDTH == 0,
                "execute_network: layer {} has {} output features, not a multiple of 32",
                l,
                nw);
            const uint32_t kt = kw / tt::constants::TILE_WIDTH, nt = nw / tt::constants::TILE_WIDTH;
            TT_FATAL(
                kt <= n::max_k_tiles,
                "execute_network: layer {} has K = {}, at most {} (K * 127 * 127 must stay below 2^30)",
                l,
                kw,
                n::max_k_tiles * tt::constants::TILE_WIDTH);
            p.Kt.push_back(kt);
            p.Nt.push_back(nt);
            p.relu.push_back(activations[l] == zkml::prim::Activation::ReLU ? 1u : 0u);
            TT_FATAL(
                activations[l] == zkml::prim::Activation::None || activations[l] == zkml::prim::Activation::ReLU,
                "execute_network: unknown activation in layer {}",
                l);
            k = nw;
        }
        for (uint32_t l = 0; l < p.layers; l++) {
            p.cap_w = std::max(p.cap_w, p.w_tiles(l));
            p.cap_acc = std::max(p.cap_acc, p.out_tiles(l));
            p.cap_act = std::max(p.cap_act, p.Mt * p.Kt[l]);  // the input of layer l; the output of layer l is the input of l + 1
        }
        p.cap_qw = p.cap_w;
        p.out_ring_pages = 2 * n::dst_tiles;

        const uint32_t in_bytes = tt::tile_size(p.fp32_in ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b);
        const uint32_t u8_bytes = tt::tile_size(tt::DataFormat::UInt8);
        const uint32_t i32_bytes = tt::tile_size(tt::DataFormat::Int32);
        const uint32_t f32_bytes = tt::tile_size(tt::DataFormat::Float32);
        p.cbs.push_back({n::c_x, p.x_tiles(), in_bytes});
        p.cbs.push_back({n::c_w, p.cap_w, in_bytes});
        p.cbs.push_back({n::c_qw, p.cap_qw, u8_bytes});
        p.cbs.push_back({n::c_acc, p.cap_acc, i32_bytes});
        p.cbs.push_back({n::c_act_a, p.cap_act, u8_bytes});
        p.cbs.push_back({n::c_act_b, p.cap_act, u8_bytes});
        p.cbs.push_back({n::c_out, p.out_ring_pages, u8_bytes});
        p.cbs.push_back({n::c_scale_x, n::scaler_pages, f32_bytes});
        p.cbs.push_back({n::c_scale_w, n::scaler_pages, f32_bytes});
        p.cbs.push_back({n::c_scale_out, n::scaler_pages, f32_bytes});
        return p;
    }
};

}  // namespace zkml::operations::network_execution
