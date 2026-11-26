// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// The public C++ API of tt-zkml-core. Everything a user of the library needs is reachable from here; the rest of src/ is internal.
//
//   zkml::Device, zkml::Model, zkml::Trace   (api/model.hpp, api/trace.hpp): open a device, load a model, run the forward pass, get the trace
//   zkml::prim::execute_network / execute_layer / pot_quantize_per_tensor: the device operations the model is built from
//   zkml::nn::Sequential                       the same network as a compile-time chain of layers (host-side lowering)
//
// The C ABI for Rust and Python (abi/ in the repository root) is a thin wrapper over zkml::Device / zkml::Model.

#pragma once

#include "api/model.hpp"
#include "api/trace.hpp"
#include "device/operations/layer_execution/execute_layer.hpp"
#include "device/operations/network_execution/execute_network.hpp"
#include "device/operations/pot_quantization/per_tensor/pot_quantize/pot_quantize_per_tensor.hpp"
#include "nn/nn.hpp"
