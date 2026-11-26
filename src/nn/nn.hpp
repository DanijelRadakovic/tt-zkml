// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#pragma once

// Host side of the quantized neural networks: layers and the Sequential container. Everything is lowered to
// zkml::prim::execute_network (src/device/operations/network_execution).

#include "nn/layers.hpp"
#include "nn/sequential.hpp"
