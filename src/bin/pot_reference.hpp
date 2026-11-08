// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Host-side reference for per-tensor PoT quantization (see
// operations/pot_quantization/per_tensor/pot_quantize_per_tensor.hpp):
//   Q = 2^(bits - 1) - 1
//   k = max{k <= 127 : absmax(x) * 2^k <= Q}
//   q = clamp(round_half_even(x * 2^k), -Q, Q)
//
// Computed in double/int64, so it is exact for every fp32 input.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace zkml::reference {

inline int64_t pot_qmax(uint32_t bits) { return (int64_t{1} << (bits - 1)) - 1; }

inline float absmax(const std::vector<float>& x) {
    float m = 0.0f;
    for (float v : x) {
        m = std::max(m, std::fabs(v));
    }
    return m;
}

// Largest k <= 127 with absmax * 2^k <= Q. absmax = 0 gives 127.
inline int32_t pot_k(float absmax, uint32_t bits) {
    const double q = static_cast<double>(pot_qmax(bits));
    int32_t k = 127;
    while (std::ldexp(static_cast<double>(absmax), k) > q) {
        k--;
    }
    return k;
}

inline std::vector<int32_t> pot_quantize(const std::vector<float>& x, int32_t k, uint32_t bits) {
    const int64_t q = pot_qmax(bits);
    std::vector<int32_t> out(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        // nearbyint uses the default rounding mode, round half to even.
        const auto r = static_cast<int64_t>(std::nearbyint(std::ldexp(static_cast<double>(x[i]), k)));
        out[i] = static_cast<int32_t>(std::clamp(r, -q, q));
    }
    return out;
}

}  // namespace zkml::reference
