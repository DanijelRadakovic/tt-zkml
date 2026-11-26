// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <tuple>

#include "operations/pot_quantization/pot_range.hpp"

namespace zkml::prim {

/**
 * @brief Width of the quantized integers of a power-of-two (PoT) quantization.
 *
 * Quantized values lie in `[-Q, Q]` with `Q = 2^(bits - 1) - 1` (127, 32767, 2^31 - 1), or with `QuantRange::M31`
 * `Q = 2^30 - 1` at 32 bits (see pot_range.hpp).
 *
 * It behaves like an enum with the values `BitWidth::Bits8`, `BitWidth::Bits16` and `BitWidth::Bits32`,
 * but it is a class, because a C++ `enum class` cannot have member functions. Its only constructor
 * takes one of those values, so a caller cannot pass a plain number like `12` by accident.
 *
 * @code
 * zkml::prim::BitWidth width = zkml::prim::BitWidth::Bits16;
 * uint32_t n = width.bits();  // 16
 * @endcode
 */
class BitWidth {
public:
    enum Value : uint32_t {
        Bits8 = 8,
        Bits16 = 16,
        Bits32 = 32,
    };

    // ttnn reflects over the attributes of an operation, so the type must be default-constructible.
    constexpr BitWidth() = default;
    constexpr BitWidth(Value value) : value_(value) {}

    /// Number of bits: 8, 16 or 32.
    constexpr uint32_t bits() const { return value_; }

    /// Largest quantized magnitude Q for the range: 127, 32767, and 2^31 - 1 (2^30 - 1 for QuantRange::M31).
    constexpr int64_t qmax(QuantRange range = QuantRange::Full) const { return pot_qmax(value_, range); }

    constexpr bool operator==(const BitWidth&) const = default;

    // ttsl reflection protocol: lets ttnn hash and print the type without further code.
    static constexpr auto attribute_names = std::forward_as_tuple("bits");
    auto attribute_values() const { return std::make_tuple(bits()); }

private:
    Value value_ = Bits8;
};

}  // namespace zkml::prim
