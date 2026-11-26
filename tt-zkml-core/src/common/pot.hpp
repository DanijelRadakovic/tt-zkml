// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// The vocabulary of a power-of-two (PoT) quantization, shared by the host and the device kernels:
//   - QuantRange, m31_modulus, pot_magnitude_bits(), pot_qmax(): the value range (plain constexpr integer code),
//   - BitWidth: the width of the quantized integers as an attribute type of the host-side operations (not compiled into
//   device
//     kernels: tt-metal defines KERNEL_BUILD for those).
//
// The value range:
//
//   Q = 2^magnitude_bits(bits, range) - 1,   quantized values lie in [-Q, Q]
//
//   bits | QuantRange::Full | QuantRange::M31
//   -----+------------------+-----------------------
//      8 |  Q = 127         | 127
//     16 |  Q = 32767       | 32767
//     32 |  Q = 2^31 - 1    | 2^30 - 1 = 1073741823
//
// Full uses every bit of the integer, so the caller can map the result into any field. M31 is for a quantization
// that is then mapped into the Mersenne field M31 (p = 2^31 - 1) by pot_quant_m31: only the 32-bit width differs. With
// Q = 2^31 - 1 = p the values -p, 0 and +p would all be the field element 0, and a residue could not be read back as
// a signed number. With Q <= (p - 1) / 2 every q in [-Q, Q] has its own residue and `m <= (p - 1) / 2 ? m : m - p`
// recovers it. The price is one bit of precision at 32 bits.

#pragma once

#include <cstdint>
#ifndef KERNEL_BUILD
#include <tuple>
#endif

namespace zkml::prim {

/// Which range of integers a quantization uses.
enum class QuantRange : uint32_t {
    /// `Q = 2^(bits - 1) - 1`: every bit of the integer.
    Full = 0,
    /// Like Full, but 32 bits is capped at `Q = 2^30 - 1`, so the values map injectively into M31.
    M31 = 1,
};

/// The Mersenne prime of the M31 field.
constexpr uint32_t m31_modulus = 0x7FFFFFFFu;

/// Number of magnitude bits of the quantized integers: bits - 1, except 30 for 32 bits with QuantRange::M31.
constexpr uint32_t pot_magnitude_bits(uint32_t bits, QuantRange range = QuantRange::Full) {
    return (bits >= 32 && range == QuantRange::M31) ? 30 : bits - 1;
}

/// Largest quantized magnitude `Q = 2^magnitude_bits - 1`.
constexpr int64_t pot_qmax(uint32_t bits, QuantRange range = QuantRange::Full) {
    return (int64_t{1} << pot_magnitude_bits(bits, range)) - 1;
}

static_assert(pot_qmax(32, QuantRange::M31) <= (int64_t{m31_modulus} - 1) / 2, "Q must fit the signed range of M31");

#ifndef KERNEL_BUILD

/**
 * @brief Width of the quantized integers of a power-of-two (PoT) quantization.
 *
 * Quantized values lie in `[-Q, Q]` with `Q = 2^(bits - 1) - 1` (127, 32767, 2^31 - 1), or with `QuantRange::M31`
 * `Q = 2^30 - 1` at 32 bits (see common/pot.hpp).
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

#endif  // KERNEL_BUILD

}  // namespace zkml::prim
