// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

//! The value range of a PoT quantization (same as `src/operations/pot_quantization/pot_range.hpp`).

/// The Mersenne prime of the M31 field.
pub const M31_P: u32 = 0x7FFF_FFFF;

/// Which range of integers a quantization uses (same as `QuantRange` in `pot_range.hpp`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantRange {
    /// `Q = 2^(bits - 1) - 1`: every bit of the integer.
    Full,
    /// Like `Full`, but 32 bits is capped at `Q = 2^30 - 1`, so that `2Q + 1 <= p` and the map `q -> q mod p` is
    /// injective on `[-Q, Q]`.
    M31,
}

/// Magnitude bits of the quantized integers: `bits - 1`, except 30 for 32 bits with `QuantRange::M31`.
pub const fn magnitude_bits(bits: u32, range: QuantRange) -> u32 {
    if bits >= 32 && matches!(range, QuantRange::M31) {
        30
    } else {
        bits - 1
    }
}

/// Largest quantized magnitude `Q = 2^magnitude_bits - 1`.
pub const fn qmax(bits: u32, range: QuantRange) -> i64 {
    (1i64 << magnitude_bits(bits, range)) - 1
}

const _: () = assert!(qmax(32, QuantRange::M31) <= (M31_P as i64 - 1) / 2);

/// Residue of `q` modulo p, in `[0, p)`.
pub const fn to_m31(q: i32) -> u32 {
    if q < 0 {
        (q as i64 + M31_P as i64) as u32
    } else {
        q as u32
    }
}

/// The signed value a verifier reads from a residue: `m <= (p - 1) / 2 ? m : m - p`.
pub const fn from_m31(m: u32) -> i32 {
    if m <= (M31_P - 1) / 2 {
        m as i32
    } else {
        (m as i64 - M31_P as i64) as i32
    }
}

/// Rank of element `(row, col)` in tile order: tiles row by row, in a tile the four 16x16 faces (top-left, top-right,
/// bottom-left, bottom-right), each row-major. `tiles_w` is the number of tiles per row of the padded tensor.
pub const fn tile_rank(row: usize, col: usize, tiles_w: usize) -> usize {
    let tile = (row / 32) * tiles_w + col / 32;
    let face = ((row / 16) % 2) * 2 + (col / 16) % 2;
    tile * 1024 + face * 256 + (row % 16) * 16 + col % 16
}
