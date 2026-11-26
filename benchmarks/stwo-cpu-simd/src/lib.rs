// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

//! CPU and SIMD reference of `pot_quantize_per_tensor(..., trace = true)` and `pot_quant_m31`.
//!
//! ```text
//! Q = 2^(bits - 1) - 1            (QuantRange::M31 caps 32 bits at 2^30 - 1 so that the M31 map is injective)
//! k = max{k <= 127 : absmax(x) * 2^k <= Q}
//! q = clamp(round_half_even(x * 2^k), -Q, Q)
//! remainder = x * 2^k - q         exact in fp32, |remainder| <= 1/2
//! argmax    = index of the first element (in tile order) with |x| = absmax, into the row-major tile-padded tensor
//! m31       = q mod (2^31 - 1)    in [0, p)
//! ```
//!
//! Two implementations of the same function:
//! - [`reference`]: scalar, in f64 and i64, written for clarity. It is the oracle.
//! - [`simd`]: `std::simd` (16 lanes of f32), optionally on all cores through rayon, and the quantized values are
//!   written into stwo's own SIMD column type (`BaseColumn` of `PackedM31`), the form a stwo prover takes.
//!
//! Both must agree with each other and with the device (see `tests/` and the `dump` binary).

#![feature(portable_simd)]

pub mod range;
pub mod reference;
pub mod simd;

/// Everything `pot_quantize_per_tensor(trace = true)` and `pot_quant_m31` produce for one tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    /// The exponent of the scaler `2^k`.
    pub k: i32,
    /// Quantized values.
    pub q: Vec<i32>,
    /// `x * 2^k - q`, per element.
    pub remainder: Vec<f32>,
    /// Index of the absmax element in the row-major tile-padded tensor.
    pub argmax: u32,
    /// `q mod (2^31 - 1)`, per element, in `[0, p)`.
    pub m31: Vec<u32>,
}
