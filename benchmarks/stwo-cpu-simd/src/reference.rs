// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

//! Scalar oracle, in f64 and i64. Written for clarity, not speed.

use crate::range::{qmax, tile_rank, to_m31, QuantRange};
use crate::Trace;

/// `2^k` as an f64, exact for `|k| <= 1022`.
pub fn pow2(k: i32) -> f64 {
    f64::from_bits(((1023 + k as i64) as u64) << 52)
}

pub fn absmax(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

/// Largest `k <= 127` with `absmax * 2^k <= Q`. An all-zero tensor gives 127.
pub fn pot_k(absmax: f32, bits: u32, range: QuantRange) -> i32 {
    let q = qmax(bits, range) as f64;
    let mut k = 127;
    while absmax as f64 * pow2(k) > q {
        k -= 1;
    }
    k
}

/// `clamp(round_half_even(x * 2^k), -Q, Q)`.
pub fn quantize(x: &[f32], k: i32, bits: u32, range: QuantRange) -> Vec<i32> {
    let q = qmax(bits, range);
    x.iter()
        .map(|&v| ((v as f64 * pow2(k)).round_ties_even() as i64).clamp(-q, q) as i32)
        .collect()
}

/// `x * 2^k - q`, computed in f64 (exact) and stored as f32 (exact too: |remainder| <= 1/2 with at most 24 bits).
pub fn remainder(x: &[f32], k: i32, q: &[i32]) -> Vec<f32> {
    x.iter().zip(q).map(|(&v, &q)| (v as f64 * pow2(k) - q as f64) as f32).collect()
}

/// Index (row-major, in the tile-padded tensor) of the first element in tile order with the largest magnitude.
/// `x` is the logical `h x w` tensor, padding counts as 0. Index 0 for an all-zero tensor.
pub fn argmax(x: &[f32], h: usize, w: usize) -> u32 {
    assert_eq!(x.len(), h * w);
    let wp = w.div_ceil(32) * 32;
    let tiles_w = wp / 32;
    let mut best = 0.0f32;
    let mut best_rank = usize::MAX;
    let mut best_index = 0u32;
    for row in 0..h {
        for col in 0..w {
            let v = x[row * w + col].abs();
            let rank = tile_rank(row, col, tiles_w);
            if v > best || (v == best && v > 0.0 && rank < best_rank) {
                best = v;
                best_rank = rank;
                best_index = (row * wp + col) as u32;
            }
        }
    }
    best_index
}

/// Everything the operation produces, for the logical `h x w` tensor `x`.
pub fn quantize_with_trace(x: &[f32], h: usize, w: usize, bits: u32, range: QuantRange) -> Trace {
    let k = pot_k(absmax(x), bits, range);
    let q = quantize(x, k, bits, range);
    let remainder = remainder(x, k, &q);
    let argmax = argmax(x, h, w);
    let m31 = q.iter().map(|&q| to_m31(q)).collect();
    Trace { k, q, remainder, argmax, m31 }
}
