// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

//! `std::simd` implementation (16 lanes of f32, the width of stwo's `PackedM31`), single-threaded or on a rayon pool.
//!
//! The arithmetic is the device's: `y = x * 2^k` is exact in fp32, rounding is round-half-even by the
//! `(|y| + 2^23) - 2^23` trick (valid for `|y| < 2^23`; above that `y` is already an integer), the clamp and the
//! convert are the same, and `k` comes from the exponent bits of the absmax as in `pot_scale_sfpu.hpp`
//! (not from the loop of the scalar oracle), so the two implementations check each other.

use std::simd::cmp::{SimdPartialEq, SimdPartialOrd};
use std::simd::num::{SimdFloat, SimdInt};
use std::simd::{f32x16, i32x16, u32x16};

use rayon::prelude::*;
use rayon::ThreadPool;
use stwo::core::fields::m31::M31;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::PackedM31;

use crate::range::{magnitude_bits, tile_rank, QuantRange, M31_P};
use crate::Trace;

pub const LANES: usize = 16;
/// Elements per parallel work item.
const CHUNK: usize = 1 << 14;

/// Maximum of `|x|`. Rayon pool or one thread.
pub fn absmax(x: &[f32], pool: Option<&ThreadPool>) -> f32 {
    fn chunk_max(c: &[f32]) -> f32 {
        let mut acc = f32x16::splat(0.0);
        let mut it = c.chunks_exact(LANES);
        for v in &mut it {
            acc = acc.simd_max(f32x16::from_slice(v).abs());
        }
        let mut m = acc.reduce_max();
        for &v in it.remainder() {
            m = m.max(v.abs());
        }
        m
    }
    match pool {
        None => chunk_max(x),
        Some(p) => p.install(|| x.par_chunks(CHUNK).map(chunk_max).reduce(|| 0.0, f32::max)),
    }
}

/// The exponent `k` from the absmax, by the exponent bits (see `pot_scale_sfpu.hpp`).
pub fn k_from_absmax(absmax: f32, bits: u32, range: QuantRange) -> i32 {
    let mag = magnitude_bits(bits, range) as i32;
    let biased = ((absmax.to_bits() >> 23) & 0xFF) as i32;
    let field = (253 + mag - biased).min(254); // biased exponent of the candidate scaler; 254 = 2^127
    let scaler = f32::from_bits((field as u32) << 23);
    let q = ((1u64 << mag) - 1) as f32; // rounds to 2^31 (2^30) at 32 bits, like the device; no fp32 lies in between
    let mut k = field - 127;
    if absmax * scaler > q {
        k -= 1;
    }
    k
}

/// Round half to even of 16 floats.
#[inline(always)]
fn rne(y: f32x16) -> f32x16 {
    const MAGIC: f32 = 8_388_608.0; // 2^23
    let sign = u32x16::splat(0x8000_0000);
    let a = y.abs();
    let magic = f32x16::splat(MAGIC);
    let r = (a + magic) - magic;
    let r = a.simd_lt(magic).select(r, a);
    f32x16::from_bits(r.to_bits() | (y.to_bits() & sign))
}

/// Quantizes `x` into `q`, `remainder` and `m31` (all of the same length as `x`). `with_trace = false` skips the
/// remainder (it stays untouched).
fn quantize_chunk(x: &[f32], scaler: f32, q_max: f32, with_trace: bool, q: &mut [i32], rem: &mut [f32], m31: &mut [u32]) {
    let scaler_v = f32x16::splat(scaler);
    let (lo, hi) = (f32x16::splat(-q_max), f32x16::splat(q_max));
    let p = i32x16::splat(M31_P as i32);
    let zero = i32x16::splat(0);
    let n_full = x.len() / LANES * LANES;
    for i in (0..n_full).step_by(LANES) {
        let y = f32x16::from_slice(&x[i..i + LANES]) * scaler_v; // exact
        let qf = rne(y).simd_clamp(lo, hi);
        let qi: i32x16 = qf.cast();
        qi.copy_to_slice(&mut q[i..i + LANES]);
        if with_trace {
            (y - qf).copy_to_slice(&mut rem[i..i + LANES]); // exact
        }
        let m = qi.simd_lt(zero).select(qi + p, qi);
        m.cast::<u32>().copy_to_slice(&mut m31[i..i + LANES]);
    }
    // Tail: pad to a full vector.
    if n_full < x.len() {
        let mut buf = [0.0f32; LANES];
        let tail = x.len() - n_full;
        buf[..tail].copy_from_slice(&x[n_full..]);
        let y = f32x16::from_array(buf) * scaler_v;
        let qf = rne(y).simd_clamp(lo, hi);
        let qi: i32x16 = qf.cast();
        let r = y - qf;
        let m = qi.simd_lt(zero).select(qi + p, qi);
        q[n_full..].copy_from_slice(&qi.to_array()[..tail]);
        if with_trace {
            rem[n_full..].copy_from_slice(&r.to_array()[..tail]);
        }
        m31[n_full..].copy_from_slice(&m.cast::<u32>().to_array()[..tail]);
    }
}

/// Index of the first element in tile order with `|x| = absmax`, into the row-major tile-padded tensor.
pub fn argmax(x: &[f32], h: usize, w: usize, absmax: f32) -> u32 {
    assert_eq!(x.len(), h * w);
    if absmax == 0.0 {
        return 0;
    }
    let wp = w.div_ceil(32) * 32;
    let tiles_w = wp / 32;
    let target = f32x16::splat(absmax);
    let mut best_rank = usize::MAX;
    let mut best_index = 0u32;
    let mut visit = |i: usize| {
        let (row, col) = (i / w, i % w);
        let rank = tile_rank(row, col, tiles_w);
        if rank < best_rank {
            best_rank = rank;
            best_index = (row * wp + col) as u32;
        }
    };
    let mut it = x.chunks_exact(LANES);
    let mut base = 0;
    for v in &mut it {
        let mask = f32x16::from_slice(v).abs().simd_eq(target).to_bitmask();
        if mask != 0 {
            for lane in 0..LANES {
                if mask >> lane & 1 == 1 {
                    visit(base + lane);
                }
            }
        }
        base += LANES;
    }
    for (j, &v) in it.remainder().iter().enumerate() {
        if v.abs() == absmax {
            visit(base + j);
        }
    }
    best_index
}

/// Packs residues into stwo's SIMD column (`PackedM31` lanes), the form a stwo trace column takes. The length is
/// padded with zeros to a multiple of 16 inside the last vector, as `BaseColumn` does.
pub fn m31_column(m31: &[u32]) -> BaseColumn {
    let mut data = Vec::with_capacity(m31.len().div_ceil(LANES));
    for c in m31.chunks(LANES) {
        let mut a = [M31(0); LANES];
        for (d, &v) in a.iter_mut().zip(c) {
            *d = M31(v);
        }
        data.push(PackedM31::from_array(a));
    }
    BaseColumn { data, length: m31.len() }
}

/// Quantizes the logical `h x w` tensor and, with `with_trace`, also produces the remainder and the argmax.
/// The `m31` residues are always produced (they are what `pot_quant_m31` adds); they are only injective with
/// `QuantRange::M31`.
pub fn quantize_with_trace(
    x: &[f32],
    h: usize,
    w: usize,
    bits: u32,
    range: QuantRange,
    with_trace: bool,
    pool: Option<&ThreadPool>,
) -> Trace {
    assert_eq!(x.len(), h * w);
    let abs = absmax(x, pool);
    let k = k_from_absmax(abs, bits, range);
    let scaler = f32::from_bits(((k + 127) as u32) << 23);
    let q_max = ((1u64 << magnitude_bits(bits, range)) - 1) as f32;

    let n = x.len();
    let mut q = vec![0i32; n];
    let mut remainder = vec![0.0f32; if with_trace { n } else { 0 }];
    let mut m31 = vec![0u32; n];
    // The remainder is an empty slice without trace: give every chunk an empty piece.
    match pool {
        None => quantize_chunk(x, scaler, q_max, with_trace, &mut q, if with_trace { &mut remainder } else { &mut [] }, &mut m31),
        Some(p) => p.install(|| {
            if with_trace {
                x.par_chunks(CHUNK)
                    .zip(q.par_chunks_mut(CHUNK))
                    .zip(remainder.par_chunks_mut(CHUNK))
                    .zip(m31.par_chunks_mut(CHUNK))
                    .for_each(|(((x, q), r), m)| quantize_chunk(x, scaler, q_max, true, q, r, m));
            } else {
                x.par_chunks(CHUNK)
                    .zip(q.par_chunks_mut(CHUNK))
                    .zip(m31.par_chunks_mut(CHUNK))
                    .for_each(|((x, q), m)| quantize_chunk(x, scaler, q_max, false, q, &mut [], m));
            }
        }),
    }
    let argmax = if with_trace { argmax(x, h, w, abs) } else { 0 };
    Trace { k, q, remainder, argmax, m31 }
}
