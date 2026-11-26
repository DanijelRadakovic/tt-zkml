// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

//! The scalar oracle, the SIMD implementation and the properties a prover relies on.

use zkml_stwo_cpu_simd::range::{from_m31, qmax, to_m31, QuantRange, M31_P};

const RANGES: [QuantRange; 2] = [QuantRange::Full, QuantRange::M31];
use zkml_stwo_cpu_simd::{reference, simd};

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed | 1;
    move || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn bf16(v: f32) -> f32 {
    f32::from_bits((v.to_bits() + 0x8000) & 0xFFFF_0000)
}

fn random_tensor(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut r = rng(seed);
    (0..n).map(|_| bf16(((r() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale)).collect()
}

#[test]
fn q_is_capped_for_m31() {
    for range in RANGES {
        assert_eq!(qmax(8, range), 127);
        assert_eq!(qmax(16, range), 32767);
    }
    assert_eq!(qmax(32, QuantRange::Full), (1 << 31) - 1);
    assert_eq!(qmax(32, QuantRange::M31), (1 << 30) - 1);
    // injective on [-Q, Q] and read back as the same signed value
    for q in [0, 1, -1, 127, -127, 32767, -32767, (1 << 30) - 1, -((1 << 30) - 1), (1 << 30) - 2] {
        let m = to_m31(q);
        assert!(m < M31_P);
        assert_eq!(from_m31(m), q);
    }
    // the reason for the M31 range: with Q = p (QuantRange::Full at 32 bits) the value q = p would map to p itself,
    // which is not a canonical residue (it is the field element 0, like q = 0 and q = -p)
    assert!(to_m31(M31_P as i32) >= M31_P);
    assert_eq!(qmax(32, QuantRange::Full), M31_P as i64);
}

#[test]
fn simd_matches_oracle_everywhere() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    // shapes: tile multiples, and padded ones (the tail of the SIMD loop and the padded argmax index)
    for &(h, w) in &[(32usize, 32usize), (32, 64), (64, 96), (40, 70), (1, 17), (33, 31), (32, 32 * 37)] {
        for &scale in &[1e-3f32, 1.0, 8.0, 3e4, 1e20] {
            for &bits in &[8u32, 16, 32] {
                let x = random_tensor(h as u64 * 1000 + w as u64, h * w, scale);
              for range in RANGES {
                let want = reference::quantize_with_trace(&x, h, w, bits, range);
                for pool in [None, Some(&pool)] {
                    let got = simd::quantize_with_trace(&x, h, w, bits, range, true, pool);
                    assert_eq!(want, got, "{h}x{w} scale {scale} bits {bits} threads {}", pool.map_or(1, |p| p.current_num_threads()));
                    // without trace the quantized values are the same
                    let got = simd::quantize_with_trace(&x, h, w, bits, range, false, pool);
                    assert_eq!(want.q, got.q);
                    assert_eq!(want.m31, got.m31);
                    assert_eq!(want.k, got.k);
                }
              }
            }
        }
    }
}

#[test]
fn k_by_exponent_bits_equals_k_by_search() {
    for (&bits, range) in [8u32, 16, 32].iter().flat_map(|b| RANGES.map(move |r| (b, r))) {
        let q = qmax(bits, range) as f32;
        let mut cases = vec![0.0, 1.0, 0.5, 3e4, 1e-30, 3.38953139e38, q, q + 0.5, q * 2.0, q - 1.0, 1e-38, 2.4e-38];
        let mut r = rng(bits as u64);
        for _ in 0..20000 {
            cases.push(bf16(f32::from_bits((r() >> 33) as u32 & 0x7F7F_0000)));
        }
        for a in cases {
            assert_eq!(simd::k_from_absmax(a, bits, range), reference::pot_k(a, bits, range), "absmax {a:e} bits {bits} {range:?}");
        }
    }
}

#[test]
fn trace_invariants_a_prover_checks() {
    let (h, w) = (64usize, 96usize);
    for (&bits, range) in [8u32, 16, 32].iter().flat_map(|b| RANGES.map(move |r| (b, r))) {
        let x = random_tensor(99 + bits as u64, h * w, 5.0);
        let t = reference::quantize_with_trace(&x, h, w, bits, range);
        let qm = qmax(bits, range);
        let p2 = |k| reference::pow2(k);
        for i in 0..x.len() {
            let y = x[i] as f64 * p2(t.k);
            // 1. x * 2^k = q + remainder, exactly, |remainder| <= 1/2, ties only at even q
            assert_eq!(y, t.q[i] as f64 + t.remainder[i] as f64);
            assert!(t.remainder[i].abs() <= 0.5);
            assert!(t.remainder[i].abs() < 0.5 || t.q[i] % 2 == 0);
            // 2. |q| <= Q
            assert!((t.q[i] as i64).abs() <= qm);
        }
        // 3. k is maximal: the element at argmax overflows with k + 1
        let wp = w.div_ceil(32) * 32;
        let (row, col) = (t.argmax as usize / wp, t.argmax as usize % wp);
        let v = x[row * w + col].abs() as f64;
        assert!(v * p2(t.k + 1) > qm as f64);
        assert_eq!(v as f32, reference::absmax(&x));
    }
}

#[test]
fn argmax_is_first_in_tile_order() {
    // three equal maxima in a 32 x 64 tensor (two tiles): row 2 col 7 is in tile 0, the other two in tile 1
    let (h, w) = (32usize, 64usize);
    let mut x = vec![0.001f32; h * w];
    x[40] = -5.0; // row 0, col 40, tile 1
    x[37] = 5.0; //  row 0, col 37, tile 1
    x[2 * 64 + 7] = 5.0; // row 2, col 7, tile 0: first in tile order, though later in row-major order
    assert_eq!(reference::argmax(&x, h, w), 2 * 64 + 7);
    assert_eq!(simd::argmax(&x, h, w, 5.0), 2 * 64 + 7);
    // all zeros: index 0
    assert_eq!(reference::argmax(&vec![0.0; h * w], h, w), 0);
    assert_eq!(simd::argmax(&vec![0.0; h * w], h, w, 0.0), 0);
    // padded 40 x 70 -> 64 x 96: the index counts the padding
    let mut y = vec![0.001f32; 40 * 70];
    y[39 * 70 + 69] = -7.0;
    assert_eq!(reference::argmax(&y, 40, 70), 39 * 96 + 69);
    // every face of a tile
    for &(row, col) in &[(0usize, 0usize), (0, 15), (0, 16), (15, 16), (16, 0), (16, 15), (31, 31), (17, 20)] {
        let mut z = vec![0.001f32; 32 * 32];
        z[row * 32 + col] = 3.0;
        assert_eq!(reference::argmax(&z, 32, 32), (row * 32 + col) as u32);
        assert_eq!(simd::argmax(&z, 32, 32, 3.0), (row * 32 + col) as u32);
    }
}

#[test]
fn all_zero_and_ties_to_even() {
    let t = reference::quantize_with_trace(&vec![0.0; 1024], 32, 32, 8, QuantRange::Full);
    assert_eq!(t.k, 127);
    assert!(t.q.iter().all(|&q| q == 0) && t.argmax == 0);
    // absmax 63.5 with Q = 127 gives k = 1 (63.5 * 2 = 127 <= Q). Then x = 0.25 -> y = 0.5 -> q = 0 (even), r = 0.5;
    // x = 0.75 -> y = 1.5 -> q = 2, r = -0.5; x = -0.25 -> y = -0.5 -> q = 0, r = -0.5
    let mut x = vec![0.0f32; 1024];
    x[0] = 63.5;
    x[1] = 0.25;
    x[2] = 0.75;
    x[3] = -0.25;
    let t = reference::quantize_with_trace(&x, 32, 32, 8, QuantRange::Full);
    assert_eq!(t.k, 1);
    assert_eq!(&t.q[..4], &[127, 0, 2, 0]);
    assert_eq!(&t.remainder[..4], &[0.0, 0.5, -0.5, -0.5]);
}

#[test]
fn stwo_column_holds_the_residues() {
    use stwo::prover::backend::Column;
    let x = random_tensor(5, 32 * 96, 7.0);
    let t = simd::quantize_with_trace(&x, 32, 96, 32, QuantRange::M31, true, None);
    let col = simd::m31_column(&t.m31);
    assert_eq!(col.len(), t.m31.len());
    for (i, &m) in t.m31.iter().enumerate() {
        assert_eq!(col.at(i).0, m);
    }
}
