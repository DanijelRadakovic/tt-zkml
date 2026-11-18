// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

//! Writes reference cases for the device test (`CrossCheckWithRust` in test_pot_quantize_per_tensor.cpp).
//!
//! Usage: dump <dir>
//!
//! For every width (8, 16, 32) and range (full, m31) it writes, for a 64 x 96 tensor of bf16-exact values,
//! `<dir>/b<bits>_<range>.{x.f32,q.i32,r.f32,m31.u32}` (raw little-endian arrays) and `<dir>/b<bits>_<range>.meta`
//! (`H W bits k argmax`). The residues are only meaningful for the m31 range.

use std::io::Write;

use zkml_stwo_cpu_simd::range::QuantRange;
use zkml_stwo_cpu_simd::reference;

/// xorshift64*, mixture of magnitudes, bf16-exact.
fn make_input(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let u = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32;
            let scale = [0.01f32, 1.0, 8.0, 300.0][i % 4];
            f32::from_bits(((((u * 2.0 - 1.0) * scale).to_bits()) + 0x8000) & 0xFFFF_0000)
        })
        .collect()
}

fn write<T: Copy>(path: &std::path::Path, v: &[T], to_bytes: impl Fn(T) -> [u8; 4]) {
    let mut f = std::fs::File::create(path).unwrap();
    for &e in v {
        f.write_all(&to_bytes(e)).unwrap();
    }
}

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("usage: dump <dir>"));
    std::fs::create_dir_all(&dir).unwrap();
    let (h, w) = (64usize, 96usize);
    for bits in [8u32, 16, 32] {
        for (range, name) in [(QuantRange::Full, "full"), (QuantRange::M31, "m31")] {
            let x = make_input(h * w, 12345 + bits as u64);
            let t = reference::quantize_with_trace(&x, h, w, bits, range);
            let b = format!("b{bits}_{name}");
            write(&dir.join(format!("{b}.x.f32")), &x, |v| v.to_le_bytes());
            write(&dir.join(format!("{b}.q.i32")), &t.q, |v| v.to_le_bytes());
            write(&dir.join(format!("{b}.r.f32")), &t.remainder, |v| v.to_le_bytes());
            write(&dir.join(format!("{b}.m31.u32")), &t.m31, |v| v.to_le_bytes());
            std::fs::write(dir.join(format!("{b}.meta")), format!("{h} {w} {bits} {} {}\n", t.k, t.argmax)).unwrap();
            println!("{b}: k={} argmax={}", t.k, t.argmax);
        }
    }
}
