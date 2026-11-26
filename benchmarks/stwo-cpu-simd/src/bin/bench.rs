// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

//! Benchmark of the CPU / SIMD trace generation, to compare with the device (tools/stwo-cpu-simd/README.md).
//!
//! Usage: bench [--tiles 16,64,256,1024,4096] [--threads N] [--csv path]
//!
//! For every width and size, two jobs:
//!   "q+m31"     what the device does: the quantized values, the scaler and the M31 residues
//!                (`pot_quantize_per_tensor` followed by `pot_quant_m31`)
//!   "+witness"  that plus the remainder and the argmax, which the host derives itself in this design (the device
//!                does not produce them)
//! on:
//!   scalar   the oracle, one thread, f64 (always with the witness)
//!   simd-1   std::simd, one thread
//!   simd-N   std::simd on a rayon pool of N threads
//! and the stwo column build (`BaseColumn` of `PackedM31`) separately. The input is `tiles` tiles (32 x 32*tiles
//! elements, bf16-exact like the device bench). Times are the median of several runs, in milliseconds.

use std::time::{Duration, Instant};

use zkml_stwo_cpu_simd::range::QuantRange;
use zkml_stwo_cpu_simd::{reference, simd};

/// The benchmark uses the range of the M31 pipeline (it only differs from `Full` at 32 bits; the cost is the same).
const RANGE: QuantRange = QuantRange::M31;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_ms(mut f: impl FnMut()) -> f64 {
    f(); // warm up
    let mut runs = Vec::new();
    let start = Instant::now();
    while runs.len() < 5 || (start.elapsed() < Duration::from_millis(400) && runs.len() < 200) {
        let t = Instant::now();
        f();
        runs.push(t.elapsed().as_secs_f64() * 1e3);
    }
    median(runs)
}

/// xorshift64*, uniform in [-8, 8), rounded to bf16 (low 16 bits cleared).
fn make_input(n: usize) -> Vec<f32> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let u = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32;
            f32::from_bits(((u * 16.0 - 8.0).to_bits() + 0x8000) & 0xFFFF_0000)
        })
        .collect()
}

fn main() {
    let mut tiles: Vec<usize> = vec![16, 64, 256, 1024, 4096];
    let mut threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut csv_path: Option<String> = None;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--tiles" => {
                i += 1;
                tiles = args[i].split(',').map(|s| s.parse().unwrap()).collect();
            }
            "--threads" => {
                i += 1;
                threads = args[i].parse().unwrap();
            }
            "--csv" => {
                i += 1;
                csv_path = Some(args[i].clone());
            }
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();

    println!("threads for simd-N: {threads}");
    let mut csv = String::from("bits,tiles,elements,impl,trace,ms,melem_per_s,ms_per_tile\n");
    println!("\n| bits | tiles | scalar +witness ms | simd-1 q+m31 ms | simd-{threads} q+m31 ms | simd-1 +witness ms | simd-{threads} +witness ms | column build ms |");
    println!("|---|---|---|---|---|---|---|---|");
    for &bits in &[8u32, 16, 32] {
        for &t in &tiles {
            let (h, w) = (32usize, 32 * t);
            let n = h * w;
            let x = make_input(n);

            // The implementations must agree before they are timed.
            let want = reference::quantize_with_trace(&x, h, w, bits, RANGE);
            let got1 = simd::quantize_with_trace(&x, h, w, bits, RANGE, true, None);
            let gotn = simd::quantize_with_trace(&x, h, w, bits, RANGE, true, Some(&pool));
            assert_eq!(want, got1, "simd-1 differs from the oracle ({bits} bits, {t} tiles)");
            assert_eq!(want, gotn, "simd-N differs from the oracle ({bits} bits, {t} tiles)");

            let scalar = if n <= 1 << 21 { Some(time_ms(|| { std::hint::black_box(reference::quantize_with_trace(&x, h, w, bits, RANGE)); })) } else { None };
            let s1_w = time_ms(|| { std::hint::black_box(simd::quantize_with_trace(&x, h, w, bits, RANGE, true, None)); });
            let sn_w = time_ms(|| { std::hint::black_box(simd::quantize_with_trace(&x, h, w, bits, RANGE, true, Some(&pool))); });
            let s1 = time_ms(|| { std::hint::black_box(simd::quantize_with_trace(&x, h, w, bits, RANGE, false, None)); });
            let sn = time_ms(|| { std::hint::black_box(simd::quantize_with_trace(&x, h, w, bits, RANGE, false, Some(&pool))); });
            let col = time_ms(|| { std::hint::black_box(simd::m31_column(&want.m31)); });

            let rate = |ms: f64| n as f64 / (ms * 1e3); // Melem/s
            // trace = 1: with the witness (remainder and argmax); 0: q and the M31 residues only
            for (name, trace, ms) in [("scalar", 1, scalar), ("simd-1", 1, Some(s1_w)), ("simd-N", 1, Some(sn_w)), ("simd-1", 0, Some(s1)), ("simd-N", 0, Some(sn))] {
                if let Some(ms) = ms {
                    csv += &format!("{bits},{t},{n},{name},{trace},{ms:.4},{:.1},{:.6}\n", rate(ms), ms / t as f64);
                }
            }
            csv += &format!("{bits},{t},{n},column,0,{col:.4},{:.1},{:.6}\n", rate(col), col / t as f64);
            println!(
                "| {bits} | {t} | {} | {s1:.3} | {sn:.3} | {s1_w:.3} | {sn_w:.3} | {col:.3} |",
                scalar.map_or("-".to_string(), |v| format!("{v:.3}")),
            );
        }
    }
    if let Some(p) = csv_path {
        std::fs::write(&p, csv).unwrap();
        println!("\nwrote {p}");
    }
}
