//! Randomized forgery test: a "malicious prover" forges random networks' witnesses and columns, and the verifier must never accept a false statement.
//!
//! The statement of a proof is fully determined by the public floats: for a fixed model and input, the true `(specs, output)` is unique. So the oracle is
//! simple: whatever the forger does, if `verify(proof, honest public inputs)` returns `Ok`, then `proof.specs` and `proof.output` must be the honest ones.
//! (Accepting a proof of the true statement is fine: a mutation may leave a free column of a padding row, say, untouched in effect.)
//!
//! Each iteration builds a random small network (random shapes, ReLU flags, dtype, values that include zeros, ties, tiny and large floats), then forges:
//!
//! * the witness: random edits of quantized tensors, accumulators, outputs, exponents, activations and the prover's own floats, with or without recomputing
//!   the rest consistently (a consistent forgery of another network is the hardest attack);
//! * the columns: random cells of random components are overwritten with attacker-chosen field elements (through the `forgery` hook);
//! * the proof after the fact: random edits of the statement, the claimed sums and the proof bytes, and of the verifier's public inputs.
//!
//! stwo's prover refuses traces that violate the constraints (it checks the composition polynomial), so for most column forgeries no proof exists; the test
//! counts those separately. What is left reaches the verifier, which must reject it (LogUp sums, preprocessed root, statement binding) or accept only the
//! true statement.
//!
//! Environment: `FUZZ_ITERS` (default 120), `FUZZ_SEED` (default fixed). A failure prints the seed and iteration to reproduce it.

use std::panic::{AssertUnwindSafe, catch_unwind};

use stwo::core::fields::m31::BaseField;
use stwo::prover::backend::Column;
use stwo::prover::backend::simd::column::BaseColumn;

use tt_zkml::air::spec::{LayerSpec, choose_ko};
use tt_zkml::air::trace::HOOK;
use tt_zkml::air::Kind;
use tt_zkml::device::Dtype;
use tt_zkml::proof::{self, ModelCommitment, Proof, ProveError, prove_unchecked, prove_witness};
use tt_zkml::air::spec::activate;
use tt_zkml::witness::{NetworkWitness, PublicInputs, matmul};

// ---- a small deterministic generator ----

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut x = self.0;
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^ (x >> 29)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo + 1) as u64) as i64
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// A float from a mix of interesting populations: ordinary values, exact zeros, exact ties of the quantization, tiny values, subnormals, powers of two.
fn float(rng: &mut Rng, scale: f32) -> f32 {
    match rng.below(10) {
        0 => 0.0,
        1 => -0.0,
        2 => (rng.range(-300, 300) as f32 + 0.5) * scale / 64.0, // many exact halves
        3 => scale * 2f32.powi(-(rng.range(8, 40) as i32)) * (rng.unit() + 0.5), // tiny
        4 => f32::from_bits(rng.range(1, 1000) as u32), // subnormal
        5 => scale * 2f32.powi(rng.range(-3, 3) as i32) * if rng.chance(50) { 1.0 } else { -1.0 },
        _ => (rng.unit() * 2.0 - 1.0) * scale,
    }
}

fn random_network(rng: &mut Rng) -> (NetworkWitness, Vec<usize>, Vec<bool>) {
    loop {
        let layers = 1 + rng.below(3);
        let m = 1 + rng.below(6);
        let dims: Vec<usize> = (0..=layers).map(|_| 1 + rng.below(10)).collect();
        let relu: Vec<bool> = (0..layers).map(|_| rng.chance(50)).collect();
        let dtype = if rng.chance(40) { Dtype::Bf16 } else { Dtype::F32 };
        let (sx, sw) = (1.0 + rng.unit() * 3.0, 0.2 + rng.unit() * 2.0);
        let x: Vec<f32> = (0..m * dims[0]).map(|_| float(rng, sx)).collect();
        let w: Vec<Vec<f32>> = (0..layers).map(|l| (0..dims[l] * dims[l + 1]).map(|_| float(rng, sw)).collect()).collect();
        if let Ok(net) = NetworkWitness::simulate(m, &dims, &relu, &x, &w, dtype) {
            if net.check().is_ok() {
                return (net, dims, relu);
            }
        }
    }
}

// ---- forging the witness ----

/// Recomputes accumulators, exponents and outputs from layer `from` on, as an honest prover of the forged tensors would.
fn recompute(w: &mut NetworkWitness, from: usize, keep_ko: bool) {
    for l in from..w.specs.len() {
        let s = w.specs[l];
        let acc = matmul(s.m, s.k, s.n, w.x(l), &w.q_w[l]);
        if !keep_ko {
            w.specs[l].ko = choose_ko(&acc);
        }
        let spec = w.specs[l];
        w.out[l] = acc.iter().map(|&a| activate(a, &spec)).collect();
        w.acc[l] = acc;
    }
}

/// Treats the accumulators of layer `l` as the truth (an attacker who skips the matmul): the exponent and the output are derived from them, the rest of
/// the network is recomputed honestly from that output.
fn reconcile_from_acc(w: &mut NetworkWitness, l: usize) {
    w.specs[l].ko = choose_ko(&w.acc[l]);
    let spec = w.specs[l];
    w.out[l] = w.acc[l].iter().map(|&a| activate(a, &spec)).collect();
    recompute(w, l + 1, false);
}

fn some_value(rng: &mut Rng) -> i32 {
    match rng.below(5) {
        0 => rng.range(-3, 3) as i32,
        1 => rng.range(-130, 130) as i32,
        2 => [127, -127, 128, -128, 255, 0, 1, -1][rng.below(8)],
        3 => rng.range(-100_000, 100_000) as i32,
        _ => rng.next() as i32,
    }
}

/// Forges the witness with 1 to 3 random edits; returns a description for failure messages.
fn forge_witness(rng: &mut Rng, w: &mut NetworkWitness) -> String {
    let mut what = Vec::new();
    for _ in 0..1 + rng.below(3) {
        let l = rng.below(w.specs.len());
        match rng.below(15) {
            0 => {
                let i = rng.below(w.q_x0.len());
                w.q_x0[i] = if rng.chance(50) { w.q_x0[i] + rng.range(-2, 2) as i32 } else { some_value(rng) };
                what.push(format!("q_x0[{i}]"));
            }
            1 => {
                let i = rng.below(w.q_w[l].len());
                w.q_w[l][i] = if rng.chance(50) { w.q_w[l][i] + rng.range(-2, 2) as i32 } else { some_value(rng) };
                what.push(format!("q_w[{l}][{i}]"));
            }
            2 => {
                let i = rng.below(w.acc[l].len());
                w.acc[l][i] = if rng.chance(50) { w.acc[l][i] + rng.range(-2, 2) as i32 } else { some_value(rng) };
                what.push(format!("acc[{l}][{i}]"));
            }
            3 => {
                let i = rng.below(w.out[l].len());
                w.out[l][i] = some_value(rng);
                what.push(format!("out[{l}][{i}]"));
            }
            4 => {
                w.specs[l].ko += rng.range(-3, 3) as i32;
                what.push(format!("ko[{l}]"));
            }
            5 => {
                w.specs[l].kw += rng.range(-3, 3) as i32;
                what.push(format!("kw[{l}]"));
            }
            6 => {
                let d = rng.range(-3, 3) as i32;
                w.specs.iter_mut().for_each(|s| s.kx += d);
                what.push("kx".to_string());
            }
            7 => {
                w.specs[l].relu = !w.specs[l].relu;
                what.push(format!("relu[{l}]"));
            }
            8 => {
                // the prover's own float differs from the public one
                let i = rng.below(w.w_float[l].len());
                w.w_float[l][i] = f32::from_bits(w.w_float[l][i].to_bits() ^ (1 << rng.below(31)));
                what.push(format!("w_float[{l}][{i}]"));
            }
            9 => {
                let i = rng.below(w.x0_float.len());
                w.x0_float[i] = f32::from_bits(w.x0_float[i].to_bits() ^ (1 << rng.below(31)));
                what.push(format!("x0_float[{i}]"));
            }
            10 => {
                // an accumulator that is not the matmul of the tensors, with everything after it derived consistently from it
                let i = rng.below(w.acc[l].len());
                w.acc[l][i] += rng.range(-300, 300) as i32;
                reconcile_from_acc(w, l);
                what.push(format!("acc[{l}][{i}] + reconcile"));
            }
            11 => {
                // an output that is not the requantization of the accumulators, the next layers computed honestly from it
                let i = rng.below(w.out[l].len());
                w.out[l][i] = some_value(rng).clamp(-127, 127);
                recompute(w, l + 1, false);
                what.push(format!("out[{l}][{i}] + propagate"));
            }
            12 => {
                // the activation flipped, everything after it derived consistently
                w.specs[l].relu = !w.specs[l].relu;
                reconcile_from_acc(w, l);
                what.push(format!("relu[{l}] + reconcile"));
            }
            13 => {
                // an exponent that is not the largest, outputs derived consistently from it
                w.specs[l].ko += rng.range(-2, 1) as i32;
                let spec = w.specs[l];
                w.out[l] = w.acc[l].iter().map(|&a| activate(a, &spec)).collect();
                recompute(w, l + 1, false);
                what.push(format!("ko[{l}] + derive outputs"));
            }
            _ => {
                // another network altogether: different weights quantized and run honestly, claimed for the public floats
                let i = rng.below(w.w_float[l].len());
                w.w_float[l][i] = -w.w_float[l][i] + 0.5;
                what.push(format!("other weights[{l}][{i}]"));
            }
        }
    }
    if rng.chance(50) {
        let from = rng.below(w.specs.len());
        let keep = rng.chance(50);
        recompute(w, from, keep);
        what.push(format!("recompute(from {from}, keep_ko {keep})"));
    }
    what.join(", ")
}

// ---- forging the columns ----

/// Overwrites `n` random cells of random components with attacker-chosen field elements.
fn install_column_forger(seed: u64) {
    HOOK.with(|h| {
        *h.borrow_mut() = Some(Box::new(move |_kind: &Kind, cols: &mut Vec<BaseColumn>| {
            let mut rng = Rng(seed ^ cols.len() as u64 * 7919);
            if !rng.chance(35) {
                return;
            }
            for _ in 0..1 + rng.below(4) {
                let c = rng.below(cols.len());
                let rows = cols[c].len();
                let r = rng.below(rows);
                let v = match rng.below(7) {
                    0 => BaseField::from(rng.next() as u32 % 2147483647),
                    1 => BaseField::from(0u32),
                    2 => BaseField::from(1u32),
                    3 => BaseField::from(2147483646u32), // -1
                    4 => BaseField::from(1u32 << 30),
                    5 => cols[c].at(r) + BaseField::from(1u32),
                    _ => cols[rng.below(cols.len())].at(rng.below(rows)), // a copy of another cell
                };
                cols[c].set(r, v);
            }
        }));
    });
}

fn clear_hook() {
    HOOK.with(|h| *h.borrow_mut() = None);
}

// ---- the oracle ----

#[derive(Default, Debug)]
struct Stats {
    iterations: usize,
    no_proof: usize,
    rejected: usize,
    accepted_true: usize,
    forged_but_identical: usize,
    tampered: usize,
}

/// `verify` must reject, or accept only the true statement.
fn judge(proof: &Proof, honest: &Proof, public: &PublicInputs, stats: &mut Stats, context: &str) {
    match proof::verify(proof, public) {
        Err(_) => stats.rejected += 1,
        Ok(()) => {
            assert!(
                proof.specs == honest.specs && proof.output == honest.output,
                "SOUNDNESS BUG: a FALSE statement was accepted.\n{context}\nhonest specs {:?}\nforged specs {:?}\nhonest output {:?}\nforged output {:?}",
                honest.specs,
                proof.specs,
                honest.output,
                proof.output
            );
            stats.accepted_true += 1;
        }
    }
}

fn attempt(forged: &NetworkWitness, honest: &Proof, public: &PublicInputs, stats: &mut Stats, context: &str) {
    match catch_unwind(AssertUnwindSafe(|| prove_unchecked(forged))) {
        Ok(Ok(proof)) => judge(&proof, honest, public, stats, context),
        Ok(Err(ProveError::Witness(_) | ProveError::Prover(_))) | Err(_) => stats.no_proof += 1,
        Ok(Err(_)) => stats.no_proof += 1,
    }
    clear_hook();
}

// ---- the test ----

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[test]
fn a_forger_cannot_make_the_verifier_accept_a_false_statement() {
    let iters = env_u64("FUZZ_ITERS", 120) as usize;
    let seed = env_u64("FUZZ_SEED", 0x5EED_2026);
    let mut stats = Stats::default();

    for it in 0..iters {
        let mut rng = Rng(seed ^ (it as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let (honest_w, dims, relu) = random_network(&mut rng);
        let public = honest_w.public_inputs();
        let honest = prove_witness(&honest_w).unwrap_or_else(|e| panic!("an honest network must prove (seed {seed}, iteration {it}, dims {dims:?}, relu {relu:?}): {e}"));
        proof::verify(&honest, &public).unwrap_or_else(|e| panic!("an honest proof must verify (seed {seed}, iteration {it}, dims {dims:?}): {e}"));
        let context = |what: &str| format!("seed {seed}, iteration {it}, dims {dims:?}, relu {relu:?}, forging: {what}");
        stats.iterations += 1;

        // 1. a forged witness, proven as if it were honest
        let mut forged = honest_w.clone();
        let what = forge_witness(&mut rng, &mut forged);
        attempt(&forged, &honest, &public, &mut stats, &context(&what));

        // 2. forged columns of the honest witness (the statement stays the true one: the hook can only win by finding another way to prove a
        //    different statement, which `judge` would catch)
        let col_seed = rng.next();
        install_column_forger(col_seed);
        attempt(&honest_w, &honest, &public, &mut stats, &context(&format!("random cells, seed {col_seed}")));

        // 3. forged witness AND forged columns
        let mut forged = honest_w.clone();
        let what = forge_witness(&mut rng, &mut forged);
        let col_seed = rng.next();
        install_column_forger(col_seed);
        attempt(&forged, &honest, &public, &mut stats, &context(&format!("{what} + random cells, seed {col_seed}")));

        // 4. the honest proof, tampered with after the fact: the statement, the claimed sums, the bytes
        for _ in 0..3 {
            let mut t = honest.clone();
            let what = match rng.below(7) {
                0 => {
                    let i = rng.below(t.output.len());
                    t.output[i] = some_value(&mut rng);
                    "output"
                }
                1 => {
                    let l = rng.below(t.specs.len());
                    t.specs[l].ko += rng.range(-2, 2) as i32;
                    "ko"
                }
                2 => {
                    let l = rng.below(t.specs.len());
                    t.specs[l].kw += rng.range(-2, 2) as i32;
                    "kw"
                }
                3 => {
                    let d = rng.range(-2, 2) as i32;
                    t.specs.iter_mut().for_each(|s| s.kx += d);
                    "kx"
                }
                4 => {
                    let l = rng.below(t.specs.len());
                    t.specs[l].relu = !t.specs[l].relu;
                    "relu"
                }
                5 => {
                    let i = rng.below(t.claimed_sums.len());
                    t.claimed_sums[i] += stwo::core::fields::qm31::SecureField::from_u32_unchecked(rng.next() as u32 % 2147483647, 0, 0, 0);
                    "claimed sum"
                }
                _ => {
                    let i = rng.below(t.bytes.len());
                    t.bytes[i] ^= 1 << rng.below(8);
                    "proof bytes"
                }
            };
            stats.tampered += 1;
            if t.specs != honest.specs || t.output != honest.output || t.claimed_sums != honest.claimed_sums || t.bytes != honest.bytes {
                judge(&t, &honest, &public, &mut stats, &context(&format!("tampered {what}")));
            } else {
                stats.forged_but_identical += 1;
            }
        }

        // 5. the honest proof against other public inputs: it is about this committed model and this input only
        let mut other = public.clone();
        let mut same_model = true;
        match rng.below(5) {
            0 => {
                let i = rng.below(other.input.len());
                other.input[i] = f32::from_bits(other.input[i].to_bits() ^ (1 << rng.range(0, 22))); // a mantissa bit
            }
            1 => {
                // the commitment of a model with one other weight
                let mut weights = honest_w.w_float.clone();
                let l = rng.below(weights.len());
                let i = rng.below(weights[l].len());
                weights[l][i] = f32::from_bits(weights[l][i].to_bits() ^ (1 << rng.range(0, 22)));
                other.model = ModelCommitment::from_floats(other.model.dtype, other.model.layers.clone(), &weights).unwrap();
                same_model = other.model.root == public.model.root;
            }
            2 => {
                let l = rng.below(other.model.layers.len());
                other.model.layers[l].relu = !other.model.layers[l].relu;
                same_model = false;
            }
            3 => {
                other.model.root[rng.below(32)] ^= 1 << rng.below(8);
                same_model = false;
            }
            _ => other.model.dtype = if other.model.dtype == Dtype::F32 { Dtype::Bf16 } else { Dtype::F32 },
        }
        // the verifier's input changed: a proof of the honest statement is not a proof about it unless the change is invisible (bf16 rounding)
        let eff = |v: &[f32], d: Dtype| tt_zkml::witness::effective(v, d);
        let same = same_model && eff(&other.input, other.model.dtype) == eff(&public.input, public.model.dtype);
        let verdict = proof::verify(&honest, &other);
        if !same {
            assert!(verdict.is_err(), "SOUNDNESS BUG: the proof verified for other public inputs. {}", context("other public inputs"));
        }
        stats.tampered += 1;
    }

    println!("forgery fuzz: {stats:?}");
    assert!(stats.rejected > 0, "no forgery reached the verifier: the fuzzer is not exercising it");
    assert!(stats.iterations == iters);
}

/// A spec that is silly on its own (the verifier must refuse it, not panic or run out of memory).
#[test]
fn nonsense_statements_are_refused_without_panicking() {
    let mut rng = Rng(7);
    let (w, _, _) = random_network(&mut rng);
    let public = w.public_inputs();
    let honest = prove_witness(&w).unwrap();
    for _ in 0..300 {
        let mut p = honest.clone();
        let l = rng.below(p.specs.len());
        let s: &mut LayerSpec = &mut p.specs[l];
        match rng.below(8) {
            0 => s.m = rng.next() as usize,
            1 => s.k = rng.next() as usize,
            2 => s.n = rng.below(1 << 30),
            3 => s.ko = rng.next() as i32,
            4 => s.kw = rng.next() as i32,
            5 => s.kx = rng.next() as i32,
            6 => p.specs = vec![p.specs[0]; rng.below(300)],
            _ => p.claimed_sums.truncate(rng.below(3)),
        }
        let result = catch_unwind(AssertUnwindSafe(|| proof::verify(&p, &public)));
        match result {
            Ok(Ok(())) => assert!(p.specs == honest.specs && p.output == honest.output, "a nonsense statement was accepted"),
            Ok(Err(_)) => {}
            Err(_) => panic!("verify panicked on a nonsense statement: {:?}", p.specs),
        }
    }
}
