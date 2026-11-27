//! Regression tests for the soundness audit (an integration test: it needs the `forgery` feature, which `cargo test` enables): a malicious prover (who is not bound by the honest trace generation) forges columns, and `verify` must reject
//! every forgery. Each test is one finding of the audit; before the fixes `verify` accepted all of them (see tt-zkml/docs/audit.md).
//!
//! The forgeries are written through `air::trace::HOOK`, which lets a test rewrite the columns of a component before they are committed.

use stwo::core::fields::m31::BaseField;
use stwo::prover::backend::Column;
use stwo::prover::backend::simd::column::BaseColumn;

use tt_zkml::air::kind::{QCols, RqCols};
use tt_zkml::air::{Kind, Which};
use tt_zkml::air::spec::*;
use tt_zkml::air::trace::{HOOK, quant_row, requant_row, requant_row_unbounded};
use tt_zkml::device::Dtype;
use tt_zkml::proof::{self, Proof, ProveError, prove_unchecked, prove_witness};
use tt_zkml::witness::*;

type Cols = Vec<BaseColumn>;

/// Pseudo-random floats in `[lo, hi)`.
fn floats(len: usize, seed: u32, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            lo + ((s >> 8) as f32 / (1u32 << 24) as f32) * (hi - lo)
        })
        .collect()
}

fn set_hook(f: impl Fn(&Kind, &mut Cols) + 'static) {
    HOOK.with(|h| *h.borrow_mut() = Some(Box::new(f)));
}

fn clear_hook() {
    HOOK.with(|h| *h.borrow_mut() = None);
}

fn digits(v: i64) -> [u32; 4] {
    assert!((0..1 << 28).contains(&v), "{v}");
    [(v & 127) as u32, ((v >> 7) & 127) as u32, ((v >> 14) & 127) as u32, ((v >> 21) & 127) as u32]
}

fn bf(v: i64) -> BaseField {
    BaseField::from(v as i32)
}

/// The forged proof must not exist or must not verify.
fn assert_rejected(forged: Result<Proof, ProveError>, public: &PublicInputs, what: &str) {
    clear_hook();
    match forged {
        Err(_) => {}
        Ok(p) => assert!(proof::verify(&p, public).is_err(), "SOUNDNESS BUG: the verifier accepted {what}"),
    }
}

/// Recomputes accumulators, exponents and outputs of the layers from `from` on, as an honest prover of the forged quantized tensors would.
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

/// Writes one complete, possibly malicious row of a float-tensor component: the quantization witness for the given values, with the field arithmetic of an
/// attacker (`r` is the remainder as the attacker claims it, `clamp`: the row says "tiny float").
#[allow(clippy::too_many_arguments)]
fn fill(cols: &mut Cols, c: &QCols, row: usize, kexp: i32, s: u32, m: BaseField, ep: u32, q: i32, qa: u32, sh: u32, r: i64, clamp: bool) {
    let n = c.nz.as_ref().unwrap();
    for i in 0..c.total {
        cols[i].set(row, BaseField::from(0u32));
    }
    let set = |cols: &mut Cols, i: usize, v: u32| cols[i].set(row, BaseField::from(v));
    cols[c.q].set(row, bf(q as i64));
    set(cols, c.s, s);
    cols[c.m].set(row, m);
    set(cols, c.ep, ep);
    let raw = 150 - ep as i64 - kexp as i64;
    set(cols, n.qa, qa);
    set(cols, n.p2, 1u32 << sh);
    set(cols, n.sh, sh);
    if clamp {
        set(cols, n.c, 1);
        set(cols, n.diff, (raw - 25).max(0) as u32);
    } else {
        set(cols, n.diff, (24 - raw).max(0) as u32);
    }
    cols[n.r].set(row, bf(r));
    let half = 1i64 << (sh - 1);
    for (&i, d) in n.v1.iter().zip(digits((half - 1 + r).max(0))) {
        set(cols, i, d);
    }
    for (&i, d) in n.v2.iter().zip(digits((half - 1 - r).max(0))) {
        set(cols, i, d);
    }
    let g = 127 - qa;
    set(cols, n.g, g);
    if g == 0 {
        set(cols, n.z, 1);
    } else {
        cols[n.ginv].set(row, BaseField::from(g).inverse());
    }
}

/// A 2 x 4 input and a 4 x 3 weight matrix with two exact zeros (they quantize to 0 with a tiny float: a clamped shift).
fn tiny_weight_setup() -> (NetworkWitness, NetworkWitness, PublicInputs) {
    let x = floats(2 * 4, 1, -2.0, 2.0);
    let mut w0 = floats(4 * 3, 2, -1.0, 1.0);
    w0[1] = 0.0;
    w0[5] = 0.0;
    let honest = NetworkWitness::simulate(2, &[4, 3], &[false], &x, &[w0], Dtype::F32).unwrap();
    honest.check().unwrap();
    let public = honest.public_inputs();
    (honest.clone(), honest, public)
}

/// F2: a clamped shift of 25 made `qa * 2^25` wrap (`64 * 2^25 = 2^31 = 1 mod p`): a zero weight was "quantized" to 64.
#[test]
fn f2_a_zero_weight_cannot_be_quantized_to_64() {
    let (honest, mut forged, public) = tiny_weight_setup();
    forged.q_w[0][1] = 64;
    recompute(&mut forged, 0, false);
    assert_ne!(forged.output(), honest.output());
    set_hook(|kind, cols| {
        if let Kind::Tensor { layer: 0, which: Which::W, kexp, .. } = kind {
            let c = QCols::new(*kexp);
            // 0.0: s = 0, m = 0, e' = 1; q = qa = 64 with sh = 25 and r = 0 - 64 * 2^25 = -1 (mod p)
            fill(cols, &c, 1, *kexp, 0, BaseField::from(0u32), 1, 64, 64, 25, -1, true);
        }
    });
    assert_rejected(prove_unchecked(&forged), &public, "a zero weight quantized to 64");
}

/// F1: `m` was only bound through `2m + s`, so `(s, m)` and `(1 - s, m + 2^30)` were the same float: the sign of a tiny float could be flipped.
#[test]
fn f1_the_mantissa_and_sign_of_a_float_are_pinned() {
    let (honest, mut forged, public) = tiny_weight_setup();
    forged.q_w[0][1] = -32;
    recompute(&mut forged, 0, false);
    assert_ne!(forged.output(), honest.output());
    set_hook(|kind, cols| {
        if let Kind::Tensor { layer: 0, which: Which::W, kexp, .. } = kind {
            let c = QCols::new(*kexp);
            let m = BaseField::from((1u32 << 30) - 1); // the other solution of 2m + s = 0
            fill(cols, &c, 1, *kexp, 1, m, 1, -32, 32, 25, -1, true);
        }
    });
    assert_rejected(prove_unchecked(&forged), &public, "a flipped sign through an unpinned mantissa");
}

/// F1, isolated from the clamped-shift fix: a float with a raw shift of 24 (not clamped), whose sign and mantissa are replaced by the other solution of
/// `2m + s = sm` (`m + 2^30`, flipped sign). The row satisfies every constraint of the tensor component; only the pinning of `s` and `m` to the public
/// float rejects it.
#[test]
fn f1b_the_mantissa_is_pinned_even_when_every_constraint_holds() {
    // the weights have absmax just below 1, so kw = 7; a float with biased exponent 119 has a raw shift 150 - 119 - 7 = 24
    let x = floats(2 * 4, 1, -2.0, 2.0);
    let mut w0 = floats(4 * 3, 2, -1.0, 1.0);
    w0[1] = 0.0046875; // 2^-8 * 1.2: scaled by 2^7 it is 0.6 -> q = 1
    let honest = NetworkWitness::simulate(2, &[4, 3], &[false], &x, &[w0.clone()], Dtype::F32).unwrap();
    honest.check().unwrap();
    assert_eq!(honest.specs[0].kw, 7);
    assert_eq!(honest.q_w[0][1], 1);
    let (s, m, ep) = float_parts(w0[1]);
    assert_eq!((s, ep), (0, 119));
    let public = honest.public_inputs();
    let mut forged = honest.clone();
    forged.q_w[0][1] = -65; // sign flipped, qa + 64
    recompute(&mut forged, 0, false);
    assert_ne!(forged.output(), honest.output());
    set_hook(move |kind, cols| {
        if let Kind::Tensor { layer: 0, which: Which::W, kexp, .. } = kind {
            let c = QCols::new(*kexp);
            // s = 1 and m' = m + 2^30 - 1 give the same 2m + s; qa = 1 + 64 with sh = 24, r = m' - 65 * 2^24 = m - 2^24 - 1 (mod p)
            let m_alt = BaseField::from(m) + BaseField::from((1u32 << 30) - 1);
            let r = m as i64 - (1i64 << 24) - 1;
            fill(cols, &c, 1, *kexp, 1, m_alt, ep, -65, 65, 24, r, false);
        }
    });
    assert_rejected(prove_unchecked(&forged), &public, "a flipped sign and shifted mantissa that satisfy every constraint");
}

/// F3: left-shift requantization wrapped: acc = 2^25 with ko = 6 gave `2^31 mod p = 1` (the true exponent is -19 and the true output 64).
#[test]
fn f3_a_left_shift_cannot_wrap() {
    let k = 2200;
    let mut x = vec![0f32; k];
    let mut w = vec![0f32; k * 2];
    for i in 0..2080 {
        x[i] = 127.0;
        w[i * 2] = 127.0;
    }
    for i in 2080..2128 {
        x[i] = 1.0;
        w[i * 2] = 127.0;
    }
    x[2128] = 1.0;
    w[2128 * 2] = 16.0;
    w[2080 * 2 + 1] = 1.0;
    let honest = NetworkWitness::simulate(1, &[k, 2], &[false], &x, &[w], Dtype::F32).unwrap();
    honest.check().unwrap();
    assert_eq!(honest.acc[0], vec![1 << 25, 1]);
    let public = honest.public_inputs();
    let mut forged = honest.clone();
    forged.specs[0].ko = 6;
    forged.out[0] = vec![1, 64];
    set_hook(|kind, cols| {
        if let Kind::Requant { layer: 0, spec, .. } = kind {
            let c = RqCols::new(spec);
            let h = c.hit.as_ref().unwrap();
            let t = spec.hit_threshold() as i64;
            cols[h.hp].set(0, BaseField::from(1u32));
            for (&i, d) in h.limbs.iter().zip(digits((1i64 << 25) - t)) {
                cols[i].set(0, BaseField::from(d));
            }
        }
    });
    assert_rejected(prove_unchecked(&forged), &public, "a wrapped left shift");
}

/// F4a: the "largest accumulator" could be a padding row, so an exponent that is not the largest was accepted.
#[test]
fn f4a_the_largest_accumulator_cannot_be_a_padding_row() {
    let x = floats(3 * 4, 11, -2.0, 2.0);
    let w = floats(4 * 3, 12, -1.0, 1.0);
    let honest = NetworkWitness::simulate(3, &[4, 3], &[false], &x, &[w], Dtype::F32).unwrap();
    assert!(honest.specs[0].ko < -1);
    let public = honest.public_inputs();
    let mut forged = honest.clone();
    forged.specs[0].ko -= 1; // not the largest exponent
    recompute(&mut forged, 0, true);
    assert_ne!(forged.output(), honest.output());
    set_hook(|kind, cols| {
        if let Kind::Requant { layer: 0, spec, .. } = kind {
            let c = RqCols::new(spec);
            let row = spec.matmul_rows(); // the first padding row
            let acc = spec.hit_threshold() as i32;
            let out = activate(acc, spec);
            for (i, v) in requant_row(spec, &c, row, acc, out, 1).unwrap().into_iter().enumerate() {
                cols[i].set(row, v);
            }
        }
    });
    assert_rejected(prove_unchecked(&forged), &public, "a padding row as the largest accumulator");
}

/// F4b: the same for the scaler of a float tensor: `kw - 1` with a padding row as the largest element.
#[test]
fn f4b_the_largest_float_cannot_be_a_padding_row() {
    let x = floats(2 * 4, 21, -2.0, 2.0);
    let w = floats(4 * 3, 22, -1.0, 1.0);
    let honest = NetworkWitness::simulate(2, &[4, 3], &[false], &x, &[w.clone()], Dtype::F32).unwrap();
    let public = honest.public_inputs();
    let mut forged = honest.clone();
    forged.specs[0].kw -= 1;
    let kw = forged.specs[0].kw;
    forged.q_w[0] = w.iter().map(|&v| quantize(v, 2f32.powi(kw), Dtype::F32)).collect();
    recompute(&mut forged, 0, false);
    assert_ne!(forged.output(), honest.output());
    set_hook(|kind, cols| {
        if let Kind::Tensor { layer: 0, which: Which::W, kexp, spec, .. } = kind {
            let c = QCols::new(*kexp);
            let row = spec.k * spec.n;
            for (i, v) in quant_row(*kexp, &c, row, (0, 254, (149 - *kexp) as u32), 127, true).unwrap().into_iter().enumerate() {
                cols[i].set(row, v);
            }
        }
    });
    assert_rejected(prove_unchecked(&forged), &public, "a padding row as the largest float");
}

/// F5: a right-shift exponent one too large was accepted when `127 * 2^s < max|acc| < 127 * 2^s + 2^(s-1)` (the output rounds to 127).
#[test]
fn f5_a_requantization_exponent_that_overflows_127_is_rejected() {
    let honest = NetworkWitness::simulate(1, &[2, 2], &[false], &[127.0, 2.0], &[vec![127.0, 1.0, 64.0, 0.0]], Dtype::F32).unwrap();
    honest.check().unwrap();
    assert_eq!(honest.acc[0], vec![16257, 127]);
    assert_eq!(honest.specs[0].ko, -8);
    let public = honest.public_inputs();
    let mut forged = honest.clone();
    forged.specs[0].ko = -7;
    recompute(&mut forged, 0, true);
    // the row of 16257 is complete and consistent except that its bound limbs stay zero: only the bound `acc <= 127 * 2^s` can reject it
    set_hook(|kind, cols| {
        if let Kind::Requant { layer: 0, spec, .. } = kind {
            let c = RqCols::new(spec);
            for (i, v) in requant_row_unbounded(spec, &c, 0, 16257, 127, 1).unwrap().into_iter().enumerate() {
                cols[i].set(0, v);
            }
        }
    });
    assert_rejected(prove_unchecked(&forged), &public, "an exponent one too large (127 * 2^s < |acc|)");
}

/// F6: under a ReLU the raw negative accumulators were not bounded by the exponent.
#[test]
fn f6_a_relu_layer_bounds_its_negative_accumulators() {
    let honest = NetworkWitness::simulate(1, &[1, 2], &[true], &[127.0], &[vec![-127.0, 1.0]], Dtype::F32).unwrap();
    honest.check().unwrap();
    assert_eq!(honest.acc[0], vec![-16129, 127]);
    let public = honest.public_inputs();
    let mut forged = honest.clone();
    forged.specs[0].ko = -1;
    recompute(&mut forged, 0, true);
    // the row of -16129 is complete and consistent except that its bound limbs stay zero (hit on its magnitude)
    set_hook(|kind, cols| {
        if let Kind::Requant { layer: 0, spec, .. } = kind {
            let c = RqCols::new(spec);
            for (i, v) in requant_row_unbounded(spec, &c, 0, -16129, 0, -1).unwrap().into_iter().enumerate() {
                cols[i].set(0, v);
            }
        }
    });
    assert_rejected(prove_unchecked(&forged), &public, "a ReLU layer with an exponent too large for its negative accumulators");
}

/// F7: the verifier took the activations and the shapes of the layers from the proof.
#[test]
fn f7_the_statement_is_bound_to_the_verifiers_model() {
    let x = floats(4 * 8, 31, -2.0, 2.0);
    let w = floats(8 * 8, 32, -1.0, 1.0);
    let honest = NetworkWitness::simulate(4, &[8, 8], &[false], &x, &[w.clone()], Dtype::F32).unwrap();
    let public = honest.public_inputs();
    // another activation
    let relu = NetworkWitness::simulate(4, &[8, 8], &[true], &x, &[w.clone()], Dtype::F32).unwrap();
    let proof = prove_witness(&relu).unwrap();
    assert_ne!(proof.output, honest.output());
    assert!(proof::verify(&proof, &public).is_err(), "a proof with another activation was accepted");
    // another split of the same flat floats (2 x 16 @ 16 x 4)
    let reshaped = NetworkWitness::simulate(2, &[16, 4], &[false], &x, &[w], Dtype::F32).unwrap();
    let proof = prove_witness(&reshaped).unwrap();
    assert!(proof::verify(&proof, &public).is_err(), "a proof of a reshaped network was accepted");
    // the honest proof still verifies, and also its own model's description
    proof::verify(&prove_witness(&honest).unwrap(), &public).unwrap();
}

/// The statement is part of the transcript: moving any piece of it (exponent, activation, output, claimed sums) invalidates the proof.
#[test]
fn the_statement_and_the_claimed_sums_are_bound_to_the_transcript() {
    let honest = NetworkWitness::simulate(4, &[8, 8], &[true], &floats(32, 5, -2.0, 2.0), &[floats(64, 6, -1.0, 1.0)], Dtype::F32).unwrap();
    let public = honest.public_inputs();
    let proof = prove_witness(&honest).unwrap();
    proof::verify(&proof, &public).unwrap();
    for edit in [|p: &mut Proof| p.output[0] ^= 1, |p: &mut Proof| p.specs[0].ko -= 1, |p: &mut Proof| p.specs[0].kx -= 1, |p: &mut Proof| p.specs[0].kw += 1] {
        let mut forged = proof.clone();
        edit(&mut forged);
        assert!(proof::verify(&forged, &public).is_err());
    }
}

/// Canonical statements and bounded work: one spelling of "all zero", no absurd shapes, no non-finite floats.
#[test]
fn statements_are_canonical_and_bounded() {
    let honest = NetworkWitness::simulate(2, &[2, 2], &[false], &[1.0, 1.0, 1.0, 1.0], &[vec![0.0; 4]], Dtype::F32).unwrap();
    assert_eq!(honest.specs[0].ko, 127);
    let public = honest.public_inputs();
    let proof = prove_witness(&honest).unwrap();
    proof::verify(&proof, &public).unwrap();
    // 41 ..= 126 also meant "all zero": only 127 is allowed
    let mut forged = proof.clone();
    forged.specs[0].ko = 100;
    assert!(proof::verify(&forged, &public).is_err());
    // shapes that would make the verifier allocate gigabytes
    let mut forged = proof.clone();
    forged.specs[0].m = 1 << 40;
    assert!(proof::verify(&forged, &public).is_err());
    let mut forged = proof.clone();
    forged.specs = vec![forged.specs[0]; 100];
    forged.claimed_sums = vec![forged.claimed_sums[0]; 100];
    assert!(proof::verify(&forged, &public).is_err());
    // a float that is finite in f32 but infinite in bf16
    let mut other = public.clone();
    other.dtype = Dtype::Bf16;
    other.input[0] = f32::MAX;
    assert!(proof::verify(&proof, &other).is_err());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Second audit round (findings G1 to G6).

/// G1: a left-shift requantization under a ReLU bounded only `max(acc, 0)`: a negative accumulator could exceed `127 >> ko`, so an exponent one too
/// large was accepted (acc = [20, -100]: the true exponent is 0, the forged one 1 with output [40, 0] instead of [20, 0]).
#[test]
fn g1_a_relu_left_shift_bounds_its_negative_accumulators() {
    let honest = NetworkWitness::simulate(1, &[2, 2], &[true], &[1.0, 63.0 / 64.0], &[vec![20. / 64., -100. / 64., -20. / 64., 100. / 64.]], Dtype::F32).unwrap();
    honest.check().unwrap();
    assert_eq!(honest.acc[0], vec![20, -100]);
    assert_eq!(honest.specs[0].ko, 0);
    let public = honest.public_inputs();
    let mut forged = honest.clone();
    forged.specs[0].ko = 1; // 100 * 2 = 200 > 127
    forged.out[0] = vec![40, 0];
    // the attacker's row for acc = -100 is complete and consistent except for the raw bound: it is built as the row of -63 (which fits) and then
    // given acc = -100, its ReLU limbs (-acc = 100), its hit limbs (|acc| - T = 100 - 32) and the scaled value -200
    set_hook(|kind, cols| {
        if let Kind::Requant { layer: 0, spec, .. } = kind {
            let c = RqCols::new(spec);
            let mut row = requant_row(spec, &c, 1, -63, 0, -1).unwrap();
            row[c.acc] = BaseField::from(-100i32);
            let relu = c.relu.as_ref().unwrap();
            for (i, d) in relu.neg_limbs.iter().zip(digits(100)) {
                row[*i] = BaseField::from(d);
            }
            let hit = c.hit.as_ref().unwrap();
            for (i, d) in hit.limbs.iter().zip(digits(100 - spec.hit_threshold() as i64)) {
                row[*i] = BaseField::from(d);
            }
            row[c.left_scaled.unwrap()] = BaseField::from(-200i32);
            for (i, v) in row.into_iter().enumerate() {
                cols[i].set(1, v);
            }
        }
    });
    assert_rejected(prove_unchecked(&forged), &public, "a ReLU left shift with an exponent too large for a negative accumulator");
}

/// G2: `i32::MIN.abs()` wraps in a release build, and `i32::MIN` is `-1` in M31: a public output of `i32::MIN` passed the range check and matched -1.
#[test]
fn g2_the_public_output_is_range_checked_without_overflow() {
    let honest = NetworkWitness::simulate(1, &[2, 2], &[false], &[1.0, 0.0], &[vec![-1. / 64., -100. / 64., 1. / 64., 100. / 64.]], Dtype::F32).unwrap();
    honest.check().unwrap();
    assert_eq!(honest.output(), &[-1, -100]);
    let public = honest.public_inputs();
    let mut forged = honest.clone();
    forged.out[0][0] = i32::MIN; // the same field element as -1
    let spec = forged.specs[0];
    set_hook(move |kind, cols| {
        if let Kind::Requant { layer: 0, .. } = kind {
            let c = RqCols::new(&spec);
            for (i, v) in requant_row(&spec, &c, 0, -64, -1, 0).unwrap().into_iter().enumerate() {
                cols[i].set(0, v);
            }
        }
    });
    let forged_proof = prove_unchecked(&forged);
    if let Ok(p) = &forged_proof {
        assert_eq!(p.output[0], i32::MIN);
    }
    assert_rejected(forged_proof, &public, "a public output of i32::MIN");
}

/// G3: a structurally malformed stwo proof (inner vectors of the wrong length) made the verifier panic, and one case aborted the process. It must be
/// a plain rejection.
#[test]
fn g3_malformed_proofs_are_rejected_without_panicking() {
    let honest = NetworkWitness::simulate(2, &[4, 3], &[true], &floats(8, 1, -2.0, 2.0), &[floats(12, 2, -1.0, 1.0)], Dtype::F32).unwrap();
    let public = honest.public_inputs();
    let proof = prove_witness(&honest).unwrap();
    proof::verify(&proof, &public).unwrap();
    let edits: Vec<(&str, fn(&mut serde_json::Value))> = vec![
        ("a sampled column with one value missing", |v| { v["sampled_values"][1][0].as_array_mut().unwrap().pop(); }),
        ("an interaction column with one value too many", |v| { let x = v["sampled_values"][2][0][0].clone(); v["sampled_values"][2][0].as_array_mut().unwrap().push(x); }),
        ("an empty sampled column", |v| { v["sampled_values"][1][0] = serde_json::json!([]); }),
        ("one decommitment missing", |v| { v["decommitments"].as_array_mut().unwrap().pop(); }),
        ("one tree of queried values missing", |v| { v["queried_values"].as_array_mut().unwrap().pop(); }),
        ("a queried column missing", |v| { v["queried_values"][1].as_array_mut().unwrap().pop(); }),
        ("a queried value missing", |v| { v["queried_values"][1][0].as_array_mut().unwrap().pop(); }),
        ("all queried values of a tree one short", |v| { for c in v["queried_values"][2].as_array_mut().unwrap() { c.as_array_mut().unwrap().pop(); } }),
        ("a composition column missing", |v| { v["sampled_values"][3].as_array_mut().unwrap().pop(); }),
    ];
    for (what, edit) in edits {
        let mut json: serde_json::Value = serde_json::from_slice(&proof.bytes).unwrap();
        edit(&mut json);
        let mut forged = proof.clone();
        forged.bytes = serde_json::to_vec(&json).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| proof::verify(&forged, &public)));
        match result {
            Ok(r) => assert!(r.is_err(), "the verifier accepted {what}"),
            Err(_) => panic!("the verifier panicked on {what}"),
        }
    }
}

/// G4: LogUp proves equality of counts modulo p only; a statement with p or more range lookups is refused.
#[test]
fn g4_statements_with_too_many_range_lookups_are_refused() {
    use tt_zkml::air::{check_lookup_budget, layout, range_lookups};
    let spec = |relu| LayerSpec { m: 2048, k: 1024, n: 1024, relu, ko: -18, kw: 7, kx: 5 };
    let big: Vec<LayerSpec> = (0..64).map(|l| spec(l % 2 == 0)).collect();
    validate(&big).unwrap(); // every limit of a single layer is respected
    assert!(range_lookups(&layout(&big)) >= (1u128 << 31) - 1);
    assert!(check_lookup_budget(&layout(&big)).is_err());
    let small = vec![LayerSpec { m: 32, k: 64, n: 32, relu: true, ko: -12, kw: 7, kx: 5 }];
    assert!(check_lookup_budget(&layout(&small)).is_ok());
}

/// G5: absurd exponents in the proof are refused without an arithmetic overflow (a panic in a debug build).
#[test]
fn g5_absurd_exponents_are_refused_without_panicking() {
    let honest = NetworkWitness::simulate(2, &[2, 2], &[false], &[1.0, 0.5, -0.25, 2.0], &[vec![0.5, -1.0, 0.25, 1.0]], Dtype::F32).unwrap();
    let public = honest.public_inputs();
    let proof = prove_witness(&honest).unwrap();
    for ko in [i32::MIN, i32::MIN + 1, -21, 7, 40, 126, i32::MAX] {
        let mut forged = proof.clone();
        forged.specs[0].ko = ko;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| proof::verify(&forged, &public)));
        assert!(matches!(result, Ok(Err(_))), "ko = {ko}");
    }
    for k in [i32::MIN, -101, 128, i32::MAX] {
        let mut forged = proof.clone();
        forged.specs[0].kw = k;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| proof::verify(&forged, &public)));
        assert!(matches!(result, Ok(Err(_))), "kw = {k}");
    }
}

/// G6: a NaN in the public floats rounded to zero in bf16 and passed the finiteness check (which looked only at the rounded values).
#[test]
fn g6_non_finite_public_floats_are_refused_before_rounding() {
    // x[1] = -0.0: a NaN that bf16 rounds to -0.0 is the "same" float for the circuit, so only the finiteness check of the raw value rejects it
    let mut x = vec![1.0f32, -0.0, -0.25, 2.0];
    let honest = NetworkWitness::simulate(2, &[2, 2], &[false], &x, &[vec![0.5, -1.0, 0.25, 1.0]], Dtype::Bf16).unwrap();
    let mut public = honest.public_inputs();
    public.dtype = Dtype::Bf16;
    let proof = prove_witness(&honest).unwrap();
    proof::verify(&proof, &public).unwrap();
    x[1] = f32::from_bits(0x7fff_ffff); // a NaN that bf16 rounding turns into -0.0
    assert_eq!(round_bf16(x[1]).to_bits(), 0x8000_0000);
    public.input = x;
    assert!(proof::verify(&proof, &public).is_err());
}
