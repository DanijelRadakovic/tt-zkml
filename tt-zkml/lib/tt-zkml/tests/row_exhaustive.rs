//! Exhaustive row-level soundness check of the constraints, with the REAL constraint code as the oracle.
//!
//! The audits found, twice, bounds that held for a derived value but not for the raw one. Those are properties of a single row, and for small parameters a
//! single row can be checked exhaustively: this test evaluates `NetEval::evaluate` (the constraints and lookups exactly as the prover and the verifier use
//! them) on one row through a small `EvalAtRow` implementation, for EVERY assignment a malicious prover could choose:
//!
//! * the interface values of the row (the accumulator and the output of a requantization row; the public float and the quantized value of a tensor row)
//!   are enumerated over ranges that go beyond every bound;
//! * the flag columns (`p`, `e`, `hp`, `hn`; `c`, `e`, `z`, `h`) take every bit pattern;
//! * every other column is forced by a constraint that is linear in it once the flags are fixed (a limb group must sum to one value, `half` must be
//!   `out / 2`, ...), so the test derives it; a limb group has a decomposition exactly if its value is at most the capacity of the limbs.
//!
//! A row is ACCEPTED if every constraint evaluates to zero and every lookup of the row is either an interface tuple of the expected form or a value that
//! is in the range table / the power-of-two table. The test then requires, for every accepted row, the row-level statement: the output is the
//! requantization of the accumulator, the accumulator respects the bound of the exponent, a "largest" flag is set only where it is true (and never on a
//! padding row), a quantized value is the rounding of its float and the float does not exceed 127 after scaling. It also checks completeness: every
//! honest row is accepted.
//!
//! `cargo test` runs a reduced domain (seconds); `ROW_FULL=1 cargo test --release --test row_exhaustive` runs the full one (about 180 million candidate
//! rows, a few minutes). Both were checked by mutation: with any one of 19 constraints or lookups removed from `air/kind.rs`, a test here fails.
//!
//! What this does NOT cover: properties of many rows together (exactly one "largest" row, the multiplicities that link the components), which are LogUp
//! arguments checked by the verifier, and large shifts / mantissas only at sampled boundary values.

use std::collections::HashMap;

use stwo::core::Fraction;
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval, Relation};

use tt_zkml::air::kind::{QCols, RqCols, tag_acc, tag_fhit, tag_hit, tag_w, tag_x};
use tt_zkml::air::spec::{LayerSpec, Mode, activate};
use tt_zkml::air::{Kind, Lookups, NetEval, Which};

const P: i64 = (1 << 31) - 1;

/// The full domain (`ROW_FULL=1`) or the reduced one.
fn full() -> bool {
    std::env::var("ROW_FULL").is_ok_and(|v| v != "0")
}

fn f(v: i64) -> BaseField {
    BaseField::from(v.rem_euclid(P) as u32)
}

/// Evaluates the constraints of one row on concrete values: records every constraint value and every LogUp fraction.
struct RowEval<'a> {
    /// constants of the statement (tree 3)
    pre: &'a [BaseField],
    /// columns of the committed model (tree 0)
    model: &'a [BaseField],
    cols: &'a [BaseField],
    next_model: usize,
    next_pre: usize,
    next_col: usize,
    constraints_hold: bool,
    fractions: Vec<(SecureField, SecureField)>,
}

impl EvalAtRow for RowEval<'_> {
    type F = BaseField;
    type EF = SecureField;

    fn next_interaction_mask<const N: usize>(&mut self, interaction: usize, offsets: [isize; N]) -> [BaseField; N] {
        assert!(N == 1 && offsets[0] == 0, "the constraints read only the current row");
        let v = match interaction {
            0 => {
                self.next_model += 1;
                self.model[self.next_model - 1]
            }
            3 => {
                self.next_pre += 1;
                self.pre[self.next_pre - 1]
            }
            1 => {
                self.next_col += 1;
                self.cols[self.next_col - 1]
            }
            other => panic!("unexpected interaction {other}"),
        };
        [v; N]
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        SecureField: std::ops::Mul<G, Output = SecureField> + From<G>,
    {
        let value: SecureField = SecureField::from(constraint);
        if value != <SecureField as num_traits::Zero>::zero() {
            self.constraints_hold = false;
        }
    }

    fn combine_ef(values: [BaseField; 4]) -> SecureField {
        SecureField::from_m31_array(values)
    }

    fn write_logup_frac(&mut self, fraction: Fraction<SecureField, SecureField>) {
        self.fractions.push((fraction.numerator, fraction.denominator));
    }

    fn finalize_logup_in_pairs(&mut self) {}
}

/// The lookups with random elements and the denominators of every valid table entry.
struct Oracle {
    lookups: Lookups,
    range: HashMap<SecureField, u32>,
    pow2: HashMap<SecureField, u32>,
}

impl Oracle {
    fn new() -> Self {
        let lookups = Lookups::draw(&mut Blake2sChannel::default());
        let range = (0..=254u32).map(|v| (lookups.range.combine(&[BaseField::from(v)]), v)).collect();
        let pow2 = (1..=25u32).map(|s| (lookups.pow2.combine(&[BaseField::from(s), BaseField::from(1u32 << s)]), s)).collect();
        Self { lookups, range, pow2 }
    }

    fn tensor(&self, tag: u32, i: u32, j: u32, v: BaseField) -> SecureField {
        self.lookups.tensor.combine(&[BaseField::from(tag), BaseField::from(i), BaseField::from(j), v])
    }

    /// Evaluates the real constraints of `kind` on the row. Returns `None` if a constraint fails or a lookup is neither one of the `interface` tuples nor
    /// in a table; otherwise the numerators of the interface tuples, in the order of `interface`.
    fn accept(&self, kind: &Kind, pre: &[BaseField], model: &[BaseField], cols: &[BaseField], interface: &[SecureField]) -> Option<Vec<SecureField>> {
        let eval = RowEval { pre, model, cols, next_model: 0, next_pre: 0, next_col: 0, constraints_hold: true, fractions: Vec::new() };
        let eval = NetEval { kind: kind.clone(), lookups: self.lookups.clone() }.evaluate(eval);
        assert_eq!((eval.next_col, eval.next_pre, eval.next_model), (cols.len(), pre.len(), model.len()), "the row must have exactly the columns of the component");
        if !eval.constraints_hold {
            return None;
        }
        let one = SecureField::from(BaseField::from(1u32));
        let mut numerators = vec![None; interface.len()];
        for (num, den) in eval.fractions {
            if let Some(slot) = interface.iter().position(|d| *d == den) {
                assert!(numerators[slot].is_none(), "an interface tuple is used twice by one row");
                numerators[slot] = Some(num);
            } else if self.range.contains_key(&den) || self.pow2.contains_key(&den) {
                if num != one {
                    return None;
                }
            } else {
                return None; // a value outside the tables
            }
        }
        Some(numerators.into_iter().map(|n| n.expect("every interface tuple of the component appears")).collect())
    }
}

/// The limbs (base 128, each at most 254) of a field element, if it has a decomposition: exactly the values up to the capacity of the limbs.
fn limbs(value: BaseField, n: usize) -> Option<Vec<BaseField>> {
    let mut v = value.0 as u64;
    let mut out = vec![BaseField::from(0u32); n];
    for t in (0..n).rev() {
        let d = (v / 128u64.pow(t as u32)).min(254);
        out[t] = BaseField::from(d as u32);
        v -= d * 128u64.pow(t as u32);
    }
    (v == 0).then_some(out)
}

/// Writes the limbs of `value`. If the value has no decomposition the limbs stay zero: with the constraint that defines the limb group in place such a
/// row is rejected by the real constraint code; if that constraint were missing, the row would be accepted and the test would see it.
fn put(cols: &mut [BaseField], idx: &[usize], value: BaseField) {
    if let Some(l) = limbs(value, idx.len()) {
        for (&i, l) in idx.iter().zip(l) {
            cols[i] = l;
        }
    }
}

/// A deviation from the forced value of one derived column: the prover writes another value there (and everything that depends on it follows the
/// written value). With every constraint in place the real code rejects all of these unless the row is a valid one anyway.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Dev {
    None,
    Ap(i64),
    Sa(i64),
    Half(i64),
    Qa(i64),
    R(i64),
    G(i64),
    Diff(i64),
    Hs(i64),
    Shift(i64),
    P2(i64),
    Ginv(i64),
    /// the committed model's sign / significand / exponent differs from the row's
    ModelS(i64),
    ModelM(i64),
    ModelE(i64),
}

fn field_int(v: SecureField) -> Option<i64> {
    let a = v.to_m31_array();
    (a[1].0 == 0 && a[2].0 == 0 && a[3].0 == 0).then(|| if a[0].0 as i64 > P / 2 { a[0].0 as i64 - P } else { a[0].0 as i64 })
}

// ------------------------------------------------------------------------------------------------------------------------------------------
// Requantization rows

/// The only row a prover can build for these interface values and flags (every other column is forced), if its limb groups have a decomposition.
fn requant_row(spec: &LayerSpec, c: &RqCols, acc: i64, out: i64, p: u32, e: u32, hp: u32, hn: u32, dev: Dev) -> Option<Vec<BaseField>> {
    let mut cols = vec![BaseField::from(0u32); c.total];
    cols[c.acc] = f(acc);
    cols[c.out] = f(out);
    if spec.mode() == Mode::Zero {
        return Some(cols);
    }
    let mut a = f(acc);
    if let Some(r) = &c.relu {
        cols[r.p] = BaseField::from(p);
        let ap = if let Dev::Ap(v) = dev { f(v) } else { BaseField::from(p) * f(acc) };
        cols[r.ap] = ap;
        put(&mut cols, &r.ap_limbs, ap);
        put(&mut cols, &r.neg_limbs, (BaseField::from(p) - BaseField::from(1u32)) * f(acc));
        a = ap;
    }
    match spec.mode() {
        Mode::Left(ko) => cols[c.left_scaled.unwrap()] = if let Dev::Sa(v) = dev { f(v) } else { f(acc) * f(1 << ko) },
        Mode::Right(s) => {
            let r = c.right.as_ref().unwrap();
            let h = f(1 << (s - 1));
            let rem = a - f(out) * f(1 << s);
            cols[r.e] = BaseField::from(e);
            if let Dev::Half(v) = dev {
                cols[r.half] = f(v);
            }
            if e == 1 {
                if !matches!(dev, Dev::Half(_)) {
                    cols[r.half] = f(out) * BaseField::from(2u32).inverse();
                }
            } else {
                put(&mut cols, &r.v1_limbs, h - f(1) + rem);
                put(&mut cols, &r.v2_limbs, h - f(1) - rem);
            }
            put(&mut cols, &r.u1_limbs, f(127 << s) - f(acc));
            put(&mut cols, &r.u2_limbs, f(127 << s) + f(acc));
        }
        Mode::Zero => unreachable!(),
    }
    let h = c.hit.as_ref().unwrap();
    let t = spec.hit_threshold() as i64;
    cols[h.hp] = BaseField::from(hp);
    cols[h.hn] = BaseField::from(hn);
    put(&mut cols, &h.limbs, BaseField::from(hp) * f(acc - t) + BaseField::from(hn) * f(-acc - t));
    Some(cols)
}

#[derive(Default)]
struct Count {
    candidates: u64,
    accepted: u64,
}

fn check_requant(oracle: &Oracle, spec: LayerSpec, accs: &[i64], outs: &[i64], deviations: bool, count: &mut Count) {
    let kind = Kind::Requant { layer: 0, spec, consumers: 3 };
    let c = RqCols::new(&spec);
    let (i, j) = (1u32, 2u32);
    let bound_ok = |acc: i64| match spec.mode() {
        Mode::Zero => acc == 0,
        Mode::Left(ko) => acc.abs() << ko <= 127,
        Mode::Right(s) => acc.abs() <= 127 << s,
    };
    let t = spec.hit_threshold() as i64;
    // An accumulator is a field element; the integer it stands for is its signed representative (the matmul pins it to an integer below 2^28 in
    // magnitude). Values given as other representatives are the same rows.
    let mut accs: Vec<i64> = accs.iter().map(|&a| { let r = a.rem_euclid(P); if r > P / 2 { r - P } else { r } }).collect();
    accs.sort();
    accs.dedup();
    let accs = &accs[..];
    let flags: Vec<(u32, u32, u32, u32)> = if spec.mode() == Mode::Zero {
        vec![(0, 0, 0, 0)]
    } else {
        let ps: &[u32] = if spec.relu { &[0, 1] } else { &[0] };
        let es: &[u32] = if matches!(spec.mode(), Mode::Right(_)) { &[0, 1] } else { &[0] };
        ps.iter().flat_map(|&p| es.iter().flat_map(move |&e| [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(hp, hn)| (p, e, hp, hn)))).collect()
    };
    for real in [1u32, 0] {
        let pre = [BaseField::from(i), BaseField::from(j), BaseField::from(real), BaseField::from(3 * real)];
        for &acc in accs {
            let mut honest_accepted = false;
            let mut honest_hit_accepted = false;
            let honest_out = if bound_ok(acc) { Some(activate(acc as i32, &spec) as i64) } else { None };
            for &out in outs {
                for &(p, e, hp, hn) in &flags {
                  let devs: Vec<Dev> = if deviations {
                      vec![Dev::None, Dev::Ap(acc), Dev::Ap(0), Dev::Ap(-acc), Dev::Ap(acc.abs()), Dev::Sa(0), Dev::Sa(out), Dev::Half(0), Dev::Half((out - 1) / 2), Dev::Half(out)]
                  } else {
                      vec![Dev::None]
                  };
                  for dev in devs {
                    let Some(cols) = requant_row(&spec, &c, acc, out, p, e, hp, hn, dev) else { continue };
                    count.candidates += 1;
                    let mut interface = vec![oracle.tensor(tag_acc(0), i, j, f(acc)), oracle.tensor(tag_x(1), i, j, f(out))];
                    if c.hit.is_some() {
                        interface.push(oracle.tensor(tag_hit(0), 0, 0, f(0)));
                    }
                    let Some(nums) = oracle.accept(&kind, &pre, &[], &cols, &interface) else { continue };
                    count.accepted += 1;
                    let ctx = format!("spec {spec:?} real {real} acc {acc} out {out} flags p={p} e={e} hp={hp} hn={hn} deviation {dev:?}");
                    assert_eq!(field_int(nums[0]), Some(real as i64), "the accumulator is consumed once per real row: {ctx}");
                    assert_eq!(field_int(nums[1]), Some(-(3 * real as i64)), "the output is yielded once per consumer: {ctx}");
                    let hit = if c.hit.is_some() { field_int(nums[2]).expect("a hit count") } else { 0 };
                    if real == 0 {
                        // a padding row is free, but it must not count as the "largest" row
                        assert_eq!(hit, 0, "SOUNDNESS: a padding row counts as the largest accumulator: {ctx}");
                        continue;
                    }
                    assert!(bound_ok(acc), "SOUNDNESS: an accumulator beyond the bound of the exponent is accepted: {ctx}");
                    assert_eq!(Some(out), honest_out, "SOUNDNESS: an output that is not the requantization is accepted: {ctx}");
                    assert!(hit == 0 || hit == 1, "SOUNDNESS: one row counts {hit} times as the largest: {ctx}");
                    if hit == 1 {
                        assert!(acc.abs() >= t, "SOUNDNESS: a row below the threshold {t} counts as the largest: {ctx}");
                        honest_hit_accepted = true;
                    } else {
                        honest_accepted = true;
                    }
                  }
                }
            }
            if real == 1 {
                if let Some(o) = honest_out.filter(|o| outs.contains(o)) {
                    assert!(honest_accepted, "COMPLETENESS: the honest row acc {acc} out {o} is not accepted ({spec:?})");
                    if spec.mode() != Mode::Zero && acc.abs() >= t {
                        assert!(honest_hit_accepted, "COMPLETENESS: the honest largest row acc {acc} is not accepted with its flag ({spec:?})");
                    }
                }
            }
        }
    }
}

fn spec(relu: bool, ko: i32) -> LayerSpec {
    LayerSpec { m: 2, k: 2, n: 4, relu, ko, kw: 0, kx: 0 }
}

#[test]
fn requantization_rows_exhaustive() {
    let oracle = Oracle::new();
    let mut count = Count::default();
    let all_outs: Vec<i64> = (-130..=130).collect();
    // a field element that is "-1 / 2", "+-2^30", and other values a wrap-around attack would use
    let odd: Vec<i64> = vec![1 << 25, -(1 << 25), 1 << 30, (1 << 30) - 1, P - 1, P / 2, (P + 1) / 2, 1 << 24, 33554431];
    for relu in [false, true] {
        // zero mode
        check_requant(&oracle, spec(relu, 127), &(-3..=3).chain([127, -127, 1 << 20]).collect::<Vec<_>>(), &all_outs, false, &mut count);
        // left shifts: every exponent, every accumulator around and beyond the range
        for ko in if full() { vec![0, 1, 2, 3, 4, 5, 6] } else { vec![0, 1, 6] } {
            let accs: Vec<i64> = (-140..=140).chain([255, -255, 256, -256, 300, -300]).chain(odd.iter().copied()).collect();
            check_requant(&oracle, spec(relu, ko), &accs, &all_outs, false, &mut count);
        }
        // right shifts: small shifts with every accumulator up to beyond the bound and every output
        for s in 1..=(if full() { 3u32 } else { 1 }) {
            let lim = (127i64 << s) + (4 << s);
            let accs: Vec<i64> = (-lim..=lim).chain(odd.iter().copied()).collect();
            check_requant(&oracle, spec(relu, -(s as i32)), &accs, &all_outs, false, &mut count);
        }
        // larger shifts (also where the number of limbs of the remainder changes: 7, 8, 14, 15, and the largest, 20): accumulators around every rounding
        // boundary of a few outputs and around the bound, every output
        for s in if full() { vec![4u32, 7, 8, 14, 15, 20] } else { vec![2, 7, 20] } {
            let (unit, half) = (1i64 << s, 1i64 << (s - 1));
            let mut accs: Vec<i64> = Vec::new();
            for q in [0i64, 1, 2, 3, 62, 63, 64, 65, 125, 126, 127, 128, 129] {
                for d in [-half - 1, -half, -half + 1, -1, 0, 1, half - 1, half, half + 1] {
                    accs.extend([q * unit + d, -(q * unit + d)]);
                }
            }
            accs.extend(odd.iter().copied());
            accs.sort();
            accs.dedup();
            check_requant(&oracle, spec(relu, -(s as i32)), &accs, &all_outs, false, &mut count);
        }
    }
    println!("requantization rows: {} candidate rows, {} accepted", count.candidates, count.accepted);
    assert!(count.accepted > 100);
}

// ------------------------------------------------------------------------------------------------------------------------------------------
// Tensor rows (the quantization of a public float)

fn tensor_row(kexp: i32, c: &QCols, sign: u32, m: i64, ep: i64, q: i64, cl: u32, e: u32, z: u32, h: u32, dev: Dev) -> Option<Vec<BaseField>> {
    let mut cols = vec![BaseField::from(0u32); c.total];
    cols[c.q] = f(q);
    cols[c.s] = BaseField::from(sign);
    cols[c.m] = f(m);
    cols[c.ep] = f(ep);
    let Some(n) = &c.nz else { return Some(cols) };
    let one = BaseField::from(1u32);
    let qa = if let Dev::Qa(v) = dev { f(v) } else if sign == 1 { -f(q) } else { f(q) };
    let raw = 150 - kexp as i64 - ep;
    let sh = if let Dev::Shift(v) = dev { v } else if cl == 1 { 25 } else { raw };
    if !(1..=25).contains(&sh) {
        return None; // no row of the power-of-two table
    }
    let p2 = if let Dev::P2(v) = dev { f(v) } else { f(1 << sh) };
    put(&mut cols, &n.ml, f(m));
    cols[n.qa] = qa;
    cols[n.p2] = p2;
    cols[n.sh] = f(sh);
    cols[n.c] = BaseField::from(cl);
    cols[n.diff] = if let Dev::Diff(v) = dev { f(v) } else if cl == 1 { f(raw - 25) } else { f(24 - raw) };
    let r = if let Dev::R(v) = dev { f(v) } else { f(m) - qa * p2 };
    cols[n.r] = r;
    let half = f(1 << (sh - 1));
    cols[n.e] = BaseField::from(e);
    if let Dev::Half(v) = dev {
        cols[n.half] = f(v);
    } else if e == 1 {
        cols[n.half] = qa * BaseField::from(2u32).inverse();
    }
    if e == 0 {
        put(&mut cols, &n.v1, half - one + r);
        put(&mut cols, &n.v2, half - one - r);
    }
    let g = if let Dev::G(v) = dev { f(v) } else { f(127) - qa };
    cols[n.g] = g;
    cols[n.z] = BaseField::from(z);
    if z == 1 {
        put(&mut cols, &n.rn, -r);
    } else if g != f(0) {
        cols[n.ginv] = g.inverse();
    }
    if let Dev::Ginv(v) = dev {
        cols[n.ginv] = f(v);
    }
    cols[n.h] = BaseField::from(h);
    cols[n.hs] = if let Dev::Hs(v) = dev { f(v) } else { BaseField::from(h) * f(20 - sh) };
    if h == 1 {
        put(&mut cols, &n.hv, f(2 * m) - f(127) * p2 - one);
    }
    Some(cols)
}

/// `round_half_even(m / 2^raw)` and whether `m / 2^raw <= 127`, exactly.
fn reference(m: i64, raw: i64) -> (i64, bool) {
    if raw >= 40 {
        return (0, true);
    }
    let (m, unit) = (m as i128, 1i128 << raw);
    let (q, rem) = (m / unit, m % unit);
    let rounded = if 2 * rem > unit || (2 * rem == unit && q % 2 == 1) { q + 1 } else { q };
    (rounded as i64, m <= 127 * unit)
}

/// Checks every tensor row over the given raw shifts. `dense`: many significands around every rounding boundary; `deviations`: also rows in which one
/// derived column is written with another value.
fn check_tensor(oracle: &Oracle, which: Which, raws: &[i64], dense: bool, deviations: bool, count: &mut Count) {
    let kexp = 3;
    let tid = if which == Which::W { 1 } else { 0 };
    let layer_spec = LayerSpec { m: 2, k: 2, n: 4, relu: false, ko: -5, kw: kexp, kx: 0 };
    let kind = Kind::Tensor { layer: 0, which, spec: layer_spec, kexp, tid };
    let c = QCols::new(kexp);
    let (i, j) = (1u32, 3u32);
    let qs: Vec<i64> = (-130..=130).collect();
    let value_tag = if which == Which::W { tag_w(0) } else { tag_x(0) };
    for real in [1u32, 0] {
        let pre = [BaseField::from(i), BaseField::from(j), BaseField::from(2 * real), BaseField::from(real)];
        for &raw in raws {
            let ep = 150 - kexp as i64 - raw;
            assert!((1..=254).contains(&ep));
            // significands: around every rounding boundary of a few quotients, and the ends of the 24-bit range
            let mut ms: Vec<i64> = vec![0, 1, 2, 3, (1 << 23) - 1, 1 << 23, (1 << 23) + 1, (1 << 24) - 1];
            if (1..=24).contains(&raw) {
                let (unit, half) = (1i64 << raw, 1i64 << (raw - 1));
                let quotients: &[i64] = if dense { &[0, 1, 2, 3, 31, 32, 33, 63, 64, 65, 126, 127, 128, 129, 191, 192, 254, 255] } else { &[0, 1, 2, 63, 64, 126, 127, 128] };
                for &q in quotients {
                    for d in [-half - 1, -half, -half + 1, -1, 0, 1, half - 1, half, half + 1] {
                        ms.push(q * unit + d);
                    }
                }
            }
            ms.retain(|m| (0..1 << 24).contains(m));
            ms.sort();
            ms.dedup();
            for &m in &ms {
                let (true_qa, fits) = if raw >= 1 { reference(m, raw) } else { (0, false) };
                for sign in [0u32, 1] {
                    let honest_q = if sign == 1 { -true_qa } else { true_qa };
                    let supported = raw >= 1 && fits;
                    let mut honest_accepted = false;
                    for &q in &qs {
                        for flags in 0..16u32 {
                            let (cl, e, z, h) = (flags & 1, (flags >> 1) & 1, (flags >> 2) & 1, (flags >> 3) & 1);
                            let devs: Vec<Dev> = if deviations {
                                let sh = if cl == 1 { 25 } else { raw.clamp(1, 25) };
                                let true_r = m - true_qa * (1i64 << sh.min(24));
                                vec![
                                    Dev::None,
                                    Dev::Qa(true_qa), Dev::Qa(0), Dev::Qa(q.abs()), Dev::Qa(64), Dev::Qa(127),
                                    Dev::R(0), Dev::R(true_r), Dev::R(1 << (sh - 1)), Dev::R(-1),
                                    Dev::G(0), Dev::G(1), Dev::G(127 - q.abs()),
                                    Dev::Diff(0), Dev::Hs(0), Dev::Half(0), Dev::Half(q.abs() / 2), Dev::Ginv(0), Dev::Ginv(1),
                                    Dev::Shift(1), Dev::Shift(17), Dev::Shift(24), Dev::Shift(25), Dev::P2(1), Dev::P2(1 << 24),
                                    // the committed model says something else than the row (weights only)
                                    Dev::ModelS(1 - sign as i64), Dev::ModelM(m + 1), Dev::ModelM(0), Dev::ModelE(ep + 1), Dev::ModelE(1),
                                ]
                            } else {
                                vec![Dev::None]
                            };
                            for dev in devs {
                                let Some(cols) = tensor_row(kexp, &c, sign, m, ep, q, cl, e, z, h, dev) else { continue };
                                count.candidates += 1;
                                // the weights are pinned by the committed model (tree 0), the input by the verifier's public float (a lookup)
                                let model: Vec<BaseField> = if which == Which::W {
                                    let (mut ms, mut mm, mut me) = (sign as i64, m, ep);
                                    match dev {
                                        Dev::ModelS(v) => ms = v,
                                        Dev::ModelM(v) => mm = v,
                                        Dev::ModelE(v) => me = v,
                                        _ => {}
                                    }
                                    vec![f(ms), f(mm), f(me)]
                                } else {
                                    if matches!(dev, Dev::ModelS(_) | Dev::ModelM(_) | Dev::ModelE(_)) {
                                        continue;
                                    }
                                    Vec::new()
                                };
                                let mut interface = vec![oracle.tensor(value_tag, i, j, f(q)), oracle.tensor(tag_fhit(tid), 0, 0, f(0))];
                                if which == Which::X {
                                    interface.push(oracle.lookups.float.combine(&[BaseField::from(tid), BaseField::from(i), BaseField::from(j), BaseField::from(sign), f(m), f(ep)]));
                                }
                                let Some(nums) = oracle.accept(&kind, &pre, &model, &cols, &interface) else { continue };
                                count.accepted += 1;
                                let ctx = format!("{which:?} real {real} raw {raw} m {m} sign {sign} q {q} flags c={cl} e={e} z={z} h={h} deviation {dev:?}");
                                assert!(!matches!(dev, Dev::ModelS(_) | Dev::ModelM(_) | Dev::ModelE(_)) || model == vec![f(sign as i64), f(m), f(ep)], "SOUNDNESS: a row that differs from the committed model is accepted: {ctx}");
                                if which == Which::X {
                                    assert_eq!(field_int(nums[2]), Some(real as i64), "the float is consumed once per real row: {ctx}");
                                }
                                assert_eq!(field_int(nums[0]), Some(-(2 * real as i64)), "the value is yielded once per use: {ctx}");
                                let hit = field_int(nums[1]).expect("a hit count");
                                if real == 0 {
                                    assert_eq!(hit, 0, "SOUNDNESS: a padding row counts as the largest float: {ctx}");
                                    continue;
                                }
                                assert!(raw >= 1, "SOUNDNESS: a float with a shift below 1 is accepted: {ctx}");
                                assert!(fits, "SOUNDNESS: a float above 127 after scaling is accepted: {ctx}");
                                assert_eq!(q, honest_q, "SOUNDNESS: a value that is not the quantization of the float is accepted: {ctx}");
                                assert!(hit == 0 || hit == 1, "SOUNDNESS: one row counts {hit} times: {ctx}");
                                if hit == 1 {
                                    // the scaler would not fit one step larger: 2 * |v| * 2^k > 127
                                    assert!(raw <= 20 && 2 * (m as i128) > 127 * (1i128 << raw), "SOUNDNESS: a small float counts as the largest: {ctx}");
                                } else {
                                    honest_accepted = true;
                                }
                            }
                        }
                    }
                    if real == 1 && supported {
                        assert!(honest_accepted, "COMPLETENESS: the honest row raw {raw} m {m} sign {sign} q {honest_q} is not accepted");
                    }
                }
            }
        }
    }
}

#[test]
fn tensor_rows_exhaustive() {
    let oracle = Oracle::new();
    let mut count = Count::default();
    let raws: &[i64] = if full() { &[-2, 0, 1, 2, 3, 7, 8, 14, 17, 18, 19, 20, 21, 23, 24, 25, 26, 40, 146] } else { &[0, 1, 17, 20, 21, 24, 25, 146] };
    check_tensor(&oracle, Which::W, raws, full(), false, &mut count);
    check_tensor(&oracle, Which::X, if full() { raws } else { &[1, 20, 25] }, false, false, &mut count);
    println!("tensor rows: {} candidate rows, {} accepted", count.candidates, count.accepted);
    assert!(count.accepted > 100);

    let kexp = 3;
    let tid = 1;
    let layer_spec = LayerSpec { m: 2, k: 2, n: 4, relu: false, ko: -5, kw: kexp, kx: 0 };
    let (i, j) = (1u32, 3u32);
    // the all-zero mode: only a zero float with q = 0
    let kind = Kind::Tensor { layer: 0, which: Which::W, spec: LayerSpec { kw: 127, ..layer_spec }, kexp: 127, tid };
    let c = QCols::new(127);
    let pre = [BaseField::from(i), BaseField::from(j), BaseField::from(2u32), BaseField::from(1u32)];
    for m in [0i64, 1, 5, 1 << 23, (1 << 24) - 1, P - 1] {
        for sign in [0u32, 1] {
            for q in -130..=130i64 {
                let cols = tensor_row(127, &c, sign, m, 1, q, 0, 0, 0, 0, Dev::None).unwrap();
                let interface = [oracle.tensor(tag_w(0), i, j, f(q))];
                if oracle.accept(&kind, &pre, &[BaseField::from(sign), f(m), f(1)], &cols, &interface).is_some() {
                    assert!(m == 0 && q == 0, "SOUNDNESS: the all-zero mode accepts m {m} q {q}");
                }
            }
        }
    }
}

/// The same checks with one derived column deviating from its forced value (see `Dev`), on a smaller domain: this is what notices a missing constraint
/// that DEFINES a column (the exhaustive tests above derive such columns, so they could not see it).
#[test]
fn rows_with_one_deviating_column() {
    let oracle = Oracle::new();
    let mut count = Count::default();
    let all_outs: Vec<i64> = (-130..=130).collect();
    for relu in [false, true] {
        for ko in if full() { vec![0, 1, 6] } else { vec![1] } {
            let accs: Vec<i64> = (-8..=8).chain([15, 16, 17, 31, 32, 33, 63, 64, 65, 126, 127, 128, 129, 200]).flat_map(|a| [a, -a]).collect();
            check_requant(&oracle, spec(relu, ko), &accs, &all_outs, true, &mut count);
        }
        for s in if full() { vec![1u32, 2, 7] } else { vec![2] } {
            let (unit, half) = (1i64 << s, 1i64 << (s - 1));
            let mut accs: Vec<i64> = Vec::new();
            for q in [0i64, 1, 2, 63, 64, 126, 127, 128] {
                for d in [-half - 1, -half, -half + 1, -1, 0, 1, half - 1, half, half + 1] {
                    accs.extend([q * unit + d, -(q * unit + d)]);
                }
            }
            check_requant(&oracle, spec(relu, -(s as i32)), &accs, &all_outs, true, &mut count);
        }
    }
    let requant = (count.candidates, count.accepted);
    check_tensor(&oracle, Which::W, if full() { &[1, 2, 17, 24, 25, 26] } else { &[1, 24, 25] }, false, true, &mut count);
    println!("deviating rows: requantization {requant:?}, with tensor rows {} candidates, {} accepted", count.candidates, count.accepted);
}

/// A committed model is not known to the verifier, so the constraints themselves must refuse committed "floats" that are not well formed: a sign that is
/// not a bit, a significand that the limbs cannot hold. (The model column and the row agree here: the malformed value is what was committed.)
#[test]
fn malformed_committed_floats_have_no_accepted_row() {
    let oracle = Oracle::new();
    let kexp = 3;
    let layer_spec = LayerSpec { m: 2, k: 2, n: 4, relu: false, ko: -5, kw: kexp, kx: 0 };
    let kind = Kind::Tensor { layer: 0, which: Which::W, spec: layer_spec, kexp, tid: 1 };
    let c = QCols::new(kexp);
    let n = c.nz.as_ref().unwrap();
    let (i, j) = (1u32, 3u32);
    let pre = [BaseField::from(i), BaseField::from(j), BaseField::from(2u32), BaseField::from(1u32)];
    let mut accepted_honest = 0;
    for raw in [1i64, 17, 24, 25, 40] {
        let ep = 150 - kexp as i64 - raw;
        for q in -130..=130i64 {
            for flags in 0..16u32 {
                let (cl, e, z, h) = (flags & 1, (flags >> 1) & 1, (flags >> 2) & 1, (flags >> 3) & 1);
                // (sign, m): honest ones as a control, then malformed ones
                for (sign, m) in [(0i64, 5i64), (2, 5), (P - 1, 5), (3, 0), (0, 1 << 29), (0, (1 << 30) + 7), (0, P - 1), (1, (254 * 2113665) + 1)] {
                    let Some(mut cols) = tensor_row(kexp, &c, 0, m, ep, q, cl, e, z, h, Dev::None) else { continue };
                    cols[c.s] = f(sign);
                    // with another sign the magnitude column that satisfies q = qa (1 - 2s) is another one
                    let factor = f(1) - f(2) * f(sign);
                    if factor != f(0) {
                        cols[n.qa] = f(q) * factor.inverse();
                    }
                    let interface = [oracle.tensor(tag_w(0), i, j, f(q)), oracle.tensor(tag_fhit(1), 0, 0, f(0))];
                    if oracle.accept(&kind, &pre, &[f(sign), f(m), f(ep)], &cols, &interface).is_some() {
                        assert!((sign, m) == (0, 5), "SOUNDNESS: a malformed committed float (sign {sign}, m {m}) has an accepted row: raw {raw} q {q}");
                        accepted_honest += 1;
                    }
                }
            }
        }
    }
    assert!(accepted_honest > 0);
}
