//! The trace generation: the CPU turns the witness (int8 tensors and int32 accumulators) into M31 columns, and builds the LogUp interaction columns.

use itertools::Itertools;
use stwo::core::ColumnVec;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::{LOG_N_LANES, PackedBaseField};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::Column;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo_constraint_framework::LogupTraceGenerator;

use super::kind::{HIT_MAX_SHIFT, Kind, Lookups, Num, POW2_MAX_SHIFT, POW2_ROWS, QCols, RqCols, Src, TABLE_MAX, TABLE_ROWS, Use, Which, layout};
use super::spec::{LIMB_BASE, LayerSpec, Mode};
use crate::witness::{NetworkWitness, WitnessError, float_parts, shift_of};

#[cfg(any(test, feature = "forgery"))]
thread_local! {
    /// Test hook: a malicious prover rewrites the columns of a component before the multiplicities of the range tables are counted.
    pub static HOOK: std::cell::RefCell<Option<Box<dyn Fn(&Kind, &mut Vec<BaseColumn>)>>> = const { std::cell::RefCell::new(None) };
}

type Eval = CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>;

/// One component with its columns materialized.
pub struct BuiltComponent {
    pub kind: Kind,
    /// Preprocessed columns (the constants of the statement).
    pub pre: Vec<BaseColumn>,
    /// Trace columns.
    pub base: Vec<BaseColumn>,
}

/// All the components of a network, in the order of their columns in the commitment trees.
pub struct NetworkTrace {
    pub components: Vec<BuiltComponent>,
}

fn column(rows: usize, values: impl IntoIterator<Item = BaseField>) -> BaseColumn {
    let mut v = values.into_iter().collect_vec();
    v.resize(rows, BaseField::from(0u32));
    v.into_iter().collect()
}

fn limbs(mut v: u32, n: usize) -> Option<Vec<u32>> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(v % LIMB_BASE);
        v /= LIMB_BASE;
    }
    (v == 0).then_some(out)
}

/// The columns of the requantization component of one layer: `acc` and `out` are its row values (padding rows are `0, 0`).
pub fn requant_row(spec: &LayerSpec, c: &RqCols, row: usize, acc: i32, out: i32, hit: i32) -> Result<Vec<BaseField>, WitnessError> {
    requant_row_inner(spec, c, row, acc, out, hit, true)
}

/// The row of a malicious prover that does not bother about the bound `|acc| <= 127 * 2^s` (its limbs stay zero): for the audit regression tests.
#[cfg(any(test, feature = "forgery"))]
pub fn requant_row_unbounded(spec: &LayerSpec, c: &RqCols, row: usize, acc: i32, out: i32, hit: i32) -> Result<Vec<BaseField>, WitnessError> {
    requant_row_inner(spec, c, row, acc, out, hit, false)
}

fn requant_row_inner(spec: &LayerSpec, c: &RqCols, row: usize, acc: i32, out: i32, hit: i32, enforce_bound: bool) -> Result<Vec<BaseField>, WitnessError> {
    let bad = |what: &str| WitnessError::Requant { row, what: what.to_string() };
    let mut cols = vec![BaseField::from(0u32); c.total];
    cols[c.acc] = BaseField::from(acc);
    cols[c.out] = BaseField::from(out);
    if spec.mode() == Mode::Zero {
        return if acc == 0 && out == 0 { Ok(cols) } else { Err(bad("the layer is declared all-zero but has a non-zero value")) };
    }
    let put = |cols: &mut Vec<BaseField>, idx: &[usize], v: u32| -> Result<(), WitnessError> {
        let l = limbs(v, idx.len()).ok_or_else(|| bad("a value does not fit its limbs"))?;
        for (&i, x) in idx.iter().zip(l) {
            cols[i] = BaseField::from(x);
        }
        Ok(())
    };
    let a = if let Some(r) = &c.relu {
        let p = acc >= 0;
        cols[r.p] = BaseField::from(p as u32);
        let ap = if p { acc } else { 0 };
        cols[r.ap] = BaseField::from(ap);
        put(&mut cols, &r.ap_limbs, ap as u32)?;
        put(&mut cols, &r.neg_limbs, if p { 0 } else { acc.unsigned_abs() })?;
        ap
    } else {
        acc
    };
    match spec.mode() {
        Mode::Left(ko) => {
            if acc.abs() > 127 {
                return Err(bad("|acc| > 127 with a left shift"));
            }
            if (a as i64) << ko != out as i64 {
                return Err(bad("out != acc * 2^ko"));
            }
        }
        Mode::Right(sh) => {
            let r = c.right.as_ref().unwrap();
            let h = 1i64 << (sh - 1);
            let rem = a as i64 - ((out as i64) << sh);
            if rem.abs() > h {
                return Err(bad("the output is not the rounded quotient"));
            }
            let bound = 127i64 << sh;
            if enforce_bound {
            put(&mut cols, &r.u1_limbs, u32::try_from(bound - acc as i64).map_err(|_| bad("an accumulator exceeds 127 * 2^s"))?)?;
            put(&mut cols, &r.u2_limbs, u32::try_from(bound + acc as i64).map_err(|_| bad("an accumulator exceeds 127 * 2^s"))?)?;
            }
            if rem.abs() == h {
                if out % 2 != 0 {
                    return Err(bad("a tie is not rounded to even"));
                }
                cols[r.e] = BaseField::from(1u32);
                cols[r.half] = BaseField::from(out / 2);
            } else {
                put(&mut cols, &r.v1_limbs, (h - 1 + rem) as u32)?;
                put(&mut cols, &r.v2_limbs, (h - 1 - rem) as u32)?;
            }
        }
        Mode::Zero => unreachable!(),
    }
    let h = c.hit.as_ref().unwrap();
    if hit != 0 {
        cols[if hit > 0 { h.hp } else { h.hn }] = BaseField::from(1u32);
        let t = spec.hit_threshold() as i64;
        let v = if hit > 0 { acc as i64 - t } else { -(acc as i64) - t };
        if v < 0 {
            return Err(bad("the largest accumulator is below its threshold"));
        }
        put(&mut cols, &h.limbs, v as u32)?;
    }
    Ok(cols)
}

/// The columns of one element of a float tensor: the float `(s, m, e')` (public), its quantization `q` and the witness of the rounding.
/// `hit`: this element is the one that justifies the scaler (`2m > 127 * 2^sh`).
pub fn quant_row(kexp: i32, c: &QCols, row: usize, parts: (u32, u32, u32), q: i32, hit: bool) -> Result<Vec<BaseField>, WitnessError> {
    let bad = |what: &str| WitnessError::Requant { row, what: format!("float quantization: {what}") };
    let (sign, m, ep) = parts;
    let mut cols = vec![BaseField::from(0u32); c.total];
    cols[c.q] = BaseField::from(q);
    cols[c.s] = BaseField::from(sign);
    cols[c.m] = BaseField::from(m);
    cols[c.ep] = BaseField::from(ep);
    let Some(n) = &c.nz else {
        return if m == 0 && q == 0 { Ok(cols) } else { Err(bad("the tensor is declared all-zero")) };
    };
    let put = |cols: &mut Vec<BaseField>, idx: &[usize], v: i64| -> Result<(), WitnessError> {
        let l = u32::try_from(v).ok().and_then(|v| limbs(v, idx.len())).ok_or_else(|| bad("a value does not fit its limbs"))?;
        for (&i, x) in idx.iter().zip(l) {
            cols[i] = BaseField::from(x);
        }
        Ok(())
    };
    let raw = shift_of(ep, kexp) as i64;
    if raw < 1 {
        return Err(bad("the quantization shift is below 1"));
    }
    // a tiny float (raw shift >= 25) quantizes to 0: sh = 25 then, and the constraints force qa = 0
    let clamped = raw >= POW2_MAX_SHIFT as i64;
    let sh = if clamped { POW2_MAX_SHIFT as i64 } else { raw };
    let p2 = 1i64 << sh;
    let half = p2 / 2;
    let m = m as i64;
    let qa = (m + half - 1 + ((m >> sh) & 1)) >> sh;
    let rem = m - qa * p2;
    if qa > 127 || (qa == 127 && rem > 0) {
        return Err(bad("the scaled float exceeds 127"));
    }
    if q != if sign == 1 { -(qa as i32) } else { qa as i32 } {
        return Err(bad("q is not the quantization of the float"));
    }
    cols[n.qa] = BaseField::from(qa as u32);
    cols[n.p2] = BaseField::from(p2 as u32);
    cols[n.sh] = BaseField::from(sh as u32);
    cols[n.r] = BaseField::from(rem as i32);
    let diff = if clamped { raw - POW2_MAX_SHIFT as i64 } else { POW2_MAX_SHIFT as i64 - 1 - raw };
    if !(0..=254).contains(&diff) {
        return Err(bad("the quantization shift is out of the range the constraints support"));
    }
    cols[n.diff] = BaseField::from(diff as u32);
    if clamped {
        cols[n.c] = BaseField::from(1u32);
    }
    if rem.abs() == half {
        cols[n.e] = BaseField::from(1u32);
        cols[n.half] = BaseField::from((qa / 2) as u32);
    } else {
        put(&mut cols, &n.v1, half - 1 + rem)?;
        put(&mut cols, &n.v2, half - 1 - rem)?;
    }
    let g = 127 - qa;
    cols[n.g] = BaseField::from(g as u32);
    if g == 0 {
        cols[n.z] = BaseField::from(1u32);
        put(&mut cols, &n.rn, -rem)?;
    } else {
        cols[n.ginv] = BaseField::from(g as u32).inverse();
    }
    if hit {
        if sh > HIT_MAX_SHIFT || 2 * m <= 127 * p2 {
            return Err(bad("this element does not justify the scaler"));
        }
        cols[n.h] = BaseField::from(1u32);
        cols[n.hs] = BaseField::from((HIT_MAX_SHIFT - sh) as u32);
        put(&mut cols, &n.hv, 2 * m - 127 * p2 - 1)?;
    }
    Ok(cols)
}

/// The first element that justifies the scaler `kexp` (`2m > 127 * 2^sh`, with a shift of at most 20).
fn first_hit(floats: &[f32], kexp: i32) -> Option<usize> {
    floats.iter().position(|&v| {
        let (_, m, ep) = float_parts(v);
        let sh = shift_of(ep, kexp) as i64;
        (1..=HIT_MAX_SHIFT).contains(&sh) && 2 * m as i64 > 127 * (1i64 << sh)
    })
}

impl BuiltComponent {
    fn new(kind: Kind, base: Vec<BaseColumn>) -> Self {
        let rows = 1usize << kind.log_size();
        let pre = kind.pre_cols().into_iter().map(|p| column(rows, p.values.into_iter().map(BaseField::from))).collect();
        Self { kind, pre, base }
    }

    pub fn pre_evals(&self) -> Vec<Eval> {
        let domain = CanonicCoset::new(self.kind.log_size()).circle_domain();
        self.pre.iter().map(|c| CircleEvaluation::new(domain, c.clone())).collect()
    }

    pub fn base_evals(&self) -> ColumnVec<Eval> {
        let domain = CanonicCoset::new(self.kind.log_size()).circle_domain();
        self.base.iter().map(|c| CircleEvaluation::new(domain, c.clone())).collect()
    }
}

/// Builds the columns of every component. With `checked` the witness must be consistent (see [`NetworkWitness::check`]); without, values outside the
/// ranges are written as they are (tests of the verifier use that to forge proofs).
pub fn build(w: &NetworkWitness, checked: bool) -> Result<NetworkTrace, WitnessError> {
    if checked {
        w.check()?;
    }
    let kinds = layout(&w.specs);
    let mut components = Vec::with_capacity(kinds.len());
    let mut mult = vec![0u32; TABLE_ROWS];
    let mut pow2_mult = vec![0u32; POW2_ROWS];
    for kind in kinds {
        if kind.is_table() {
            continue;
        }
        let rows = 1usize << kind.log_size();
        let base: Vec<BaseColumn> = match &kind {
            Kind::Matmul { layer, spec } => {
                let (k, n) = (spec.k, spec.n);
                let (qx, qw, acc) = (w.x(*layer), &w.q_w[*layer], &w.acc[*layer]);
                let mut cols = Vec::with_capacity(2 * k + 1);
                for kk in 0..k {
                    cols.push(column(rows, (0..spec.matmul_rows()).map(|r| BaseField::from(qx[(r / n) * k + kk]))));
                }
                for kk in 0..k {
                    cols.push(column(rows, (0..spec.matmul_rows()).map(|r| BaseField::from(qw[kk * n + r % n]))));
                }
                cols.push(column(rows, acc.iter().map(|&a| BaseField::from(a))));
                cols
            }
            Kind::Tensor { layer, which, kexp, .. } => {
                let (floats, qs) = match which {
                    Which::X => (&w.x0_float, &w.q_x0),
                    Which::W => (&w.w_float[*layer], &w.q_w[*layer]),
                };
                let c = QCols::new(*kexp);
                let hit_row = if c.nz.is_some() { first_hit(floats, *kexp) } else { None };
                if c.nz.is_some() && hit_row.is_none() && checked {
                    return Err(WitnessError::ScalerExponent { what: format!("{which:?} tensor of layer {layer}"), expected: *kexp, found: *kexp });
                }
                let mut table = vec![Vec::with_capacity(rows); c.total];
                for r in 0..rows {
                    let (parts, q) = if r < floats.len() { (float_parts(floats[r]), qs[r]) } else { ((0, 0, 1), 0) };
                    match quant_row(*kexp, &c, r, parts, q, Some(r) == hit_row) {
                        Ok(row) => row.into_iter().enumerate().for_each(|(i, v)| table[i].push(v)),
                        Err(e) if checked => return Err(e),
                        Err(_) => {
                            // unchecked: keep the float and q, whatever the rest holds
                            let mut row = vec![BaseField::from(0u32); c.total];
                            row[c.q] = BaseField::from(q);
                            row[c.s] = BaseField::from(parts.0);
                            row[c.m] = BaseField::from(parts.1);
                            row[c.ep] = BaseField::from(parts.2);
                            row.into_iter().enumerate().for_each(|(i, v)| table[i].push(v));
                        }
                    }
                }
                table.into_iter().map(|col| column(rows, col)).collect()
            }
            Kind::Requant { layer, spec, .. } => {
                let c = RqCols::new(spec);
                let (acc, out) = (&w.acc[*layer], &w.out[*layer]);
                let t = spec.hit_threshold();
                let hit_row = (spec.mode() != Mode::Zero).then(|| acc.iter().position(|a| a.unsigned_abs() >= t)).flatten();
                if spec.mode() != Mode::Zero && hit_row.is_none() && checked {
                    return Err(WitnessError::NoLargestAccumulator { layer: *layer });
                }
                let mut table = vec![Vec::with_capacity(rows); c.total];
                for r in 0..rows {
                    let (a, o) = if r < acc.len() { (acc[r], out[r]) } else { (0, 0) };
                    let hit = if Some(r) == hit_row { a.signum() } else { 0 };
                    let hit = if hit == 0 && Some(r) == hit_row { 1 } else { hit };
                    match requant_row(spec, &c, r, a, o, hit) {
                        Ok(row) => row.into_iter().enumerate().for_each(|(i, v)| table[i].push(v)),
                        Err(e) if checked => return Err(e),
                        // unchecked: keep the row, whatever it holds
                        Err(_) => {
                            let mut row = vec![BaseField::from(0u32); c.total];
                            row[c.acc] = BaseField::from(a);
                            row[c.out] = BaseField::from(o);
                            row.into_iter().enumerate().for_each(|(i, v)| table[i].push(v));
                        }
                    }
                }
                table.into_iter().map(|col| column(rows, col)).collect()
            }
            Kind::Table | Kind::Pow2Table => unreachable!(),
        };
        #[cfg(any(test, feature = "forgery"))]
        let base = {
            let mut base = base;
            HOOK.with(|h| {
                if let Some(f) = h.borrow().as_ref() {
                    f(&kind, &mut base)
                }
            });
            base
        };
        let built = BuiltComponent::new(kind, base);
        count_range_uses(&built, &mut mult, &mut pow2_mult);
        components.push(built);
    }
    // The table: its multiplicity column (the value 254 occurs in two rows; all its uses are booked on the first).
    let table_kind = Kind::Table;
    let column_mult = column(TABLE_ROWS, (0..TABLE_ROWS).map(|i| BaseField::from(if i < TABLE_ROWS - 1 { mult[i] } else { 0 })));
    components.push(BuiltComponent::new(table_kind, vec![column_mult]));
    // The power-of-two table: shifts 1 ..= 25, the last one repeated; all its uses are booked on its first row.
    let column_pow2 = column(POW2_ROWS, (0..POW2_ROWS).map(|i| BaseField::from(if i < POW2_MAX_SHIFT as usize { pow2_mult[i] } else { 0 })));
    components.push(BuiltComponent::new(Kind::Pow2Table, vec![column_pow2]));
    Ok(NetworkTrace { components })
}

impl Kind {
    pub fn is_table(&self) -> bool {
        matches!(self, Kind::Table | Kind::Pow2Table)
    }
}

/// Counts how often each table value is looked up by the component (padding rows included).
fn count_range_uses(c: &BuiltComponent, mult: &mut [u32], pow2_mult: &mut [u32]) {
    let rows = 1usize << c.kind.log_size();
    for u in c.kind.uses() {
        match u {
            Use::Range { value: Src::Col(i), offset } => {
                for row in 0..rows {
                    let v = (c.base[i].at(row) + BaseField::from(offset)).0; // in M31: a negative value is p - |x|
                    if v <= TABLE_MAX {
                        mult[v as usize] += 1;
                    }
                }
            }
            Use::Pow2 { sh: Src::Col(i), .. } => {
                for row in 0..rows {
                    let v = c.base[i].at(row).0;
                    if (1..=POW2_MAX_SHIFT).contains(&v) {
                        pow2_mult[v as usize - 1] += 1;
                    }
                }
            }
            _ => {}
        }
    }
}

fn packed(c: &BuiltComponent, s: Src, vec_row: usize) -> PackedBaseField {
    match s {
        Src::Pre(i) => c.pre[i].data[vec_row],
        Src::Col(i) => c.base[i].data[vec_row],
        Src::Const(v) => PackedBaseField::broadcast(BaseField::from(v)),
    }
}

fn numerator(c: &BuiltComponent, n: Num, vec_row: usize) -> PackedSecureField {
    match n {
        Num::One => PackedSecureField::from(PackedBaseField::broadcast(BaseField::from(1u32))),
        Num::Pre(i) => c.pre[i].data[vec_row].into(),
        Num::NegPre(i) => (-c.pre[i].data[vec_row]).into(),
        Num::Col(i) => c.base[i].data[vec_row].into(),
        Num::NegCol(i) => (-c.base[i].data[vec_row]).into(),
        Num::Sum(i, j) => (c.base[i].data[vec_row] + c.base[j].data[vec_row]).into(),
    }
}

/// The LogUp interaction columns of every component and their claimed sums, in the order of the components.
pub fn interaction_traces(t: &NetworkTrace, lookups: &Lookups) -> (Vec<Eval>, Vec<SecureField>) {
    let mut all = Vec::new();
    let mut sums = Vec::new();
    for c in &t.components {
        let log_size = c.kind.log_size();
        let uses = c.kind.uses();
        let mut logup = LogupTraceGenerator::new(log_size);
        for chunk in &uses.iter().chunks(2) {
            let chunk = chunk.collect_vec();
            let mut col = logup.new_col();
            for vec_row in 0..(1usize << (log_size - LOG_N_LANES)) {
                let frac = |u: &Use| -> (PackedSecureField, PackedSecureField) {
                    match *u {
                        Use::Range { value, offset } => {
                            let v = packed(c, value, vec_row) + PackedBaseField::broadcast(BaseField::from(offset));
                            (numerator(c, Num::One, vec_row), lookups.range_elements().combine(&[v]))
                        }
                        Use::RangeYield { value, num } => (numerator(c, num, vec_row), lookups.range_elements().combine(&[packed(c, value, vec_row)])),
                        Use::Tensor { tag, i, j, value, num } => {
                            let tag = PackedBaseField::broadcast(BaseField::from(tag));
                            let den = lookups.tensor_elements().combine(&[tag, packed(c, i, vec_row), packed(c, j, vec_row), packed(c, value, vec_row)]);
                            (numerator(c, num, vec_row), den)
                        }
                        Use::Float { tag, i, j, s, m, ep, num } => {
                            let tag = PackedBaseField::broadcast(BaseField::from(tag));
                            let den = lookups.float_elements().combine(&[tag, packed(c, i, vec_row), packed(c, j, vec_row), packed(c, s, vec_row), packed(c, m, vec_row), packed(c, ep, vec_row)]);
                            (numerator(c, num, vec_row), den)
                        }
                        Use::Pow2 { sh, p2 } => (numerator(c, Num::One, vec_row), lookups.pow2_elements().combine(&[packed(c, sh, vec_row), packed(c, p2, vec_row)])),
                        Use::Pow2Yield { sh, p2, num } => (numerator(c, num, vec_row), lookups.pow2_elements().combine(&[packed(c, sh, vec_row), packed(c, p2, vec_row)])),
                    }
                };
                let (n0, d0) = frac(chunk[0]);
                match chunk.get(1) {
                    Some(u1) => {
                        let (n1, d1) = frac(u1);
                        col.write_frac(vec_row, n0 * d1 + n1 * d0, d0 * d1);
                    }
                    None => col.write_frac(vec_row, n0, d0),
                }
            }
            col.finalize_col();
        }
        let (trace, sum) = logup.finalize_last();
        all.extend(trace);
        sums.push(sum);
    }
    (all, sums)
}
