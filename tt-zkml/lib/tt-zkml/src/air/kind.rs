//! The components of the AIR: what each one is, its preprocessed columns, its lookups and its constraints.

use itertools::Itertools;
use num_traits::{One, Zero};
use stwo::core::fields::FieldExpOps;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval, RelationEntry, relation};

use super::spec::{ACC_LIMBS, LIMB_BASE, LayerSpec, Mode};
use crate::witness::{PublicInputs, effective, float_parts};

relation!(RangeLookup, 1);
relation!(TensorLookup, 4);
relation!(FloatLookup, 6);
relation!(Pow2Lookup, 2);

/// The LogUp relations: the range table, the quantized tensors `(tag, i, j, value)`, the public float tensors `(tag, i, j, s, m, e')` and the table of
/// powers of two `(shift, 2^shift)`.
#[derive(Clone)]
pub struct Lookups {
    pub range: RangeLookup,
    pub tensor: TensorLookup,
    pub float: FloatLookup,
    pub pow2: Pow2Lookup,
}

impl Lookups {
    pub fn draw(channel: &mut impl stwo::core::channel::Channel) -> Self {
        Self { range: RangeLookup::draw(channel), tensor: TensorLookup::draw(channel), float: FloatLookup::draw(channel), pow2: Pow2Lookup::draw(channel) }
    }

    pub(super) fn range_elements(&self) -> &stwo_constraint_framework::logup::LookupElements<1> {
        &self.range.0
    }

    pub(super) fn tensor_elements(&self) -> &stwo_constraint_framework::logup::LookupElements<4> {
        &self.tensor.0
    }

    pub(super) fn float_elements(&self) -> &stwo_constraint_framework::logup::LookupElements<6> {
        &self.float.0
    }

    pub(super) fn pow2_elements(&self) -> &stwo_constraint_framework::logup::LookupElements<2> {
        &self.pow2.0
    }

    /// Placeholder elements: the structure of the components does not depend on them.
    pub fn dummy() -> Self {
        Self { range: RangeLookup::dummy(), tensor: TensorLookup::dummy(), float: FloatLookup::dummy(), pow2: Pow2Lookup::dummy() }
    }
}

/// stwo's SIMD backend works on 16 lanes: a trace has at least 2^4 rows.
pub const MIN_LOG_SIZE: u32 = 4;
/// The range table has 256 rows: `0 ..= 254` and `254` once more.
pub const TABLE_LOG_SIZE: u32 = 8;
pub(super) const TABLE_ROWS: usize = 1 << TABLE_LOG_SIZE;
/// The table of powers of two: shifts `1 ..= POW2_MAX_SHIFT` (the last one repeated up to 32 rows).
pub const POW2_LOG_SIZE: u32 = 5;
pub(super) const POW2_ROWS: usize = 1 << POW2_LOG_SIZE;
pub const POW2_MAX_SHIFT: u32 = 25;
/// A float whose quantization shift is above this is "tiny": it quantizes to zero.
pub(super) const HIT_MAX_SHIFT: i64 = 20;
/// Tags of the "largest float" counters in the tensor relation (one per float tensor).
pub fn tag_fhit(tensor: u32) -> u32 {
    (1 << 20) + tensor
}
pub(super) const TABLE_MAX: u32 = 254;
/// A signed value `v` in `[-127, 127]` is range checked as `v + 127` in the table.
pub(super) const OFFSET: u32 = 127;

pub(super) fn log_rows(rows: usize) -> u32 {
    rows.next_power_of_two().ilog2().max(MIN_LOG_SIZE)
}

/// Which tensor of a layer a `Tensor` component commits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Which {
    /// The input of layer 0, `[m, k]`.
    X,
    /// The weights, `[k, n]`.
    W,
}

#[derive(Debug, Clone)]
pub enum Kind {
    Matmul { layer: usize, spec: LayerSpec },
    /// A float tensor and its quantization: `tid` is 0 for the input, `l + 1` for the weights of layer `l`; `kexp` the exponent of its scaler.
    Tensor { layer: usize, which: Which, spec: LayerSpec, kexp: i32, tid: u32 },
    /// `consumers`: how often the next matmul reads each output (its `n`; 1 for the last layer, whose output is public).
    Requant { layer: usize, spec: LayerSpec, consumers: u32 },
    Table,
    /// The powers of two used by the quantization.
    Pow2Table,
}

/// A preprocessed column: constants of the statement.
#[derive(Debug, Clone)]
pub struct PreCol {
    pub id: String,
    pub log_size: u32,
    pub values: Vec<u32>,
}

pub fn table_pre_id() -> String {
    "tt_zkml_range_table".to_string()
}

/// The components of a network, in the order of their columns in the commitment trees.
pub fn layout(specs: &[LayerSpec]) -> Vec<Kind> {
    let mut kinds = Vec::new();
    for (l, &spec) in specs.iter().enumerate() {
        kinds.push(Kind::Matmul { layer: l, spec });
        kinds.push(Kind::Tensor { layer: l, which: Which::W, spec, kexp: spec.kw, tid: l as u32 + 1 });
        if l == 0 {
            kinds.push(Kind::Tensor { layer: 0, which: Which::X, spec, kexp: spec.kx, tid: 0 });
        }
        let consumers = specs.get(l + 1).map_or(1, |next| next.n as u32);
        kinds.push(Kind::Requant { layer: l, spec, consumers });
    }
    kinds.push(Kind::Table);
    kinds.push(Kind::Pow2Table);
    kinds
}

/// The number of lookups into the range table of all components (each row of each component, padding included). LogUp proves equality of counts
/// modulo p only, per tuple: if one value outside the range table could be used p times, it would cancel. Statements must stay below the modulus
/// (checked by `check_lookup_budget`). The other relations need no budget, for another reason than their total number of uses (the tensor relation
/// is used more often than the range relation): the index part of each of their tuples is a preprocessed constant, so a wrong VALUE at one index can
/// be used at most as often as that index occurs (n, m or once; the hit counters at most once per row), far below p. Power-of-two tuples are used once
/// per tensor row, fewer than the range lookups.
pub fn range_lookups(kinds: &[Kind]) -> u128 {
    kinds
        .iter()
        .filter(|k| !matches!(k, Kind::Table | Kind::Pow2Table))
        .map(|k| (1u128 << k.log_size()) * k.uses().iter().filter(|u| matches!(u, Use::Range { .. })).count() as u128)
        .sum()
}

/// Refuses statements whose range lookups could reach the field size.
pub fn check_lookup_budget(kinds: &[Kind]) -> Result<(), String> {
    let total = range_lookups(kinds);
    if total >= (1u128 << 31) - 1 {
        return Err(format!("the statement needs {total} range lookups, at least the field size: LogUp counts would wrap"));
    }
    Ok(())
}

// ---- tags of the tensor relation: the first element of every tuple ----
pub fn tag_x(layer: usize) -> u32 {
    4 * layer as u32
}
pub fn tag_w(layer: usize) -> u32 {
    4 * layer as u32 + 1
}
pub fn tag_acc(layer: usize) -> u32 {
    4 * layer as u32 + 2
}
pub fn tag_hit(layer: usize) -> u32 {
    4 * layer as u32 + 3
}

/// Where a lookup takes a base-field value from.
#[derive(Debug, Clone, Copy)]
pub(super) enum Src {
    Pre(usize),
    Col(usize),
    Const(u32),
}

/// The multiplicity of a lookup.
#[derive(Debug, Clone, Copy)]
pub(super) enum Num {
    One,
    Pre(usize),
    NegPre(usize),
    Col(usize),
    NegCol(usize),
    /// The sum of two columns.
    Sum(usize, usize),
}

#[derive(Debug, Clone, Copy)]
pub(super) enum Use {
    /// `value + offset` is looked up in the range table.
    Range { value: Src, offset: u32 },
    /// The table's own row: yields its value `-mult` times.
    RangeYield { value: Src, num: Num },
    /// A tuple of the tensor relation.
    Tensor { tag: u32, i: Src, j: Src, value: Src, num: Num },
    /// A tuple of the public float relation.
    Float { tag: u32, i: Src, j: Src, s: Src, m: Src, ep: Src, num: Num },
    /// `(shift, 2^shift)` is a row of the power-of-two table.
    Pow2 { sh: Src, p2: Src },
    /// The power-of-two table's own row.
    Pow2Yield { sh: Src, p2: Src, num: Num },
}

/// Column indices of a requantization component.
#[derive(Debug, Clone)]
pub struct RqCols {
    pub acc: usize,
    pub out: usize,
    pub relu: Option<RqRelu>,
    pub right: Option<RqRight>,
    /// Left shift: `sa = acc * 2^ko` of the RAW accumulator, range checked, so that `|acc| * 2^ko <= 127` holds also under a ReLU (where `out` only
    /// bounds the non-negative part).
    pub left_scaled: Option<usize>,
    pub hit: Option<RqHit>,
    pub total: usize,
}

#[derive(Debug, Clone)]
pub struct RqRelu {
    pub p: usize,
    pub ap: usize,
    pub ap_limbs: Vec<usize>,
    pub neg_limbs: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct RqRight {
    pub shift: u32,
    pub e: usize,
    pub half: usize,
    pub v1_limbs: Vec<usize>,
    pub v2_limbs: Vec<usize>,
    /// `127 * 2^shift -+ acc >= 0`: no accumulator exceeds what the exponent allows.
    pub u1_limbs: Vec<usize>,
    pub u2_limbs: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct RqHit {
    pub hp: usize,
    pub hn: usize,
    pub limbs: Vec<usize>,
}

/// Number of limbs for a value below `2^bits`.
fn limbs_for(bits: u32) -> usize {
    bits.div_ceil(super::spec::LIMB_BITS) as usize
}

/// Column indices of a float tensor component (the quantization of `q` from the public float `(s, m, e')`).
#[derive(Debug, Clone)]
pub struct QCols {
    pub q: usize,
    pub s: usize,
    pub m: usize,
    pub ep: usize,
    /// `None` for an all-zero tensor.
    pub nz: Option<QNonZero>,
    pub total: usize,
}

#[derive(Debug, Clone)]
pub struct QNonZero {
    pub qa: usize,
    pub p2: usize,
    pub sh: usize,
    pub c: usize,
    pub diff: usize,
    pub r: usize,
    pub e: usize,
    pub half: usize,
    pub v1: Vec<usize>,
    pub v2: Vec<usize>,
    pub z: usize,
    pub g: usize,
    pub ginv: usize,
    pub rn: Vec<usize>,
    pub h: usize,
    pub hv: Vec<usize>,
    pub hs: usize,
}

impl QCols {
    pub fn new(kexp: i32) -> Self {
        if kexp >= 127 {
            return Self { q: 0, s: 1, m: 2, ep: 3, nz: None, total: 4 };
        }
        let mut next = 4;
        let mut take = |n: usize| -> Vec<usize> {
            let v = (next..next + n).collect_vec();
            next += n;
            v
        };
        let [qa, p2, sh, c, diff, r, e, half] = take(8)[..] else { unreachable!() };
        let v1 = take(ACC_LIMBS);
        let v2 = take(ACC_LIMBS);
        let [z, g, ginv] = take(3)[..] else { unreachable!() };
        let rn = take(ACC_LIMBS);
        let [h] = take(1)[..] else { unreachable!() };
        let hv = take(ACC_LIMBS);
        let [hs] = take(1)[..] else { unreachable!() };
        Self { q: 0, s: 1, m: 2, ep: 3, nz: Some(QNonZero { qa, p2, sh, c, diff, r, e, half, v1, v2, z, g, ginv, rn, h, hv, hs }), total: next }
    }
}

impl RqCols {
    pub fn new(spec: &LayerSpec) -> Self {
        let mut next = 2;
        let mut take = |n: usize| -> Vec<usize> {
            let v = (next..next + n).collect_vec();
            next += n;
            v
        };
        let mode = spec.mode();
        if mode == Mode::Zero {
            return Self { acc: 0, out: 1, relu: None, right: None, left_scaled: None, hit: None, total: 2 };
        }
        let relu = spec.relu.then(|| {
            let [p, ap] = take(2)[..] else { unreachable!() };
            RqRelu { p, ap, ap_limbs: take(ACC_LIMBS), neg_limbs: take(ACC_LIMBS) }
        });
        let right = match mode {
            Mode::Right(shift) => {
                let [e, half] = take(2)[..] else { unreachable!() };
                let n = limbs_for(shift);
                Some(RqRight { shift, e, half, v1_limbs: take(n), v2_limbs: take(n), u1_limbs: take(ACC_LIMBS), u2_limbs: take(ACC_LIMBS) })
            }
            _ => None,
        };
        let left_scaled = matches!(mode, Mode::Left(_)).then(|| take(1)[0]);
        let [hp, hn] = take(2)[..] else { unreachable!() };
        let hit = Some(RqHit { hp, hn, limbs: take(ACC_LIMBS) });
        Self { acc: 0, out: 1, relu, right, left_scaled, hit, total: next }
    }
}

impl Kind {
    pub fn log_size(&self) -> u32 {
        match self {
            Kind::Matmul { spec, .. } => log_rows(spec.matmul_rows()),
            Kind::Tensor { spec, which: Which::X, .. } => log_rows(spec.m * spec.k),
            Kind::Tensor { spec, which: Which::W, .. } => log_rows(spec.k * spec.n),
            Kind::Requant { spec, .. } => log_rows(spec.matmul_rows()),
            Kind::Table => TABLE_LOG_SIZE,
            Kind::Pow2Table => POW2_LOG_SIZE,
        }
    }

    fn prefix(&self) -> String {
        match self {
            Kind::Matmul { layer, .. } => format!("tt_zkml_l{layer}_matmul"),
            Kind::Tensor { layer, which: Which::X, .. } => format!("tt_zkml_l{layer}_x"),
            Kind::Tensor { layer, which: Which::W, .. } => format!("tt_zkml_l{layer}_w"),
            Kind::Requant { layer, .. } => format!("tt_zkml_l{layer}_requant"),
            Kind::Table => "tt_zkml_range".to_string(),
            Kind::Pow2Table => "tt_zkml_pow2".to_string(),
        }
    }

    pub fn pre_ids(&self) -> Vec<String> {
        let names: &[&str] = match self {
            Kind::Matmul { .. } => &["m", "n", "real"],
            Kind::Tensor { .. } => &["i", "j", "mult", "real"],
            Kind::Requant { .. } => &["i", "j", "real", "mult"],
            Kind::Table => return vec![table_pre_id()],
            Kind::Pow2Table => &["sh", "p2"],
        };
        names.iter().map(|n| format!("{}_{n}", self.prefix())).collect()
    }

    /// The preprocessed columns, in the order of [`Kind::pre_ids`].
    pub fn pre_cols(&self) -> Vec<PreCol> {
        let log_size = self.log_size();
        let rows = 1usize << log_size;
        let ids = self.pre_ids();
        let build = |f: &dyn Fn(usize) -> u32| (0..rows).map(f).collect_vec();
        let values: Vec<Vec<u32>> = match self {
            Kind::Matmul { spec, .. } => {
                let real = spec.matmul_rows();
                vec![build(&|r| if r < real { (r / spec.n) as u32 } else { 0 }), build(&|r| if r < real { (r % spec.n) as u32 } else { 0 }), build(&|r| (r < real) as u32)]
            }
            Kind::Tensor { spec, which, .. } => {
                let (rows_t, cols_t, mult) = match which {
                    Which::X => (spec.m, spec.k, spec.n as u32),
                    Which::W => (spec.k, spec.n, spec.m as u32),
                };
                let real = rows_t * cols_t;
                vec![
                    build(&|r| if r < real { (r / cols_t) as u32 } else { 0 }),
                    build(&|r| if r < real { (r % cols_t) as u32 } else { 0 }),
                    build(&|r| if r < real { mult } else { 0 }),
                    build(&|r| (r < real) as u32),
                ]
            }
            Kind::Requant { spec, consumers, .. } => {
                let real = spec.matmul_rows();
                vec![
                    build(&|r| if r < real { (r / spec.n) as u32 } else { 0 }),
                    build(&|r| if r < real { (r % spec.n) as u32 } else { 0 }),
                    build(&|r| (r < real) as u32),
                    build(&|r| if r < real { *consumers } else { 0 }),
                ]
            }
            Kind::Table => vec![build(&|r| r.min(TABLE_MAX as usize) as u32)],
            Kind::Pow2Table => vec![build(&|r| (r as u32 + 1).min(POW2_MAX_SHIFT)), build(&|r| 1u32 << (r as u32 + 1).min(POW2_MAX_SHIFT))],
        };
        ids.into_iter().zip(values).map(|(id, values)| PreCol { id, log_size, values }).collect()
    }

    pub fn n_cols(&self) -> usize {
        match self {
            Kind::Matmul { spec, .. } => 2 * spec.k + 1,
            Kind::Tensor { kexp, .. } => QCols::new(*kexp).total,
            Kind::Requant { spec, .. } => RqCols::new(spec).total,
            Kind::Table | Kind::Pow2Table => 1,
        }
    }

    /// The lookups of a row, in the order the LogUp fractions are batched in pairs.
    pub(super) fn uses(&self) -> Vec<Use> {
        match self {
            Kind::Matmul { layer, spec } => {
                let k = spec.k;
                let mut uses = Vec::with_capacity(2 * k + 1);
                for kk in 0..k {
                    uses.push(Use::Tensor { tag: tag_x(*layer), i: Src::Pre(0), j: Src::Const(kk as u32), value: Src::Col(kk), num: Num::Pre(2) });
                    uses.push(Use::Tensor { tag: tag_w(*layer), i: Src::Const(kk as u32), j: Src::Pre(1), value: Src::Col(k + kk), num: Num::Pre(2) });
                }
                uses.push(Use::Tensor { tag: tag_acc(*layer), i: Src::Pre(0), j: Src::Pre(1), value: Src::Col(2 * k), num: Num::NegPre(2) });
                uses
            }
            Kind::Tensor { layer, which, kexp, tid, .. } => {
                let tag = match which {
                    Which::X => tag_x(*layer),
                    Which::W => tag_w(*layer),
                };
                let c = QCols::new(*kexp);
                let mut uses = vec![
                    Use::Float { tag: *tid, i: Src::Pre(0), j: Src::Pre(1), s: Src::Col(c.s), m: Src::Col(c.m), ep: Src::Col(c.ep), num: Num::Pre(3) },
                    Use::Tensor { tag, i: Src::Pre(0), j: Src::Pre(1), value: Src::Col(c.q), num: Num::NegPre(2) },
                    Use::Range { value: Src::Col(c.q), offset: OFFSET },
                ];
                if let Some(n) = &c.nz {
                    let range0 = |i: usize| Use::Range { value: Src::Col(i), offset: 0 };
                    uses.push(Use::Pow2 { sh: Src::Col(n.sh), p2: Src::Col(n.p2) });
                    uses.extend([range0(n.qa), range0(n.g), range0(n.diff), range0(n.hs)]);
                    uses.push(Use::Range { value: Src::Col(n.half), offset: OFFSET });
                    uses.extend(n.v1.iter().chain(&n.v2).chain(&n.rn).chain(&n.hv).map(|&i| range0(i)));
                    uses.push(Use::Tensor { tag: tag_fhit(*tid), i: Src::Const(0), j: Src::Const(0), value: Src::Const(0), num: Num::Col(n.h) });
                }
                uses
            }
            Kind::Requant { layer, spec, .. } => {
                let c = RqCols::new(spec);
                let mut uses = vec![
                    Use::Tensor { tag: tag_acc(*layer), i: Src::Pre(0), j: Src::Pre(1), value: Src::Col(c.acc), num: Num::Pre(2) },
                    Use::Tensor { tag: tag_x(*layer + 1), i: Src::Pre(0), j: Src::Pre(1), value: Src::Col(c.out), num: Num::NegPre(3) },
                    Use::Range { value: Src::Col(c.out), offset: OFFSET },
                ];
                let limbs = |idx: &[usize]| idx.iter().map(|&i| Use::Range { value: Src::Col(i), offset: 0 }).collect_vec();
                if let Some(r) = &c.relu {
                    uses.extend(limbs(&r.ap_limbs));
                    uses.extend(limbs(&r.neg_limbs));
                }
                if let Some(r) = &c.right {
                    uses.push(Use::Range { value: Src::Col(r.half), offset: OFFSET });
                    uses.extend(limbs(&r.v1_limbs));
                    uses.extend(limbs(&r.v2_limbs));
                    uses.extend(limbs(&r.u1_limbs));
                    uses.extend(limbs(&r.u2_limbs));
                } else if let Some(sa) = c.left_scaled {
                    // the raw accumulator is at most 127 in magnitude, so `acc * 2^ko` cannot wrap, and `|acc| * 2^ko <= 127` (also under a ReLU)
                    uses.push(Use::Range { value: Src::Col(c.acc), offset: OFFSET });
                    uses.push(Use::Range { value: Src::Col(sa), offset: OFFSET });
                }
                if let Some(h) = &c.hit {
                    uses.extend(limbs(&h.limbs));
                    uses.push(Use::Tensor { tag: tag_hit(*layer), i: Src::Const(0), j: Src::Const(0), value: Src::Const(0), num: Num::Sum(h.hp, h.hn) });
                }
                uses
            }
            Kind::Table => vec![Use::RangeYield { value: Src::Pre(0), num: Num::NegCol(0) }],
            Kind::Pow2Table => vec![Use::Pow2Yield { sh: Src::Pre(0), p2: Src::Pre(1), num: Num::NegCol(0) }],
        }
    }

    /// The polynomial constraints of a row. `pre` and `cols` are the preprocessed and the trace columns at the row, in the order of their ids / indices.
    fn constraints<E: EvalAtRow>(&self, eval: &mut E, pre: &[E::F], cols: &[E::F]) {
        let c = |v: u64| E::F::from(BaseField::from((v % ((1u64 << 31) - 1)) as u32));
        let limb_sum = |idx: &[usize]| -> E::F {
            let mut base = 1u32;
            let mut sum = E::F::zero();
            for &i in idx {
                sum = sum + cols[i].clone() * c(base as u64);
                base *= LIMB_BASE;
            }
            sum
        };
        match self {
            Kind::Matmul { spec, .. } => {
                let k = spec.k;
                let dot = (0..k).map(|kk| cols[kk].clone() * cols[k + kk].clone()).reduce(|a, b| a + b).expect("K >= 1");
                eval.add_constraint(cols[2 * k].clone() - dot);
            }
            Kind::Table | Kind::Pow2Table => {}
            Kind::Tensor { kexp, .. } => {
                let qc = QCols::new(*kexp);
                let one = E::F::one();
                // s, m and e' are pinned to the public float by the lookup (as field elements: m < 2^24, s a bit, e' in 1..=254)
                let (q, sgn, m, ep) = (cols[qc.q].clone(), cols[qc.s].clone(), cols[qc.m].clone(), cols[qc.ep].clone());
                let real = pre[3].clone();
                let Some(n) = &qc.nz else {
                    // an all-zero tensor
                    eval.add_constraint(q);
                    eval.add_constraint(m);
                    return;
                };
                let signed = |v: i64| -> E::F { E::F::from(BaseField::from(v.rem_euclid((1i64 << 31) - 1) as u32)) };
                let (qa, p2, sh, cl, diff, r) = (cols[n.qa].clone(), cols[n.p2].clone(), cols[n.sh].clone(), cols[n.c].clone(), cols[n.diff].clone(), cols[n.r].clone());
                // q = (-1)^s * qa, qa >= 0 (range checked)
                eval.add_constraint(q - qa.clone() * (one.clone() - sgn * c(2)));
                // the shift: raw = 150 - e' - kexp. Normally sh = raw <= 24 (diff = 24 - raw >= 0). A tiny float (raw >= 25) quantizes to 0: then
                // c = 1, qa = 0, sh = 25 (so that |r| = m < 2^24 fits) and diff = raw - 25 >= 0. Either way qa * 2^sh stays below the modulus.
                let raw = signed(150 - *kexp as i64) - ep;
                eval.add_constraint(cl.clone() * (cl.clone() - one.clone()));
                eval.add_constraint((one.clone() - cl.clone()) * (sh.clone() - raw.clone()));
                eval.add_constraint((one.clone() - cl.clone()) * (diff.clone() - (c(24) - raw.clone())));
                eval.add_constraint(cl.clone() * (sh.clone() - c(POW2_MAX_SHIFT as u64)));
                eval.add_constraint(cl.clone() * (diff - (raw - c(POW2_MAX_SHIFT as u64))));
                eval.add_constraint(cl * qa.clone());
                // m = qa * 2^sh + r, round half to even: |r| <= 2^(sh-1), a tie has an even qa
                eval.add_constraint(r.clone() - (m.clone() - qa.clone() * p2.clone()));
                let half_p2 = p2.clone() * c(1 << 30); // 2^(sh-1): the inverse of 2 is 2^30 in M31
                let e = cols[n.e].clone();
                eval.add_constraint(e.clone() * (e.clone() - one.clone()));
                eval.add_constraint(e.clone() * (r.clone() - half_p2.clone()) * (r.clone() + half_p2.clone()));
                eval.add_constraint(e.clone() * (qa.clone() - cols[n.half].clone() * c(2)));
                let not_e = one.clone() - e;
                eval.add_constraint(not_e.clone() * (limb_sum(&n.v1) - (half_p2.clone() - one.clone() + r.clone())));
                eval.add_constraint(not_e * (limb_sum(&n.v2) - (half_p2 - one.clone() - r.clone())));
                // the value is at most 127 exactly: g = 127 - qa >= 0, and qa = 127 needs r <= 0
                let (z, g, ginv) = (cols[n.z].clone(), cols[n.g].clone(), cols[n.ginv].clone());
                eval.add_constraint(g.clone() - (c(127) - qa));
                eval.add_constraint(z.clone() * (z.clone() - one.clone()));
                eval.add_constraint(z.clone() * g.clone());
                eval.add_constraint((one.clone() - z.clone()) * (g * ginv - one.clone()));
                eval.add_constraint(limb_sum(&n.rn) + z * r);
                // the scaler is the largest: one element has 2m > 127 * 2^sh (at a shift of at most 20, so that nothing wraps)
                let (h, hs) = (cols[n.h].clone(), cols[n.hs].clone());
                eval.add_constraint(h.clone() * (h.clone() - one.clone()));
                eval.add_constraint(h.clone() * (one.clone() - real)); // only a real element can be the largest (padding rows are free)
                eval.add_constraint(hs - h.clone() * (c(HIT_MAX_SHIFT as u64) - sh));
                eval.add_constraint(h * (limb_sum(&n.hv) - (m * c(2) - p2 * c(127) - one)));
            }
            Kind::Requant { spec, .. } => {
                let _ = pre;
                let rc = RqCols::new(spec);
                let (acc, out) = (cols[rc.acc].clone(), cols[rc.out].clone());
                let one = E::F::one();
                if spec.mode() == Mode::Zero {
                    eval.add_constraint(acc);
                    eval.add_constraint(out);
                    return;
                }
                // the value the requantization works on: the ReLU of the accumulator, or the accumulator
                let a = if let Some(r) = &rc.relu {
                    let p = cols[r.p].clone();
                    let ap = cols[r.ap].clone();
                    eval.add_constraint(p.clone() * (p.clone() - one.clone())); // p is a bit
                    eval.add_constraint(ap.clone() - p.clone() * acc.clone()); // ap = p ? acc : 0
                    eval.add_constraint(ap.clone() - limb_sum(&r.ap_limbs)); // ap >= 0
                    eval.add_constraint((p - one.clone()) * acc.clone() - limb_sum(&r.neg_limbs)); // p = 0  =>  -acc >= 0
                    ap
                } else {
                    acc.clone()
                };
                match spec.mode() {
                    Mode::Left(ko) => {
                        eval.add_constraint(out.clone() - a * c(1u64 << ko));
                        // the raw accumulator, scaled, fits too: under a ReLU `out` bounds only max(acc, 0)
                        eval.add_constraint(cols[rc.left_scaled.unwrap()].clone() - acc.clone() * c(1u64 << ko));
                    }
                    Mode::Right(_) => {
                        let r = rc.right.as_ref().unwrap();
                        let sh = r.shift;
                        let half_range = c(1u64 << (sh - 1)); // H
                        let rem = a - out.clone() * c(1u64 << sh);
                        let e = cols[r.e].clone();
                        eval.add_constraint(e.clone() * (e.clone() - one.clone())); // e is a bit
                        // e = 1: a tie, |rem| = H and the output is even
                        eval.add_constraint(e.clone() * (rem.clone() * rem.clone() - half_range.clone() * half_range.clone()));
                        eval.add_constraint(e.clone() * (out - cols[r.half].clone() * c(2)));
                        // e = 0: |rem| <= H - 1, as two non-negative numbers: v1 = H - 1 + rem, v2 = H - 1 - rem (their sum is fixed)
                        let not_e = one.clone() - e;
                        eval.add_constraint(not_e.clone() * (limb_sum(&r.v1_limbs) - (half_range.clone() - one.clone() + rem.clone())));
                        eval.add_constraint(not_e * (limb_sum(&r.v2_limbs) - (half_range - one.clone() - rem)));
                        // no accumulator exceeds the exponent: -127 * 2^s <= acc <= 127 * 2^s (the raw accumulator, also under a ReLU)
                        let bound = c(127u64 << sh);
                        eval.add_constraint(limb_sum(&r.u1_limbs) - (bound.clone() - acc.clone()));
                        eval.add_constraint(limb_sum(&r.u2_limbs) - (bound + acc.clone()));
                    }
                    Mode::Zero => unreachable!(),
                }
                // the exponent is the largest one: exactly one row has |acc| >= T (counted in the tensor relation against the public -1)
                let h = rc.hit.as_ref().unwrap();
                let (hp, hn) = (cols[h.hp].clone(), cols[h.hn].clone());
                let t = c(spec.hit_threshold() as u64);
                eval.add_constraint(hp.clone() * (hp.clone() - one.clone()));
                eval.add_constraint(hn.clone() * (hn.clone() - one.clone()));
                let real = pre[2].clone();
                eval.add_constraint((hp.clone() + hn.clone()) * (one - real)); // only a real row can be the largest (padding rows are free)
                eval.add_constraint(hp * (acc.clone() - t.clone()) + hn * (-acc - t) - limb_sum(&h.limbs));
            }
        }
    }
}

/// The evaluation of a component for stwo: preprocessed columns, trace columns, constraints, then the lookups.
#[derive(Clone)]
pub struct NetEval {
    pub kind: Kind,
    pub lookups: Lookups,
}

impl FrameworkEval for NetEval {
    fn log_size(&self) -> u32 {
        self.kind.log_size()
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size() + 1 // constraints and LogUp pairs have degree <= 3
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let pre = self.kind.pre_ids().into_iter().map(|id| eval.get_preprocessed_column(PreProcessedColumnId { id })).collect_vec();
        let cols = (0..self.kind.n_cols()).map(|_| eval.next_trace_mask()).collect_vec();
        self.kind.constraints(&mut eval, &pre, &cols);

        let value = |s: Src| -> E::F {
            match s {
                Src::Pre(i) => pre[i].clone(),
                Src::Col(i) => cols[i].clone(),
                Src::Const(v) => E::F::from(BaseField::from(v)),
            }
        };
        let num = |n: Num| -> E::EF {
            match n {
                Num::One => E::EF::one(),
                Num::Pre(i) => pre[i].clone().into(),
                Num::NegPre(i) => (-pre[i].clone()).into(),
                Num::Col(i) => cols[i].clone().into(),
                Num::NegCol(i) => (-cols[i].clone()).into(),
                Num::Sum(i, j) => (cols[i].clone() + cols[j].clone()).into(),
            }
        };
        for u in self.kind.uses() {
            match u {
                Use::Range { value: v, offset } => {
                    let shifted = value(v) + E::F::from(BaseField::from(offset));
                    eval.add_to_relation(RelationEntry::new(&self.lookups.range, E::EF::one(), &[shifted]));
                }
                Use::RangeYield { value: v, num: n } => {
                    eval.add_to_relation(RelationEntry::new(&self.lookups.range, num(n), &[value(v)]));
                }
                Use::Tensor { tag, i, j, value: v, num: n } => {
                    eval.add_to_relation(RelationEntry::new(&self.lookups.tensor, num(n), &[E::F::from(BaseField::from(tag)), value(i), value(j), value(v)]));
                }
                Use::Float { tag, i, j, s, m, ep, num: n } => {
                    eval.add_to_relation(RelationEntry::new(&self.lookups.float, num(n), &[E::F::from(BaseField::from(tag)), value(i), value(j), value(s), value(m), value(ep)]));
                }
                Use::Pow2 { sh, p2 } => {
                    eval.add_to_relation(RelationEntry::new(&self.lookups.pow2, E::EF::one(), &[value(sh), value(p2)]));
                }
                Use::Pow2Yield { sh, p2, num: n } => {
                    eval.add_to_relation(RelationEntry::new(&self.lookups.pow2, num(n), &[value(sh), value(p2)]));
                }
            }
        }
        eval.finalize_logup_in_pairs();
        eval
    }
}

/// The part of the LogUp sum the verifier computes itself, from what it knows:
///
/// * the public output tensor (the last requantization yields it once),
/// * the "largest accumulator" rows (every requantization with a hit consumes one, the public side yields one),
/// * the public float tensors, the model's weights and the input: every quantization consumes the float tuple `(tag, i, j, 2m + s, e')` of its element,
///   the public side yields it, so a quantized tensor is the quantization of exactly these floats,
/// * the "largest float" rows of every non-zero float tensor.
pub fn public_sum(specs: &[LayerSpec], output: &[i32], inputs: &PublicInputs, lookups: &Lookups) -> SecureField {
    let tensor = |tag: u32, i: u32, j: u32, v: BaseField| -> SecureField {
        lookups.tensor_elements().combine::<BaseField, SecureField>(&[BaseField::from(tag), BaseField::from(i), BaseField::from(j), v]).inverse()
    };
    let float = |tag: u32, i: usize, j: usize, v: f32| -> SecureField {
        let (s, m, ep) = float_parts(v);
        lookups
            .float_elements()
            .combine::<BaseField, SecureField>(&[BaseField::from(tag), BaseField::from(i as u32), BaseField::from(j as u32), BaseField::from(s), BaseField::from(m), BaseField::from(ep)])
            .inverse()
    };
    let last = specs.len() - 1;
    let n = specs[last].n;
    let mut sum = SecureField::zero();
    for (r, &v) in output.iter().enumerate() {
        sum += tensor(tag_x(last + 1), (r / n) as u32, (r % n) as u32, BaseField::from(v));
    }
    for (l, spec) in specs.iter().enumerate() {
        if spec.mode() != Mode::Zero {
            sum -= tensor(tag_hit(l), 0, 0, BaseField::zero());
        }
    }
    // the floats the device sees (bf16 rounded for a bf16 model)
    let x = effective(&inputs.input, inputs.dtype);
    for (r, &v) in x.iter().enumerate() {
        sum -= float(0, r / specs[0].k, r % specs[0].k, v);
    }
    if specs[0].kx < 127 {
        sum -= tensor(tag_fhit(0), 0, 0, BaseField::zero());
    }
    for (l, spec) in specs.iter().enumerate() {
        for (r, &v) in effective(&inputs.weights[l], inputs.dtype).iter().enumerate() {
            sum -= float(l as u32 + 1, r / spec.n, r % spec.n, v);
        }
        if spec.kw < 127 {
            sum -= tensor(tag_fhit(l as u32 + 1), 0, 0, BaseField::zero());
        }
    }
    sum
}
