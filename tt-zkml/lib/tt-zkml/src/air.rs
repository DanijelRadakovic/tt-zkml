//! The constraints (AIR) of the execution trace of a quantized network, over the M31 field of stwo.
//!
//! Implemented so far: the int8 matmul of a layer ([`MatmulEval`]) with the range checks of its quantized values (a lookup into the table of
//! [`TableEval`]). Still to come, in this order: the requantization and ReLU, the link between the layers (the output of layer `l` is the input of
//! layer `l + 1`) and the quantization of the input.
//!
//! A layer computes `acc[m, n] = sum_k q_x[m, k] * q_w[k, n]` with `q_x`, `q_w` in `[-127, 127]`, exactly, in the int32 accumulator of the device. In M31
//! the same equation holds without wrap-around because `|acc| <= 127 * 127 * K < 2^31 - 1` for `K <= 133_000`, so it is a plain degree-2 constraint.
//!
//! Trace layout of one layer: one row per output element `(m, n)` (row `m * N + n`), `2K + 1` columns
//! `x_0 .. x_{K-1}, w_0 .. w_{K-1}, acc`. The constraint of every row is `acc - sum_k x_k * w_k = 0`. Rows beyond `M * N` (padding to a power of two)
//! are all zero, which satisfies it.
//!
//! Range check: every `x_k` and `w_k` of every row is looked up as `v = value + 127` in the table `0 ..= 254` (LogUp, the relation [`RangeLookup`]).
//! The table is a preprocessed column (a constant the verifier knows) of 256 rows: `0 ..= 254` and `254` once more, so `-127 ..= 127` is exact. The
//! table component carries the multiplicities and yields them back; the sum of the two components' claimed sums must be zero.
//!
//! What this does not prove yet (the reason it is a first slice, not a sound proof of inference): nothing ties `x` and `w` to the committed input
//! and weights; that is the next lookup.

use itertools::Itertools;
use num_traits::One;
use stwo::core::ColumnVec;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::{LOG_N_LANES, PackedBaseField};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, LogupTraceGenerator, RelationEntry, relation};

relation!(RangeLookup, 1);

/// Size of the range table: values `0 ..= 254` (a quantized value plus 127) and one repeat of `254`.
pub const TABLE_LOG_SIZE: u32 = 8;
const TABLE_ROWS: usize = 1 << TABLE_LOG_SIZE;
const OFFSET: u32 = 127;
const MAX_VALUE: u32 = 254;

/// The id of the preprocessed column that holds the table.
pub fn table_column_id() -> PreProcessedColumnId {
    PreProcessedColumnId { id: "tt_zkml_range_table".to_string() }
}

/// The preprocessed column of the table: row `i` holds `min(i, 254)`.
pub fn table_values() -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    let values: BaseColumn = (0..TABLE_ROWS).map(|i| BaseField::from(i.min(MAX_VALUE as usize) as u32)).collect();
    CircleEvaluation::new(CanonicCoset::new(TABLE_LOG_SIZE).circle_domain(), values)
}

/// stwo's SIMD backend works on 16 lanes: a trace has at least 2^4 rows.
const MIN_LOG_SIZE: u32 = 4;
/// The largest `K` for which `127 * 127 * K` stays below the M31 modulus.
pub const MAX_K: usize = (((1u64 << 31) - 1) / (127 * 127)) as usize;

/// The dimensions of one layer: `x [m, k] * w [k, n]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MatmulShape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

impl MatmulShape {
    /// Number of rows that carry data.
    pub fn rows(&self) -> usize {
        self.m * self.n
    }

    /// log2 of the trace height.
    pub fn log_size(&self) -> u32 {
        self.rows().next_power_of_two().ilog2().max(MIN_LOG_SIZE)
    }

    /// Number of trace columns.
    pub fn columns(&self) -> usize {
        2 * self.k + 1
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitnessError {
    /// A buffer does not have the size its shape says.
    Size { what: &'static str, expected: usize, found: usize },
    /// A quantized value is outside `[-127, 127]`.
    Range { what: &'static str, index: usize, value: i32 },
    /// `K` is so large that the accumulator could wrap around in M31.
    TooLarge { k: usize },
    /// The accumulator of the device is not `q_x * q_w`.
    Accumulator { index: usize, expected: i32, found: i32 },
}

impl std::fmt::Display for WitnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WitnessError::Size { what, expected, found } => write!(f, "{what} has {found} elements, expected {expected}"),
            WitnessError::Range { what, index, value } => write!(f, "{what}[{index}] = {value} is outside [-127, 127]"),
            WitnessError::TooLarge { k } => write!(f, "K = {k} exceeds {MAX_K}: the int32 accumulator could wrap around in M31"),
            WitnessError::Accumulator { index, expected, found } => {
                write!(f, "accumulator[{index}] = {found}, but q_x * q_w = {expected}")
            }
        }
    }
}

impl std::error::Error for WitnessError {}

/// Decodes the sign-magnitude int32 the device writes its accumulator in (bit 31 is the sign, the low 31 bits the magnitude).
pub fn sign_magnitude_to_i32(bits: u32) -> i32 {
    let magnitude = (bits & 0x7fff_ffff) as i32;
    if bits >> 31 == 1 { -magnitude } else { magnitude }
}

/// The values of one layer's matmul: the quantized input and weights and the accumulator.
#[derive(Debug, Clone)]
pub struct MatmulWitness {
    pub shape: MatmulShape,
    /// `[m, k]`, row-major.
    pub q_x: Vec<i32>,
    /// `[k, n]`, row-major.
    pub q_w: Vec<i32>,
    /// `[m, n]`, row-major.
    pub acc: Vec<i32>,
}

impl MatmulWitness {
    /// Checks that the buffers fit the shape, that `q_x` and `q_w` are in range and that `acc` is their product (the device's accumulator is checked
    /// against the host this way before anything is proven).
    pub fn new(shape: MatmulShape, q_x: Vec<i32>, q_w: Vec<i32>, acc: Vec<i32>) -> Result<Self, WitnessError> {
        let w = Self { shape, q_x, q_w, acc };
        w.check()?;
        Ok(w)
    }

    /// The accumulator the host computes from `q_x` and `q_w`.
    pub fn reference_acc(shape: MatmulShape, q_x: &[i32], q_w: &[i32]) -> Vec<i32> {
        let MatmulShape { m, k, n } = shape;
        let mut acc = vec![0i32; m * n];
        for i in 0..m {
            for kk in 0..k {
                let x = q_x[i * k + kk];
                for j in 0..n {
                    acc[i * n + j] += x * q_w[kk * n + j];
                }
            }
        }
        acc
    }

    pub(crate) fn check(&self) -> Result<(), WitnessError> {
        let MatmulShape { m, k, n } = self.shape;
        if k > MAX_K {
            return Err(WitnessError::TooLarge { k });
        }
        for (what, expected, found) in [("q_x", m * k, self.q_x.len()), ("q_w", k * n, self.q_w.len()), ("acc", m * n, self.acc.len())] {
            if expected != found {
                return Err(WitnessError::Size { what, expected, found });
            }
        }
        for (what, values) in [("q_x", &self.q_x), ("q_w", &self.q_w)] {
            if let Some((index, &value)) = values.iter().enumerate().find(|(_, v)| v.abs() > 127) {
                return Err(WitnessError::Range { what, index, value });
            }
        }
        let expected = Self::reference_acc(self.shape, &self.q_x, &self.q_w);
        match expected.iter().zip(&self.acc).position(|(e, a)| e != a) {
            Some(index) => Err(WitnessError::Accumulator { index, expected: expected[index], found: self.acc[index] }),
            None => Ok(()),
        }
    }

    /// The base columns of the layer: `x_0 .. x_{K-1}, w_0 .. w_{K-1}, acc` (`2^log_size` rows, zero padded).
    pub fn columns(&self) -> Vec<BaseColumn> {
        let MatmulShape { m: _, k, n } = self.shape;
        let log_size = self.shape.log_size();
        let mut columns = (0..self.shape.columns()).map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size)).collect_vec();
        for row in 0..self.shape.rows() {
            let (i, j) = (row / n, row % n);
            for kk in 0..k {
                columns[kk].set(row, BaseField::from(self.q_x[i * k + kk]));
                columns[k + kk].set(row, BaseField::from(self.q_w[kk * n + j]));
            }
            columns[2 * k].set(row, BaseField::from(self.acc[row]));
        }
        columns
    }

    /// The base trace of the layer, over the canonic domain of its size.
    pub fn trace(&self) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        let domain = CanonicCoset::new(self.shape.log_size()).circle_domain();
        self.columns().into_iter().map(|c| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, c)).collect_vec()
    }

    /// The interaction trace of the layer (the LogUp columns of its range checks) and its claimed sum.
    pub fn interaction_trace(
        &self,
        columns: &[BaseColumn],
        lookup: &RangeLookup,
    ) -> (ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>, SecureField) {
        let k = self.shape.k;
        let log_size = self.shape.log_size();
        let offset = PackedBaseField::broadcast(BaseField::from(OFFSET));
        let mut logup = LogupTraceGenerator::new(log_size);
        for pair in 0..k {
            let mut col = logup.new_col();
            for vec_row in 0..(1usize << (log_size - LOG_N_LANES)) {
                let q0: PackedSecureField = lookup.0.combine(&[columns[2 * pair].data[vec_row] + offset]);
                let q1: PackedSecureField = lookup.0.combine(&[columns[2 * pair + 1].data[vec_row] + offset]);
                col.write_frac(vec_row, q0 + q1, q0 * q1);
            }
            col.finalize_col();
        }
        logup.finalize_last()
    }
}

/// The base trace of the table: the multiplicities `mult[v]` (how often `v` was looked up), the table's value being the preprocessed column.
pub fn table_trace(mult: &[u32]) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    // The value 254 occurs in two rows; all its uses are booked on the first.
    let column: BaseColumn = (0..TABLE_ROWS).map(|i| BaseField::from(if i < TABLE_ROWS - 1 { mult[i] } else { 0 })).collect();
    vec![CircleEvaluation::new(CanonicCoset::new(TABLE_LOG_SIZE).circle_domain(), column)]
}

/// The interaction trace of the table and its claimed sum.
pub fn table_interaction_trace(
    mult: &[u32],
    lookup: &RangeLookup,
) -> (ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>, SecureField) {
    let values = table_values().values;
    let multiplicities = table_trace(mult).remove(0).values;
    let mut logup = LogupTraceGenerator::new(TABLE_LOG_SIZE);
    let mut col = logup.new_col();
    for vec_row in 0..(1usize << (TABLE_LOG_SIZE - LOG_N_LANES)) {
        let q: PackedSecureField = lookup.0.combine(&[values.data[vec_row]]);
        col.write_frac(vec_row, (-multiplicities.data[vec_row]).into(), q);
    }
    col.finalize_col();
    logup.finalize_last()
}

/// The table of the range check: yields every value `mult` times.
#[derive(Clone)]
pub struct TableEval {
    pub lookup: RangeLookup,
}

pub type TableComponent = FrameworkComponent<TableEval>;

impl FrameworkEval for TableEval {
    fn log_size(&self) -> u32 {
        TABLE_LOG_SIZE
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        TABLE_LOG_SIZE + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let value = eval.get_preprocessed_column(table_column_id());
        let mult = eval.next_trace_mask();
        eval.add_to_relation(RelationEntry::new(&self.lookup, (-mult).into(), &[value]));
        eval.finalize_logup_in_pairs();
        eval
    }
}

/// The constraints of the matmul of one layer: for every row, `acc = sum_k x_k * w_k`, and every `x_k`, `w_k` is in `[-127, 127]` (looked up as `value + 127`).
#[derive(Clone)]
pub struct MatmulEval {
    pub shape: MatmulShape,
    pub lookup: RangeLookup,
}

pub type MatmulComponent = FrameworkComponent<MatmulEval>;

impl FrameworkEval for MatmulEval {
    fn log_size(&self) -> u32 {
        self.shape.log_size()
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size() + 1 // degree 2
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let k = self.shape.k;
        let x = (0..k).map(|_| eval.next_trace_mask()).collect_vec();
        let w = (0..k).map(|_| eval.next_trace_mask()).collect_vec();
        let acc = eval.next_trace_mask();
        let offset = E::F::from(BaseField::from(OFFSET));
        for value in x.iter().chain(&w) {
            eval.add_to_relation(RelationEntry::new(&self.lookup, E::EF::one(), &[value.clone() + offset.clone()]));
        }
        let dot = x.into_iter().zip(w).map(|(x, w)| x * w).reduce(|a, b| a + b).expect("K >= 1");
        eval.add_constraint(acc - dot);
        eval.finalize_logup_in_pairs();
        eval
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn pseudo_values(len: usize, seed: u32) -> Vec<i32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 16) % 255) as i32 - 127
            })
            .collect()
    }

    #[allow(dead_code)]
    pub(crate) fn witness(m: usize, k: usize, n: usize, seed: u32) -> MatmulWitness {
        let shape = MatmulShape { m, k, n };
        let q_x = pseudo_values(m * k, seed);
        let q_w = pseudo_values(k * n, seed + 1);
        let acc = MatmulWitness::reference_acc(shape, &q_x, &q_w);
        MatmulWitness::new(shape, q_x, q_w, acc).unwrap()
    }

    #[test]
    fn sign_magnitude_decoding() {
        assert_eq!(sign_magnitude_to_i32(0), 0);
        assert_eq!(sign_magnitude_to_i32(5), 5);
        assert_eq!(sign_magnitude_to_i32(0x8000_0005), -5);
        assert_eq!(sign_magnitude_to_i32(0x7fff_ffff), i32::MAX);
        assert_eq!(sign_magnitude_to_i32(0xffff_ffff), -i32::MAX);
    }

    #[test]
    fn witness_rejects_bad_values() {
        let shape = MatmulShape { m: 2, k: 3, n: 2 };
        let q_x = pseudo_values(6, 1);
        let q_w = pseudo_values(6, 2);
        let acc = MatmulWitness::reference_acc(shape, &q_x, &q_w);
        assert!(MatmulWitness::new(shape, q_x.clone(), q_w.clone(), acc.clone()).is_ok());
        let mut bad = acc.clone();
        bad[3] -= 1;
        assert!(matches!(MatmulWitness::new(shape, q_x.clone(), q_w.clone(), bad), Err(WitnessError::Accumulator { index: 3, .. })));
        let mut big = q_x.clone();
        big[0] = 128;
        assert!(matches!(MatmulWitness::new(shape, big, q_w.clone(), acc.clone()), Err(WitnessError::Range { what: "q_x", .. })));
        assert!(matches!(MatmulWitness::new(shape, q_x, q_w, vec![0; 3]), Err(WitnessError::Size { what: "acc", .. })));
    }
}
