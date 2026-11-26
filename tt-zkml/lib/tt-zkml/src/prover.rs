//! Proof generation and verification with stwo.
//!
//! [`prove_layers`] proves the matmul constraints and the range checks of [`crate::air`] for a list of layers (one stwo component each, plus the range
//! table, all in one proof);
//! [`verify`] checks such a proof. [`prove`], from the [`crate::Trace`] of the device, is still missing: the trace does not carry `q_x` / `q_w` yet.
//!
//! The statement a proof makes today is only "these shapes have a witness satisfying `acc = x * w`": the accumulator and the quantized tensors are not
//! public inputs yet (see the limits listed in [`crate::air`]).

use itertools::Itertools;
use num_traits::Zero;
use stwo::core::air::Component;
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fri::FriParams;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::core::verifier::verify as stwo_verify;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::{CommitmentSchemeProver, ComponentProver, prove as stwo_prove};
use stwo_constraint_framework::TraceLocationAllocator;

use crate::air::{
    MatmulComponent, MatmulEval, MatmulShape, MatmulWitness, RangeLookup, TABLE_LOG_SIZE, TableComponent, TableEval, table_column_id, table_trace,
    table_interaction_trace, table_values,
};
use crate::device::Trace;

/// A proof of the matmuls of a network: the public shapes and the serialized stwo proof.
#[derive(Debug, Clone)]
pub struct Proof {
    pub shapes: Vec<MatmulShape>,
    /// The LogUp claimed sums: one per layer, then the table's.
    pub claimed_sums: Vec<SecureField>,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProveError {
    /// The proof from the device trace needs `q_x` / `q_w` in the trace, which it does not carry yet.
    NotImplemented,
    /// The witness is not consistent (see [`crate::air::WitnessError`]).
    Witness(String),
    /// Nothing to prove.
    Empty,
    /// stwo failed to prove.
    Prover(String),
    /// The proof is not valid.
    Verification(String),
    /// The proof could not be decoded.
    Decode(String),
}

impl std::fmt::Display for ProveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProveError::NotImplemented => write!(f, "proving the device trace is not implemented yet: the trace does not carry the quantized input and weights (q_x, q_w), and the constraints for the requantization are missing"),
            ProveError::Witness(e) => write!(f, "invalid witness: {e}"),
            ProveError::Empty => write!(f, "no layers to prove"),
            ProveError::Prover(e) => write!(f, "proving failed: {e}"),
            ProveError::Verification(e) => write!(f, "the proof is not valid: {e}"),
            ProveError::Decode(e) => write!(f, "cannot decode the proof: {e}"),
        }
    }
}

impl std::error::Error for ProveError {}

const LOG_BLOWUP_FACTOR: u32 = 1;

/// The commitment parameters for traces of at most `2^max_log_size` rows. 70 queries at blowup 2 plus 10 bits of proof of work: about 80 bits.
fn pcs_config(max_log_size: u32) -> PcsConfig {
    let fri_params = FriParams::new(10, 0, LOG_BLOWUP_FACTOR, 70, 1);
    PcsConfig {
        fri_params,
        trace_lifting_log_size: max_log_size + LOG_BLOWUP_FACTOR,
        preprocessed_lifting_log_size: max_log_size + LOG_BLOWUP_FACTOR,
    }
}

/// The largest trace of a proof of `shapes` (the table included).
fn max_log_size(shapes: &[MatmulShape]) -> u32 {
    shapes.iter().map(|s| s.log_size()).max().unwrap().max(TABLE_LOG_SIZE)
}

/// One component per layer and the table, in the order of their columns in the trees.
fn components(shapes: &[MatmulShape], lookup: &RangeLookup, claimed_sums: &[SecureField]) -> (Vec<MatmulComponent>, TableComponent) {
    let mut allocator = TraceLocationAllocator::new_with_preprocessed_columns(&[table_column_id()]);
    let layers = shapes
        .iter()
        .zip(claimed_sums)
        .map(|(&shape, &sum)| MatmulComponent::new(&mut allocator, MatmulEval { shape, lookup: lookup.clone() }, sum))
        .collect_vec();
    let table = TableComponent::new(&mut allocator, TableEval { lookup: lookup.clone() }, claimed_sums[shapes.len()]);
    (layers, table)
}

fn twiddles(max_log_size: u32) -> stwo::prover::poly::twiddles::TwiddleTree<SimdBackend> {
    SimdBackend::precompute_twiddles(CanonicCoset::new(max_log_size + 1 + LOG_BLOWUP_FACTOR).circle_domain().half_coset)
}

/// Proves the matmul constraints of `layers` and the range checks of their quantized values.
pub fn prove_layers(layers: &[MatmulWitness]) -> Result<Proof, ProveError> {
    prove_impl(layers, true)
}

/// `checked = false` also accepts values outside `[-127, 127]` (the lookups for them do not cancel): for the test that the verifier rejects them.
fn prove_impl(layers: &[MatmulWitness], checked: bool) -> Result<Proof, ProveError> {
    if layers.is_empty() {
        return Err(ProveError::Empty);
    }
    let shapes = layers.iter().map(|l| l.shape).collect_vec();
    let max_log = max_log_size(&shapes);
    let config = pcs_config(max_log);
    let twiddles = twiddles(max_log);

    let channel = &mut Blake2sChannel::default();
    let mut scheme = CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    // Preprocessed tree: the range table.
    let mut tree = scheme.tree_builder();
    tree.extend_evals(vec![table_values()]);
    tree.commit(channel);

    // Base trace: the layers, then the multiplicities of the table.
    let columns = layers.iter().map(|l| l.columns()).collect_vec();
    if checked {
        for l in layers {
            l.check().map_err(|e| ProveError::Witness(e.to_string()))?;
        }
    }
    let mut mult = vec![0u32; 1 << TABLE_LOG_SIZE];
    // The multiplicities do not depend on the lookup elements (drawn after the base trace is committed), so they are counted first.
    for (l, cols) in layers.iter().zip(&columns) {
        count_lookups(l.shape, cols, &mut mult);
    }
    let mut tree = scheme.tree_builder();
    tree.extend_evals(layers.iter().flat_map(|l| l.trace()).chain(table_trace(&mult)).collect_vec());
    tree.commit(channel);

    // Interaction trace.
    let lookup = RangeLookup::draw(channel);
    let mut claimed_sums = Vec::new();
    let mut interaction = Vec::new();
    for (l, cols) in layers.iter().zip(&columns) {
        let (trace, sum) = l.interaction_trace(cols, &lookup);
        interaction.extend(trace);
        claimed_sums.push(sum);
    }
    let (trace, sum) = table_interaction_trace(&mult, &lookup);
    interaction.extend(trace);
    claimed_sums.push(sum);
    let mut tree = scheme.tree_builder();
    tree.extend_evals(interaction);
    tree.commit(channel);

    let (layer_components, table_component) = components(&shapes, &lookup, &claimed_sums);
    let refs = layer_components
        .iter()
        .map(|c| c as &dyn ComponentProver<SimdBackend>)
        .chain([&table_component as &dyn ComponentProver<SimdBackend>])
        .collect_vec();
    let proof = stwo_prove::<SimdBackend, Blake2sMerkleChannel>(&refs, channel, scheme).map_err(|e| ProveError::Prover(format!("{e:?}")))?;
    let bytes = serde_json::to_vec(&proof).map_err(|e| ProveError::Prover(e.to_string()))?;
    Ok(Proof { shapes, claimed_sums, bytes })
}

/// Counts how often every table value is looked up by the layer (padding rows included).
fn count_lookups(shape: MatmulShape, columns: &[stwo::prover::backend::simd::column::BaseColumn], mult: &mut [u32]) {
    use stwo::prover::backend::Column;
    for column in &columns[..2 * shape.k] {
        for row in 0..(1usize << shape.log_size()) {
            let v = (column.at(row) + stwo::core::fields::m31::BaseField::from(127u32)).0; // in M31: a negative value is p - |x|
            if v <= 254 {
                mult[v as usize] += 1;
            }
        }
    }
}

/// Verifies a proof made by [`prove_layers`].
pub fn verify(proof: &Proof) -> Result<(), ProveError> {
    if proof.shapes.is_empty() {
        return Err(ProveError::Empty);
    }
    if proof.claimed_sums.len() != proof.shapes.len() + 1 {
        return Err(ProveError::Decode("one claimed sum per layer and one for the table are expected".into()));
    }
    // Every lookup must be answered by the table: the LogUp sums of all the components cancel.
    if proof.claimed_sums.iter().copied().sum::<SecureField>() != SecureField::zero() {
        return Err(ProveError::Verification("the lookups into the range table do not cancel".into()));
    }
    let stark: StarkProof<Blake2sMerkleHasher> = serde_json::from_slice(&proof.bytes).map_err(|e| ProveError::Decode(e.to_string()))?;
    let max_log = max_log_size(&proof.shapes);
    let config = pcs_config(max_log);

    // The preprocessed commitment must be the one of the range table: the verifier knows the constant, so it recomputes the root.
    {
        let twiddles = twiddles(max_log);
        let mut scheme = CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);
        let mut tree = scheme.tree_builder();
        tree.extend_evals(vec![table_values()]);
        tree.commit(&mut Blake2sChannel::default());
        if scheme.roots()[0] != stark.commitments[0] {
            return Err(ProveError::Verification("the preprocessed commitment is not the range table".into()));
        }
    }

    let channel = &mut Blake2sChannel::default();
    let scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
    // The component structure does not depend on the lookup elements, which are drawn after the base trace is committed.
    let placeholder = RangeLookup::dummy();
    let sizes = {
        let (layers, table) = components(&proof.shapes, &placeholder, &proof.claimed_sums);
        TreeVec::concat_cols(layers.iter().map(|c| c.trace_log_degree_bounds()).chain([table.trace_log_degree_bounds()]))
    };
    // The shapes are the verifier's statement: a proof of other shapes has other column counts. Refuse it here, stwo would panic on it.
    let shaped = stark.commitments.len() == 4 // + the composition polynomial
        && stark.sampled_values.len() == 4
        && (0..3).all(|tree| stark.sampled_values[tree].len() == sizes[tree].len());
    if !shaped {
        return Err(ProveError::Verification("the proof does not have the columns of these shapes".into()));
    }
    scheme.commit(stark.commitments[0], &sizes[0], channel);
    scheme.commit(stark.commitments[1], &sizes[1], channel);
    let lookup = RangeLookup::draw(channel);
    scheme.commit(stark.commitments[2], &sizes[2], channel);

    let (layers, table) = components(&proof.shapes, &lookup, &proof.claimed_sums);
    let refs = layers.iter().map(|c| c as &dyn Component).chain([&table as &dyn Component]).collect_vec();
    stwo_verify(&refs, channel, scheme, stark).map_err(|e| ProveError::Verification(format!("{e:?}")))
}

/// Generates the proof that `trace` is the execution of the quantized network.
pub fn prove(_trace: &Trace) -> Result<Proof, ProveError> {
    Err(ProveError::NotImplemented)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::MatmulWitness;

    fn layer(m: usize, k: usize, n: usize, seed: u32) -> MatmulWitness {
        let shape = MatmulShape { m, k, n };
        let values = |len: usize, s: u32| -> Vec<i32> {
            let mut x = s.wrapping_mul(2654435761).wrapping_add(1);
            (0..len)
                .map(|_| {
                    x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((x >> 16) % 255) as i32 - 127
                })
                .collect()
        };
        let q_x = values(m * k, seed);
        let q_w = values(k * n, seed + 1);
        let acc = MatmulWitness::reference_acc(shape, &q_x, &q_w);
        MatmulWitness::new(shape, q_x, q_w, acc).unwrap()
    }

    #[test]
    fn prove_and_verify_one_layer() {
        let proof = prove_layers(&[layer(8, 16, 8, 1)]).unwrap();
        verify(&proof).unwrap();
    }

    #[test]
    fn prove_and_verify_layers_of_different_sizes() {
        let proof = prove_layers(&[layer(32, 64, 32, 1), layer(32, 32, 16, 2), layer(32, 16, 8, 3)]).unwrap();
        assert_eq!(proof.shapes.len(), 3);
        verify(&proof).unwrap();
    }

    #[test]
    fn a_wrong_accumulator_does_not_verify() {
        // The checked API refuses a wrong accumulator; the unchecked path hands it to stwo, whose prover refuses it.
        let mut bad = layer(8, 16, 8, 1);
        bad.acc[10] += 1;
        assert!(matches!(prove_layers(std::slice::from_ref(&bad)), Err(ProveError::Witness(_))));
        // stwo's prover refuses a trace that breaks the constraints (ConstraintsNotSatisfied).
        assert!(matches!(prove_impl(&[bad], false), Err(ProveError::Prover(_))));
    }

    #[test]
    fn a_tampered_proof_does_not_verify() {
        let mut proof = prove_layers(&[layer(8, 16, 8, 1)]).unwrap();
        let mid = proof.bytes.len() / 2;
        proof.bytes[mid] = if proof.bytes[mid] == b'1' { b'2' } else { b'1' };
        assert!(verify(&proof).is_err());
    }

    #[test]
    fn a_proof_for_other_shapes_does_not_verify() {
        let mut proof = prove_layers(&[layer(8, 16, 8, 1)]).unwrap();
        proof.shapes[0].k = 8;
        assert!(verify(&proof).is_err());
    }

    #[test]
    fn a_value_outside_the_range_does_not_verify() {
        // q_x[0] = 200 with a consistent accumulator: the matmul constraint holds, only the range lookup can catch it.
        let mut w = layer(8, 16, 8, 1);
        w.q_x[0] = 200;
        w.acc = MatmulWitness::reference_acc(w.shape, &w.q_x, &w.q_w);
        assert!(matches!(prove_layers(std::slice::from_ref(&w)), Err(ProveError::Witness(_))));
        let proof = prove_impl(&[w], false).unwrap();
        assert!(verify(&proof).is_err());
    }

    #[test]
    fn the_extremes_of_the_range_verify() {
        let shape = MatmulShape { m: 4, k: 4, n: 4 };
        let q_x = vec![127, -127, 127, -127].repeat(4);
        let q_w = vec![-127, 127, 127, -127].repeat(4);
        let acc = MatmulWitness::reference_acc(shape, &q_x, &q_w);
        let proof = prove_layers(&[MatmulWitness::new(shape, q_x, q_w, acc).unwrap()]).unwrap();
        verify(&proof).unwrap();
    }

    #[test]
    fn a_forged_claimed_sum_does_not_verify() {
        let mut proof = prove_layers(&[layer(8, 16, 8, 1)]).unwrap();
        proof.claimed_sums[0] += SecureField::from_u32_unchecked(1, 0, 0, 0);
        assert!(verify(&proof).is_err());
    }

    #[test]
    fn nothing_to_prove() {
        assert_eq!(prove_layers(&[]).unwrap_err(), ProveError::Empty);
    }
}
