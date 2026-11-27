//! Proof generation and verification with stwo.
//!
//! [`prove`] takes the model, the float input and the device [`Trace`], rebuilds the witness on the CPU ([`NetworkWitness::from_trace`]), converts it to
//! M31 columns and proves the constraints of [`crate::air`]. [`verify`] checks a [`Proof`] against its public statement (the layer specs and the output).

use std::panic::{AssertUnwindSafe, catch_unwind};

use itertools::Itertools;
use num_traits::Zero;
use stwo::core::air::Component;
use stwo::core::channel::{Blake2sChannel, Channel};
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
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

use crate::air::spec::{self, LayerSpec};
use crate::air::{Kind, Lookups, NetEval, build, check_lookup_budget, interaction_traces, layout, public_sum};
use crate::device::Trace;
use crate::modelfile::ModelFile;
use crate::witness::{NetworkWitness, PublicInputs, WitnessError, effective};

type NetComponent = stwo_constraint_framework::FrameworkComponent<NetEval>;

/// A proof of one forward pass with its public statement.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Proof {
    /// The layers: shapes, ReLU, requantization exponent.
    pub specs: Vec<LayerSpec>,
    /// The output of the network, `[m, n_last]` int8 values.
    pub output: Vec<i32>,
    /// The LogUp claimed sums, one per component of [`layout`].
    pub claimed_sums: Vec<SecureField>,
    /// The serialized stwo proof.
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProveError {
    /// The witness is not a valid execution.
    Witness(String),
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
            ProveError::Witness(e) => write!(f, "invalid witness: {e}"),
            ProveError::Prover(e) => write!(f, "proving failed: {e}"),
            ProveError::Verification(e) => write!(f, "the proof is not valid: {e}"),
            ProveError::Decode(e) => write!(f, "cannot decode the proof: {e}"),
        }
    }
}

impl std::error::Error for ProveError {}

impl From<WitnessError> for ProveError {
    fn from(e: WitnessError) -> Self {
        ProveError::Witness(e.to_string())
    }
}

const LOG_BLOWUP_FACTOR: u32 = 2;
/// log2 of the number of coefficients of the last FRI layer.
const LOG_LAST_LAYER_DEGREE_BOUND: u32 = 0;

const POW_BITS: u32 = 16;
const N_QUERIES: usize = 64;
const FOLD_STEP: u32 = 1;

/// The commitment parameters for traces of at most `2^max_log_size` rows: blowup 4, 64 queries and 16 bits of proof of work. Security: about
/// 64 * 2 + 16 = 144 bits under the conjectured (capacity-level) FRI soundness, which is disputed; under proven analyses it is far lower: 80 bits is the
/// limit of the Johnson bound that no concrete parameter reaches, about 66 bits with a concrete proximity parameter, about 59 bits in the
/// unique-decoding regime, and the commit-phase term at the largest supported domain is about 2^-58 (see docs/audit.md).
fn pcs_config(max_log_size: u32) -> PcsConfig {
    PcsConfig {
        fri_params: FriParams::new(POW_BITS, LOG_LAST_LAYER_DEGREE_BOUND, LOG_BLOWUP_FACTOR, N_QUERIES, FOLD_STEP),
        trace_lifting_log_size: max_log_size + LOG_BLOWUP_FACTOR,
        preprocessed_lifting_log_size: max_log_size + LOG_BLOWUP_FACTOR,
    }
}

fn twiddles(max_log_size: u32) -> stwo::prover::poly::twiddles::TwiddleTree<SimdBackend> {
    SimdBackend::precompute_twiddles(CanonicCoset::new(max_log_size + 1 + LOG_BLOWUP_FACTOR).circle_domain().half_coset)
}

fn max_log_size(kinds: &[Kind]) -> u32 {
    kinds.iter().map(|k| k.log_size()).max().unwrap()
}

/// The stwo components of `kinds`; their columns follow the order of `kinds` in every tree.
fn components(kinds: &[Kind], lookups: &Lookups, claimed_sums: &[SecureField]) -> Vec<NetComponent> {
    let ids = kinds.iter().flat_map(|k| k.pre_ids()).map(|id| PreProcessedColumnId { id }).collect_vec();
    let mut allocator = TraceLocationAllocator::new_with_preprocessed_columns(&ids);
    kinds.iter().zip(claimed_sums).map(|(k, &sum)| NetComponent::new(&mut allocator, NetEval { kind: k.clone(), lookups: lookups.clone() }, sum)).collect()
}

/// Binds the whole public statement to the Fiat-Shamir transcript before anything else: the layers (shapes, activation, all exponents), the output
/// and the float input and weights as the device sees them. The preprocessed root alone would not bind the exponents or the activations.
fn mix_statement(channel: &mut Blake2sChannel, specs: &[LayerSpec], output: &[i32], input: &[f32], weights: &[Vec<f32>]) {
    let mut words: Vec<u32> = b"tt-zkml proof v3".chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    // the parameters of the proof system are part of what is proven with
    words.extend([POW_BITS, LOG_LAST_LAYER_DEGREE_BOUND, LOG_BLOWUP_FACTOR, N_QUERIES as u32, FOLD_STEP]);
    words.push(specs.len() as u32);
    for s in specs {
        words.extend([s.m as u32, s.k as u32, s.n as u32, s.relu as u32, s.ko as u32, s.kw as u32, s.kx as u32]);
    }
    words.push(output.len() as u32);
    words.extend(output.iter().map(|&v| v as u32));
    words.push(input.len() as u32);
    words.extend(input.iter().map(|v| v.to_bits()));
    for w in weights {
        words.push(w.len() as u32);
        words.extend(w.iter().map(|v| v.to_bits()));
    }
    channel.mix_u32s(&words);
}

/// The preprocessed tree (constants of the statement) of `kinds`.
fn preprocessed_evals(kinds: &[Kind]) -> Vec<stwo::prover::poly::circle::CircleEvaluation<SimdBackend, stwo::core::fields::m31::BaseField, stwo::prover::poly::BitReversedOrder>> {
    use stwo::core::fields::m31::BaseField;
    use stwo::prover::poly::circle::CircleEvaluation;
    kinds
        .iter()
        .flat_map(|k| {
            let domain = CanonicCoset::new(k.log_size()).circle_domain();
            k.pre_cols().into_iter().map(move |p| CircleEvaluation::new(domain, p.values.into_iter().map(BaseField::from).collect()))
        })
        .collect()
}

impl Proof {
    /// The output as floats, approximately what the unquantized network computes: the public int8 output divided by `2^(kx + sum(kw + ko))`, the exponents
    /// of the statement. Anyone can compute it from the proof alone; it is the same as [`crate::Trace::dequantized_output`] of the execution.
    ///
    /// Meaningful only for a proof that [`verify`] accepted: the exponents of an unverified proof are whatever its author wrote.
    pub fn dequantized_output(&self) -> Vec<f32> {
        let Some(first) = self.specs.first() else { return Vec::new() };
        let exponent: i64 = first.kx as i64 + self.specs.iter().map(|s| s.kw as i64 + s.ko as i64).sum::<i64>();
        let scale = 2f64.powi(exponent.clamp(-2000, 2000) as i32);
        self.output.iter().map(|&v| (v as f64 / scale) as f32).collect()
    }

    /// The proof and its statement as one JSON document.
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("a proof serializes")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ProveError> {
        serde_json::from_slice(bytes).map_err(|e| ProveError::Decode(e.to_string()))
    }
}

/// Rebuilds the witness of the device forward pass on the CPU and proves it.
pub fn prove(model: &ModelFile, input: &[f32], trace: &Trace) -> Result<Proof, ProveError> {
    let witness = NetworkWitness::from_trace(model, input, trace)?;
    prove_witness(&witness)
}

/// Proves a witness: every constraint of the network.
pub fn prove_witness(witness: &NetworkWitness) -> Result<Proof, ProveError> {
    prove_impl(witness, true)
}

/// A proof of a witness that is not checked for consistency: for the tests of a malicious prover (feature `forgery`).
#[cfg(feature = "forgery")]
#[doc(hidden)]
pub fn prove_unchecked(witness: &NetworkWitness) -> Result<Proof, ProveError> {
    prove_impl(witness, false)
}

/// `checked = false` skips the consistency checks of the witness: for the tests that a forged witness cannot be proven or verified.
pub(crate) fn prove_impl(witness: &NetworkWitness, checked: bool) -> Result<Proof, ProveError> {
    check_lookup_budget(&layout(&witness.specs)).map_err(ProveError::Witness)?;
    let trace = build(witness, checked)?;
    let kinds = trace.components.iter().map(|c| c.kind.clone()).collect_vec();
    let max_log = max_log_size(&kinds);
    let config = pcs_config(max_log);
    let twiddles = twiddles(max_log);

    let channel = &mut Blake2sChannel::default();
    mix_statement(channel, &witness.specs, witness.output(), &witness.x0_float, &witness.w_float);
    let mut scheme = CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    let mut tree = scheme.tree_builder();
    tree.extend_evals(preprocessed_evals(&kinds));
    tree.commit(channel);

    let mut tree = scheme.tree_builder();
    tree.extend_evals(trace.components.iter().flat_map(|c| c.base_evals()).collect_vec());
    tree.commit(channel);

    let lookups = Lookups::draw(channel);
    let (interaction, claimed_sums) = interaction_traces(&trace, &lookups);
    channel.mix_felts(&claimed_sums);
    let mut tree = scheme.tree_builder();
    tree.extend_evals(interaction);
    tree.commit(channel);

    let comps = components(&kinds, &lookups, &claimed_sums);
    let refs = comps.iter().map(|c| c as &dyn ComponentProver<SimdBackend>).collect_vec();
    let proof = stwo_prove::<SimdBackend, Blake2sMerkleChannel>(&refs, channel, scheme).map_err(|e| ProveError::Prover(format!("{e:?}")))?;
    let bytes = serde_json::to_vec(&proof).map_err(|e| ProveError::Prover(e.to_string()))?;
    Ok(Proof { specs: witness.specs.clone(), output: witness.output().to_vec(), claimed_sums, bytes })
}

/// Verifies a proof of the execution of the model `inputs.weights` on `inputs.input`: against its public statement (`specs`, `output`) and the public
/// floats.
pub fn verify(proof: &Proof, inputs: &PublicInputs) -> Result<(), ProveError> {
    let fail = |s: &str| Err(ProveError::Verification(s.to_string()));
    // ---- cheap checks first: the statement against the verifier's model ----
    spec::validate(&proof.specs).map_err(|e| ProveError::Verification(e.to_string()))?;
    // The statement is the verifier's model, not the proof's claim: layers, activations and shapes come from `inputs`; only the exponents (which the
    // constraints prove to be the largest ones) are taken from the proof.
    let first = &proof.specs[0];
    let layers = &inputs.layers;
    let in0 = layers.first().map_or(0, |l| l.in_features as usize);
    if layers.len() != proof.specs.len()
        || in0 == 0
        || inputs.input.is_empty()
        || inputs.input.len() % in0 != 0
        || first.m != inputs.input.len() / in0
        || inputs.weights.len() != layers.len()
        || layers.iter().zip(&proof.specs).any(|(l, s)| l.in_features as usize != s.k || l.out_features as usize != s.n || l.relu != s.relu)
        || inputs.weights.iter().zip(&proof.specs).any(|(w, s)| w.len() != s.k * s.n)
    {
        return fail("the statement is not about this model: the layers, their activations or the shapes differ");
    }
    // finite as given (a NaN would round to zero in bf16) and after the dtype's rounding (a large f32 rounds to infinity in bf16)
    let eff_input = effective(&inputs.input, inputs.dtype);
    let eff_weights: Vec<Vec<f32>> = inputs.weights.iter().map(|w| effective(w, inputs.dtype)).collect();
    let raw = inputs.input.iter().chain(inputs.weights.iter().flatten());
    if raw.chain(eff_input.iter()).chain(eff_weights.iter().flatten()).any(|v| !v.is_finite()) {
        return fail("the public input and weights must be finite");
    }
    let last = proof.specs.last().unwrap();
    // (not `v.abs()`: i32::MIN.abs() overflows, and i32::MIN is -1 in M31)
    if proof.output.len() != last.m * last.n || proof.output.iter().any(|v| !(-spec::QMAX..=spec::QMAX).contains(v)) {
        return fail("the public output has the wrong size or values out of range");
    }
    let kinds = layout(&proof.specs);
    check_lookup_budget(&kinds).map_err(ProveError::Verification)?;
    if proof.claimed_sums.len() != kinds.len() {
        return Err(ProveError::Decode("one claimed sum per component is expected".into()));
    }
    if !proof.claimed_sums.iter().all(canonical) {
        return Err(ProveError::Decode("a claimed sum is a field element that is not reduced (not below the modulus)".into()));
    }
    let max_log = max_log_size(&kinds);
    let config = pcs_config(max_log);
    let shape_comps = components(&kinds, &Lookups::dummy(), &proof.claimed_sums);
    let sizes = TreeVec::concat_cols(shape_comps.iter().map(|c| c.trace_log_degree_bounds()));
    // The size of a proof is bounded by its statement: nothing larger is parsed (the verifier's memory must not depend on the proof file).
    let limit = max_proof_bytes(sizes.iter().map(|t| t.len()).sum(), max_log);
    if proof.bytes.len() > limit {
        return Err(ProveError::Decode(format!("the proof has {} bytes, more than a proof of this statement can have ({limit})", proof.bytes.len())));
    }
    let stark: StarkProof<Blake2sMerkleHasher> = serde_json::from_slice(&proof.bytes).map_err(|e| ProveError::Decode(e.to_string()))?;
    // One encoding per proof: no unknown fields, no other spelling of the same values (the bytes are exactly what the prover's serializer writes).
    if serde_json::to_vec(&stark).map_err(|e| ProveError::Decode(e.to_string()))? != proof.bytes {
        return Err(ProveError::Decode("the proof is not in its canonical encoding".into()));
    }
    // The parameters written into the proof are not used (the verifier's own are), but a proof that states others is refused rather than misread.
    if stark.config != config {
        return Err(ProveError::Decode("the proof states other commitment parameters than the verifier's".into()));
    }
    check_last_layer(&proof.bytes).map_err(ProveError::Decode)?;

    // ---- the structure of the stwo proof, so that stwo never meets a shape it would assert on ----
    check_structure(&stark, &shape_comps, &sizes, kinds.iter().map(|k| k.pre_ids().len()).sum(), max_log).map_err(ProveError::Verification)?;

    // ---- the preprocessed commitment must be the one of the statement's constants (index columns, multiplicities, the tables) ----
    {
        let twiddles = twiddles(max_log);
        let mut scheme = CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);
        let mut tree = scheme.tree_builder();
        tree.extend_evals(preprocessed_evals(&kinds));
        tree.commit(&mut Blake2sChannel::default());
        if scheme.roots()[0] != stark.commitments[0] {
            return fail("the preprocessed commitment is not the one of this statement");
        }
    }

    let channel = &mut Blake2sChannel::default();
    mix_statement(channel, &proof.specs, &proof.output, &eff_input, &eff_weights);
    let scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
    scheme.commit(stark.commitments[0], &sizes[0], channel);
    scheme.commit(stark.commitments[1], &sizes[1], channel);
    let lookups = Lookups::draw(channel);
    // Every lookup is answered: the LogUp sums of all components and of the public part (the output, the largest-accumulator rows) cancel.
    let total: SecureField = proof.claimed_sums.iter().copied().sum::<SecureField>() + public_sum(&proof.specs, &proof.output, inputs, &lookups);
    if total != SecureField::zero() {
        return fail("the lookups do not cancel: the tensors do not match the matmuls, the requantizations, the quantization of the public floats or the public output");
    }
    channel.mix_felts(&proof.claimed_sums);
    scheme.commit(stark.commitments[2], &sizes[2], channel);

    let comps = components(&kinds, &lookups, &proof.claimed_sums);
    let refs = comps.iter().map(|c| c as &dyn Component).collect_vec();
    // The number of values per queried column must be the number of distinct query positions, which only the transcript gives: replay it on a copy.
    let (pre_queries, queries) = query_counts(&stark, &sizes, config, max_log, channel.clone()).map_err(ProveError::Verification)?;
    for (tree, columns) in stark.queried_values.iter().enumerate() {
        let expected = if tree == 0 { pre_queries } else { queries };
        if columns.iter().any(|c| c.len() != expected) {
            return fail("the proof does not have one queried value per query position");
        }
    }
    // Defense in depth: the structure was checked above, but a panic inside stwo on attacker data must be a rejection, not a crash.
    match catch_unwind(AssertUnwindSafe(|| stwo_verify(&refs, channel, scheme, stark))) {
        Ok(result) => result.map_err(|e| ProveError::Verification(format!("{e:?}"))),
        Err(_) => fail("the proof is malformed (the stwo verifier rejected its structure)"),
    }
}

/// A field element of a proof must be given by its canonical representative: stwo deserializes M31 as a raw `u32` and its arithmetic assumes values below
/// the modulus (an addition of two raw values would otherwise wrap modulo 2^32, which is not field arithmetic). Every field element of an untrusted proof is
/// therefore checked before anything is computed with it.
fn canonical(v: &SecureField) -> bool {
    v.to_m31_array().iter().all(|c| c.0 < stwo::core::fields::m31::P)
}

/// The last FRI layer's polynomial: its `log_size` is a field of the proof that the transcript does not cover (only the coefficients are hashed) and that
/// stwo uses as a loop bound and a shift amount. Its fields are private, so it is checked on the JSON: exactly `2^LOG_LAST_LAYER_DEGREE_BOUND` coefficients
/// and the matching `log_size`.
fn check_last_layer(bytes: &[u8]) -> Result<(), String> {
    let json: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    let poly = &json["fri_proof"]["last_layer_poly"];
    let coeffs = poly["coeffs"].as_array().map(|c| c.len());
    if coeffs != Some(1 << LOG_LAST_LAYER_DEGREE_BOUND) || poly["log_size"].as_u64() != Some(LOG_LAST_LAYER_DEGREE_BOUND as u64) {
        return Err("the last FRI layer of the proof does not have the size of the parameters".into());
    }
    Ok(())
}

/// An upper bound of the size of the serialized stwo proof of a statement with `columns` committed columns and traces of at most `2^max_log` rows:
/// every number is counted with its longest decimal form and separator.
fn max_proof_bytes(columns: usize, max_log: u32) -> usize {
    let height = (max_log + LOG_BLOWUP_FACTOR + 2) as usize;
    let columns = columns + 8; // + the composition polynomial
    let field_elements = columns * 2 * 4 + columns * N_QUERIES + (height + 2) * N_QUERIES * 2 * 4 + 16;
    let hashes = (4 + height + 2) * N_QUERIES * (height + 1) + 64;
    field_elements * 14 + hashes * 32 * 4 + 16 * 1024
}

/// The number of distinct query positions of the preprocessed tree and of the other trees. They follow from the transcript only after stwo has absorbed
/// the composition commitment, the sampled values and the FRI layers, so this replays those steps of stwo's verifier (`verify_ex`, `verify_values`) on a
/// copy of the channel, with a second commitment-scheme verifier that only carries the trees' sizes. stwo then does the real work on the original.
fn query_counts(stark: &StarkProof<Blake2sMerkleHasher>, sizes: &TreeVec<Vec<u32>>, config: PcsConfig, max_log: u32, mut channel: Blake2sChannel) -> Result<(usize, usize), String> {
    use stwo::core::circle::CirclePoint;
    use stwo::core::fri::{CirclePolyDegreeBound, FriVerifier};
    use stwo::core::pcs::utils::prepare_preprocessed_query_positions;
    use stwo::core::proof_of_work::OODS_POW_BITS;
    let mut scheme = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
    let scratch = &mut Blake2sChannel::default();
    for tree in 0..3 {
        scheme.commit(stark.commitments[tree], &sizes[tree], scratch);
    }
    let _random_coeff = channel.draw_secure_felt();
    scheme.commit(stark.commitments[3], &[max_log; 8], &mut channel);
    if !channel.verify_pow_nonce(OODS_POW_BITS, stark.oods_proof_of_work) {
        return Err("the proof of work before the out-of-domain point is wrong".into());
    }
    channel.mix_u64(stark.oods_proof_of_work);
    let _oods_point = CirclePoint::<SecureField>::get_random_point(&mut channel);
    channel.mix_felts(&stark.sampled_values.clone().flatten_cols());
    let _random_coeff = channel.draw_secure_felt();
    let lifting_log_size = scheme.trees.last().unwrap().height;
    let bound = CirclePolyDegreeBound::new(lifting_log_size - config.fri_params.log_blowup_factor);
    let mut fri = FriVerifier::<Blake2sMerkleChannel>::commit(&mut channel, config.fri_params, stark.fri_proof.clone(), bound).map_err(|e| format!("{e:?}"))?;
    fri.verify_proof_of_work(&mut channel).map_err(|e| format!("{e:?}"))?;
    let positions = fri.sample_query_positions(&mut channel);
    let pre_positions = prepare_preprocessed_query_positions(&positions, lifting_log_size, scheme.trees[0].height);
    Ok((pre_positions.len(), positions.len()))
}

/// Checks the shapes of a stwo proof against the statement before stwo reads it: the number of trees, the sampled values per column (the number of
/// mask points of each column; a mismatch makes the constraint framework panic during an unwinding, which aborts the process), the decommitments and
/// the queried values per tree and column.
fn check_structure(stark: &StarkProof<Blake2sMerkleHasher>, comps: &[NetComponent], sizes: &TreeVec<Vec<u32>>, n_preprocessed: usize, max_log: u32) -> Result<(), String> {
    use stwo::core::air::Components;
    use stwo::core::circle::CirclePoint;
    const COMPOSITION_COLUMNS: usize = 8; // 2 * SECURE_EXTENSION_DEGREE
    let p = &stark.0;
    // every field element is canonical (see `canonical`)
    let base_ok = p.queried_values.iter().flatten().flatten().all(|v| v.0 < stwo::core::fields::m31::P);
    let fri = &p.fri_proof;
    let secure_ok = p
        .sampled_values
        .iter()
        .flatten()
        .flatten()
        .chain(&fri.first_layer.fri_witness)
        .chain(fri.inner_layers.iter().flat_map(|l| &l.fri_witness))
        .chain(fri.last_layer_poly.clone().into_ordered_coefficients().iter())
        .all(canonical);
    if !base_ok || !secure_ok {
        return Err("the proof contains a field element that is not reduced (not below the modulus)".into());
    }
    if p.commitments.len() != 4 || p.sampled_values.len() != 4 || p.decommitments.len() != 4 || p.queried_values.len() != 4 {
        return Err("the proof does not have the four trees of this statement".into());
    }
    let refs = comps.iter().map(|c| c as &dyn Component).collect_vec();
    // The number of points per column does not depend on the point itself.
    let masks = Components { components: refs, n_preprocessed_columns: n_preprocessed }.mask_points(CirclePoint::<SecureField>::zero(), max_log + 1, false);
    for tree in 0..3 {
        let expected: Vec<usize> = masks[tree].iter().map(|points| points.len()).collect();
        let found: Vec<usize> = p.sampled_values[tree].iter().map(|values| values.len()).collect();
        if expected != found || sizes[tree].len() != p.queried_values[tree].len() {
            return Err(format!("tree {tree} of the proof does not have the columns or samples of this statement"));
        }
    }
    if p.sampled_values[3].len() != COMPOSITION_COLUMNS || p.sampled_values[3].iter().any(|v| v.len() != 1) || p.queried_values[3].len() != COMPOSITION_COLUMNS {
        return Err("the composition tree of the proof has the wrong shape".into());
    }
    // every column of a tree is queried at the same positions
    for tree in 0..4 {
        let lens: Vec<usize> = p.queried_values[tree].iter().map(|c| c.len()).collect();
        if lens.windows(2).any(|w| w[0] != w[1]) {
            return Err(format!("the queried values of tree {tree} are ragged"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Dtype;
    use crate::witness::tests::{floats, network};

    fn check(original: &NetworkWitness) -> Proof {
        let proof = prove_witness(original).unwrap();
        assert_eq!(proof.output, original.output());
        verify(&proof, &original.public_inputs()).unwrap();
        proof
    }

    /// The float network on the CPU: `relu?(x @ w)` per layer.
    fn float_reference(w: &NetworkWitness) -> Vec<f32> {
        let mut x = w.x0_float.clone();
        for (l, s) in w.specs.iter().enumerate() {
            let mut y = vec![0f32; s.m * s.n];
            for i in 0..s.m {
                for k in 0..s.k {
                    for j in 0..s.n {
                        y[i * s.n + j] += x[i * s.k + k] * w.w_float[l][k * s.n + j];
                    }
                }
            }
            if s.relu {
                y.iter_mut().for_each(|v| *v = v.max(0.0));
            }
            x = y;
        }
        x
    }

    #[test]
    fn the_dequantized_output_approximates_the_float_network() {
        for (m, dims, relu, seed) in [(8, vec![32, 64, 16], vec![true, false], 3), (8, vec![64, 32], vec![false], 5), (4, vec![16, 32, 32, 8], vec![true, true, false], 7)] {
            let original = network(m, &dims, &relu, seed);
            let proof = prove_witness(&original).unwrap();
            let got = proof.dequantized_output();
            let want = float_reference(&original);
            let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
            let worst = got.iter().zip(&want).fold(0f32, |a, (g, w)| a.max((g - w).abs()));
            assert!(worst <= 0.1 * scale, "dims {dims:?}: the worst error {worst} is more than 10% of the largest output {scale}");
        }
    }

    #[test]
    fn prove_and_verify_one_layer() {
        check(&network(4, &[8, 8], &[false], 1));
    }

    #[test]
    fn prove_and_verify_relu_networks() {
        for (m, dims, relu) in [(4, vec![8, 16, 8], vec![true, false]), (8, vec![32, 16, 32, 8], vec![true, true, false]), (3, vec![5, 7, 3], vec![false, true])] {
            check(&network(m, &dims, &relu, 3));
        }
    }

    #[test]
    fn prove_and_verify_a_bf16_network() {
        let x = floats(4 * 8, 1, -2.0, 2.0);
        let w = vec![floats(8 * 16, 2, -1.0, 1.0), floats(16 * 4, 3, -1.0, 1.0)];
        check(&NetworkWitness::simulate(4, &[8, 16, 4], &[true, false], &x, &w, Dtype::Bf16).unwrap());
    }

    #[test]
    fn small_requantization_and_zero_modes_verify() {
        // x = 1.0 (q 64), w = [1.0, -0.984375] (q 64, -63): acc = 64, the requantization is a left shift
        let original = NetworkWitness::simulate(1, &[2, 1], &[false], &[1.0, 1.0], &[vec![1.0, -0.984375]], Dtype::F32).unwrap();
        assert_eq!(original.acc[0], vec![64]);
        assert!(original.specs[0].ko >= 0 && original.specs[0].ko < 41);
        check(&original);
        // the ReLU of negatives
        check(&NetworkWitness::simulate(2, &[2, 2], &[true], &[1.0, 1.0, 1.0, 1.0], &[vec![-1.0, -1.0, -1.0, -1.0]], Dtype::F32).unwrap());
        // all-zero input and all-zero weights
        let zero_x = NetworkWitness::simulate(2, &[2, 2], &[false], &[0.0; 4], &[vec![0.5, 0.25, 0.125, 1.0]], Dtype::F32).unwrap();
        assert_eq!((zero_x.specs[0].kx, zero_x.specs[0].ko), (127, 127));
        check(&zero_x);
        let zero_w = NetworkWitness::simulate(2, &[2, 2], &[false], &[0.5, 1.5, -2.0, 0.25], &[vec![0.0; 4]], Dtype::F32).unwrap();
        assert_eq!(zero_w.specs[0].kw, 127);
        check(&zero_w);
    }

    #[test]
    fn extremes_and_ties_verify() {
        // the largest value is 127, so the scaler is 1 and the others are quantized with ties: 2.5 -> 2, -3.5 -> -4, 126.5 -> 126, 0.5 -> 0
        let tied = NetworkWitness::simulate(2, &[4, 2], &[false], &[127.0, 2.5, -3.5, 0.5, -127.0, 126.5, 1.5, -0.5], &[vec![127.0, 2.5, -3.5, 0.5, 1.5, -126.5, 5.5, -127.0]], Dtype::F32).unwrap();
        assert_eq!(&tied.q_x0, &[127, 2, -4, 0, -127, 126, 2, 0]);
        check(&tied);
        // tiny floats (a large shift) next to large ones, subnormals and zeros
        let wide = NetworkWitness::simulate(2, &[4, 2], &[false], &[1.0, 1e-30, 0.0, 1e-45, -1.0, 2e-20, 0.25, -1e-10], &[vec![1.0, -1e-38, 0.5, 0.0, 3e-5, 1e-3, -1.0, 1e-45]], Dtype::F32).unwrap();
        check(&wide);
        for seed in 0..10 {
            check(&network(2, &[4, 4, 2], &[seed % 2 == 0, true], seed));
        }
    }

    #[test]
    fn a_tampered_proof_does_not_verify() {
        let original = network(4, &[8, 8], &[false], 1);
        let mut forged = prove_witness(&original).unwrap();
        let mid = forged.bytes.len() / 2;
        forged.bytes[mid] = if forged.bytes[mid] == b'1' { b'2' } else { b'1' };
        assert!(verify(&forged, &original.public_inputs()).is_err());
    }

    #[test]
    fn a_wrong_public_output_does_not_verify() {
        let original = network(4, &[8, 8], &[false], 1);
        let mut forged = prove_witness(&original).unwrap();
        forged.output[0] += 1;
        assert!(verify(&forged, &original.public_inputs()).is_err());
    }

    #[test]
    fn a_wrong_statement_does_not_verify() {
        let original = network(4, &[8, 8], &[false], 1);
        let public = original.public_inputs();
        let proof = prove_witness(&original).unwrap();
        for edit in [
            |s: &mut LayerSpec| s.ko -= 1,
            |s: &mut LayerSpec| s.relu = true,
            |s: &mut LayerSpec| s.k = 16,
            |s: &mut LayerSpec| s.kw -= 1,
            |s: &mut LayerSpec| s.kx += 1,
        ] {
            let mut forged = proof.clone();
            edit(&mut forged.specs[0]);
            assert!(verify(&forged, &public).is_err());
        }
        let mut forged = proof;
        forged.claimed_sums[0] += SecureField::from_u32_unchecked(1, 0, 0, 0);
        assert!(verify(&forged, &public).is_err());
    }

    #[test]
    fn a_proof_does_not_verify_for_other_public_floats() {
        let original = network(4, &[8, 16, 8], &[true, false], 6);
        let proof = prove_witness(&original).unwrap();
        let public = original.public_inputs();
        verify(&proof, &public).unwrap();
        // another input
        let mut forged = public.clone();
        forged.input[0] += 0.001;
        assert!(verify(&proof, &forged).is_err());
        // another weight, even by one ulp, in either layer
        for (l, i) in [(0, 0), (0, 77), (1, 5)] {
            let mut forged = public.clone();
            forged.weights[l][i] = f32::from_bits(forged.weights[l][i].to_bits() + 1);
            assert!(verify(&proof, &forged).is_err(), "weight {l}/{i}");
        }
        // swapped weights, wrong shapes, not finite
        let mut forged = public.clone();
        forged.weights.swap(0, 1);
        assert!(verify(&proof, &forged).is_err());
        let mut forged = public.clone();
        forged.input.pop();
        assert!(verify(&proof, &forged).is_err());
        let mut forged = public;
        forged.weights[1][0] = f32::INFINITY;
        assert!(verify(&proof, &forged).is_err());
    }

    /// A forged witness is refused by the consistency checks, and if it is forced past them stwo's prover refuses it or the verifier rejects it.
    fn refuse(forged: &NetworkWitness, public: &PublicInputs) {
        assert!(matches!(prove_witness(forged), Err(ProveError::Witness(_))));
        match prove_impl(forged, false) {
            Err(_) => {}
            Ok(proof) => assert!(verify(&proof, public).is_err(), "the verifier accepted a forged witness"),
        }
    }

    #[test]
    fn reject_forged_witnesses() {
        let original = network(4, &[8, 16, 8], &[true, false], 2);
        let public = original.public_inputs();

        let mut forged = original.clone();
        forged.acc[0][5] += 1;
        refuse(&forged, &public);

        let mut forged = original.clone();
        forged.q_w[0][3] = 200; // outside the range, with a consistent accumulator
        forged.acc[0] = crate::witness::matmul(forged.specs[0].m, forged.specs[0].k, forged.specs[0].n, &forged.q_x0, &forged.q_w[0]);
        refuse(&forged, &public);

        let mut forged = original.clone();
        forged.out[0][1] = forged.out[0][1].wrapping_add(1); // not the requantization of acc
        refuse(&forged, &public);

        let mut forged = original.clone();
        forged.specs[1].ko -= 1; // an exponent that is not the largest
        refuse(&forged, &public);

        // a quantized weight that is not the quantization of its float, with a consistent accumulator
        let mut forged = original.clone();
        forged.q_w[1][4] += 1;
        forged.acc[1] = crate::witness::matmul(forged.specs[1].m, forged.specs[1].k, forged.specs[1].n, forged.x(1), &forged.q_w[1]);
        refuse(&forged, &public);

        // the same for the input
        let mut forged = original.clone();
        forged.q_x0[2] -= 1;
        refuse(&forged, &public);

        // a scaler that is not the largest, and one that is too large
        let mut forged = original.clone();
        forged.specs.iter_mut().for_each(|s| s.kx -= 1);
        refuse(&forged, &public);
        let mut forged = original.clone();
        forged.specs[0].kw += 1;
        refuse(&forged, &public);

        // a float that does not match its quantization
        let mut forged = original.clone();
        forged.w_float[0][5] = -forged.w_float[0][5];
        refuse(&forged, &public);
    }

    #[test]
    fn a_mismatched_tensor_is_not_provable() {
        // the matmul rows must read the committed weights: other weights than the ones the accumulators come from
        let original = network(4, &[8, 8, 8], &[false, false], 4);
        let mut forged = original.clone();
        forged.q_w[1][0] += 1;
        refuse(&forged, &original.public_inputs());
    }

    #[test]
    fn a_proof_of_one_model_is_not_a_proof_of_another() {
        let first = network(4, &[8, 8], &[false], 1);
        let second = network(4, &[8, 8], &[false], 2);
        let proof = prove_witness(&first).unwrap();
        assert!(verify(&proof, &second.public_inputs()).is_err());
    }
}
