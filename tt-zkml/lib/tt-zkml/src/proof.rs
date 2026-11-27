//! Proof generation and verification with stwo.
//!
//! [`prove`] takes the model, the float input and the device [`Trace`], rebuilds the witness on the CPU ([`NetworkWitness::from_trace`]), converts it to
//! M31 columns and proves the constraints of [`crate::air`]. [`verify`] checks a [`Proof`] against its public statement (the layer specs and the output).

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
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

use crate::air::spec::{self, LayerSpec};
use crate::air::{Kind, Lookups, NetEval, build, interaction_traces, layout, public_sum};
use crate::device::Trace;
use crate::modelfile::ModelFile;
use crate::witness::{NetworkWitness, PublicInputs, WitnessError};

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

const LOG_BLOWUP_FACTOR: u32 = 1;

/// The commitment parameters for traces of at most `2^max_log_size` rows. 70 queries at blowup 2 plus 10 bits of proof of work: about 80 bits.
fn pcs_config(max_log_size: u32) -> PcsConfig {
    PcsConfig {
        fri_params: FriParams::new(10, 0, LOG_BLOWUP_FACTOR, 70, 1),
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
    pub fn dequantized_output(&self) -> Vec<f32> {
        let exponent: i64 = self.specs[0].kx as i64 + self.specs.iter().map(|s| s.kw as i64 + s.ko as i64).sum::<i64>();
        self.output.iter().map(|&v| (v as f64 / 2f64.powi(exponent as i32)) as f32).collect()
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

/// `checked = false` skips the consistency checks of the witness: for the tests that a forged witness cannot be proven or verified.
pub(crate) fn prove_impl(witness: &NetworkWitness, checked: bool) -> Result<Proof, ProveError> {
    let trace = build(witness, checked)?;
    let kinds = trace.components.iter().map(|c| c.kind.clone()).collect_vec();
    let max_log = max_log_size(&kinds);
    let config = pcs_config(max_log);
    let twiddles = twiddles(max_log);

    let channel = &mut Blake2sChannel::default();
    let mut scheme = CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    let mut tree = scheme.tree_builder();
    tree.extend_evals(preprocessed_evals(&kinds));
    tree.commit(channel);

    let mut tree = scheme.tree_builder();
    tree.extend_evals(trace.components.iter().flat_map(|c| c.base_evals()).collect_vec());
    tree.commit(channel);

    let lookups = Lookups::draw(channel);
    let (interaction, claimed_sums) = interaction_traces(&trace, &lookups);
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
    spec::validate(&proof.specs).map_err(|e| ProveError::Verification(e.to_string()))?;
    let first = &proof.specs[0];
    if inputs.input.len() != first.m * first.k
        || inputs.weights.len() != proof.specs.len()
        || inputs.weights.iter().zip(&proof.specs).any(|(w, s)| w.len() != s.k * s.n)
        || inputs.input.iter().chain(inputs.weights.iter().flatten()).any(|v| !v.is_finite())
    {
        return fail("the public input and weights do not have the shapes of the statement");
    }
    let last = proof.specs.last().unwrap();
    if proof.output.len() != last.m * last.n || proof.output.iter().any(|v| v.abs() > spec::QMAX) {
        return fail("the public output has the wrong size or values out of range");
    }
    let kinds = layout(&proof.specs);
    if proof.claimed_sums.len() != kinds.len() {
        return Err(ProveError::Decode("one claimed sum per component is expected".into()));
    }
    let stark: StarkProof<Blake2sMerkleHasher> = serde_json::from_slice(&proof.bytes).map_err(|e| ProveError::Decode(e.to_string()))?;
    let max_log = max_log_size(&kinds);
    let config = pcs_config(max_log);

    // The preprocessed commitment must be the one of the statement's constants (index columns, multiplicities, the range table).
    {
        let twiddles = twiddles(max_log);
        let mut scheme = CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);
        let mut tree = scheme.tree_builder();
        tree.extend_evals(preprocessed_evals(&kinds));
        tree.commit(&mut Blake2sChannel::default());
        if stark.commitments.is_empty() || scheme.roots()[0] != stark.commitments[0] {
            return fail("the preprocessed commitment is not the one of this statement");
        }
    }

    let channel = &mut Blake2sChannel::default();
    let scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
    let sizes = TreeVec::concat_cols(components(&kinds, &Lookups::dummy(), &proof.claimed_sums).iter().map(|c| c.trace_log_degree_bounds()));
    // Refuse a proof with other column counts: stwo would panic on it.
    let shaped = stark.commitments.len() == 4 // + the composition polynomial
        && stark.sampled_values.len() == 4
        && (0..3).all(|tree| stark.sampled_values[tree].len() == sizes[tree].len());
    if !shaped {
        return fail("the proof does not have the columns of this statement");
    }
    scheme.commit(stark.commitments[0], &sizes[0], channel);
    scheme.commit(stark.commitments[1], &sizes[1], channel);
    let lookups = Lookups::draw(channel);
    // Every lookup is answered: the LogUp sums of all components and of the public part (the output, the largest-accumulator rows) cancel.
    let total: SecureField = proof.claimed_sums.iter().copied().sum::<SecureField>() + public_sum(&proof.specs, &proof.output, inputs, &lookups);
    if total != SecureField::zero() {
        return fail("the lookups do not cancel: the tensors do not match the matmuls, the requantizations, the quantization of the public floats or the public output");
    }
    scheme.commit(stark.commitments[2], &sizes[2], channel);

    let comps = components(&kinds, &lookups, &proof.claimed_sums);
    let refs = comps.iter().map(|c| c as &dyn Component).collect_vec();
    stwo_verify(&refs, channel, scheme, stark).map_err(|e| ProveError::Verification(format!("{e:?}")))
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
