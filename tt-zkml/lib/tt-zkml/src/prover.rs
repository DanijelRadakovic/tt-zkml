//! Proof generation and verification with stwo.
//!
//! Not implemented yet: it will take the [`crate::Trace`], build the trace columns over M31 for the components of [`crate::air`] and call the stwo
//! prover.

use crate::device::Trace;

/// A proof of one forward pass (to be defined with the prover).
#[derive(Debug, Clone)]
pub struct Proof {
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProveError {
    /// The constraints of the execution trace are the next step of the project.
    NotImplemented,
}

impl std::fmt::Display for ProveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProveError::NotImplemented => write!(f, "proof generation is not implemented yet: the AIR constraints of the trace are the next step"),
        }
    }
}

impl std::error::Error for ProveError {}

/// Generates the proof that `trace` is the execution of the quantized network.
pub fn prove(_trace: &Trace) -> Result<Proof, ProveError> {
    Err(ProveError::NotImplemented)
}

/// Verifies a proof.
pub fn verify(_proof: &Proof) -> Result<(), ProveError> {
    Err(ProveError::NotImplemented)
}
