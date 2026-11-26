//! tt-zkml: zero-knowledge proofs of quantized neural-network inference executed on Tenstorrent hardware.
//!
//! * [`device`]: safe API over the C ABI: open a [`device::Device`], create a [`device::Model`], load its parameters, run the forward pass and get the
//!   [`device::Trace`].
//! * [`modelfile`]: the on-disk format of a model (a directory with `model.json` and the weights) and of an input.
//! * [`air`]: the constraints of the execution trace over M31 (the int8 matmul of a layer so far).
//! * [`prover`]: proof generation and verification with stwo ([`prover::prove_layers`], [`prover::verify`]); from the device [`Trace`] not yet.

pub mod air;
pub mod device;
pub mod modelfile;
pub mod prover;

pub use device::{Device, Dtype, Error, Layer, LayerTrace, Model, Trace};
