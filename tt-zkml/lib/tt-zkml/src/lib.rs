//! tt-zkml: zero-knowledge proofs of quantized neural-network inference executed on Tenstorrent hardware.
//!
//! * [`device`]: safe API over the C ABI: open a [`device::Device`], create a [`device::Model`], load its parameters, run the forward pass and get the
//!   [`device::Trace`].
//! * [`modelfile`]: the on-disk format of a model (a directory with `model.json` and the weights) and of an input.
//! * [`witness`]: the CPU rebuilds the int8 tensors of a forward pass from the device trace, the model and the input.
//! * [`air`]: the constraints of the execution over M31: matmul, committed tensors, requantization + ReLU, range checks, linked by LogUp.
//! * [`proof`]: proof generation from the device [`Trace`] and verification with stwo.

pub mod air;
pub mod device;
pub mod modelfile;
pub mod proof;
pub mod witness;

pub use device::{Device, Dtype, Error, Layer, LayerTrace, Model, Trace};
