//! The witness of a forward pass: the floats, the int8 tensors and the int32 accumulators of every layer, reconstructed on the CPU.
//!
//! The device returns, per layer, the accumulator and the power-of-two scalers; the quantized input `q_x0` and weights `q_w` follow from the float
//! input / weights and `scaler_x` / `scaler_w` (`round_half_even(v * scaler)`), and the quantized input of every next layer from the previous
//! accumulator (the requantization of [`crate::air::spec`]). So the device does not export `q_x` / `q_w`; the CPU rebuilds them, and
//! [`NetworkWitness::check`] verifies that they reproduce the device's accumulators and output.
//!
//! The floats are the values the device quantizes: for a bf16 model the f32 values rounded to bf16. In the proof they are public inputs
//! ([`PublicInputs`]): the verifier supplies them, and the constraints tie every quantized value to its float.

use crate::air::spec::{self, LayerSpec, QMAX, SpecError, activate, choose_ko, exponent_of, pot_exponent};
use crate::device::{Dtype, Layer, Trace};
use crate::modelfile::ModelFile;
use crate::proof::{ModelCommitment, ProveError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitnessError {
    Spec(SpecError),
    Size { what: String, expected: usize, found: usize },
    Range { what: String, index: usize, value: i32 },
    /// `acc` is not `q_x @ q_w`.
    Accumulator { layer: usize, index: usize, expected: i32, found: i32 },
    /// `out` is not the requantization of `acc`.
    Output { layer: usize, index: usize, expected: i32, found: i32 },
    /// The requantization exponent is not the one for these accumulators.
    Exponent { layer: usize, expected: i32, found: i32 },
    /// No accumulator is large enough to justify the exponent.
    NoLargestAccumulator { layer: usize },
    /// A row of the requantization does not satisfy its constraints.
    Requant { row: usize, what: String },
    /// A quantized value is not the quantization of its float with the declared exponent.
    Quantization { what: String, index: usize, expected: i32, found: i32 },
    /// The quantization exponent is not the one for these floats.
    ScalerExponent { what: String, expected: i32, found: i32 },
    /// A float cannot be represented by the constraints (not finite, or its scaled magnitude is out of the supported range).
    Float { what: String, index: usize },
    /// The trace has something this version cannot handle (a scaler that is not a power of two, ...).
    Trace(String),
}

impl std::fmt::Display for WitnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WitnessError::Spec(e) => write!(f, "{e}"),
            WitnessError::Size { what, expected, found } => write!(f, "{what} has {found} elements, expected {expected}"),
            WitnessError::Range { what, index, value } => write!(f, "{what}[{index}] = {value} is outside [-127, 127]"),
            WitnessError::Accumulator { layer, index, expected, found } => {
                write!(f, "layer {layer}: accumulator[{index}] = {found}, but q_x * q_w = {expected}")
            }
            WitnessError::Output { layer, index, expected, found } => write!(f, "layer {layer}: output[{index}] = {found}, but the requantization gives {expected}"),
            WitnessError::Exponent { layer, expected, found } => write!(f, "layer {layer}: the requantization exponent is {found}, the largest that fits is {expected}"),
            WitnessError::NoLargestAccumulator { layer } => write!(f, "layer {layer}: no accumulator justifies the requantization exponent"),
            WitnessError::Quantization { what, index, expected, found } => write!(f, "{what}[{index}] = {found}, but the quantization of its float is {expected}"),
            WitnessError::ScalerExponent { what, expected, found } => write!(f, "{what}: the quantization exponent is {found}, the largest that fits is {expected}"),
            WitnessError::Float { what, index } => write!(f, "{what}[{index}]: this float cannot be handled (not finite, or its quantization shift is out of range)"),
            WitnessError::Requant { row, what } => write!(f, "requantization row {row}: {what}"),
            WitnessError::Trace(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for WitnessError {}

impl From<SpecError> for WitnessError {
    fn from(e: SpecError) -> Self {
        WitnessError::Spec(e)
    }
}

/// The values of every layer of a forward pass.
#[derive(Debug, Clone)]
pub struct NetworkWitness {
    pub specs: Vec<LayerSpec>,
    /// The float input as the device sees it (bf16 rounded for a bf16 model), `[m, k_0]`.
    pub x0_float: Vec<f32>,
    /// The float weights as the device sees them, `[k, n]` each.
    pub w_float: Vec<Vec<f32>>,
    /// The quantized input, `[m, k_0]`.
    pub q_x0: Vec<i32>,
    /// The quantized weights of each layer, `[k, n]`.
    pub q_w: Vec<Vec<i32>>,
    /// The accumulators of each layer, `[m, n]`.
    pub acc: Vec<Vec<i32>>,
    /// The output of each layer (`relu?(requant(acc))`), `[m, n]`: the quantized input of the next layer; the last one is the output of the network.
    pub out: Vec<Vec<i32>>,
}

/// The reference matmul `[m, k] @ [k, n]` of int8 values in int32.
pub fn matmul(m: usize, k: usize, n: usize, x: &[i32], w: &[i32]) -> Vec<i32> {
    let mut acc = vec![0i32; m * n];
    for i in 0..m {
        for kk in 0..k {
            let xv = x[i * k + kk];
            for j in 0..n {
                acc[i * n + j] = acc[i * n + j].wrapping_add(xv.wrapping_mul(w[kk * n + j])); // exact for int8 inputs; wraps (never panics) for forged values
            }
        }
    }
    acc
}

/// `value` rounded to bf16 (round half to even), as a float.
pub fn round_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) & 0xffff_0000)
}

/// The PoT quantization of one float: `round_half_even(value * scaler)`; bf16 devices round the float to bf16 first.
pub fn quantize(value: f32, scaler: f32, dtype: Dtype) -> i32 {
    let v = if dtype == Dtype::Bf16 { round_bf16(value) } else { value };
    (v * scaler).round_ties_even() as i32
}

/// The value of a float tensor as the device sees it.
pub fn effective(values: &[f32], dtype: Dtype) -> Vec<f32> {
    values.iter().map(|&v| if dtype == Dtype::Bf16 { round_bf16(v) } else { v }).collect()
}

/// A finite float as `(-1)^s * m * 2^(ep - 150)`: `m` is the 24-bit significand (0 for zero), `ep` the biased exponent (at least 1: subnormals and zero use 1).
pub fn float_parts(v: f32) -> (u32, u32, u32) {
    let bits = v.to_bits();
    let exp = (bits >> 23) & 0xff;
    let frac = bits & 0x007f_ffff;
    let m = if exp == 0 { frac } else { frac | 0x0080_0000 };
    (bits >> 31, m, exp.max(1))
}

/// The quantization shift of a float: `v * 2^k = m * 2^-sh` with `sh = 150 - ep - k`.
pub fn shift_of(ep: u32, k: i32) -> i32 {
    150 - ep as i32 - k
}

/// What a verifier knows: the model as a commitment (its architecture and the root of its weights, not the weights themselves) and the float input.
/// The proof is about exactly this committed model on exactly this input; nothing about the network's structure is taken from the proof.
#[derive(Debug, Clone)]
pub struct PublicInputs {
    pub model: ModelCommitment,
    pub input: Vec<f32>,
}

impl PublicInputs {
    /// From the model itself (computes its commitment, which needs the weights). A verifier that only holds a commitment builds the struct directly.
    pub fn new(model: &ModelFile, input: &[f32]) -> Result<Self, ProveError> {
        Ok(Self { model: ModelCommitment::new(model)?, input: input.to_vec() })
    }
}

fn absmax(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |a, v| a.max(v.abs()))
}

impl NetworkWitness {
    /// The `q_x` of layer `l`: the input for layer 0, else the output of the previous layer.
    pub fn x(&self, l: usize) -> &[i32] {
        if l == 0 { &self.q_x0 } else { &self.out[l - 1] }
    }

    /// The public inputs of this witness: the commitment of its model (the floats the device saw, already rounded to the dtype, so the dtype is
    /// irrelevant here) and its input.
    pub fn public_inputs(&self) -> PublicInputs {
        let layers = self.specs.iter().map(|s| Layer { in_features: s.k as u32, out_features: s.n as u32, relu: s.relu }).collect();
        let model = ModelCommitment::from_floats(Dtype::F32, layers, &self.w_float).expect("the weights of a witness have the shapes of its layers");
        PublicInputs { model, input: self.x0_float.clone() }
    }

    /// The output of the network (public).
    pub fn output(&self) -> &[i32] {
        self.out.last().expect("at least one layer")
    }

    /// Runs the network on the CPU the way the device does: the PoT quantization of the floats, integer layers, the exponent the device would choose.
    /// `relu[l]` says whether layer `l` has a ReLU.
    pub fn simulate(m: usize, dims: &[usize], relu: &[bool], x: &[f32], w: &[Vec<f32>], dtype: Dtype) -> Result<Self, WitnessError> {
        let x0_float = effective(x, dtype);
        let w_float: Vec<Vec<f32>> = w.iter().map(|v| effective(v, dtype)).collect();
        let kx = pot_exponent(absmax(&x0_float));
        let q_x0: Vec<i32> = x0_float.iter().map(|&v| quantize(v, 2f32.powi(kx), Dtype::F32)).collect();
        let mut specs = Vec::new();
        let (mut q_w, mut acc, mut out) = (Vec::new(), Vec::new(), Vec::new());
        let mut cur = q_x0.clone();
        for l in 0..dims.len() - 1 {
            let (k, n) = (dims[l], dims[l + 1]);
            let kw = pot_exponent(absmax(&w_float[l]));
            let qw: Vec<i32> = w_float[l].iter().map(|&v| quantize(v, 2f32.powi(kw), Dtype::F32)).collect();
            let a = matmul(m, k, n, &cur, &qw);
            let spec = LayerSpec { m, k, n, relu: relu[l], ko: choose_ko(&a), kw, kx };
            cur = a.iter().map(|&v| activate(v, &spec)).collect();
            specs.push(spec);
            q_w.push(qw);
            acc.push(a);
            out.push(cur.clone());
        }
        Ok(Self { specs, x0_float, w_float, q_x0, q_w, acc, out })
    }

    /// Rebuilds the witness of a device forward pass from the model, the float input and the device trace.
    pub fn from_trace(model: &ModelFile, input: &[f32], trace: &Trace) -> Result<Self, WitnessError> {
        let layers = &model.layers;
        if trace.layers.len() != layers.len() {
            return Err(WitnessError::Trace(format!("the trace has {} layers, the model {}", trace.layers.len(), layers.len())));
        }
        let m = trace.batch as usize;
        let k0 = layers[0].in_features as usize;
        if input.len() != m * k0 {
            return Err(WitnessError::Size { what: "input".into(), expected: m * k0, found: input.len() });
        }
        if let Some(index) = input.iter().position(|v| !v.is_finite()) {
            return Err(WitnessError::Float { what: "input".into(), index });
        }
        for (l, w) in model.weights.iter().enumerate() {
            if let Some(index) = w.iter().position(|v| !v.is_finite()) {
                return Err(WitnessError::Float { what: format!("weights[{l}]"), index });
            }
        }
        let x0_float = effective(input, model.dtype);
        let kx = exponent_of(trace.scaler_x).ok_or_else(|| WitnessError::Trace(format!("scaler_x {} is not a power of two", trace.scaler_x)))?;
        let q_x0 = x0_float.iter().map(|&v| quantize(v, trace.scaler_x, Dtype::F32)).collect();
        let mut specs = Vec::new();
        let (mut w_float, mut q_w, mut acc, mut out) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (l, (layer, lt)) in layers.iter().zip(&trace.layers).enumerate() {
            let ko = exponent_of(lt.scaler_out).ok_or_else(|| WitnessError::Trace(format!("layer {l}: scaler_out {} is not a power of two", lt.scaler_out)))?;
            let kw = exponent_of(lt.scaler_w).ok_or_else(|| WitnessError::Trace(format!("layer {l}: scaler_w {} is not a power of two", lt.scaler_w)))?;
            let spec = LayerSpec { m, k: layer.in_features as usize, n: layer.out_features as usize, relu: layer.relu, ko, kw, kx };
            let wf = effective(&model.weights[l], model.dtype);
            q_w.push(wf.iter().map(|&v| quantize(v, lt.scaler_w, Dtype::F32)).collect::<Vec<_>>());
            w_float.push(wf);
            let a = lt.acc_values();
            out.push(a.iter().map(|&v| activate(v, &spec)).collect::<Vec<_>>());
            acc.push(a);
            specs.push(spec);
        }
        let w = Self { specs, x0_float, w_float, q_x0, q_w, acc, out };
        // the CPU's requantization must reproduce the device's output
        let device_out = trace.output_values();
        if let Some(index) = device_out.iter().zip(w.output()).position(|(d, c)| d != c).or((device_out.len() != w.output().len()).then_some(0)) {
            let last = w.specs.len() - 1;
            return Err(WitnessError::Output { layer: last, index, expected: *w.output().get(index).unwrap_or(&0), found: *device_out.get(index).unwrap_or(&0) });
        }
        Ok(w)
    }

    /// The quantized values are the quantization of their floats with the exponent the device chooses for them.
    fn check_quantization(&self, what: &str, floats: &[f32], q: &[i32], k: i32) -> Result<(), WitnessError> {
        if let Some(index) = floats.iter().position(|v| !v.is_finite()) {
            return Err(WitnessError::Float { what: what.into(), index });
        }
        let expected = pot_exponent(absmax(floats));
        if expected >= 127 && floats.iter().any(|&v| v != 0.0) {
            // the exponent is capped at 127, which the constraints reserve for "all zero": a tensor of only tiny non-zero floats is not supported
            let index = floats.iter().position(|&v| v != 0.0).unwrap();
            return Err(WitnessError::Float { what: what.into(), index });
        }
        if expected != k {
            return Err(WitnessError::ScalerExponent { what: what.into(), expected, found: k });
        }
        for (index, (&v, &found)) in floats.iter().zip(q).enumerate() {
            let want = quantize(v, 2f32.powi(k.min(126)), Dtype::F32);
            let (_, m, ep) = float_parts(v);
            if k < 127 && m != 0 && shift_of(ep, k) < 1 {
                return Err(WitnessError::Float { what: what.into(), index });
            }
            if want != found {
                return Err(WitnessError::Quantization { what: what.into(), index, expected: want, found });
            }
        }
        Ok(())
    }

    /// Checks that the witness is a valid execution: shapes, ranges, `acc = q_x @ q_w`, `out = requant(acc)` and the exponents.
    pub fn check(&self) -> Result<(), WitnessError> {
        spec::validate(&self.specs)?;
        let l_count = self.specs.len();
        if self.q_w.len() != l_count || self.acc.len() != l_count || self.out.len() != l_count {
            return Err(WitnessError::Size { what: "layers".into(), expected: l_count, found: self.q_w.len().min(self.acc.len()).min(self.out.len()) });
        }
        let size = |what: String, expected: usize, found: usize| if expected == found { Ok(()) } else { Err(WitnessError::Size { what, expected, found }) };
        let range = |what: String, values: &[i32]| match values.iter().position(|v| v.abs() > QMAX) {
            Some(index) => Err(WitnessError::Range { what, index, value: values[index] }),
            None => Ok(()),
        };
        size("q_x0".into(), self.specs[0].m * self.specs[0].k, self.q_x0.len())?;
        size("x0_float".into(), self.q_x0.len(), self.x0_float.len())?;
        range("q_x0".into(), &self.q_x0)?;
        self.check_quantization("q_x0", &self.x0_float, &self.q_x0, self.specs[0].kx)?;
        for (l, s) in self.specs.iter().enumerate() {
            size(format!("q_w[{l}]"), s.k * s.n, self.q_w[l].len())?;
            size(format!("acc[{l}]"), s.m * s.n, self.acc[l].len())?;
            size(format!("out[{l}]"), s.m * s.n, self.out[l].len())?;
            range(format!("q_w[{l}]"), &self.q_w[l])?;
            size(format!("w_float[{l}]"), s.k * s.n, self.w_float[l].len())?;
            self.check_quantization(&format!("q_w[{l}]"), &self.w_float[l], &self.q_w[l], s.kw)?;
            range(format!("out[{l}]"), &self.out[l])?;
            let expected = matmul(s.m, s.k, s.n, self.x(l), &self.q_w[l]);
            if let Some(index) = expected.iter().zip(&self.acc[l]).position(|(e, a)| e != a) {
                return Err(WitnessError::Accumulator { layer: l, index, expected: expected[index], found: self.acc[l][index] });
            }
            if let Some(index) = self.acc[l].iter().zip(&self.out[l]).position(|(&a, &o)| activate(a, s) != o) {
                return Err(WitnessError::Output { layer: l, index, expected: activate(self.acc[l][index], s), found: self.out[l][index] });
            }
            let ko = choose_ko(&self.acc[l]);
            if ko != s.ko {
                return Err(WitnessError::Exponent { layer: l, expected: ko, found: s.ko });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Pseudo-random floats in `[lo, hi)`.
    pub fn floats(len: usize, seed: u32, lo: f32, hi: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                lo + ((s >> 8) as f32 / (1u32 << 24) as f32) * (hi - lo)
            })
            .collect()
    }

    /// A network `dims[0] -> dims[1] -> ...` of batch `m` with pseudo-random float input and weights.
    pub fn network(m: usize, dims: &[usize], relu: &[bool], seed: u32) -> NetworkWitness {
        let x = floats(m * dims[0], seed, -2.0, 2.0);
        let w: Vec<Vec<f32>> = (0..dims.len() - 1).map(|l| floats(dims[l] * dims[l + 1], seed + 1 + l as u32, -1.0, 1.0)).collect();
        NetworkWitness::simulate(m, dims, relu, &x, &w, Dtype::F32).unwrap()
    }

    #[test]
    fn simulated_networks_are_valid() {
        for (m, dims, relu) in [(2, vec![3, 5], vec![false]), (4, vec![8, 16, 8], vec![true, false]), (3, vec![5, 4, 6, 2], vec![true, true, true])] {
            network(m, &dims, &relu, 5).check().unwrap();
        }
        let x = floats(6, 1, -2.0, 2.0);
        let w = vec![floats(12, 2, -1.0, 1.0)];
        NetworkWitness::simulate(2, &[3, 4], &[false], &x, &w, Dtype::Bf16).unwrap().check().unwrap();
    }

    #[test]
    fn reject_forged_witnesses() {
        let original = network(4, &[8, 16, 8], &[true, false], 9);
        let mut forged = original.clone();
        forged.acc[0][3] += 1;
        assert!(matches!(forged.check(), Err(WitnessError::Accumulator { layer: 0, index: 3, .. })));
        let mut forged = original.clone();
        forged.q_w[1][0] = 128;
        assert!(matches!(forged.check(), Err(WitnessError::Range { .. })));
        let mut forged = original.clone();
        forged.out[0][2] = if forged.out[0][2] == 5 { 6 } else { 5 };
        assert!(matches!(forged.check(), Err(WitnessError::Output { layer: 0, index: 2, .. })));
        let mut forged = original.clone();
        forged.specs[1].ko -= 1;
        assert!(matches!(forged.check(), Err(WitnessError::Output { layer: 1, .. } | WitnessError::Exponent { layer: 1, .. })));
        // a quantized value that is not the quantization of its float
        let mut forged = original.clone();
        forged.q_x0[0] += 1;
        assert!(matches!(forged.check(), Err(WitnessError::Quantization { .. } | WitnessError::Accumulator { .. })));
        let mut forged = original.clone();
        forged.w_float[0][5] = -forged.w_float[0][5];
        assert!(matches!(forged.check(), Err(WitnessError::Quantization { .. })));
        // an exponent that is not the largest one that fits
        let mut forged = original.clone();
        forged.specs.iter_mut().for_each(|s| s.kx -= 1);
        assert!(matches!(forged.check(), Err(WitnessError::ScalerExponent { .. })));
        let mut forged = original.clone();
        forged.specs[1].kx -= 1; // the layers must agree on the scaler of the input
        assert!(matches!(forged.check(), Err(WitnessError::Spec(_))));
        let mut forged = original.clone();
        forged.specs[1].kw += 1;
        assert!(matches!(forged.check(), Err(WitnessError::ScalerExponent { .. })));
        // a float that is not finite
        let mut forged = original;
        forged.x0_float[1] = f32::NAN;
        assert!(matches!(forged.check(), Err(WitnessError::Float { .. })));
    }

    #[test]
    fn float_parts_and_shifts() {
        assert_eq!(float_parts(1.0), (0, 1 << 23, 127));
        assert_eq!(float_parts(-0.5), (1, 1 << 23, 126));
        assert_eq!(float_parts(0.0), (0, 0, 1));
        assert_eq!(float_parts(f32::from_bits(1)), (0, 1, 1)); // the smallest subnormal
        // 1.0 * 2^6 = 64 = m * 2^-sh with m = 2^23: sh = 17
        assert_eq!(shift_of(127, 6), 17);
    }

    #[test]
    fn bf16_rounding_and_quantization() {
        assert_eq!(round_bf16(1.0), 1.0);
        assert_eq!(round_bf16(1.0 + 1.0 / 256.0), 1.0); // a tie rounds to even
        assert_eq!(round_bf16(1.0 + 3.0 / 256.0), 1.0 + 2.0 / 128.0);
        assert_eq!(quantize(0.5, 64.0, Dtype::F32), 32);
        assert_eq!(quantize(2.5 / 64.0, 64.0, Dtype::F32), 2); // 2.5 rounds to even
        assert_eq!(quantize(-3.5 / 64.0, 64.0, Dtype::F32), -4);
    }
}
