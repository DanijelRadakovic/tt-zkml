//! The public description of a quantized network (the statement a proof is about) and the CPU reference of its integer arithmetic.
//!
//! A layer is `acc = q_x @ q_w` (int32, exact), then `out = relu?(requant(acc))`, and `out` is the `q_x` of the next layer. The requantization is the
//! one of the device: `out = round_half_even(acc / 2^shift)` (or `acc * 2^ko` when `ko >= 0`) with `ko` chosen as the largest exponent for which
//! `max|acc| * 2^ko <= 127`. `ko` is the exponent of `scaler_out` of the device trace.

use serde::{Deserialize, Serialize};

/// Largest `|q|` of an int8 value of the PoT quantization.
pub const QMAX: i32 = 127;
/// Bits of a range-check limb: limbs are looked up in the table `0 ..= 254`, so `value = sum limb_i * 128^i` with `limb_i` in `[0, 254]`.
pub const LIMB_BITS: u32 = 7;
pub const LIMB_BASE: u32 = 1 << LIMB_BITS;
/// Limbs of a non-negative accumulator-sized value (`< 2^28`).
pub const ACC_LIMBS: usize = 4;
/// `127 * 127 * K < 2^28` keeps every accumulator-sized value inside the limbs (and far below the M31 modulus).
pub const MAX_K: usize = 8192;
/// Largest right shift of the requantization. `max|acc| <= 127 * 127 * MAX_K < 127 * 2^20`, so no honest layer needs more, and `254 * 2^20` fits the
/// four limbs of the bound check (`< 2^29`) with no wrap-around in M31.
pub const MAX_SHIFT: u32 = 20;
/// Limits that keep a verifier's work and memory bounded by the statement it is given.
pub const MAX_LAYERS: usize = 64;
pub const MAX_TENSOR_ELEMENTS: usize = 1 << 24;

/// A layer of the statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerSpec {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub relu: bool,
    /// The exponent of `scaler_out`: `out` is `acc * 2^ko` rounded. `ko >= 41` stands for "all accumulators are zero" (the device writes 127).
    pub ko: i32,
    /// The exponent of `scaler_w`: `q_w = round_half_even(w * 2^kw)`. `kw >= 127` stands for "all weights are zero".
    pub kw: i32,
    /// The exponent of `scaler_x`, the quantization of the network input (every layer carries the same value; only layer 0's input is quantized from
    /// floats). `kx >= 127` stands for "the input is all zero".
    pub kx: i32,
}

/// How the requantization of a layer is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// `max|acc| = 0`: every accumulator and output is zero.
    Zero,
    /// `out = acc * 2^ko`, `ko` in `0 ..= 6`.
    Left(u32),
    /// `out = round_half_even(acc / 2^shift)`, `shift` in `1 ..= MAX_SHIFT`.
    Right(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecError {
    Shape(String),
    Ko { layer: usize, ko: i32 },
    TooLarge { layer: usize, k: usize },
    Chain { layer: usize },
}

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpecError::Shape(s) => write!(f, "{s}"),
            SpecError::Ko { layer, ko } => write!(f, "layer {layer}: the requantization exponent {ko} is outside what the constraints support"),
            SpecError::TooLarge { layer, k } => write!(f, "layer {layer}: K = {k} exceeds {MAX_K}"),
            SpecError::Chain { layer } => write!(f, "layer {layer}: its input is not the output of the previous layer"),
        }
    }
}

impl std::error::Error for SpecError {}

impl LayerSpec {
    pub fn mode(&self) -> Mode {
        match self.ko {
            ko if ko >= 41 => Mode::Zero,
            ko if ko >= 0 => Mode::Left(ko as u32),
            ko => Mode::Right((-ko) as u32),
        }
    }

    /// Smallest `|acc|` for which `ko` is the largest allowed exponent: `|acc| * 2^(ko + 1) > 127`. The layer must have one such accumulator.
    pub fn hit_threshold(&self) -> u32 {
        let s = self.ko + 1;
        if s >= 0 { (QMAX as u32 >> s.min(31)) + 1 } else { ((QMAX as u32) << (-s)) + 1 }
    }

    pub fn matmul_rows(&self) -> usize {
        self.m * self.n
    }

    pub fn validate(&self, layer: usize) -> Result<(), SpecError> {
        if self.m == 0 || self.k == 0 || self.n == 0 {
            return Err(SpecError::Shape(format!("layer {layer}: empty shape")));
        }
        for (a, b) in [(self.m, self.k), (self.k, self.n), (self.m, self.n)] {
            if a.checked_mul(b).is_none_or(|e| e > MAX_TENSOR_ELEMENTS) {
                return Err(SpecError::Shape(format!("layer {layer}: a tensor has more than {MAX_TENSOR_ELEMENTS} elements")));
            }
        }
        if self.ko >= 41 && self.ko != 127 {
            return Err(SpecError::Ko { layer, ko: self.ko }); // "all zero" has one spelling
        }
        if self.k > MAX_K {
            return Err(SpecError::TooLarge { layer, k: self.k });
        }
        for (what, k) in [("kw", self.kw), ("kx", self.kx)] {
            if !(-100..=127).contains(&k) {
                return Err(SpecError::Shape(format!("layer {layer}: the quantization exponent {what} = {k} is not supported")));
            }
        }
        match self.mode() {
            Mode::Right(s) if s > MAX_SHIFT => Err(SpecError::Ko { layer, ko: self.ko }),
            Mode::Left(k) if k > 6 => Err(SpecError::Ko { layer, ko: self.ko }),
            _ => Ok(()),
        }
    }
}

/// Checks every layer and that the layers chain.
pub fn validate(specs: &[LayerSpec]) -> Result<(), SpecError> {
    if specs.is_empty() || specs.len() > MAX_LAYERS {
        return Err(SpecError::Shape(format!("the number of layers must be in 1..={MAX_LAYERS}")));
    }
    for (l, s) in specs.iter().enumerate() {
        s.validate(l)?;
        if s.kx != specs[0].kx {
            return Err(SpecError::Shape(format!("layer {l}: kx differs from layer 0")));
        }
        if l > 0 && (specs[l - 1].n != s.k || specs[l - 1].m != s.m) {
            return Err(SpecError::Chain { layer: l });
        }
    }
    Ok(())
}

/// The exponent of a power-of-two float (`scaler_out` of the device).
pub fn exponent_of(scaler: f32) -> Option<i32> {
    let bits = scaler.to_bits();
    if bits >> 31 != 0 || bits & 0x007f_ffff != 0 || bits >> 23 == 0 || bits >> 23 == 0xff {
        return None;
    }
    Some((bits >> 23) as i32 - 127)
}

/// The exponent `k` of the PoT quantization of floats: the largest `k` with `absmax * 2^k <= 127` (at most 127). Exact: scaling by a power of two is
/// exact in f64.
pub fn pot_exponent(absmax: f32) -> i32 {
    if absmax == 0.0 {
        return 127;
    }
    let a = absmax as f64;
    let mut k = (127.0 / a).log2().floor() as i32;
    while a * 2f64.powi(k + 1) <= 127.0 {
        k += 1;
    }
    while a * 2f64.powi(k) > 127.0 {
        k -= 1;
    }
    k.min(127)
}

/// The requantization of one accumulator, as the device does it (before the ReLU is applied: see [`activate`]).
pub fn requant(acc: i32, mode: Mode) -> i32 {
    match mode {
        Mode::Zero => 0,
        Mode::Left(ko) => acc << ko,
        Mode::Right(sh) => {
            let mag = acc.unsigned_abs() as i64;
            let q = (mag + (1i64 << (sh - 1)) - 1 + ((mag >> sh) & 1)) >> sh;
            if acc < 0 { -(q as i32) } else { q as i32 }
        }
    }
}

/// `out` of a layer for the accumulator `acc`: the ReLU, then the requantization.
pub fn activate(acc: i32, spec: &LayerSpec) -> i32 {
    let a = if spec.relu { acc.max(0) } else { acc };
    requant(a, spec.mode())
}

/// The accumulator exponent `ko` the device chooses for these accumulators: the largest with `max|acc| * 2^ko <= 127`.
pub fn choose_ko(accs: &[i32]) -> i32 {
    let amax = accs.iter().map(|a| a.unsigned_abs() as i64).max().unwrap_or(0);
    if amax == 0 {
        return 127;
    }
    let t = 64 - amax.leading_zeros() as i32;
    let mut ko = 7 - t;
    if ko < 0 && amax > (127i64 << -ko) {
        ko -= 1;
    }
    ko
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponent_of_powers_of_two() {
        assert_eq!(exponent_of(1.0), Some(0));
        assert_eq!(exponent_of(0.125), Some(-3));
        assert_eq!(exponent_of(64.0), Some(6));
        assert_eq!(exponent_of(2f32.powi(127)), Some(127));
        assert_eq!(exponent_of(3.0), None);
        assert_eq!(exponent_of(-1.0), None);
        assert_eq!(exponent_of(0.0), None);
    }

    #[test]
    fn pot_exponent_is_the_largest_that_fits() {
        assert_eq!(pot_exponent(1.0), 6);
        assert_eq!(pot_exponent(127.0), 0);
        assert_eq!(pot_exponent(128.0), -1);
        assert_eq!(pot_exponent(0.0), 127);
        assert_eq!(pot_exponent(0.5), 7);
        for a in [0.001f32, 0.3, 1.7, 63.9, 64.0, 100.0, 12345.0] {
            let k = pot_exponent(a);
            assert!(a as f64 * 2f64.powi(k) <= 127.0 && a as f64 * 2f64.powi(k + 1) > 127.0, "{a}: {k}");
        }
    }

    #[test]
    fn requant_rounds_half_to_even() {
        let m = Mode::Right(2);
        assert_eq!([0, 1, 2, 3, 4, 5, 6].map(|a| requant(a, m)), [0, 0, 0, 1, 1, 1, 2]);
        assert_eq!(requant(-6, m), -2);
        assert_eq!(requant(-2, m), 0);
        assert_eq!(requant(-3, m), -1);
        assert_eq!(requant(5, Mode::Left(2)), 20);
    }

    #[test]
    fn chosen_ko_is_the_largest_that_fits() {
        for amax in [1, 2, 63, 64, 126, 127, 128, 200, 1000, 12345, 16_000_000] {
            let ko = choose_ko(&[amax, -amax / 2]);
            let fits = |k: i32| if k >= 0 { (amax as i64) << k <= 127 } else { amax as i64 <= 127i64 << -k };
            assert!(fits(ko), "amax {amax}: ko {ko} does not fit");
            assert!(!fits(ko + 1), "amax {amax}: ko {ko} is not the largest");
            let spec = LayerSpec { m: 1, k: 1, n: 1, relu: false, ko, kw: 0, kx: 0 };
            assert!(amax as u32 >= spec.hit_threshold(), "amax {amax} ko {ko} threshold {}", spec.hit_threshold());
        }
        assert_eq!(choose_ko(&[0, 0]), 127);
    }
}
