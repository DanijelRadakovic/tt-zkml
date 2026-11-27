//! The constraints (AIR) of the execution of a quantized network, over the M31 field of stwo (Circle STARK), with LogUp lookups between the components.
//!
//! The statement: a network of layers (`LayerSpec`: shapes, ReLU, scaler exponents, requantization exponent) maps the public float input and the public float
//! weights to the public output tensor: `q = round_half_even(float * 2^k)` for the input and the weights, `acc[l] = q_x[l] @ q_w[l]` and
//! `q_x[l + 1] = relu?(requant(acc[l]))`.
//!
//! Components (all [`NetEval`], they differ in [`Kind`]):
//!
//! * `Matmul(l)`: one row per output element `(m, n)`, columns `x_0 .. x_{K-1}, w_0 .. w_{K-1}, acc`; `acc = sum_k x_k * w_k`. The `x_k` / `w_k` are not
//!   free: each is looked up in the tensor relation as `(tag, i, j, value)` of the committed tensor, the `acc` is produced into the relation.
//! * `Tensor(W, l)` and `Tensor(X, 0)`: the quantized weights and input, one row per element. Each row consumes the public float `(sign, m, e')` of its
//!   element and proves `q = round_half_even(float * 2^k)` (`m = |q| * 2^sh + r`, `|r| <= 2^(sh-1)`, ties to even, `2^sh` from the `Pow2Table`), that
//!   no scaled float exceeds 127 and that `k` is the largest exponent (one element has `|float| * 2^(k+1) > 127`). It yields `q` once per use by the
//!   matmul (the multiplicity is a preprocessed constant) and range checks it to `[-127, 127]`.
//! * `Requant(l)`: one row per output element: consumes `acc` from the matmul, applies the ReLU and the requantization (see [`requant`]) and yields the
//!   result as the `q_x` of the next layer (the tensor of the next matmul). It proves the rounding (`acc = out * 2^s + rem`, `|rem| <= 2^(s-1)`, ties to
//!   even) and that the exponent `ko` is the largest one for which no output exceeds 127 (one row has `|acc| * 2^(ko+1) > 127`).
//! * `Table`: the preprocessed range table `0 ..= 254` with the multiplicities of the range checks. `Pow2Table`: the powers of two `2^1 ..= 2^25`.
//!
//! All lookups use LogUp; the claimed sums of all components plus the public part ([`public_sum`]: the output tensor and the count of "largest accumulator"
//! rows) must be zero. The index columns, the "real row" flags and the multiplicities are preprocessed: constants the verifier recomputes.
//!
//! The public part of the LogUp sum ([`public_sum`]) is computed by the verifier from the public floats and the output; a proof is about exactly that model
//! and that input. What is not hidden: the weights and the input are public inputs here (the proof is an integrity proof, not a zero-knowledge one for them).

pub mod spec;
#[cfg(feature = "forgery")]
#[doc(hidden)]
pub mod kind;
#[cfg(not(feature = "forgery"))]
mod kind;
#[cfg(feature = "forgery")]
#[doc(hidden)]
pub mod trace;
#[cfg(not(feature = "forgery"))]
mod trace;

pub use kind::{Kind, Lookups, NetEval, PreCol, Which, check_lookup_budget, layout, public_sum, range_lookups, table_pre_id, TABLE_LOG_SIZE};
pub use spec::{ACC_LIMBS, LayerSpec, Mode, SpecError};
pub use trace::{BuiltComponent, NetworkTrace, build, interaction_traces};
