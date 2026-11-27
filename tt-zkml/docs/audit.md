# Soundness audit of the tt-zkml AIR

Scope: `tt-zkml/lib/tt-zkml/src/{air/*.rs, proof.rs, witness.rs}` over stwo (`starkware-libs/proving` rev `c87ea10`, Circle STARK over M31, constraint
framework, LogUp). Date: 2026-11-27. Status of the code audited: the AIR as first written (the "v3" with float quantization binding). The findings below
were fixed in the same pass; every fix has a regression test.

**Bottom line.** The first version was not sound: 7 distinct ways for a malicious prover to make `proof::verify` accept a false statement were found
(three of them critical) and all were confirmed by running forged proofs. All are fixed. After the fixes, the review below finds no further way to
violate the statement, under the assumptions and limits listed in section 6. This is a manual review plus an independent adversarial review plus mutation
testing, **not** a machine-checked proof and not a third-party audit: see section 7.

## 1. The statement

`verify(proof, inputs)` returns `Ok` only if, for the verifier's `inputs` (model layers with their ReLU flags, float weights, float input, dtype), the
prover's `proof.specs` (per layer: `m, k, n, relu, ko, kw, kx`) and the public `proof.output`:

1. `q_x0 = round_half_even(X * 2^kx)` and `q_w[l] = round_half_even(W_l * 2^kw_l)`, where `X`, `W_l` are the floats the device sees (rounded to bf16 first for
   a bf16 model) and `kx`, `kw_l` are the LARGEST exponents with `absmax * 2^k <= 127` (`k = 127`: the tensor is all zero);
2. `acc[l] = q_x[l] @ q_w[l]` exactly (int32);
3. `out[l] = relu?(requant(acc[l]))`, `requant` = `round_half_even(acc / 2^s)` for `ko = -s < 0`, `acc * 2^ko` for `0 <= ko <= 6`, `0` for `ko = 127`;
   `ko` is the LARGEST exponent with `max|acc| * 2^ko <= 127` (over the raw accumulators, also under a ReLU); `q_x[l+1] = out[l]`;
4. `proof.output == out[last]`.

The layers, activations and shapes come from the verifier's model, not from the proof (finding F7). The exponents come from the proof and are proven to be
the largest ones.

Not in the statement: zero knowledge (the float weights and input are public inputs of the verifier), and anything about how the device produced the trace
(the trace is untrusted; only the proof matters).

## 2. Method

1. **Manual review** by the author of every constraint, lookup and verifier check, with the explicit question "which quantities are integers, what are
   their bounds, can a field wrap-around (mod p = 2^31 - 1) or a free column change the meaning?".
2. **Independent adversarial review** by a separate reviewer that was given only the code and the statement (not the author's conclusions). It worked on a
   copy, built forged proofs by rewriting columns of an otherwise honest trace (a malicious prover is not bound by the honest trace generator), and
   reported only findings it confirmed with a running test that `verify` accepted. It had a negative control (the same forgery without column forging is
   rejected).
3. **Regression tests** (`src/audit_tests.rs`): every finding is a test in which a forged proof must not verify. The forgeries are written through a
   test-only hook (`air::trace::HOOK`) that rewrites the columns of a component.
4. **Mutation testing of the regression tests**: each fix was switched off in turn, and the matching test must then FAIL (the forgery is accepted). This
   showed that three tests (`f1`, `f5`, `f6`) were first vacuous (rejected for an unrelated reason) and they were rewritten, or accompanied by an
   isolated one (`f1b`), until the fix under test was the only thing rejecting the forgery. Result: every fix below is covered by a test that fails without it (table, section 3).

Both reviews found the same critical issues independently (F1 to F4), which is some evidence of coverage; the independent reviewer found F5 to F7 and the
transcript and robustness points as well.

## 3. Findings

Severity: critical = a false statement is accepted; high = accepted under a plausible adversary; medium/low = robustness or defense in depth.

| ID | Sev. | Where | Problem | Fix | Test (fails without the fix) |
|---|---|---|---|---|---|
| F1 | critical | Float tuple | The public float was bound through `sm = 2m + s` only; `(s, m)` and `(1-s, m + 2^30)` give the same `sm` (`1/2 = 2^30` in M31), so the sign and mantissa of a tiny float could be replaced: a zero weight became `-32`. | The tuple is `(tag, i, j, s, m, e')` (arity 6): `s`, `m`, `e'` are pinned as field elements to the public values. | `f1_*`, `f1b_*` (mutation M6) |
| F2 | critical | Tensor, shift 25 | `qa * 2^25` wraps (`64 * 2^25 = 2^31 = 1 mod p`): `r = m - 1` passes the nearest-rounding check, so ANY float whose scaled magnitude is below 0.5 (every zero weight, every tiny value) could be quantized to ±64. | Tiny floats (raw shift `>= 25`) take a branch that forces `qa = 0` (`c * qa = 0`); a normal shift is `<= 24`, so `qa * 2^sh <= 127 * 2^24 < p` never wraps. | `f2_*` (M5) |
| F3 | critical | Requant, left shift | `acc * 2^ko` wraps for `ko >= 4` (`acc = 2^25`, `ko = 6` gives `1`); nothing bounded `|acc|` for a left shift. | `Left` mode range checks the raw accumulator to `[-127, 127]`. | `f3_*` (M4) |
| F4 | high | Hit rows | The "largest accumulator/float" counter rows were not gated by the `real` flag; padding rows have free columns and could carry the one hit, so a smaller (not largest) exponent `ko`, `kw` or `kx` was accepted. | `(hp + hn) * (1 - real) = 0` and `h * (1 - real) = 0`. | `f4a_*` (M1), `f4b_*` (M2) |
| F5 | high | Requant, right shift | `out = ±127` with a positive remainder was accepted: `ko` one too large when `127 * 2^s < max|acc| < 127 * 2^s + 2^(s-1)`. | `-127 * 2^s <= acc <= 127 * 2^s` as two limb-checked non-negative numbers. | `f5_*` (M3) |
| F6 | high | Requant under ReLU | The negative raw accumulators were not bounded by the exponent (only the ReLU'd values were), though the device chooses `ko` over the raw values. | The same bound is on the raw `acc`. | `f6_*` (M3) |
| F7 | high | Verifier | `relu`, `m`, `k`, `n` of every layer were taken from the proof; a proof for another activation or another split of the same flat floats verified. | `PublicInputs` carries the layers; `verify` compares `(k, n, relu)` and derives `m`. | `f7_*` |
| F8 | medium | Transcript | The statement (exponents, activations, output, floats) and the claimed sums were not mixed into the Fiat-Shamir transcript; the exponents and ReLU flags are constants of the constraints and are in no committed tree. | The statement is mixed in first, the claimed sums before the interaction commitment. | `the_statement_and_the_claimed_sums_*` |
| F9 | low | Statement | `ko in 41..=127` all meant "all zero" (malleable, and `dequantized_output` reported the claimed exponent). | Only 127 is accepted. | `statements_are_canonical_*` |
| F10 | low | Verifier | Shapes and the number of layers from the proof were unbounded (memory DoS). | `MAX_LAYERS = 64`, tensors of at most `2^24` elements, `K <= 8192`. | same |
| F11 | low | Verifier | A float finite in f32 can round to infinity in bf16. | Finiteness is checked after the dtype's rounding. | same |
| F12 | info | PCS | The configuration was blowup 2, 70 queries, 10 bits of work: about 80 bits only under the conjectured FRI bound, about 45 bits under the proven bound. | Blowup 4, 64 queries, 16 bits of work: about 144 bits conjectured, about 80 bits under the proven (Johnson bound) analysis (an estimate: `64 * log2(1/sqrt(rate = 1/4)) + 16`). Cost: proving time about 1.4x. | (config) |

Other changes made for the fixes: the largest right shift is 20 (was 22) so that the bound check fits four 7-bit limbs; `K <= 8192` (the honest
accumulator then never exceeds `127 * 2^20`).

## 4. Why the fixed AIR is sound (the review)

Notation: p = 2^31 - 1; "integer" means the canonical representative in `[0, p)` or `(-p/2, p/2)` as the context says. A *limb decomposition* of a
column `v` is `v = sum_{t<4} l_t * 128^t` with every `l_t` looked up in the range table `[0, 254]`.

**L1 (lookups are multiset equality).** A LogUp relation with random challenges `(z, alpha)` drawn after the base commitment makes `sum use_i / (z - h(t_i))
= 0` over all components and the public part imply, except with probability about `N / 2^124` (QM31), that the multiset of used tuples equals the multiset of
yielded tuples with the multiplicities as field elements; since the number of rows is far below p, this is equality of counts. Every component's claimed
sum is bound by the framework's cumulative-sum constraint (`cumsum_shift = claimed_sum / 2^log_size`, cyclic `[-1, 0]` mask), and the verifier requires
`sum(claimed) + public_sum = 0` for each relation (relations have separate challenges, and the check on the total forces per-relation cancellation).

**L2 (range table).** The table is the preprocessed column `0..=254` (and `254` again) with prover multiplicities; by L1 every looked-up value is in
`[0, 254]` as a field element. A column `v` looked up as `v + 127` is an integer in `[-127, 127]`.

**L3 (no wrap).** Four limbs sum to at most `254 * (1 + 128 + 128^2 + 128^3) = 536 870 910 < 2^29`. A limb-checked column is therefore an integer in
`[0, 2^29)`; if two such columns satisfy `a + b = c` (field) with `c < 2^29`, then `a + b = c` as integers (the sum is `< 2^30 < p`). This gives, e.g.,
`v1 + v2 = 2H - 2` with both non-negative, hence `v1, v2 in [0, 2H-2]`.

**L4 (public floats and the tensor quantization).** Each real tensor row consumes `(tag, i, j, s, m, e')` and the public side yields the verifier's value for
every `(i, j)`; by L1 the row's `s, m, e'` equal the public ones as field elements (`s` a bit, `m < 2^24`, `e' in [1, 254]`; `0 <= m` and `m < 2^24` are
therefore integers). Then:

* `raw = 150 - kexp - e'` is a small integer (`|raw| < 400`). With the bit `c`: `c = 0`: `sh = raw` and `diff = 24 - raw >= 0` (range checked), so
  `sh in [1, 24]` (the Pow2 table lookup excludes `sh <= 0`); `c = 1`: `sh = 25`, `diff = raw - 25 >= 0`, and `qa = 0`. `(sh, p2)` is a row of the Pow2 table, so
  `p2 = 2^sh` exactly.
* `qa` is range checked to `[0, 254]` and `g = 127 - qa` too, so `qa in [0, 127]` (integers). Then `qa * p2 <= 127 * 2^24 = 2 130 706 432 < p` for `c = 0` and
  `qa * p2 = 0` for `c = 1`: **no wrap**, so `r = m - qa * p2` is the exact integer.
* `|r| = H` (a tie, `H = 2^(sh-1)`) with `qa` even (`qa = 2 * half`, `half` range checked), or `|r| <= H - 1` by L3 on `v1 = H - 1 + r`, `v2 = H - 1 - r`. A tie
  cannot be reached for `c = 1` (`m < 2^24 = H`). So `qa = round_half_even(m / 2^sh)`; for `c = 1`, `m / 2^raw < 1/2` makes this `0`, the correct value.
  `q = (1 - 2s) * qa`, with the sign `s` pinned.
* `qa = 127` forces `r <= 0` (`z`, `g`, `ginv`, `rn` limbs: `z = 1` iff `g = 0`, then `-r` is a non-negative limb sum), i.e. `|float| * 2^k <= 127` exactly,
  not only `< 127.5`.
* Largest scaler: `h` is a bit, only on real rows, `hs = h * (20 - sh)` range checked forces `sh <= 20` when `h = 1`, and `hv = 2m - 127 * 2^sh - 1 >= 0`
  (`127 * 2^20 < 2^27`: no wrap); the counter `sum h` is checked to be exactly 1 by the public `-1`, so exactly one real element has `2m > 127 * 2^sh`, i.e.
  `|float| * 2^(k+1) > 127`. Together with the `<= 127` bound this makes `k` the largest exponent. An all-zero tensor (`k = 127`) is forced by `m = 0` on every
  real row, and a non-zero tensor cannot claim it.

**L5 (matmul).** `acc = sum x_k w_k` with `|x_k|, |w_k| <= 127` (they are the range-checked tensor entries, by the tuple lookups `(tag, i, k, x)` /
`(tag, k, j, w)`), `K <= 8192`, so `|acc| <= 127^2 * 8192 < 2^28 < p`: exact. The tensor tuples are used by the matmul rows exactly `n` (resp. `m`) times with
the entry's `(i, j)` in the tuple, so by L1 every row reads the committed entry, not a value of its own. The accumulator is yielded once to the requantization.

**L6 (requantization).** Rows consume `acc` from the matmul (L1), so `acc` is the exact matmul result.

* ReLU: `p` a bit, `ap = p * acc`, `ap` and `(p - 1) * acc` limb checked: `p = 1` gives `acc = ap >= 0`, `p = 0` gives `acc <= 0`, `ap = 0`; so `a' = max(acc, 0)`.
* Right shift `s in [1, 20]`: with `|out| <= 127` (range check), `rem = a' - out * 2^s` is an exact integer (`< 2^29 + 127 * 2^20`); then as in L4:
  `out = round_half_even(a' / 2^s)`, tie handled by evenness. `|acc| <= 127 * 2^s` on the RAW accumulator by `u1 = 127 * 2^s - acc >= 0`,
  `u2 = 127 * 2^s + acc >= 0` (limb checked; `u <= 254 * 2^20 < 2^29`), so `max|acc| * 2^ko <= 127` exactly (F5, F6).
* Left shift `ko in [0, 6]`: the raw `acc` is range checked to `[-127, 127]`, so `out = a' * 2^ko` cannot wrap (F3), and `|out| <= 127` is range checked.
* Zero mode: `acc = out = 0` on every row (`ko = 127` only).
* Largest exponent: exactly one real row has `|acc| >= T` where `T` is the smallest integer with `T * 2^(ko+1) > 127` (`hp`/`hn` bits, gated by `real`, `hv`
  limb checked, counted against the public `-1`); with the bound above, `ko` is the largest exponent (the device's `choose_ko`).

**L7 (links).** Each requantization yields `out` as the tensor `X_{l+1}` once per use by the next matmul (`n_{l+1}` times, a preprocessed constant); the last
layer's output is consumed by the public side, which computes the tuples from `proof.output`. So the matmul reads exactly the previous layer's output, and
the public output is the network's output.

**L8 (padding rows).** All fractions that are not self-balancing use `real` or `mult` preprocessed columns, which are zero on padding rows; the remaining
unconditional lookups (range checks, powers of two) are self-balancing (each use is counted in the table multiplicities). The hit counters are gated by
`real` (F4). Padding-row columns are otherwise free but constrained like any row, and no constraint of a padding row feeds a real one.

**L9 (verifier).** The preprocessed root is recomputed from the verifier's statement (index columns, `real`, multiplicities, tables) and compared; the PCS
and FRI parameters are the verifier's, not read from the proof; the number of commitments and sampled columns is checked; the statement and claimed sums
are in the transcript before the challenges they feed. Constraint degrees are at most 3, so the composition polynomial bound `log_size + 1` suffices
(the quotient has degree `< 2^(L+1)`).

## 5. Integer ranges (where wrap-around was possible)

| Quantity | Bound | Prime p = 2^31 - 1 | Why safe |
|---|---|---|---|
| `x`, `w` | `[-127, 127]` | | range checked |
| `acc` | `127^2 * 8192 < 2^28` | yes | `K <= 8192` |
| `m` (float significand) | `< 2^24` | | pinned to the public value |
| `qa * 2^sh` (quantization) | `<= 127 * 2^24 = 2.13e9` | `< p = 2.147e9` | `sh <= 24`, else `qa = 0` |
| `out * 2^s`, `s <= 20` (requant) | `<= 127 * 2^20 = 1.33e8` | yes | `MAX_SHIFT = 20` |
| `acc * 2^ko`, `ko <= 6` (requant) | `<= 127 * 64` | yes | `|acc| <= 127` is checked |
| limb sums | `< 2^29` | yes | four limbs of at most 254 |
| `u1, u2 = 127 * 2^s -+ acc` | `< 254 * 2^20 + ...` | yes | `< 2^29` |
| `hv = 2m - 127 * 2^sh - 1` (`sh <= 20`) | `< 2^25` | yes | `sh <= 20` is forced when `h = 1` |
| `2m + s` | no longer used | | removed (F1) |

## 6. Assumptions and limits

* stwo's DEEP-ALI + FRI soundness holds with the verifier-chosen parameters, and the lifted-Merkle PCS binds columns of mixed sizes. The bound claimed
  (144 bits) is the conjectured one; the proven (Johnson bound) estimate is about 80 bits. The LogUp and DEEP errors over QM31 are
  negligible.
* The Fiat-Shamir transcript is Blake2s with the statement mixed in first.
* Supported ranges: `kx, kw in [-100, 127]`, a quantization shift of at least 1 for every non-zero float, `K <= 8192`, right shift of the requantization at most 20,
  at most 64 layers, every tensor at most `2^24` elements. Statements outside these are refused (by the prover and by the verifier), not unsound.
* The proof is an integrity proof, not zero knowledge: the weights and the input are public inputs of the verifier. The output is public.
* `verify` must be given `PublicInputs` from a trusted source (the model the verifier cares about). Everything about the network's structure is taken from it.
* The device and its trace are not trusted; a trace that does not reproduce an honest execution simply cannot be proven.

## 7. What this audit is not

* Not a machine-checked proof: the arguments in section 4 are manual. The constraints were not modelled in a proof assistant or checked with a dedicated
  ZK-circuit analysis tool (for example an under-constraint solver).
* Not an audit by a third party. Two independent readers (the author and the independent reviewer) plus a test suite is reasonable evidence, not a
  guarantee; the first version contained seven findings that passed all of its original tests.
* stwo itself (the constraint framework, the prover and verifier, FRI and the PCS) was relied on, not audited.

Recommended next steps before any production use: an external review of `air/kind.rs` and `proof.rs`; a randomized forgery test (mutate honest columns
at random and require rejection) run in CI; running the mutation procedure above in CI so that the regression tests keep failing when a constraint is
removed; and a model of the constraints in an under-constraint checker.

## 8. Reproducing

```bash
cd tt-zkml
cargo test --release -p tt-zkml --lib                 # all unit tests, including src/audit_tests.rs (the forgeries)
cargo test --release -p tt-zkml --test proof          # the end-to-end test on a device
```

Mutation check of one fix (example, F4): in `src/air/kind.rs` delete the line
`eval.add_constraint((hp.clone() + hn.clone()) * (one - real));` and run `cargo test --release --lib audit_tests`: `f4a_*` must fail. The other mutations: delete
`eval.add_constraint(h.clone() * (one.clone() - real));` (`f4b`), the two `u1_limbs` / `u2_limbs` constraints (`f5`, `f6`), the `Use::Range` on `c.acc` in
the left-shift branch of `uses()` (`f3`), `eval.add_constraint(cl * qa.clone());` (`f2`), and pin `(s, m)` only through `2m + s` in the Float tuple (`f1b`).
