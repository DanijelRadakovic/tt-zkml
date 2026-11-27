// The circuits (AIR) of tt-zkml: every column, constraint and lookup, with worked examples.
// Build: typst compile docs/zk-circuits.typ docs/zk-circuits.pdf   (from lib/tt-zkml/)
// The numbers of the examples are printed by the trace generator: cargo run --release --example circuit_examples
#set document(title: "tt-zkml: the circuits", author: "tt-zkml")
#set page(paper: "a4", margin: (x: 18mm, y: 20mm), numbering: "1", header: context {
  if counter(page).get().first() > 1 [#set text(size: 8pt, fill: gray); tt-zkml: the circuits #h(1fr) AIR over M31, stwo Circle STARK]
})
#set text(size: 9.5pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { v(0.9em); text(size: 14pt, it); v(0.3em) }
#show heading.where(level: 2): it => { v(0.6em); text(size: 11.5pt, it); v(0.2em) }
#show raw.where(block: false): set text(size: 8.8pt)
#show raw.where(block: true): set text(size: 8.3pt)
#show table.cell.where(y: 0): set text(weight: "bold")

#let P = $p$
#let cell(body) = text(size: 8.6pt, body)
#let ctable(columns, ..args) = table(columns: columns, stroke: 0.4pt + gray, inset: (x: 4pt, y: 3.5pt), ..args)
#let box(body, fill: luma(245)) = block(fill: fill, inset: 6pt, radius: 2pt, width: 100%, body)
#let note(body) = block(fill: rgb("#fff7e0"), inset: 6pt, radius: 2pt, width: 100%, text(size: 9pt, body))

#align(center)[
  #text(size: 20pt, weight: "bold")[The circuits of tt-zkml]\
  #v(2pt)
  #text(size: 11pt)[Every column, constraint and lookup of the AIR, with worked examples]
]
#v(6pt)

= What is proven

A quantized neural network runs on a Tenstorrent device. This document describes the *circuits* that prove that the output is exactly what the quantized
network computes on the verifier's own model and input. The proof system is a Circle STARK (stwo) over the Mersenne field
$FF_p$, $p = 2^31 - 1$, with LogUp lookups between the components.

*Statement.* For the verifier's model (layers with their ReLU flags, float weights $W_l$), float input $X$, and dtype (f32, or bf16 which rounds the floats first),
and for the prover's per-layer exponents $(k_x, k_(w,l), k_(o,l))$ and public output $Y$, the proof shows:

+ $q^x_0 = op("rne")(X dot 2^(k_x))$ and $q^w_l = op("rne")(W_l dot 2^(k_(w,l)))$, where $op("rne")$ is round-half-to-even and each $k$ is the *largest*
  exponent with $op("absmax") dot 2^k <= 127$ ($k = 127$ means "all zero");
+ $a_l = q^x_l dot q^w_l$ exactly, in integers ($a_l$ is the int32 accumulator of the device);
+ $o_l = op("relu")^"?" (op("requant")(a_l))$ with $op("requant")(a) = op("rne")(a / 2^s)$ if $k_(o,l) = -s < 0$, $a dot 2^(k_o)$ if $0 <= k_o <= 6$, $0$ if $k_o = 127$,
  and $k_o$ the *largest* exponent with $max |a| dot 2^(k_o) <= 127$ (over the raw accumulators, also under a ReLU); $q^x_(l+1) = o_l$;
+ $Y = o_(L-1)$.

The weights and the input are public inputs of the verifier (this is an integrity proof, not zero knowledge for them). The device and its trace are not
trusted: only the proof matters. The soundness of the first version of these circuits was audited (see docs/audit.md); every constraint below is the fixed one.

== The pieces and how they are linked

#figure(
  {
    let b(t, w: 70pt) = rect(width: w, inset: 5pt, radius: 2pt, stroke: 0.6pt, fill: luma(245), align(center, text(size: 8pt, t)))
    let a = text(size: 11pt, $arrow.r$)
    align(center, grid(columns: 7, column-gutter: 4pt, align: horizon + center,
      b[public floats\ $X$, $W_l$], a, b[*Tensor*\ quantize], a, b[*Matmul*\ $a = x w$], a, b[*Requant*\ ReLU, round]))
  },
  caption: [Data flow of one layer. The arrows are LogUp lookups on tuples. The output $o_l$ of `Requant` is the input $q^x_(l+1)$ read by the next layer's `Matmul`; the
    last output is public.],
)

#ctable((auto, 1fr, auto),
  [Component], [One row per], [Role],
  [`Tensor` ($X_0$, $W_l$)], [element of the tensor], [consumes the public float, proves $q = op("rne")("float" dot 2^k)$, range checks $q$, yields it to the matmul],
  [`Matmul`($l$)], [output element $(m, n)$], [proves $a = sum_k x_k w_k$ and reads $x_k, w_k$ from the tensors, yields $a$],
  [`Requant`($l$)], [output element $(m, n)$], [consumes $a$, applies ReLU and the rounding, yields $o$ as the next layer's $x$],
  [`Table`], [value $0 dots 254$], [the range table (multiplicities)],
  [`Pow2Table`], [shift $1 dots 25$], [the powers of two $2^s$ (multiplicities)],
)

= Conventions

== Columns, rows and padding

A component has a trace of $2^L$ rows ($L >= 4$, the smallest power of two that holds its data). Columns are of two kinds:
*preprocessed* columns are constants of the statement that the verifier recomputes and checks against the commitment (index columns `i`, `j`, `m`, `n`,
the flag `real`, the multiplicity `mult`, the tables); *trace* columns are the prover's witness. Rows beyond the data are *padding rows*: their
preprocessed `real` and `mult` are $0$ and their trace columns are free (but constrained like any row). Every lookup that is not self-balancing carries
`real` or `mult` as its multiplicity, so padding rows cannot be used to prove anything; the "largest" counters below are explicitly gated by `real`.

All constraints have degree $<= 3$ (the quotient has degree $< 2^(L+1)$). A constraint is written $c = 0$; "$[c = 0]$" in prose means the condition holds.

== Lookups (LogUp)

A *use* of a tuple $t$ with multiplicity $mu$ adds the fraction $mu / h(t)$ to the component's sum, where $h(t) = sum_i alpha_i t_i - z$ with random $z, alpha_i$ drawn after
the base trace is committed. A *yield* is a use with negative multiplicity. Over all components and the verifier's public part, the sum of every
relation must be $0$; with overwhelming probability this holds iff the multiset of used tuples equals the multiset of yielded tuples
(counted with multiplicities). A lookup is therefore "this tuple appears in that other place the right number of times".

#ctable((auto, auto, 1fr),
  [Relation], [Tuple], [Meaning],
  [`Range`], [$(v)$], [$v$ is in the table $0 dots 254$],
  [`Pow2`], [$(s, p_2)$], [$p_2 = 2^s$, $1 <= s <= 25$],
  [`Tensor`], [$("tag", i, j, v)$], [element $(i, j)$ of a quantized tensor has value $v$ (also: $("tag", 0, 0, 0)$ counts "largest" rows)],
  [`Float`], [$("tag", i, j, s, m, e')$], [element $(i, j)$ of a public float tensor is $(-1)^s dot m dot 2^(e' - 150)$],
)

Tags of `Tensor`: $x_l = 4l$ (the input of layer $l$), $w_l = 4l + 1$, $a_l = 4l + 2$ (the accumulator), $"hit"_l = 4l + 3$, and
$"fhit"_t = 2^(20) + t$ for the "largest float" counter of the float tensor $t$. Tags of `Float`: $0$ for $X$, $l + 1$ for $W_l$.

== Gadgets used everywhere

#ctable((auto, 1fr),
  [Gadget], [What it guarantees],
  [$op("Range")(v)$], [lookup of the value $v + 127$ in `Range`: $v in [-127, 127]$ (an integer). Written $op("Range")_0(v)$ without the offset: $v in [0, 254]$.],
  [$op("Limbs")(v; l_0 dots l_3)$], [$v = sum_(t=0)^3 l_t 128^t$ with every limb $op("Range")_0$-checked. Then $v$ is an integer in $[0, 2^(29))$ (limbs are at most 254, so the sum is at most
    $536 thin 870 thin 910$). This is the "non-negative and small" check; it cannot wrap in $FF_p$.],
  [bit], [$b (b - 1) = 0$],
  [two-sided bound], [$v_1 + v_2 = 2H - 2$ with $v_1, v_2$ both $op("Limbs")$-checked gives $0 <= v_1, v_2 <= 2H - 2$ in the integers (the sum is $< 2^30 < p$), i.e. $|v_1 - (H - 1)| <= H - 1$.],
  [is-zero ($z, g, g^(-1)$)], [$z (z - 1) = 0$, $z g = 0$, $(1 - z)(g dot g^(-1) - 1) = 0$ force $z = 1 <=> g = 0$.],
)

= Component `Tensor`: quantizing a public float

One component for the input $X$ (layer 0) and one for every weight matrix $W_l$. For the tensor $t$ with scaler exponent $k$, row $r$ is element $(i, j)$ with
the *public* float $v = (-1)^s dot m dot 2^(e' - 150)$ ($m$ the 24-bit significand, $e'$ the biased exponent, at least 1; subnormals and zero use $e' = 1$, zero has
$m = 0$). The quantization is $q = (-1)^s dot op("rne")(m dot 2^(-"sh"))$ with the shift $"raw" = 150 - e' - k$ (because $v dot 2^k = m dot 2^(-"raw")$).

*Preprocessed*: `i`, `j`, `mult` ($n$ for $X$, $m$ for $W$: how often the matmul reads each element), `real`.
*Trace* (33 columns): $q, s, m, e'$ and
$q_a, p_2, "sh", c, "diff", r, e, "half", v_1[4], v_2[4], z, g, g^(-1), r_n[4], h, h_v[4], h_s$.
(An all-zero tensor, $k = 127$, has only $q, s, m, e'$.)

*Lookups of a row.*
#ctable((auto, 1fr),
  [use], [meaning],
  [`Float`$(op("tag"), i, j, s, m, e')$, mult $+"real"$], [consumes the public float of the element: $s$, $m$, $e'$ are *equal* to the verifier's values],
  [`Tensor`$(op("tag"), i, j, q)$, mult $-"mult"$], [yields the quantized element, once per use by the matmul],
  [$op("Range")(q)$], [$q in [-127, 127]$],
  [`Pow2`$("sh", p_2)$], [$p_2 = 2^("sh")$ with $"sh" in [1, 25]$],
  [$op("Range")_0(q_a), op("Range")_0(g), op("Range")_0("diff"), op("Range")_0(h_s), op("Range")("half")$], [small non-negative columns],
  [$op("Limbs")(v_1), op("Limbs")(v_2), op("Limbs")(r_n), op("Limbs")(h_v)$], [the four limb sets (16 `Range` lookups)],
  [`Tensor`$("fhit", 0, 0, 0)$, mult $+h$], [counts the "largest float" row (the verifier subtracts one)],
)

*Constraints* ($"raw" := 150 - k - e'$ is a small integer; $H := p_2 slash 2$, computed as $2^30 p_2$ in $FF_p$):

#ctable((auto, 1fr, 1fr),
  [], [Constraint], [Meaning],
  [T1], [$q - q_a (1 - 2 s) = 0$], [$q$ is the signed value of the magnitude $q_a$],
  [T2], [$c (c - 1) = 0$], [$c = 1$ marks a tiny float],
  [T3], [$(1 - c)("sh" - "raw") = 0$], [normal case: the shift is the raw shift],
  [T4], [$(1 - c)("diff" - (24 - "raw")) = 0$], [with $op("Range")_0("diff")$: normal means $"raw" <= 24$],
  [T5], [$c ("sh" - 25) = 0$], [tiny case: the shift is 25],
  [T6], [$c ("diff" - ("raw" - 25)) = 0$], [with $op("Range")_0("diff")$: tiny means $"raw" >= 25$],
  [T7], [$c dot q_a = 0$], [a tiny float (scaled value $< 1 slash 2$) quantizes to $0$],
  [T8], [$r - (m - q_a p_2) = 0$], [the remainder of $m$ divided by $2^("sh")$],
  [T9], [$e (e - 1) = 0$], [$e = 1$ marks a tie],
  [T10], [$e (r - H)(r + H) = 0$], [a tie has $|r| = H$],
  [T11], [$e (q_a - 2 "half") = 0$], [and an even $q_a$ (round half to even)],
  [T12], [$(1 - e)(sum v_1 - (H - 1 + r)) = 0$], [not a tie: $v_1 = H - 1 + r >= 0$],
  [T13], [$(1 - e)(sum v_2 - (H - 1 - r)) = 0$], [not a tie: $v_2 = H - 1 - r >= 0$; with T12 and the two-sided bound $|r| <= H - 1$],
  [T14], [$g - (127 - q_a) = 0$], [with $op("Range")_0(q_a), op("Range")_0(g)$: $q_a in [0, 127]$],
  [T15-T17], [is-zero gadget on $g$ with flag $z$], [$z = 1 <=> q_a = 127$],
  [T18], [$sum r_n + z r = 0$], [$q_a = 127 ==> r <= 0$: the scaled float is $<= 127$ *exactly*, not only $< 127.5$],
  [T19], [$h (h - 1) = 0$], [$h$ marks the element that justifies the scaler],
  [T20], [$h (1 - "real") = 0$], [only a real element can be that element (padding rows are free)],
  [T21], [$h_s - h (20 - "sh") = 0$], [with $op("Range")_0(h_s)$: $h = 1 ==> "sh" <= 20$],
  [T22], [$h (sum h_v - (2 m - 127 p_2 - 1)) = 0$], [with $op("Limbs")(h_v)$: $h = 1 ==> 2m > 127 dot 2^("sh")$, i.e. $|v| dot 2^(k+1) > 127$],
)

*Why this is the quantization.* T3-T7 fix the shift: for a float with $"raw" <= 24$ it is $"raw"$ itself, for a tiny one it is 25 and the result is forced to 0 (the true
value is $m slash 2^("raw") < 2^(24) slash 2^(25) = 1 slash 2$, which rounds to 0). T8-T13 make $q_a$ the nearest integer to $m slash 2^("sh")$ with ties to even.
The bound $q_a p_2 <= 127 dot 2^(24) < p$ (T14 and $"sh" <= 24$ when $c = 0$; $q_a = 0$ when $c = 1$) means $q_a p_2$ never wraps modulo $p$, so $r$ is the true
integer remainder. T18 makes $127$ an exact upper bound; T19-T22 together with the verifier's "exactly one" (section 8) make $k$ the *largest* exponent: some real
element exceeds $127 slash 2$ after scaling by $2^k$.

== Example: the weights $W = [127.0, 2.5, -3.5, 0.0]$, $k_w = 0$

The largest absolute value is 127, so $127 dot 2^0 <= 127 < 127 dot 2^1$ and $k_w = 0$. Rows computed by the trace generator ($H = p_2 slash 2$):

#ctable((auto,) * 11,
  [float], [$s$], [$m$], [$e'$], [raw], [$q_a$], [sh], [$p_2$], [$r$], [$e$ / half], [$q$],
  [127.0], [0], [16646144], [133], [17], [127], [17], [131072], [0], [0 / 0], [127],
  [2.5], [0], [10485760], [128], [22], [2], [22], [4194304], [2097152 = $H$], [1 / 1], [2],
  [-3.5], [1], [14680064], [128], [22], [4], [22], [4194304], [-2097152 = $-H$], [1 / 2], [-4],
  [0.0], [0], [0], [1], [149], [0], [25], [33554432], [0], [0 / 0], [0],
)

- $2.5$ is a tie: $m slash p_2 = 2.5$, $r = +H$, so $e = 1$ and $q_a$ must be even: $q_a = 2$ with half $= 1$ ($q_a = 2 dot "half"$). $-3.5$: $q_a = 4$ (half $= 2$), $s = 1$ gives $q = -4$.
- $127.0$: raw $= 17 <= 24$, so $c = 0$, $"diff" = 24 - 17 = 7$. $q_a = 127$ gives $g = 0$, $z = 1$ and $r_n = -r = 0$ (the value is exactly $127$, not above).
  This is also the *largest* element: $2m = 33 thin 292 thin 288 > 127 dot 2^(17) = 16 thin 646 thin 144$, so $h = 1$, $h_s = 20 - 17 = 3$,
  $h_v = 2m - 127 p_2 - 1 = 16 thin 646 thin 143$ with limbs $[127, 127, 119, 7]$. Exactly one element has $h = 1$.
- $0.0$: raw $= 150 - 1 - 0 = 149 >= 25$: tiny, $c = 1$, sh $= 25$, "diff" $= 149 - 25 = 124$, $q_a = 0$, $r = m = 0$, and for $e = 0$ the limbs of
  $v_1 = v_2 = H - 1 = 2^(24) - 1 = 16 thin 777 thin 215$ are $[127, 127, 127, 7]$.
- For the other rows $v_1, v_2$ of ties are $0$ (the $(1 - e)$ factor switches T12/T13 off).

== Example: the input $X = [0.3, -1.5, 0.02, 2.0]$

The largest is $2.0$: $2 dot 2^5 = 64 <= 127 < 128 = 2 dot 2^6$, so $k_x = 5$.

#ctable((auto,) * 7,
  [float], [$s$], [$m$], [$e'$], [raw $= 150 - e' - 5$], [$m slash 2^("raw")$], [$q$],
  [0.3], [0], [10066330], [125], [20], [9.6], [10],
  [-1.5], [1], [12582912], [127], [18], [48.0], [-48],
  [0.02], [0], [10737418], [121], [24], [0.64], [1],
  [2.0], [0], [8388608], [128], [17], [64.0], [64],
)

Here $2.0$ is the element with $h = 1$ ($2m = 16 thin 777 thin 216 > 127 dot 2^(17) = 16 thin 646 thin 144$), and $0.02$ has raw $= 24$, the largest
shift without clamping ($q_a = 1$ since $0.64 > 1 slash 2$).

= Component `Matmul`

One row per output element $(m, n)$ of $[M, K] dot [K, N]$; row index $m N + n$. *Preprocessed*: `m`, `n`, `real`. *Trace*: $x_0 dots x_(K-1)$, $w_0 dots w_(K-1)$, $a$
($2K + 1$ columns).

#ctable((auto, 1fr, 1fr),
  [], [Constraint / lookup], [Meaning],
  [M1], [$a - sum_(k=0)^(K-1) x_k w_k = 0$], [the accumulator is the dot product (degree 2)],
  [L1], [for each $k$: use `Tensor`$(x_l, m, k, x_k)$, mult $+"real"$], [$x_k$ is the quantized input element $(m, k)$],
  [L2], [for each $k$: use `Tensor`$(w_l, k, n, w_k)$, mult $+"real"$], [$w_k$ is the quantized weight element $(k, n)$],
  [L3], [yield `Tensor`$(a_l, m, n, a)$, mult $-"real"$], [the accumulator, consumed by `Requant`],
)

The tuples contain the *position* $(m, k)$ or $(k, n)$, and the tensor yields each element exactly $N$ (resp. $M$) times, so a row cannot read another value than the
committed element: the used multiset must equal the yielded one, and a different value at $(m, k)$ in any of the $N$ rows breaks the balance.
Since $|x_k|, |w_k| <= 127$ and $K <= 8192$, $|a| <= 127^2 dot 8192 < 2^(28) < p$: the dot product is exact in $FF_p$.

*Example.* $X = mat(1, 2; 3, -1)$, $W = mat(2, 0; -1, 3)$, so $A = X W = mat(0, 6; 7, -3)$ ($M = K = N = 2$). The four data rows (the other 12 rows of the 16-row trace are padding with $"real" = 0$):

#ctable((auto,) * 8,
  [row], [$(m, n)$], [$x_0$], [$x_1$], [$w_0$], [$w_1$], [$a$], [check],
  [0], [(0,0)], [1], [2], [2], [-1], [0], [$1 dot 2 + 2 dot (-1) = 0$],
  [1], [(0,1)], [1], [2], [0], [3], [6], [$1 dot 0 + 2 dot 3 = 6$],
  [2], [(1,0)], [3], [-1], [2], [-1], [7], [$3 dot 2 + (-1)(-1) = 7$],
  [3], [(1,1)], [3], [-1], [0], [3], [-3], [$3 dot 0 + (-1) dot 3 = -3$],
)

Tuples used: row 0 uses $(x_l, 0, 0, 1), (x_l, 0, 1, 2), (w_l, 0, 0, 2), (w_l, 1, 0, -1)$; row 1 uses $(x_l, 0, 0, 1), (x_l, 0, 1, 2), (w_l, 0, 1, 0), (w_l, 1, 1, 3)$; and so on.
The `Tensor` component of $X$ yields $(x_l, 0, 0, 1)$ with multiplicity $N = 2$, and indeed rows 0 and 1 use it once each. If row 1 claimed $x_0 = 5$, the tuple $(x_l, 0, 0, 5)$ would be
used once with no matching yield, and the LogUp sum would not vanish.

= Component `Requant`: ReLU and the requantization

One row per output element of layer $l$ (the same shape as the matmul output). *Preprocessed*: `i`, `j`, `real`, `mult` (how often the next matmul reads
the element: $n_(l+1)$, and $1$ for the last layer whose output is public). Columns depend on the mode of $k_o$:

#ctable((auto, 1fr),
  [Mode], [Condition and columns],
  [Zero], [$k_o = 127$ ($max |a| = 0$). Columns: $a, o$.],
  [Left($k_o$)], [$0 <= k_o <= 6$. Columns: $a, o$, [ReLU: $p, a_p, "ap"[4], "ng"[4]$], $h_p, h_n, h_v[4]$.],
  [Right($s$)], [$k_o = -s$, $1 <= s <= 20$. Columns: $a, o$, [ReLU columns], $e, "half", v_1[n_s], v_2[n_s], u_1[4], u_2[4]$, $h_p, h_n, h_v[4]$ with $n_s = ceil(s slash 7)$.],
)

Let $a' = a_p$ with a ReLU and $a' = a$ without; $H = 2^(s-1)$; $"rem" := a' - o dot 2^s$; $T$ = the smallest integer with $T dot 2^(k_o + 1) > 127$
(`hit_threshold`: $T = floor(127 slash 2^(k_o+1)) + 1$ if $k_o + 1 >= 0$, else $127 dot 2^(-k_o-1) + 1$).

#ctable((auto, 1fr, 1fr),
  [], [Constraint], [Meaning],
  [R0], [Zero: $a = 0$, $o = 0$], [all accumulators are zero],
  [R1], [ReLU: $p (p - 1) = 0$], [$p$ is a bit: "non-negative"],
  [R2], [ReLU: $a_p - p a = 0$], [$a_p = a$ or $0$],
  [R3], [ReLU: $a_p - sum "ap" = 0$ ($op("Limbs")$)], [$a_p >= 0$],
  [R4], [ReLU: $(p - 1) a - sum "ng" = 0$ ($op("Limbs")$)], [$p = 0 ==> a <= 0$; so $a_p = max(a, 0)$],
  [R5], [Left: $o - a' dot 2^(k_o) = 0$], [exact (no wrap: $|a| <= 127$ is range checked)],
  [R6], [Right: $e (e - 1) = 0$], [$e$ marks a tie],
  [R7], [Right: $e ("rem"^2 - H^2) = 0$], [a tie has $|"rem"| = H$],
  [R8], [Right: $e (o - 2 "half") = 0$], [and an even $o$],
  [R9, R10], [Right: $(1 - e)(sum v_1 - (H - 1 + "rem")) = 0$, $(1 - e)(sum v_2 - (H - 1 - "rem")) = 0$], [not a tie: $|"rem"| <= H - 1$ (two-sided bound)],
  [R11, R12], [Right: $sum u_1 - (127 dot 2^s - a) = 0$, $sum u_2 - (127 dot 2^s + a) = 0$], [$|a| <= 127 dot 2^s$ for the *raw* accumulator ($op("Limbs")$)],
  [R13, R14], [$h_p (h_p - 1) = 0$, $h_n (h_n - 1) = 0$], [bits: the "largest" row has a positive / negative accumulator],
  [R15], [$(h_p + h_n)(1 - "real") = 0$], [only a real row can be it],
  [R16], [$h_p (a - T) + h_n (-a - T) - sum h_v = 0$], [$|a| >= T$ on that row ($op("Limbs")$), i.e. $|a| dot 2^(k_o+1) > 127$],
)

*Lookups of a row.* use `Tensor`$(a_l, i, j, a)$ with mult $+"real"$ (the accumulator from the matmul); yield `Tensor`$(x_(l+1), i, j, o)$ with mult $-"mult"$
(the next layer's input); $op("Range")(o)$; with a ReLU the limbs of $a_p$ and $"ng"$; Right: $op("Range")("half")$ and the limbs of $v_1, v_2, u_1, u_2$;
Left: $op("Range")(a)$ (this is what keeps $|a| <= 127$); for the modes with a hit: the limbs of $h_v$ and `Tensor`$("hit"_l, 0, 0, 0)$ with mult $h_p + h_n$.

*Why the exponent is the largest.* $o$ is range checked to $[-127, 127]$, which only says the *rounded* value fits. R11/R12 (and the range check of $a$ in Left mode)
bound the *exact* value: $|a| <= 127 dot 2^s$, so $max|a| dot 2^(k_o) <= 127$. R13-R16 and the verifier's "exactly one" give a real row with $|a| dot 2^(k_o+1) > 127$, so $k_o + 1$ would not fit.
Together $k_o$ is the largest exponent, which is what the device chooses.

== Example: $a = [1000, -2500, 300]$, $k_o = -5$

$max|a| = 2500$: $2500 dot 2^(-5) = 78.1 <= 127$ and $2500 dot 2^(-4) = 156 > 127$, so $k_o = -5$ ($s = 5$, $H = 16$, $T = 127 dot 2^4 + 1 = 2033$).

#ctable((auto,) * 9,
  [$a$], [$o$ (no ReLU)], [rem], [$v_1 = 15 + "rem"$], [$v_2 = 15 - "rem"$], [$u_1 = 4064 - a$], [$u_2 = 4064 + a$], [hit], [$h_v$],
  [1000], [31], [$1000 - 992 = 8$], [23], [7], [3064], [5064], [], [],
  [-2500], [-78], [$-2500 + 2496 = -4$], [11], [19], [6564], [1564], [$h_n = 1$], [$2500 - 2033 = 467$],
  [300], [9], [$300 - 288 = 12$], [27], [3], [3764], [4364], [], [],
)

($1000 slash 32 = 31.25 arrow 31$, $-2500 slash 32 = -78.125 arrow -78$, $300 slash 32 = 9.375 arrow 9$. The limbs of $u_1, u_2$ and $h_v$ are the base-128 digits, e.g. $u_1(1000) = 3064 = [120, 23, 0, 0]$, $h_v = 467 = [83, 3, 0, 0]$.)
The row with $a = -2500$ is the one flagged as "largest": $|a| >= 2033$. With a ReLU the same rows give $o = [31, 0, 9]$: for $a = -2500$, $p = 0$, $a_p = 0$,
$"ng" = 2500$ (limbs $[68, 19, 0, 0]$), $"rem" = 0 - 0 = 0$, so $v_1 = v_2 = 15$.

*A tie.* $a = 24$, $s = 3$ ($H = 4$): $24 slash 8 = 3$ exactly, no tie. $a = 12$: $12 slash 8 = 1.5$, $"rem"$ for $o = 2$ is $12 - 16 = -4 = -H$: a tie, $e = 1$, and $o = 2$ is even ($"half" = 1$); $o = 1$ would give
$"rem" = +4 = H$ but is odd, so R8 rejects it.

*An exponent that is too large (rejected).* $a = [16257, 127]$: $s = 8$ is the largest ($127 dot 2^8 = 32512 >= 16257$). Claiming $s = 7$ would round $16257 slash 128 = 127.0078$ to $127$
(which passes $|o| <= 127$), but $u_1 = 127 dot 128 - 16257 = -1 < 0$ cannot be limb-checked, so R11 fails.

== Example: left shift and zero mode

$a = [20, -30, 5]$: $max|a| = 30$, $30 dot 2^2 = 120 <= 127 < 240$, so $k_o = 2$ (Left(2)), $o = [80, -120, 20]$, $T = floor(127 slash 8) + 1 = 16$. Every $a$ is range checked
to $[-127, 127]$ (otherwise $a dot 2^(k_o)$ could wrap modulo $p$). The first row with $|a| >= 16$ (here $a = 20$) is flagged $h_p = 1$; the verifier only needs exactly one such row.
$a = [0, 0, 0]$: $k_o = 127$, Zero mode: R0 forces $a = o = 0$ in every row and there is no hit.

= Tables

*Range table* (component `Table`, preprocessed $v_i = min(i, 254)$ for $i = 0 dots 255$; trace: $mu_i$): yields `Range`$(v_i)$ with multiplicity $-mu_i$. All uses of `Range` in the whole
network must be matched, so every checked value is in $[0, 254]$ (the table has $254$ twice so that it has 256 rows; the prover books the uses of $254$ on the first row).

*Power-of-two table* (component `Pow2Table`, 32 rows: $s_i = min(i + 1, 25)$, $p_(2,i) = 2^(s_i)$; trace: $mu_i$): yields `Pow2`$(s_i, p_(2,i))$ with $-mu_i$. It fixes the pair
$("sh", p_2)$ of every quantized float: $p_2$ is exactly $2^("sh")$ and $1 <= "sh" <= 25$.

= The public part and the verifier

The verifier adds to the claimed sums of the components the part it can compute itself, with $h_T, h_F$ the combinations of the `Tensor` and `Float` relations:
$ P = sum_((i, j)) 1 / h_T(x_L, i, j, Y_(i j)) #h(0.4em) - #h(0.4em) sum_(l : k_(o,l) != 127) 1 / h_T("hit"_l, 0, 0, 0)
  #h(0.4em) - #h(0.4em) sum_((i, j)) 1 / h_F(0, i, j, s, m, e')_(X) #h(0.4em) - #h(0.4em) [k_x != 127] / h_T("fhit"_0, 0, 0, 0)
  #h(0.4em) - #h(0.4em) sum_l (sum_((i, j)) 1 / h_F(l + 1, i, j, s, m, e')_(W_l) + [k_(w,l) != 127] / h_T("fhit"_(l+1), 0, 0, 0)) $
(the $(s, m, e')$ come from the verifier's own floats, rounded to bf16 first for a bf16 model). The terms mean: the last `Requant` yields the output once and the public side
consumes it (so the circuit's output is $Y$); every "largest" counter is checked to be exactly one; every quantization consumes the float tuple of its element and the
public side yields it (so a quantized tensor is the quantization of exactly the verifier's floats). `verify` returns `Ok` only if:

+ the statement is the verifier's: the layers, activations and shapes ($m, k, n$, relu) equal the verifier's model (only the exponents $k_x, k_w, k_o$ come from the proof);
  the floats are finite after the dtype's rounding; the public output is in $[-127, 127]$;
+ the statement (layers, output, floats) is mixed into the Fiat-Shamir transcript first, then the preprocessed root is recomputed from the statement and compared with the proof's;
+ the number of commitments and sampled columns is that of the statement; the lookup elements are drawn after the base trace and the claimed sums are mixed into the transcript
  before the interaction trace is committed;
+ $sum_c "claimed"_c + P = 0$ (it is bound to the per-row LogUp constraints by the framework, and must vanish for each relation: they have independent random elements);
+ the stwo verification (composition polynomial at the out-of-domain point, FRI, proof of work) succeeds with the verifier's parameters: blowup 4, 64 queries, 16 bits of proof of work.

= Supported ranges and the integer bounds that prevent wrap-around

Statements outside these ranges are refused by prover and verifier (never unsound):
$k_x, k_w in [-100, 127]$ with a quantization shift of at least 1 for every non-zero float; $K <= 8192$; right shift $s <= 20$; at most 64 layers; every tensor at most $2^(24)$ elements;
a tensor of only tiny non-zero floats (scaler capped at 127) is not supported.

#ctable((auto, auto, 1fr),
  [Quantity], [Bound], [Why it is below $p = 2^31 - 1$ (no wrap-around)],
  [$x$, $w$, $o$], [$[-127, 127]$], [range checked],
  [$a$], [$|a| <= 127^2 dot 8192 < 2^(28)$], [$K <= 8192$],
  [$m$], [$< 2^(24)$], [pinned to the public float by the `Float` lookup ($s$, $m$, $e'$ are separate tuple entries)],
  [$q_a p_2$], [$<= 127 dot 2^(24) = 2.13 dot 10^9$], [$"sh" <= 24$ unless $c = 1$, where $q_a = 0$],
  [$o dot 2^s$, $s <= 20$], [$<= 127 dot 2^(20)$], [$o$ is range checked],
  [$a dot 2^(k_o)$, $k_o <= 6$], [$<= 127 dot 64$], [$|a| <= 127$ is range checked in Left mode],
  [limb sums], [$< 2^(29)$], [four limbs of at most 254],
  [$u_1, u_2 = 127 dot 2^s plus.minus a$], [$<= 254 dot 2^(20) < 2^(29)$], [$|a| <= 127 dot 2^s$ and $s <= 20$],
  [$h_v = 2m - 127 p_2 - 1$], [$< 2^(25)$], [$"sh" <= 20$ is forced when $h = 1$],
)
