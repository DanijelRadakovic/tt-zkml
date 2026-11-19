#set page(paper: "a4", margin: (x: 1.7cm, y: 1.8cm), numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.")
#show raw: set text(size: 9pt)
#let node(t, fill: luma(245)) = box(stroke: 0.6pt, inset: 5pt, radius: 3pt, fill: fill, align(center, text(size: 8pt, t)))
#let arr = text(size: 11pt)[#sym.arrow.r]
#let note(t) = block(fill: rgb("#fff7e0"), stroke: 0.5pt + rgb("#d9b44a"), inset: 7pt, radius: 3pt, width: 100%, t)

#align(center)[
  #text(size: 17pt, weight: "bold")[One quantized zkML layer: two workflows, one worked example] \
  #v(2pt)
  #text(size: 10pt)[`pot_quantize_per_tensor`, matmul, requantize or dequantize, activation. 3×3 tensors, 8 bit, power-of-two scales.]
]

= The idea in one page

A real tensor `x` is stored as an integer tensor `q` and one exponent `k` per tensor: $x approx q dot 2^(-k)$. The exponent is chosen as large as possible so that the largest value still fits into the integer range $[-Q, Q]$, with $Q = 2^7 - 1 = 127$ for 8 bit. All steps use #emph[round half to even] and a clamp to $[-Q, Q]$. The multiplication of two quantized tensors produces an int32 accumulator whose scale is the #emph[sum] of the two exponents. That accumulator is too wide to feed into the next layer, and the two workflows differ in how they bring it back.

#v(4pt)
#align(center)[
  #stack(dir: ltr, spacing: 5pt,
    node[`x` (float)], arr, node[`pot_quantize`\ `_per_tensor`], arr, node[`x_q` int8, `k_x`], arr,
    node[`matmul`\ (int32)], arr, node(fill: rgb("#e8f0ff"))[`acc` int32\ scale $k_x + k_w$])
  #v(2pt)
  #stack(dir: ltr, spacing: 5pt, node[`W` (float)], arr, node[`pot_quantize`\ `_per_tensor`], arr, node[`W_q` int8, `k_w`], arr, node[#text(size: 8pt)[goes into the matmul above]])
]
#v(6pt)
*Workflow A, float between layers*
#align(center)[
  #stack(dir: ltr, spacing: 5pt,
    node(fill: rgb("#e8f0ff"))[`acc` int32], arr, node[`dequantize`\ $"acc" dot 2^(-(k_x+k_w))$], arr, node[`y` float], arr,
    node[activation\ (float)], arr, node[`pot_quantize`\ `_per_tensor`], arr, node(fill: rgb("#e6f7e6"))[next `x_q`, `k`])
]
*Workflow B, integer between layers*
#align(center)[
  #stack(dir: ltr, spacing: 5pt,
    node(fill: rgb("#e8f0ff"))[`acc` int32], arr, node[`pot_requantize`\ `_per_tensor`\ (dynamic `s`)], arr, node[`q_out` int8,\ $k_"out" = k_x + k_w - s$], arr,
    node[activation\ on integers], arr, node(fill: rgb("#e6f7e6"))[next `x_q`, `k_out`])
]
#v(4pt)
In both workflows the last box is the input of the next layer. The next layer quantizes its own weights, multiplies, and repeats the same tail.

= Rules used in the example

- $Q = 127$. The exponent is $k = max{k <= 127 : "absmax" dot 2^k <= Q}$.
- $q = "clamp"("round"_("half-even")(x dot 2^k), -Q, Q)$. A remainder is $r = x dot 2^k - q$ with $|r| <= 1/2$. A tie (exactly $1/2$) goes to the even integer.
- Requantize of an integer accumulator by $s$ bits: $q = "round"_("half-even")(a / 2^s)$, remainder $r = a - q dot 2^s$, $|r| <= 2^(s-1)$. The shift is the smallest $s >= 0$ for which every $|q| <= Q$, which makes the new exponent $k_"out" = k_"acc" - s$ the largest possible.
- $s$ is computed from the bit length of the accumulator absmax: $s = "bitlen"("absmax") - 7$, then one check (below) because rounding can carry into the next bit.

= Worked example

== Step 1: the inputs

#grid(columns: (1fr, 1fr), align: center,
  [`x` (activations)\ $ mat(
    "0.50", "−1.25", "0.75";
    "2.00", "0.10", "−0.60";
    "−0.30", "1.10", "0.90"
) $],
  [`W` (weights)\ $ mat(
    "0.40", "−0.20", "0.10";
    "0.30", "0.80", "−0.50";
    "−0.70", "0.25", "0.60"
) $])

== Step 2: quantize `x` and `W` with `pot_quantize_per_tensor`

*`x`:* absmax $= 2.00$. Try $k = 6$: $2.00 dot 64 = 128 > 127$, too big. Try $k = 5$: $2.00 dot 32 = 64 <= 127$. So *$k_x = 5$*. Multiply by $2^5 = 32$ and round half to even:

#grid(columns: (1fr, 1fr), align: center,
  [$x dot 32$\ $ mat(
    "16.0", "−40.0", "24.0";
    "64.0", "3.2", "−19.2";
    "−9.6", "35.2", "28.8"
) $],
  [$x_q$ (rounded)\ $ mat(
    "16", "−40", "24";
    "64", "3", "−19";
    "−10", "35", "29"
) $])

*`W`:* absmax $= 0.80$. Try $k = 8$: $0.80 dot 256 = 204.8 > 127$. Try $k = 7$: $0.80 dot 128 = 102.4 <= 127$. So *$k_w = 7$*.

#grid(columns: (1fr, 1fr), align: center,
  [$W dot 128$\ $ mat(
    "51.2", "−25.6", "12.8";
    "38.4", "102.4", "−64.0";
    "−89.6", "32.0", "76.8"
) $],
  [$W_q$ (rounded)\ $ mat(
    "51", "−26", "13";
    "38", "102", "−64";
    "−90", "32", "77"
) $])

Each entry has a remainder $r$ with $|r| <= 1/2$, e.g. $x_(1,1) dot 32 = 3.2 arrow.r q = 3$, $r = 0.2$.

== Step 3: integer matmul

$"acc" = x_q W_q$, computed exactly in int32. The scale of the result is $k_x + k_w = 5 + 7 = 12$, i.e. $"acc" dot 2^(-12) approx x W$. Entry $(0,0)$:
$ (16)(51) + (-40)(38) + (24)(-90) = 816 + (−1520) + (−2160) = −2864 $

#grid(columns: (1fr, 1fr), align: center,
  [$"acc"$ (int32, scale $2^(-12)$)\ $ mat(
    "−2864", "−3728", "4616";
    "5088", "−1966", "−823";
    "−1790", "4758", "−137"
) $],
  [float reference $x W$ (for comparison)\ $ mat(
    "−0.7000", "−0.9125", "1.1250";
    "1.2500", "−0.4700", "−0.2100";
    "−0.4200", "1.1650", "−0.0400"
) $])

The largest magnitude is $5088$ (row 1, column 0). It needs $13$ bits, while an int8 holds $7$ magnitude bits. This is why the accumulator cannot go to the next layer as it is.

== Workflow A: dequantize, float activation, quantize again

*Dequantize.* Multiply by $2^(-12)$ (exact in fp32, since $|"acc"| < 2^24$):
#align(center)[$ mat(
    "−0.6992", "−0.9102", "1.1270";
    "1.2422", "−0.4800", "−0.2009";
    "−0.4370", "1.1616", "−0.0334"
) $]
*Activation.* ReLU in float, $y = max(0, "deq")$:
#align(center)[$ mat(
    "0.0000", "0.0000", "1.1270";
    "1.2422", "0.0000", "0.0000";
    "0.0000", "1.1616", "0.0000"
) $]
*Quantize for the next layer* with `pot_quantize_per_tensor`: absmax $= 1.2422$. Try $k=7$: $1.2422 dot 128 = 159 > 127$. Try $k = 6$: $1.2422 dot 64 = 79.5 <= 127$. So *$k = 6$*.
#grid(columns: (1fr, 1fr), align: center,
  [$y dot 64$\ $ mat(
    "0.000", "0.000", "72.125";
    "79.500", "0.000", "0.000";
    "0.000", "74.344", "0.000"
) $],
  [next $x_q$, $k = 6$\ $ mat(
    "0", "0", "72";
    "80", "0", "0";
    "0", "74", "0"
) $])
The value $79.5$ is an exact tie. Half to even rounds it to $80$ (even), not to $79$.

== Workflow B: requantize the accumulator, integer activation

*Absmax and shift.* $"absmax"(|"acc"|) = 5088$. In binary it has $13$ bits ($4096 <= 5088 < 8192$). The shift is $s = 13 - 7 = 6$.
Check that $s$ is the smallest one that works: with $s = 5$, $5088 / 32 = 159 > 127$, too big. With $s = 6$, $5088 / 64 = 79.5 -> 80 <= 127$. So *$s = 6$* and *$k_"out" = 12 - 6 = 6$*.

*Rounded shift.* $q_"out" = "round"_("half-even")("acc" / 64)$, remainder $r = "acc" - 64 dot q_"out"$:
#grid(columns: (1fr, 1fr), align: center,
  [$q_"out"$ (int8)\ $ mat(
    "−45", "−58", "72";
    "80", "−31", "−13";
    "−28", "74", "−2"
) $],
  [remainder $r$, bound $|r| <= 32$\ $ mat(
    "16", "−16", "8";
    "−32", "18", "9";
    "2", "22", "−9"
) $])
The tie shows up here as $r = -32$ for the entry $5088 = 80 dot 64 - 32$: it rounds to the even value $80$.

*Integer activation.* ReLU on the integers, $q = max(0, q_"out")$, scale unchanged ($k = 6$):
#align(center)[$ mat(
    "0", "0", "72";
    "80", "0", "0";
    "0", "74", "0"
) $]
This is the next layer's $x_q$ with $k = 6$. In this example it matches workflow A exactly (same matrix). In general the two can differ by $plus.minus 1$ in an entry, because A ends with its own choice of $k$ and rounding on the post-activation tensor, while B rounds once from the accumulator. For comparison, $q dot 2^(-6)$ gives $1.125, 1.25, 1.156$ against the float reference $1.125, 1.25, 1.165$.

#note[
  Summary of the numbers: $k_x = 5$, $k_w = 7$, $"acc"$ scale $12$, absmax of acc $= 5088$ (13 bits), $s = 6$, $k_"out" = 6$. In A, the dequantize and quantize steps go through float; in B they are replaced by one integer shift.
]

= What the prover must check

All values are mapped into M31 (residues mod $p = 2^31 - 1$). For that to be injective, all quantized values must satisfy $|q| <= Q <= 2^30 - 1$. The checks below are constraints on the trace columns. The host (or device) only has to produce the witness (`q`, `r`, `k`, `s`, the index of the absmax element). It is not trusted.

== Quantize (both workflows)

+ *Rounding.* $x dot 2^k = q + r$ with $|r| <= 1/2$, and if $|r| = 1/2$ then $q$ is even. $x$ enters as a significand and an exponent (a float is not a field element), so $x dot 2^k$ becomes an integer multiplication with a shift. $2^k$ comes from a lookup table.
+ *Range.* $|q| <= Q$ (a range check, e.g. by bit decomposition or lookup limbs). For an honest $k$ the clamp never fires.
+ *Maximality of $k$.* There is an element $i^*$ with $|x_(i^*)| dot 2^(k+1) > Q$, unless $k = 127$. Without it a cheating prover could pick a very small $k$ (everything rounds to $0$) or a smaller $k$ that loses precision, and the proof would still pass the first two checks.

== Matmul (both)

+ $"acc"_(i j) = sum_t x_(q,i t) dot W_(q,t j)$ as an exact field equation. It is only equal to the integer result if no wrap-around happens: $n dot Q_x dot Q_w < p / 2$. For $n = 3$ and 8 bit that is $48387$, far below $2^30$. For int16 $times$ int16 it already fails for realistic $n$, so the matmul must limit the bit widths or use limbs.
+ The scale is bookkeeping: $k_"acc" = k_x + k_w$ must be the same public value used in the next step.

== Workflow A, after the matmul

+ *Dequantize:* is a relabeling in the integer world ($y = "acc" dot 2^(-k_"acc")$). In float it is exact only when $|"acc"| < 2^24$, so that bound is a constraint.
+ *Float activation:* ReLU needs the sign bit and a select. Any other float activation (GELU, exp, sigmoid) has to be proven bit-exactly in float arithmetic, or approximated by a table with an error bound. Neither is cheap in M31.
+ *Re-quantize:* the quantize checks from above, on the activation output.

== Workflow B, after the matmul

+ *Requantize rounding:* $"acc" = q dot 2^s + r$ with $|r| <= 2^(s-1)$, ties to even ($r = -2^(s-1)$ or $+2^(s-1)$ only if $q$ is even). $2^s$ is a lookup.
+ *Range:* $|q| <= Q$.
+ *Maximality of $s$:* for the absmax element $i^*$ of the accumulator, $"round"(abs("acc"_(i^*)) / 2^(s-1)) > Q$ (for $s = 0$ no check). This is the integer version of condition 3 above and it needs the same witness: the index of the absmax element. The other elements are covered by the range check, since none of them can be larger than $|"acc"_(i^*)|$.
+ *Scale:* $k_"out" = k_"acc" - s$.
+ *Integer activation:* ReLU is $q' = q$ if $q >= 0$ else $0$ (sign bit plus a select). Any other activation is a table lookup: $(q, q')$ must be a row of a table with $2^8$ entries (preprocessed column), which stwo supports directly. The table itself is public and fixed.

== Linking the steps

The output column of one step is the input column of the next (same trace cells), with the exponents $k_x, k_w, s, k_"out"$ as public or committed values. Otherwise a prover could prove each step on different tensors.

#pagebreak(weak: true)
= Tradeoffs

#table(columns: (auto, 1fr, 1fr), inset: 6pt, align: (left, left, left), stroke: 0.5pt + luma(170),
  [], [*A: float between layers*], [*B: integer between layers*],
  [Ops after matmul], [dequantize, float activation, quantize (two passes over data)], [requantize (two passes over the int32 accumulator), integer activation],
  [Rounding steps], [matmul is exact; one rounding when quantizing the activation], [one rounding in the requantize; activation tables add their own quantization],
  [Proving cost], [High: float semantics per element, float activation, plus the full quantize witness], [Lower: integer shifts, lookups, range checks only],
  [Activations], [Any float function, but proving it exactly is hard], [ReLU is trivial; others must be a table over int8 or int16 inputs (one lookup argument)],
  [Accuracy], [Best: the activation sees the full 13 bit accumulator in float], [The activation sees only 8 bits after requantize; a table has a fixed error],
  [Device work], [Existing `pot_quantize` can be reused; needs `dequantize` and float activation], [Needs `pot_requantize` (int32 absmax, SFPU int ops); activation is cheap],
  [Overflow risk], [int32 accumulator, same limit in both], [int32 accumulator, same limit in both],
  [Developer inspection], [Float values visible at every layer], [Floats only by dequantizing on demand],
)

*Recommendation.* B is the better fit for a sound proof: every step is an integer relation with a small witness, and the activation is a lookup. A is useful as a reference to measure how much accuracy B loses, and it is the natural way to check a layer by eye. Both start with the same quantize and matmul, so `pot_quantize` and the matmul are shared, and the separate `pot_requantize_per_tensor` and `dequantize` operations let a developer check each step on its own.

*Open points for B.* (1) The post-ReLU tensor can have a smaller absmax than the accumulator (if the largest entry was negative), so the next layer's $k_"out"$ is valid but not maximal for the data it sees. Applying ReLU to the int32 accumulator before the requantize avoids this, and is exact. (2) int16 $times$ int16 products overflow the int32 accumulator; decide the supported width pairs first. (3) For a table activation, the table's input scale must be fixed or part of the table's key.

#pagebreak()
= The layer as one operation: `execute_layer`, workflow B

The diagram follows one layer from the float inputs to the next layer's input, with the numbers of the example above. The five integer tensors in the last box are the outputs of the device operations. The columns of the proof trace are built by the host from them (next part).

#grid(columns: (1fr, 1fr), column-gutter: 8pt,
[#block(breakable: false, width: 100%, stroke: 0.6pt, radius: 4pt, inset: 7pt, fill: luma(248))[
  #text(weight: "bold", size: 9.5pt)[Input tensor `x` (float)]
  #v(2pt)
  #text(size: 8.5pt)[$ mat(
    "0.50", "−1.25", "0.75";
    "2.00", "0.10", "−0.60";
    "−0.30", "1.10", "0.90"
) $]
]
],
[#block(breakable: false, width: 100%, stroke: 0.6pt, radius: 4pt, inset: 7pt, fill: luma(248))[
  #text(weight: "bold", size: 9.5pt)[Weight tensor `W` (float)]
  #v(2pt)
  #text(size: 8.5pt)[$ mat(
    "0.40", "−0.20", "0.10";
    "0.30", "0.80", "−0.50";
    "−0.70", "0.25", "0.60"
) $]
]
])
#align(center)[#text(size: 15pt)[#sym.arrow.b]]
#grid(columns: (1fr, 1fr), column-gutter: 8pt,
[#block(breakable: false, width: 100%, stroke: 0.6pt, radius: 4pt, inset: 7pt, fill: luma(248))[
  #text(weight: "bold", size: 9.5pt)[`pot_quantize_per_tensor(x)`]
  #v(2pt)
  absmax $= 2.00$ \ $k = 5$ because $2.00 dot 2^5 = 64 <= 127$ and $2.00 dot 2^6 = 128 > 127$ \ scaler $2^5 = 32$ \ $x_q = "round"(x dot 32)$#text(size: 8.5pt)[$ mat(
    "16", "−40", "24";
    "64", "3", "−19";
    "−10", "35", "29"
) $]
]
],
[#block(breakable: false, width: 100%, stroke: 0.6pt, radius: 4pt, inset: 7pt, fill: luma(248))[
  #text(weight: "bold", size: 9.5pt)[`pot_quantize_per_tensor(W)`]
  #v(2pt)
  absmax $= 0.80$ \ $k = 7$ because $0.80 dot 2^7 = 102.4 <= 127$ and $0.80 dot 2^8 = 204.8 > 127$ \ scaler $2^7 = 128$ \ $W_q = "round"(W dot 128)$#text(size: 8.5pt)[$ mat(
    "51", "−26", "13";
    "38", "102", "−64";
    "−90", "32", "77"
) $]
]
])
#align(center)[#text(size: 15pt)[#sym.arrow.b]]
#block(breakable: false, width: 100%, stroke: 0.6pt, radius: 4pt, inset: 7pt, fill: luma(248))[
  #text(weight: "bold", size: 9.5pt)[`matmul` (int32 accumulator)]
  #v(2pt)
  $"acc" = x_q W_q$, scale $k_x + k_w = 12$. Example: entry $(0,0) = 16 dot 51 + (-40) dot 38 + 24 dot (-90) = -2864$. \ largest $|"acc"| = 5088$ (13 bits)#text(size: 8.5pt)[$ mat(
    "−2864", "−3728", "4616";
    "5088", "−1966", "−823";
    "−1790", "4758", "−137"
) $]
]
#align(center)[#text(size: 15pt)[#sym.arrow.b]]
#block(breakable: false, width: 100%, stroke: 0.6pt, radius: 4pt, inset: 7pt, fill: luma(248))[
  #text(weight: "bold", size: 9.5pt)[`pot_requantize_per_tensor` (dynamic)]
  #v(2pt)
  absmax $= 5088$, bit length $13$, $s = 13 - 7 = 6$. Check: $5088 / 2^5 = 159 > 127$, so $s = 5$ is too small. $k_"out" = 12 - 6 = 6$, scaler $2^(-6)$. \ $q_"out" = "round"("acc" / 64)$, ties to even ($5088 / 64 = 79.5 arrow.r 80$)#text(size: 8.5pt)[$ mat(
    "−45", "−58", "72";
    "80", "−31", "−13";
    "−28", "74", "−2"
) $]
]
#align(center)[#text(size: 15pt)[#sym.arrow.b]]
#block(breakable: false, width: 100%, stroke: 0.6pt, radius: 4pt, inset: 7pt, fill: luma(248))[
  #text(weight: "bold", size: 9.5pt)[integer activation: ReLU]
  #v(2pt)
  $"out" = max(0, q_"out")$, scale unchanged, $k = 6$#text(size: 8.5pt)[$ mat(
    "0", "0", "72";
    "80", "0", "0";
    "0", "74", "0"
) $]
]
#align(center)[#text(size: 15pt)[#sym.arrow.b]]
#block(breakable: false, width: 100%, stroke: 0.6pt, radius: 4pt, inset: 7pt, fill: rgb("#eef7ee"))[
  #text(weight: "bold", size: 9.5pt)[Outputs of the device operations (all integer, the host builds the trace from them)]
  #v(2pt)
  #table(columns: (auto, 1fr, auto), inset: 4pt, stroke: 0.4pt + luma(170), align: (left, left, left),
  [*tensor*], [*values*], [*scalers*],
  [`x_q` int8], [$((16,-40,24),(64,3,-19),(-10,35,29))$], [$32$ ($k_x = 5$)],
  [`W_q` int8], [$((51,-26,13),(38,102,-64),(-90,32,77))$], [$128$ ($k_w = 7$)],
  [`acc` int32], [$((-2864,-3728,4616),(5088,-1966,-823),(-1790,4758,-137))$], [scale $2^(-12)$],
  [`q_out` int8], [$((-45,-58,72),(80,-31,-13),(-28,74,-2))$], [$2^(-6)$ ($s = 6$)],
  [`out` int8 (ReLU)], [$((0,0,72),(80,0,0),(0,74,0))$], [$k_"out" = 6$]
)
]

= Trace the host builds and its constraints

The host adds the remainders, the significand and exponent of each float, the flags and the lookup values to the five tensors above. All values are residues mod $p = 2^31 - 1$ (a negative $v$ is stored as $p - |v|$). The tables show the signed values for readability.

== C1: quantize `x` (9 rows)

A float is split into a signed 24-bit significand and an exponent, $x = X dot 2^E$, and $u = -(E + k)$, so that $X = q dot 2^(u) + R$.

#align(center)[#text(size: 7.5pt)[#table(columns: 9, inset: 3.5pt, stroke: 0.4pt + luma(170), align: center,
  [*i*], [*x*], [*X*], [*E*], [*u*], [*q*], [*R*], [*tie*], [*is_max*],
  [0], [0.5], [8388608], [−24], [19], [16], [0], [0], [0],
  [1], [−1.25], [−10485760], [−23], [18], [−40], [0], [0], [0],
  [2], [0.75], [12582912], [−24], [19], [24], [0], [0], [0],
  [3], [2], [8388608], [−22], [17], [64], [0], [0], [1],
  [4], [0.1], [13421773], [−27], [22], [3], [838861], [0], [0],
  [5], [−0.6], [−10066330], [−24], [19], [−19], [−104858], [0], [0],
  [6], [−0.3], [−10066330], [−25], [20], [−10], [419430], [0], [0],
  [7], [1.1], [9227469], [−23], [18], [35], [52429], [0], [0],
  [8], [0.9], [15099494], [−24], [19], [29], [−104858], [0], [0]
)]]

Constraints (per row unless stated):
+ $X = q dot 2^(u) + R$, with $2^(u)$ read from the pow2 lookup table (preprocessed column).
+ $|R| <= 2^(u-1)$: range check of $2^(u-1) - R$ and $2^(u-1) + R$ (both $>= 0$).
+ Tie: $"tie" dot ("tie" - 1) = 0$, and if $|R| = 2^(u-1)$ then $"tie" = 1$ and $q$ is even ($q = 2h$, $h$ a witness column).
+ $|q| <= 127$ (range check).
+ Maximality (the row with $"is_max" = 1$, one per tensor): $"is_max"$ is boolean, $sum "is_max" = 1$, and for that row $|X| dot 2^(E+k+1) > 127$. Here: $2.0 dot 2^6 = 128 > 127$. Skipped when $k = 127$.
+ Public value: $k = 5$.

== C2: quantize `W` (9 rows)

Same columns and the same constraints, with $k = 7$. $q$ column: $51, -26, 13, 38, 102, -64, -90, 32, 77$. The maximality row is element 4 ($0.8 dot 2^8 = 204.8 > 127$).

== C3: matmul (one row per output element, 9 rows)

#align(center)[#text(size: 7.5pt)[#table(columns: 9, inset: 3.5pt, stroke: 0.4pt + luma(170), align: center,
  [*(i,j)*], [*a0*], [*b0*], [*a1*], [*b1*], [*a2*], [*b2*], [*acc*], [*acc mod p*],
  [0,0], [16], [51], [−40], [38], [24], [−90], [−2864], [2147480783],
  [0,1], [16], [−26], [−40], [102], [24], [32], [−3728], [2147479919],
  [0,2], [16], [13], [−40], [−64], [24], [77], [4616], [4616],
  [1,0], [64], [51], [3], [38], [−19], [−90], [5088], [5088],
  [1,1], [64], [−26], [3], [102], [−19], [32], [−1966], [2147481681],
  [1,2], [64], [13], [3], [−64], [−19], [77], [−823], [2147482824],
  [2,0], [−10], [51], [35], [38], [29], [−90], [−1790], [2147481857],
  [2,1], [−10], [−26], [35], [102], [29], [32], [4758], [4758],
  [2,2], [−10], [13], [35], [−64], [29], [77], [−137], [2147483510]
)]]

Constraints:
+ $"acc" = a_0 b_0 + a_1 b_1 + a_2 b_2$ in the field. It equals the integer result only if nothing wraps: $n dot Q_x dot Q_w = 3 dot 127 dot 127 = 48387 < 2^30$.
+ Links: $a_t$ is the $q$ column of C1 at $(i, t)$ and $b_t$ is the $q$ column of C2 at $(t, j)$ (lookup into the other component, same values).
+ Public values: $k_"acc" = k_x + k_w = 12$.

== C4: requantize (9 rows plus one maximality row)

#align(center)[#text(size: 7.5pt)[#table(columns: 8, inset: 3.5pt, stroke: 0.4pt + luma(170), align: center,
  [*i*], [*acc*], [*q_out*], [*R*], [*R + 32*], [*tie*], [*is_max*], [*q_out mod p*],
  [0], [−2864], [−45], [16], [48], [0], [0], [2147483602],
  [1], [−3728], [−58], [−16], [16], [0], [0], [2147483589],
  [2], [4616], [72], [8], [40], [0], [0], [72],
  [3], [5088], [80], [−32], [0], [1], [1], [80],
  [4], [−1966], [−31], [18], [50], [0], [0], [2147483616],
  [5], [−823], [−13], [9], [41], [0], [0], [2147483634],
  [6], [−1790], [−28], [2], [34], [0], [0], [2147483619],
  [7], [4758], [74], [22], [54], [0], [0], [74],
  [8], [−137], [−2], [−9], [23], [0], [0], [2147483645]
)]]

Constraints:
+ $"acc" = q_"out" dot 2^s + R$ with $s = 6$ and $2^s = 64$ from the lookup.
+ $|R| <= 2^(s-1) = 32$: range check of $R + 32$ and $32 - R$ ($R + 32$ is in the table above).
+ Tie: if $|R| = 32$ then $q_"out"$ is even (row 3: $R = -32$, $q_"out" = 80$).
+ $|q_"out"| <= 127$.
+ Maximality of $s$ (extra row for the `is_max` element, row 3): $|"acc"| = 5088 = 159 dot 2^5 + 0$, and $159 >= 128$, so $s - 1 = 5$ would have overflowed. Skipped when $s = 0$.
+ $k_"out" = k_"acc" - s = 12 - 6 = 6$ (public).
+ Link: `acc` is the $"acc"$ column of C3.

== C5: integer ReLU (9 rows)

#align(center)[#text(size: 7.5pt)[#table(columns: 5, inset: 3.5pt, stroke: 0.4pt + luma(170), align: center,
  [*i*], [*q*], [*neg*], [*a*], [*out*],
  [0], [−45], [1], [45], [0],
  [1], [−58], [1], [58], [0],
  [2], [72], [0], [72], [72],
  [3], [80], [0], [80], [80],
  [4], [−31], [1], [31], [0],
  [5], [−13], [1], [13], [0],
  [6], [−28], [1], [28], [0],
  [7], [74], [0], [74], [74],
  [8], [−2], [1], [2], [0]
)]]

Constraints:
+ $"neg" dot ("neg" - 1) = 0$.
+ $q = (1 - 2 dot "neg") dot a$ and $0 <= a <= 127$ (range check). The sign is then determined by $a$ and $q$, so a cheating prover cannot flip it.
+ $"out" = (1 - "neg") dot a$.
+ Link: $q$ is the $q_"out"$ column of C4. The $"out"$ column is the $x_q$ column of the next layer (the same cells, not a copy).

== Tables that are not trace rows

- *pow2 lookup* (preprocessed): $(e, 2^e)$ for the exponents that occur, used by C1, C2 and C4.
- *range-check tables* (preprocessed): small limbs (for example 8 or 11 bit) used to prove the bounds above.
- *Public inputs and outputs:* $k_x = 5$, $k_w = 7$, $s = 6$, $k_"out" = 6$, the commitment to `x` and `W` (their $X$ and $E$ columns), and the commitment to the `out` column.

== Final result of the host

#note[
  Layer output: $"out" = ((0,0,72),(80,0,0),(0,74,0))$ with $k_"out" = 6$. As floats, $"out" dot 2^(-6) = ((0,0,1.125),(1.25,0,0),(0,1.15625,0))$. All residues of `out` are the integers themselves, since none is negative. Trace size for this layer: 5 components of 9 rows each (plus the maximality rows), about 8 columns per quantize row, 9 for the matmul and 8 for the requantize.
]
