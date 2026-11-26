#set page(paper: "a4", margin: (x: 1.7cm, y: 1.8cm), numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.")
#show raw: set text(size: 9pt)
#let note(t) = block(fill: rgb("#fff7e0"), stroke: 0.5pt + rgb("#d9b44a"), inset: 7pt, radius: 3pt, width: 100%, t)
#let node(t, fill: luma(245)) = box(stroke: 0.6pt, inset: 5pt, radius: 3pt, fill: fill, align(center, text(size: 8pt, t)))

#align(center)[
  #text(size: 17pt, weight: "bold")[Smaller matmul traces for the prover] \
  #v(2pt)
  #text(size: 10pt)[Why the naive trace of $x_q W_q$ is too big, two ways to shrink it (random check, sumcheck), and what sumcheck and GKR are.]
]

= The problem

The matmul of a layer is $"acc"_(i j) = sum_(k) x_(i k) w_(k j)$ for an $M times K$ matrix $x$ and a $K times N$ matrix $w$. Written as a trace in the straightforward way, every single product $x_(i k) w_(k j)$ is a constraint: one row (or one group of columns) per product. That is $M N K$ products. The prover must keep the trace columns in memory, extend them (the low-degree extension, a blow-up of 2 or more) and commit to them, so its memory grows with the trace size.

The trace of everything else in a layer (quantize, requantize, activation) has one row per tensor element: $M K + K N + M N$ rows. The matmul is the only part that is cubic. The two approaches below make it linear too: they check the matmul against the three tensors $x_q$, $w_q$, $"acc"$ without writing down the $M N K$ products.

#table(columns: (auto, auto, auto, auto, auto, auto), inset: 4pt, stroke: 0.4pt + luma(170), align: center,
  [*M x K x N*], [*products* $M N K$], [*linear* $M K + K N + M N$], [*ratio*], [*naive trace*], [*linear trace*],
  [2 x 2 x 2], [8], [12], [0.667], [96 B], [432 B],
  [128 x 512 x 128], [8 388 608], [147 456], [56.9], [101 MB], [5.31 MB],
  [1024 x 1024 x 1024], [1 073 741 824], [3 145 728], [341], [12.9 GB], [113 MB],
  [4096 x 4096 x 4096], [68 719 476 736], [50 331 648], [1.37e+03], [825 GB], [1.81 GB],
  [1 x 4096 x 4096], [16 777 216], [16 785 408], [1], [201 MB], [604 MB]
)
#text(size: 8.5pt)[Assumptions of the memory columns (rough): the naive trace has 3 cells per product (the two factors and a running sum) of 4 bytes, before the low-degree extension; the linear one has about 9 cells per multiplication, because the random value it multiplies by is an element of a degree-4 extension of the field (see below). The linear approaches gain about $min(M, N, K)$: none for a matrix times a vector (last row, $M = 1$), a lot for square layers.]

= Four terms you need

*Trace and AIR.* The prover fills a table (the trace). An AIR is a set of polynomial equations that every row, and every pair of neighbouring rows, must satisfy. stwo proves that such a table exists.

*Challenge (Fiat-Shamir).* A random number the verifier would send after seeing the prover's commitments. The prover cannot know it in advance, so it cannot build a cheating table that happens to pass for that number. In a non-interactive proof it is derived by hashing the commitments. With the field M31 ($p = 2^31 - 1$) a single challenge only gives a cheating chance of about $2^(-31)$, too large for a proof system; stwo therefore draws challenges from the degree-4 extension QM31 (about $2^(-124)$).

*Multilinear extension (MLE).* A matrix with $2^a$ rows and $2^b$ columns is a function on bits, $x(i, k)$ for $i in {0,1}^a$, $k in {0,1}^b$. Its MLE $tilde(x)(u, t)$ is the unique polynomial of degree at most 1 in each variable that agrees with $x$ on the bits and can be evaluated at any field elements $u, t$. For a $2 times 2$ matrix: $tilde(x)(u,t) = (1-u)(1-t) x_(00) + (1-u) t x_(01) + u (1-t) x_(10) + u t x_(11)$. Evaluating at random points is a random linear combination of all entries, with weights $"eq"(u, i) = u i + (1-u)(1-i)$ per index bit.

*LogUp.* A lookup argument: it proves that the multiset of values in a column of one table equals what is read from another table, so a value is stored once and the same cell is used by every component (for example, the $q$ column of the quantize table is the $x$ column of the matmul table). It is also a sum, which is also why LogUp can be proven with a GKR-style protocol (below; I believe stwo has such a prover but have not rechecked it).

= Example data

The top-left $2 times 2$ corner of the $x_q$ and $W_q$ of the earlier example. All values are signed integers; in the trace they are residues mod $p$ (a negative $v$ is stored as $p - |v|$), and every equation below holds in the field.

#grid(columns: (1fr, 1fr, 1fr), align: center,
  [$x_q$\ $ mat(
    "16", "−40";
    "64", "3"
) $],
  [$W_q$\ $ mat(
    "51", "−26";
    "38", "102"
) $],
  [$"acc" = x_q W_q$\ $ mat(
    "−704", "−4496";
    "3378", "−1358"
) $])

The accumulator entry $(0,0)$ is $16 dot 51 + (-40)(38) = -704$. Naively that is $2 dot 2 dot 2 = 8$ products to constrain.

= Approach A: a random check of the matmul (Freivalds)

*Idea.* If $"acc" = x W$ then for every vector $r$ with $N$ entries, $"acc" r = x (W r)$. The right side needs only two matrix-vector products instead of a matrix-matrix product. If $"acc" != x W$, a random $r$ makes the two sides differ with probability at least $1 - 1 slash abs(F)$ (a nonzero row of $"acc" - x W$ has a nonzero dot product with a random vector). The prover commits $x_q$, $W_q$, $"acc"$ first; the challenge $r$ comes from the hash of those commitments; then the table below is filled.

*Example.* A stand-in challenge $r = (3, 5)$ (in the real protocol: random QM31 values).
$ v = W r = (51 dot 3 + (-26) dot 5,\ 38 dot 3 + 102 dot 5) = (23, 624) $
$ u = x v = (16 dot 23 + (-40) dot 624, 64 dot 23 + 3 dot 624) = (−24592, 3344) $
$ "acc" r = (-704 dot 3 + (-4496) dot 5,\ 3378 dot 3 + (-1358) dot 5) = (−24592, 3344) $
Both sides agree, $u = "acc" r$. If the prover had cheated with $"acc"_(00)$ off by one, the first entry would differ by $r_0 = 3$ and the check fails (it passes only when the challenge component happens to be $0$).

*Tables.* Two components, plus lookups into the quantize tables.

#text(weight: "bold")[Table V: $v = W r$, one row per $k$ (K rows)]
#align(center)[#text(size: 8.5pt)[#table(columns: 6, inset: 4pt, stroke: 0.4pt + luma(170), align: center,
  [*k*], [*$w_(k,0)$*], [*$w_(k,1)$*], [*$r_0$*], [*$r_1$*], [*$v_k$*],
  [0], [51], [−26], [3], [5], [23],
  [1], [38], [102], [3], [5], [624]
)]]

Constraint: $v_k = w_(k,0) r_0 + w_(k,1) r_1$ (one multiplication per element of $W$). The $w$ cells are looked up from the quantize component of $W$; $r$ is the challenge (public).

#text(weight: "bold")[Table U: $u = x v$ and $"acc" r$, one row per $i$ (M rows)]
#align(center)[#text(size: 8pt)[#table(columns: 9, inset: 4pt, stroke: 0.4pt + luma(170), align: center,
  [*i*], [*$x_(i,0)$*], [*$x_(i,1)$*], [*$v_0$*], [*$v_1$*], [*$u_i$*], [*$"acc"_(i,0)$*], [*$"acc"_(i,1)$*], [*$s_i$*],
  [0], [16], [−40], [23], [624], [−24592], [−704], [−4496], [−24592],
  [1], [64], [3], [23], [624], [3344], [3378], [−1358], [3344]
)]]

Constraints: $u_i = x_(i,0) v_0 + x_(i,1) v_1$ and $s_i = "acc"_(i,0) r_0 + "acc"_(i,1) r_1$ and *$u_i = s_i$*. The $v$ cells are looked up from table V (LogUp), the $x$ cells from the quantize component of $x$, the $"acc"$ cells from the matmul output that the next component (the activation) also reads.

*Size.* $K N + M K + M N$ multiplications, no cubic term, nothing but the three tensors and two small vectors committed. Each multiplication is an M31 value times a QM31 challenge (4 M31 multiplications). *Soundness* is about $2^(-124)$ per check with QM31 challenges. *Condition:* the field equation equals the integer equation only if no product sum wraps: $K dot 127^2 < p slash 2$, i.e. $K <= 66000$.

= Approach B: sumcheck

*Sumcheck in one paragraph.* The prover wants to convince the verifier that $H = sum_(b in {0,1}^n) g(b)$ for a polynomial $g$ that the verifier can evaluate at one point. Round 1: the prover sends the univariate polynomial $s_1(t) = sum_(b_2, ..., b_n) g(t, b_2, ..., b_n)$. The verifier checks $s_1(0) + s_1(1) = H$, picks a random $r_1$, and the claim becomes "$s_1(r_1) = sum_(b_2, ..., b_n) g(r_1, b_2, ..., b_n)$": the same kind of claim with one variable fewer. After $n$ rounds, a claim about the single value $g(r_1, ..., r_n)$ is left, which the verifier checks by evaluating $g$ once. The verifier does $O(n)$ work instead of $2^n$ and a cheating prover is caught with probability about $n d slash abs(F)$ ($d$ = degree of $g$ in each variable).

*Sumcheck for a matmul.* With the MLEs, for random $u_a$ (row index) and $u_c$ (column index):
$ tilde("acc")(u_a, u_c) = sum_(b in {0,1}^(log K)) tilde(x)(u_a, b) dot tilde(w)(b, u_c). $
The left side is one evaluation of $"acc"$. The sum has $K$ terms and is a sumcheck with $log K$ rounds, each a degree-2 polynomial. The end of the sumcheck leaves one claim about $tilde(x)$ and one about $tilde(w)$ at random points; those three evaluations are the only places where the full tensors are touched.

*Example* ($K = 2$, so $n = 1$ variable $b$). Challenges $u_a = 3$, $u_c = 5$.
- $f(b) = tilde(x)(3, b)$: $(1-3) dot "row"_0 + 3 dot "row"_1 = (160, 89)$ for $b = 0, 1$.
- $g(b) = tilde(w)(b, 5)$: $(1-5) dot "col"_0 + 5 dot "col"_1 = (−334, 358)$.
- Claim: $H = tilde("acc")(3,5) = (1-3)(1-5)(-704) + (1-3)5(-4496) + 3(1-5)3378 + 3 dot 5 (-1358) = −21578$.
- Check $H = f(0) g(0) + f(1) g(1) = −53440 + 31862 = −21578$.
- Round 1: the prover sends $s(t) = f(t) g(t)$ through $s(0) = −53440, s(1) = 31862, s(2) = 18900$. The verifier checks $s(0) + s(1) = H$, draws $r_b = 7$, and the claim becomes $s(7) = f(7) g(7) = −337 dot 4510 = −1519870$.
- Left to prove: $tilde(x)(3, 7) = −337$ and $tilde(w)(7, 5) = 4510$.

*Tables.* A tiny sumcheck table, and one evaluation table per tensor.

#text(weight: "bold")[Table S: the sumcheck rounds ($log K$ rows)]
#align(center)[#text(size: 8.5pt)[#table(columns: 7, inset: 4pt, stroke: 0.4pt + luma(170), align: center,
  [*round*], [*claim in*], [*$s(0)$*], [*$s(1)$*], [*$s(2)$*], [*challenge*], [*claim out $= s("r")$*],
  [1], [−21578], [−53440], [31862], [18900], [7], [−1519870]
)]]

Constraints per row: $s(0) + s(1) = "claim in"$; $"claim out" = s("challenge")$ by quadratic interpolation; the next row's claim in is this row's claim out. The last claim out equals the product of the two evaluations below.

#text(weight: "bold")[Table EX: $tilde(x)(3, 7) = sum_(i,k) "eq" dot x_(i k)$, one row per element of $x$ ($M K$ rows)]
#align(center)[#text(size: 8.5pt)[#table(columns: 7, inset: 4pt, stroke: 0.4pt + luma(170), align: center,
  [*i*], [*k*], [*$"eq"_a(i)$*], [*$"eq"_b(k)$*], [*weight $w$*], [*$x_(i k)$*], [*running sum*],
  [0], [0], [−2], [−6], [12], [16], [192],
  [0], [1], [−2], [7], [−14], [−40], [752],
  [1], [0], [3], [−6], [−18], [64], [−400],
  [1], [1], [3], [7], [21], [3], [−337]
)]]

Constraints: $w = "eq"_a(i) dot "eq"_b(k)$, running sum $+=$ $w dot x_(i k)$. The factors $"eq"(u, i)$ depend on the challenge and the index only (they are computed incrementally from the bits of the index, or looked up). The last running sum is $−337$, which must equal $f(7)$ from table S.

#text(weight: "bold")[Table EW: $tilde(w)(7, 5)$, one row per element of $W$ ($K N$ rows)]
#align(center)[#text(size: 8.5pt)[#table(columns: 7, inset: 4pt, stroke: 0.4pt + luma(170), align: center,
  [*k*], [*j*], [*$"eq"_b(k)$*], [*$"eq"_c(j)$*], [*weight $w$*], [*$w_(k j)$*], [*running sum*],
  [0], [0], [−6], [−4], [24], [51], [1224],
  [0], [1], [−6], [5], [−30], [−26], [2004],
  [1], [0], [7], [−4], [−28], [38], [940],
  [1], [1], [7], [5], [35], [102], [4510]
)]]

Last running sum $4510$ $= g(7)$. Table EA is the same for the accumulator, with the weights $"eq"_a(i) "eq"_c(j)$ and last sum $−21578$ $= H$.

*Size.* $M K + K N + M N + log K$ rows, each with one multiplication. Like approach A, no cubic term. The sumcheck itself costs the verifier $O(log K)$ and the proof only $3 log K$ field elements.

= Approach A or approach B?

#table(columns: (auto, 1fr, 1fr), inset: 5pt, stroke: 0.4pt + luma(170), align: (left, left, left),
  [], [*A: random check (Freivalds)*], [*B: sumcheck*],
  [Rows], [$K N + M K + M N$], [$M K + K N + M N + log K$],
  [What the AIR does], [two matrix-vector products with a QM31 vector; compares two vectors], [three evaluations of MLEs (one multiplication per element) plus $log K$ small rounds],
  [Challenges], [one vector $r$ with $N$ entries], [$log M + log N + log K$ field elements],
  [Fits stwo], [a plain AIR with a LogUp link between the tables], [an AIR for the evaluation tables and the round table; the sumcheck itself is a few rows],
  [Simplicity], [simplest: no polynomials, no interpolation], [more pieces, but they generalize (below)],
  [Strength], [easy to audit, soundness from one challenge], [chains across layers (GKR), so intermediate tensors need not be committed]
)

For a single layer I would start with A: it is the smaller change, and the trace is already linear. Sumcheck pays off when you want to prove several layers without committing to every intermediate tensor, which is what GKR does.

= GKR in one page

*What it is.* The prover shows that a layered circuit was evaluated correctly. The circuit has layers $L_0$ (inputs) to $L_d$ (outputs). GKR starts from a claim about the output layer, usually "the output MLE at a random point equals $c$", and uses one sumcheck per layer to turn the claim about layer $L_(i+1)$ into a claim about layer $L_i$: the claimed value at a random point of the layer above is a sum over the values of the layer below combined by the gates of the layer (for a matmul layer: the identity of approach B). After the last sumcheck only a claim about the inputs is left, and the verifier checks it against the committed inputs. Only the inputs and the outputs are ever committed.

#align(center)[
  #stack(dir: ttb, spacing: 4pt,
    node(fill: rgb("#e6f7e6"))[output layer $L_d$: claim "$tilde(L_d)(r) = c$" (from the committed output)],
    text(size: 11pt)[#sym.arrow.b #h(4pt) sumcheck over the gates of layer $d$],
    node[layer $L_(d-1)$: new claim "$tilde(L_(d-1))(r') = c'$"],
    text(size: 11pt)[#sym.arrow.b #h(4pt) sumcheck],
    node[...],
    text(size: 11pt)[#sym.arrow.b #h(4pt) sumcheck],
    node(fill: rgb("#e8f0ff"))[inputs $L_0$: claim checked against the committed input tensors])
]

*Why it helps here.* For a network of many layers, the prover never has to commit to (or keep the LDE of) the activations in the middle; it only needs them to run the sumchecks. The memory of the proof is then dominated by the input and weights, not by the depth. The nonlinear parts of a quantized layer (rounding, clamp, ReLU, requantize) are not sums of products: a GKR proof handles them with lookup arguments, which is the use of GKR in stwo: its LogUp lookups are proven as a sum, by a GKR-style protocol, instead of committing to the whole fraction columns (I have not checked the details of how much of this is available in stwo's API, so confirm that before relying on it).

= What the op has to emit

Both approaches need the three tensors as the prover's witness: $x_q$, $W_q$ and the raw accumulator $"acc"$ (before the activation), plus the scalers. Producing them costs the device one write each. What the host must not do is recompute $"acc"$, which is the cubic part. With these tensors the host builds the remaining columns in $O(M K + K N + M N)$ work: the remainders, the tie flags, the absmax witness, the activation bits and the requantize remainders.
