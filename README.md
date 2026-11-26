# tt-zkml
TT kernels for executing ZKML on Tenstorren hardware.


## Setup

```bash
git clone tt-metal ...
export TT_METAL_HOME=$(pwd)/tt-metal

# PUT THIS into the script scripts/setup.sh
cd $TT_METAL_HOME
git submodule update --init --recursive
# TODO add steps for configuring TT_METAL 
./build_metal.sh -c --release --build-all
# for developent:
./build_metal.sh -c -e --enable-fake-kernels-target --release --build-all
```





## Host execution

```bash
rm -rf build
cmake --preset default
cmake --build .build/default
./build/zkml_quant
```

## Running Tets

```bash
ctest --test-dir .build/default --output-on-failure
```

## API documentation

The doc comments in `src/` are turned into HTML by Doxygen (Graphviz adds the include graphs):

```bash
cmake --build .build/default --target docs
xdg-open .build/default/docs/html/index.html
```

The `docs` target exists only when `doxygen` is installed. Write doc comments as `/** @brief ... @param ... */`.

## PoT per-tensor quantization

`zkml::prim::pot_quantize_per_tensor(input, bit_width, output_dtype)` computes the scaler on the
device from the data and returns `(quantized, scaler)`:

```
Q = 2^(bit_width - 1) - 1          bit_width: 8, 16 or 32     (QuantRange::M31: 32 bits capped at 2^30 - 1, see below)
k = max{k <= 127 : absmax(x) * 2^k <= Q}
q = clamp(round_half_even(x * 2^k), -Q, Q)      scaler = 2^k, so x ~= q / scaler
```

The host program runs it for the given bit widths (default: 8 16 32), with FLOAT32 and BFLOAT16 input:

```bash
.build/default/zkml_pot_quant_per_tensor 8 16
```

**Math unit.** The whole operation runs on the SFPU. Pass 1 finds the absmax and leaves the scaler `2^k` in an SFPU register
(LReg7); pass 2 is one fused sweep per tile that multiplies by it, rounds half to even and converts to int32
(`quant_sfpu.hpp`). An FPU multiply (`mul_tiles`) existed as an option and was removed: with the fused sweeps it is no
faster (equal at 16 bits, 4 % slower at 32 bits). The absmax cannot run on the FPU anyway: it has no abs and no min.
Measurements: `benchmarks/quant_tensor_fpu_vs_sfpu/` (`results/lreg7_scaler.md` for the current kernels).

### `execute_layer`: one quantized layer

`zkml::prim::execute_layer(x, w, Activation::ReLU)` (`src/operations/layer_execution/`) runs `out = quantize(relu(quantize(x) @ quantize(w)))`
on one core: both inputs are quantized per tensor to 8-bit sign-magnitude bytes (the quantizer's two passes, the scaler in LReg7), the
FPU multiplies them at HiFi4 into an exact int32 accumulator (the `Int8` circular buffer is a second view of the bytes the packer wrote
through a `UInt8` one), the activation runs on the accumulator, and the accumulator is requantized per tensor. It returns the `UINT8`
output (sign-magnitude bytes) and the three scalers. FLOAT32 inputs, `K <= 1024`, everything resident in L1. Tests:
`test_execute_layer` (every byte against a host model); cycles: `benchmarks/execute_layer/README.md`, which also explains why the
activation does not overlap the matmul yet.

### M31 and the witness for the stwo prover

`zkml::prim::pot_quant_m31(q)` maps a quantized tensor into the field M31 (`p = 2^31 - 1`): INT32 or INT8 in, UINT32
residues `q mod p` in `[0, p)` out (`q >= 0` stays, `q < 0` becomes `p + q`). It is a separate operation, so the quantizer
stays field-agnostic.

**The range of `q` (`QuantRange`).** By default (`QuantRange::Full`) the quantizer uses every bit, `Q = 2^(bits - 1) - 1`
(2^31 - 1 at 32 bits), so the result can be mapped into any field later. That is too large for M31: with `Q = p` the
values `-p`, `0` and `p` are all the field element 0, and a signed value could not be read back from a residue. A
quantization that goes through `pot_quant_m31` must therefore pass `QuantRange::M31`, which caps 32 bits at
`Q = 2^30 - 1 <= (p - 1) / 2` (8 and 16 bits are the same in both). Then the map is injective and
`m <= (p - 1) / 2 ? m : m - p` recovers `q`. The price is one bit of precision at 32 bits. A two-limb
representation (e.g. 16 + 16 bits, each range-checked) would keep all 32 bits; it is not implemented. The device
cannot check the range of the tensor it is given, so `pot_quant_m31` trusts the caller (the test `FullRangeDoesNotFitM31`
shows what goes wrong otherwise).

```cpp
auto [q, scaler] = zkml::prim::pot_quantize_per_tensor(x, BitWidth::Bits32, DataType::INT32, std::nullopt, 16, 2,
                                                       QuantRange::M31);
ttnn::Tensor m31 = zkml::prim::pot_quant_m31(q);
```

**What the prover has to check.** Without any of these a prover can cheat:
1. `x * 2^k = q + remainder` for every element, `|remainder| <= 1/2`, and `|remainder| = 1/2` only if `q` is even (round
   half to even). Without it `q` is not tied to `x`: a prover can pick any `q` (a floor instead of a round to bias
   borderline values, or values unrelated to `x`).
2. `|q| <= Q` for every element, by a range check. In M31 every `q` is just a field element, so without it a prover can claim
   `q = 400` for an int8 weight, or `p - 5` for `-5` and let a later step read a huge positive number.
3. `|x[argmax]| * 2^(k+1) > Q` for one element `argmax`, unless `k = 127` (capped: all-zero or tiny input). Without it `k`
   could be any smaller exponent: a valid but coarser quantization (a prover that wants a different model output picks
   `k - 5` and every weight collapses to 0 or +-1), since `|q| <= Q` alone does not pin `k`. The clamp never fires for an
   honest `k` (`absmax * 2^k <= Q`), so condition 2 replaces the clamp.

**Who builds the witness: the host.** The device returns `q` and `2^k` (and, through `pot_quant_m31`, the residues). The
remainder `x * 2^k - q` and the argmax are functions of `(x, q, k)`, which the host has anyway because `x` and `q` are columns
of the trace. The host derives them: measured, the remainder and the argmax cost 0.1 to 0.4 ns per element on one CPU thread (the other
columns, the sign, significand and exponent of `x`, the shift and the limbs of the range checks, are not built or measured
yet), and a wrong witness cannot be accepted because the constraints check it. Doing it on the device was built and measured (4.4 to 11 times the cost of the quantization itself, mostly an argmax
scan on a RISC-V core) and then removed: the expanded trace is about 20 columns, 80 bytes per element, against 2 bytes of
bf16 input, so it should cross PCIe as raw values, not as columns. It would make sense on the device only if the prover or a
later M31 step ran there.

`tools/stwo-cpu-simd` is the Rust crate that does this on the CPU (a scalar oracle and a `std::simd` version that fills stwo's
own `BaseColumn`), and `CrossCheckWithRust` in `tests/cpp/operations/pot_quantization/test_pot_quant_m31.cpp` compares the
device with it bit for bit (`q`, `k` and the residues, for every width, both ranges and both input dtypes). Same job, same tensor
(quantize, scaler and M31 residues): the device takes 1.6 to 2.1 us per tile at 16 and 32 bits and about 1 us at 8 bits on
one core. One CPU thread takes 0.75 to 0.9 us (1.3 at 8 bits) and 16 threads 0.2 to 0.6 us; the tables are in
`tools/stwo-cpu-simd/README.md` and `tools/quant_tensor_trace_bench/README.md`.

## NoC / DRAM microbenchmark

`tools/noc_bench` measures DRAM latency per bank, core-to-core round-trip and one-way latency, and
DRAM bandwidth for 1..all cores. It is the source of the chip-level numbers in
`docs/typst/dataflow.typ`. Run one part or everything (all takes a few minutes):

```bash
.build/default/zkml_noc_bench bw          # all | bank | c2c | oneway | bw
```

Output is one line per measurement; the format is described at the top of `tools/noc_bench/noc_bench.cpp`.

## SFPU against FPU benchmark

`tools/quant_tensor_fpu_vs_sfpu` compares the two math units on device cycles, host time and power. Run it inside
the container that has the toolchain and the device (`run_all.sh cycles|wall|power|all`); the README there has the
method, the raw results and the numbers.

## Debuggin

```bash
TT_METAL_DPRINT_CORES=0,0 \
TT_METAL_DPRINT_ONE_FILE_PER_RISC=1 \
.build/default/zkml_pot_quant_per_tensor 8
```