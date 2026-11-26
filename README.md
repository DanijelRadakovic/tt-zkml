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
Q = 2^(bit_width - 1) - 1          bit_width: 8, 16 or 32
k = max{k <= 127 : absmax(x) * 2^k <= Q}
q = clamp(round_half_even(x * 2^k), -Q, Q)      scaler = 2^k, so x ~= q / scaler
```

The host program runs it for the given bit widths (default: 8 16 32), BFLOAT16 on both math units:

```bash
.build/default/zkml_pot_quant_per_tensor 8 16
```

**Math unit.** Pass 2 multiplies by the scaler on the FPU or the SFPU (`compute_unit` argument). The default,
`Auto`, uses the FPU for BFLOAT16 input at 16 and 32 bits (about 10 % faster and 9 % less energy, bit-identical result)
and the SFPU otherwise (at 8 bits the FPU would be 2.3x slower). The absmax of pass 1 always runs on the SFPU: the FPU
has no abs and no min. Measurements: `tools/quant_tensor_fpu_vs_sfpu/README.md`.

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