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

## PoT per-tensor quantization

`zkml::prim::pot_quantize_per_tensor(input, bit_width, output_dtype)` computes the scaler on the
device from the data and returns `(quantized, scaler)`:

```
Q = 2^(bit_width - 1) - 1          bit_width: 8, 16 or 32
k = max{k <= 127 : absmax(x) * 2^k <= Q}
q = clamp(round_half_even(x * 2^k), -Q, Q)      scaler = 2^k, so x ~= q / scaler
```

The host program runs it for the given bit widths (default: 8 16 32):

```bash
.build/default/zkml_pot_quant_per_tensor 8 16
```

## Debuggin

```bash
TT_METAL_DPRINT_CORES=0,0 \
TT_METAL_DPRINT_ONE_FILE_PER_RISC=1 \
.build/default/zkml_pot_quant_per_tensor 8
```