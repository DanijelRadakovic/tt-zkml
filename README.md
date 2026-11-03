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

## Debuggin

```bash
TT_METAL_DPRINT_CORES=0,0 \
TT_METAL_DPRINT_ONE_FILE_PER_RISC=1 \
.build/default/zkml_pot_quant_per_tensor
```