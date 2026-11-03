# tt-zkml
TT kernels for executing ZKML on Tenstorren hardware.


## Setup

```bash
export TT_METAL_HOME=<path to tt-metal repo>
```


## Host execution

```bash
rm -rf build
cmake --preset default
cmake --build build -j
./build/zkml_quant
```