# Step-by-step diagrams of execute_network on one Tensix core

`docs/32x64l8.pdf`: one page per step of the
execution of a 32 x 64, 8-layer network (bf16 input), with the circular buffers in L1, SrcA / SrcB, FPU, DST, SFPU with its registers, the packer, the
unpackers, the five RISC-V cores and a profiler timeline with a cursor.

- **Measured** (device profiler): the zones of the five RISC-V cores; the time of every step and the timeline come from them.
- **Modelled**: the state of the coprocessor units and the fill of the circular buffers (they follow from the kernel code; the profiler cannot see them).

## Regenerate

```bash
# 1. profile (fine zones only for 2 layers: they fill the profiler buffer of a RISC after about 6 layers)
for mode in device; do
  ZKML_FINE_ZONES=1 TT_METAL_DEVICE_PROFILER=1 .build/default/zkml_nn_network_bench $mode bf16 32 64 2 5   # -> results/trace/<mode>_32x64_L2_fine.csv
  TT_METAL_DEVICE_PROFILER=1 .build/default/zkml_nn_network_bench $mode bf16 32 64 8 5                      # -> results/trace/<mode>_32x64_L8_coarse.csv
done                      # copy $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv after every run
# 2. render
python3 benchmarks/nn_network/trace/build.py /tmp/trace_out      # writes docs/32x64l8.pdf
```

`ZKML_FINE_ZONES=1` makes the program factory build the compute kernel with the profiler-only zones of `layer_stages.hpp` (`SP_*`, `QP_*`, `MM_*`,
`RQ_SCALER`); without it they compile out.

Files: `tracelib.py` (profiler CSV -> zones), `model.py` (state and steps), `draw.py` (one step as SVG), `pages.py` (guide, hardware map, circular
buffers, whole run, middle layers, why the requantization is bound), `build.py` (assembly with typst). `model.py` / `pages.py` still contain a `pre`
flag for a pre-quantized variant that was removed from the project; it is unused.
