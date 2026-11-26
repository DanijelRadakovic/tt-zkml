# A chained quantized network: CPU against the device (`execute_network`)

`bench.cpp` runs a chain of `L` square layers (`W x W`, `L <= 6`, ReLU between, none after the last) for batch size `M`: on the CPU
(1, 6, 12 threads, bit-exact with the device), on the device as one `zkml::prim::execute_network` call, and, for comparison, as `L`
separate `execute_layer` calls (not chained: each re-quantizes `x`).

```bash
cmake --preset default && cd .build/default && ninja zkml_nn_network_bench && cd ../..
TT_METAL_HOME=... benchmarks/nn_network/run_all.sh                       # sweep, writes results/net.txt, net.md, check.txt (about 20 min)
.build/default/zkml_nn_network_bench check 128 256 6                     # CPU against the device: bytes, accumulators, scalers
.build/default/zkml_nn_network_bench device bf16 128 256 4 20            # M W L reps
```

## Profiling why the fused network is not faster in wall time

1. **Device time** (profiler zones `QUANT_X`, `QUANT_W`, `MATMUL`, `REQUANT` are in the compute kernel of `execute_network`; they compile out
   without the profiler). The profiled kernel is larger, so use `L <= 4`:
   ```bash
   rm -f $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv
   TT_METAL_DEVICE_PROFILER=1 .build/default/zkml_nn_network_bench device bf16 128 256 4 10
   cp $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv device.csv     # same for `layers`, then:
   python3 -I benchmarks/nn_network/analyze_profile.py device.csv layers.csv 4
   ```
2. **Host time**: `device` and `layers` print `host-enqueue` (host time until the last call returned, before waiting for the device).
3. **Host time by piece**: `zkml_nn_network_bench hostprof bf16 128 256 4 100` times the pieces of `ttnn::device_operation::launch` for
   `execute_network` and `execute_layer`: input validation, `create_output_tensors`, program hash, canonical cache key, and the rest
   (`place_output_tensors`, `override_runtime_arguments`, the enqueue).

## Findings (M = 128, W = 256, bf16)

- Device: fused 180 us against 203 us for 4 `execute_layer` (the input is quantized once, not four times).
- Wall: fused 240 us, loop 229 us. The loop's host work overlaps the device (the next layer is enqueued while one runs); the fused call is
  one enqueue, so its host time is in front of the kernel.
- Host time of one call (us), `hostprof`:

  | call | outputs | create_output_tensors | hash + key | rest | total |
  |---|---|---|---|---|---|
  | `execute_layer` | 5 | 17.4 | 0.5 | 6.3 | 24.2 |
  | `execute_network` L=1 | 5 | 17.9 | 0.5 | 6.5 | 25.0 |
  | `execute_network` L=4 | 14 | 53.6 | 1.0 | 7.5 | 62.2 |
  | `execute_network` L=6 | 20 | 76.4 | 1.4 | 10.7 | 88.5 |

  About 3.6 us per output tensor, 85% of the call. The program hash, the canonical key and the validation are below 1 us each; the rest is
  6 to 11 us. Creating the `2 + 3L` outputs (`acc`, `scaler_w`, `scaler_out` per layer) is the bottleneck of the host side.
