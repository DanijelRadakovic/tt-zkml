# A network of identical quantized layers: CPU (1 thread, N threads) against the device

How long does a network of `L` identical layers take for different output widths `N`, on one CPU thread, on several CPU threads, and on
the device. One layer is exactly what `zkml::prim::execute_layer` computes: `x` `[M, K]` and `w` `[K, N]` (fp32) are quantized per tensor to
8 bits with a power-of-two scaler (Q = 127, round half to even), multiplied exactly into int32, ReLU is applied on the accumulator, and
the accumulator is requantized per tensor to 8 bits (integer arithmetic). The CPU implementation in `bench.cpp` computes the same thing
bit for bit (`results/check.txt`: every output byte and the three scalers equal the device's).

A network is `L = 8` such layers, each with its own weights `w_i` (own random data); the input `x` is the same for all layers. The
layers are independent of each other: the device op takes fp32 or bf16 tensors, so chaining layer outputs on the device would need a
host round trip (int8 bytes to floats) that is not part of the op, and the work per layer is the same either way.

## Run

```bash
cmake --preset default && cd .build/default && ninja zkml_nn_bench && cd ../..
TT_METAL_HOME=/proj_sw/user_dev/ctr-dradakovic/tt-metal benchmarks/nn_layers/run_all.sh        # about 5 minutes, writes results/
.build/default/zkml_nn_bench cpu 16 64 256 256 8 20       # threads M K N L reps
.build/default/zkml_nn_bench device bf16 64 256 256 8 20  # fp32|bf16 M K N L reps
.build/default/zkml_nn_bench check 64 256 128             # CPU against the device
```

Environment variables of `run_all.sh`: `K`, `L`, `REPS`, `BATCHES`, `WIDTHS`, `THREADS`. Run it inside the IRD container (device needed).

## What is measured

- **CPU**: the whole layer: both quantizations (absmax pass + quantize pass for `x` and for `w`, `w` is written transposed), the int8
  matmul into int32 (4 output columns at a time, contiguous int8 rows, compiled with `-O3 -march=native`, auto-vectorized by the compiler,
  not hand-written VNNI intrinsics), the ReLU with the integer absmax, and the integer requantization. Threads come from a persistent
  spinning thread pool; every pass is split statically over the threads (matmul over rows of `x`). Median of 20 runs of the whole
  network after 2 warm-up runs.
- **Device**: `L` calls of `execute_layer` (fp32 or bf16 tensors, ReLU) back to back, then a sync, program cached, input and weights already
  on the device, outputs not read back. The time includes the host's per-call work (program-cache lookup, output allocation, enqueue),
  about 20 us per call, which is why the device time per layer flattens at the small sizes. One Tensix core of 110 is used.
- Per-layer time = network time / `L`.

Machine: CPU AMD Ryzen 5 7600X, 6 cores / 12 hardware threads, AVX-512 with VNNI (this container sees 12 CPUs); device Blackhole p100a,
one core at 1.35 GHz.

## Results


### x = 64 x 256, w = 256 x N, 8 layers (time per layer in us; speedup against 1 CPU thread)

| N | MACs/layer | CPU 1 thr | CPU 6 thr | CPU 12 thr | CPU 16 thr | device fp32 | device bf16 | device fp32 vs CPU 1 | device bf16 vs CPU 1 | device bf16 vs best CPU |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.52 M | 44.5 | 21.4 | 30.6 | 56.7 | 23.4 | 22.9 | 1.9x | 1.9x | 0.9x |
| 64 | 1.05 M | 73.4 | 28.4 | 42.4 | 82.1 | 23.1 | 23.1 | 3.2x | 3.2x | 1.2x |
| 128 | 2.10 M | 133.1 | 49.1 | 63.0 | 113.6 | 29.7 | 23.6 | 4.5x | 5.6x | 2.1x |
| 256 | 4.19 M | 255.5 | 86.1 | 101.1 | 188.1 | 48.4 | 35.6 | 5.3x | 7.2x | 2.4x |
| 512 | 8.39 M | 497.8 | 153.5 | 184.9 | 340.7 | 84.3 | 62.2 | 5.9x | 8.0x | 2.5x |

### x = 128 x 256, w = 256 x N, 8 layers (time per layer in us; speedup against 1 CPU thread)

| N | MACs/layer | CPU 1 thr | CPU 6 thr | CPU 12 thr | CPU 16 thr | device fp32 | device bf16 | device fp32 vs CPU 1 | device bf16 vs CPU 1 | device bf16 vs best CPU |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 1.05 M | 76.2 | 30.8 | 38.4 | 73.7 | 24.0 | 23.6 | 3.2x | 3.2x | 1.3x |
| 64 | 2.10 M | 123.1 | 44.7 | 54.0 | 107.6 | 29.9 | 23.9 | 4.1x | 5.2x | 1.9x |
| 128 | 4.19 M | 213.2 | 58.4 | 84.5 | 172.5 | 42.8 | 32.3 | 5.0x | 6.6x | 1.8x |
| 256 | 8.39 M | 401.5 | 132.2 | 140.7 | 297.2 | 67.2 | 52.5 | 6.0x | 7.6x | 2.5x |
| 512 | 16.78 M | 773.0 | 251.6 | 242.1 | 490.2 | 117.4 | 92.9 | 6.6x | 8.3x | 2.6x |

(`results/nn.txt` has the raw lines, `results/nn.md` the tables.)

## Reading the table

- **Device against one CPU thread**: 1.9x to 6.6x faster in fp32, 1.9x to 8.3x in bf16, growing with `N` (more work per call hides the
  fixed per-call cost). bf16 is faster than fp32 at the same shape because the unpack of a bf16 tile reads half the bytes.
- **Device against the best multi-threaded CPU run**: 0.9x to 2.6x. At the smallest width (`N = 32`) six CPU threads are as fast as the
  device; from `N = 128` the device is about 1.8x to 2.6x faster than the best CPU configuration while using one core of 110.
- **CPU thread scaling is poor and non-monotonic**: 6 threads are the best here in all but one case (2.1x to 3.6x over one thread; the layers are small and
  every pass has a synchronization), 12 threads (all hardware threads, two per core) are slower than 6 except at the largest size (242 against 252 us), and *16 threads are slower than
  both because the machine has only 12 hardware threads*: with spinning workers the extra four threads fight for CPUs. The 16-thread
  column is what you get when asking for 16 threads on this machine, not a 16-core result.
- **The CPU kernel is a plain auto-vectorized loop.** A hand-written AVX-512 VNNI matmul (`vpdpbusd`) would make the CPU matmul several
  times faster; the quantization and requantization passes are memory-bound and would not change. So the CPU columns are a baseline
  for "reasonable C++", not the fastest possible CPU implementation.
- **Not measured here**: more than one device core (the op is single-core; the layers are independent, so they could run on different
  cores at the same time), and layers that do not fit in the L1 of one core (the op throws; `K = 256`, `N = 512`, `M = 128` is about the
  limit).
