# quant_tensor_fit_in_l1_bench

**Question.** `pot_quantize_per_tensor` reads the input from DRAM twice (pass 1 absmax, pass 2
quantize). If the whole tensor fits in L1, the reader could send it once and compute could re-read
the tiles from L1. Is that faster?

**Answer (this card, one core): no. 0 – 1.7 % faster, in most configs under 1 %.** The DRAM read of
the whole tensor takes 3 – 10 % of the kernel time and already overlaps compute; compute is the
bottleneck. Numbers and the reasoning are below. Everything here is reproducible with one script.

## Contents

| File | What it is |
|---|---|
| `bench.cpp` | Host program. Builds reader + compute + writer on core (0,0) in `stream` or `resident` mode, runs it N times, prints a correctness checksum. |
| `kernels/reader.cpp` | Reads blocks of `BLOCK` tiles; `NUM_PASSES` = 2 (stream) or 1 (resident). Zone `READ_BLOCK` per block. |
| `kernels/compute.cpp` | The real `quantize.cpp` (Int32 output only) + `RESIDENT` mode + profiler zones `PASS1`, `SCALER`, `PASS2`, `WAIT_C0`. |
| `kernels/writer.cpp` | Same as the real writer. Zone `WRITE_ALL`. |
| `run_all.sh` | Runs the configuration matrix with the profiler on, keeps one raw log per configuration. |
| `analyze.py` | Reads the raw logs, prints the comparison table, writes `results/summary.csv`. |
| `results/raw/*.csv` | **The raw device-profiler logs** (one per configuration, 11 runs each). Open one to learn the format. |
| `results/summary.csv`, `results/summary.md` | Reduced numbers. |
| `results/checks.txt` | Per-configuration scaler + FNV checksum of the output (host-side correctness check). |

Build and run (from the repo root):

```bash
export TT_METAL_HOME=/proj_sw/user_dev/ctr-dradakovic/tt-metal
cmake --build .build/default --target zkml_quant_l1_bench
tools/quant_tensor_fit_in_l1_bench/run_all.sh            # ~2 min, writes results/
python3 -I tools/quant_tensor_fit_in_l1_bench/analyze.py tools/quant_tensor_fit_in_l1_bench/results/raw/*.csv
```

One configuration by hand:

```bash
rm -f $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv
TT_METAL_DEVICE_PROFILER=1 .build/default/zkml_quant_l1_bench resident f32 32 128 16 11
#                                                    mode      in  bits tiles block iters
less $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv
```

## The two modes

| | `stream` (today's op) | `resident` (the idea) |
|---|---|---|
| reader | sends the tensor twice (`2 * num_tiles` reads) | sends it once |
| `c_0` | ring of `2 * BLOCK` pages | `num_tiles` pages: the whole tensor |
| pass 1 | `wait_front(n)`, `copy_tile(cb, i, i)`, **pop** | `wait_front(done + n)`, `copy_tile(cb, done + i, i)`, **no pop** |
| pass 2 | same as pass 1 | tiles are still in L1; `wait_front` returns immediately, pop as it goes |

No "reset the counter" is needed: `copy_tile(cb, tile_index, dst)` reads page `read_ptr + tile_index`.
As long as nothing is popped, `read_ptr` stays at the start. The CB is exactly as big as the tensor,
so it never wraps and the input-side ring cutting becomes a no-op. Both modes give identical
outputs (`results/checks.txt`: same checksum for all 24 configurations).

Limit: the tensor must fit in L1 next to `c_1`, `c_16`, `c_17`. On this card about 1.3 MB free, so the
bench goes to 128 fp32 tiles (512 KB) / 128 bf16 tiles (256 KB).

## How to measure this properly

The point is to avoid measuring the wrong thing. In order:

1. **Measure on the device, not on the host.** A host timer around `EnqueueMeshWorkload` + `Finish`
   includes dispatch, launch and completion latency (tens of µs on top of the kernel here: 23 µs host vs ~12 µs kernel for 16 f32 tiles, against a
   10 – 190 µs kernel). The device profiler timestamps the kernels themselves, on the chip's own
   counter (1350 MHz = 0.74 ns per cycle on this card; the log header says `CHIP_FREQ[MHz]: 1350`).
2. **Compare like with like.** Same kernel code, same CB sizes except `c_0`, same DST chunking,
   same data. The only differences are the reader's pass count and whether pass 1 pops. I copied the real compute
   kernel instead of writing a toy so that the SFPU cost (the thing that decides the answer) is
   real. Outputs are checksummed so a "faster" variant that computes the wrong thing cannot slip through.
3. **Run it many times, drop the first.** Run 0 has cold instruction caches and cold NoC/DRAM state.
   `iters = 11`, run 0 dropped, median of 10. The run-to-run spread of the kernel time is 0.05 – 0.1 %
   (`spread_pct` in `summary.csv`), far below the 0.5 – 1.7 % differences, so those are real, but small.
4. **Don't just compare totals; find out *why*.** A total says "no gain". Zones say where the time
   goes, which tells you whether a different config could change the answer:
   - `PASS1`, `PASS2`: compute time per pass.
   - `WAIT_C0`: time compute sat blocked in `wait_front` on `c_0`, i.e. *starved by the reader*. This is
     the number that decides it: if compute never waits for the reader, a faster reader (fewer reads)
     cannot help.
   - `READ_BLOCK`: reader time per block.
   - `WRITE_ALL`: writer total (it is almost exactly `PASS2`, i.e. the writer just follows compute).
5. **Sweep the knobs that change who is the bottleneck**: tensor size (16/64/128 tiles), dtype (bf16
   tiles are half the bytes and have a cheaper `copy_tile`), bit width (8 = `quant_tile`, 32 = four SFPU
   passes, so a much heavier pass 2), and reader block size (4/16/128 tiles). A fast pass 2 and a small
   block are the cases where the reader is *most* likely to matter, so they are the fair test.
6. **Keep the profiler from perturbing the result.** Zones cost a few cycles each and the L1 marker
   buffer holds 250 markers per RISC (a zone is 2 markers), so zones are per block / per chunk, never per
   tile, and sizes stop at 128 tiles. The profiler log was checked for completeness (`WAIT_C0` count =
   chunks per run × runs).

## How the profiling was done

**Instrumentation.** In kernels:

```cpp
#include "api/debug/kernel_profiler.hpp"
...
{ DeviceZoneScopedN("PASS1"); ... }        // RAII: records a start marker here, an end marker at the }
```

A zone is a scope, and two zones in one scope clash (the macro declares a variable), hence the extra
braces. In the compute kernel the code is compiled three times (UNPACK, MATH, PACK TRISC), so every
zone is recorded by all three; `WAIT_C0` is under `#ifdef TRISC_UNPACK` because `wait_front` only blocks
there.

**Enabling.** `TT_METAL_DEVICE_PROFILER=1` at run time (the kernels are built with `-DPROFILE_KERNEL=1`
then; without it the macros compile to nothing). The host calls
`tt::tt_metal::ReadMeshDeviceProfilerResults(*dev)` before closing the device, which copies the device
marker buffers out and writes
`$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv`. The file is overwritten each
process, so `run_all.sh` deletes it before and copies it after every configuration.

**Automatic zones.** Besides ours, the runtime adds `BRISC-FW`/`NCRISC-FW`/`TRISC-FW` (firmware around the
kernel) and `BRISC-KERNEL`/`NCRISC-KERNEL`/`TRISC-KERNEL` (the kernel body). The kernel duration used below is
the earliest `*-KERNEL` start to the latest `*-KERNEL` end on the core.

## Reading the profiler log

Look at `results/raw/resident_f32_32b_16t_4blk.csv` (16 tiles, so it is short). The header line gives the
chip frequency; then one row per marker:

```
PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, run host ID, ..., zone name, type, ...
0,1,2,NCRISC,48403,167835888260,..., NCRISC-KERNEL, ZONE_START
0,1,2,NCRISC,18052,167835888328,..., READ_BLOCK,     ZONE_START
0,1,2,NCRISC,18052,167835889163,..., READ_BLOCK,     ZONE_END
0,1,2,NCRISC,18052,167835889202,..., READ_BLOCK,     ZONE_START
```

- `core_x,core_y` = `1,2`: the worker core (virtual coordinates of logical (0,0)).
- `RISC processor type`: `NCRISC` = reader, `BRISC` = writer, `TRISC_0/1/2` = unpack / math / pack.
- `time[cycles since reset]`: one chip-wide counter, so rows of different RISCs on the same core are
  directly comparable. Divide cycle counts by 1350 for µs.
- `zone name` + `type` (`ZONE_START` / `ZONE_END`): a zone's duration is END − START of a matched pair.

Example: the first `READ_BLOCK` above runs from `…888328` to `…889163` = **835 cycles = 0.62 µs** for
4 fp32 tiles (16 KB), ~26 GB/s for a block that is latency-bound (one barrier per 4 reads).

What `analyze.py` does with it:

1. Pair START/END per RISC with a stack (zones nest: `WAIT_C0` is inside `PASS1`).
2. The *i*-th `*-KERNEL` start on a RISC is run *i*; drop run 0.
3. Per run: `kernel` = max(KERNEL end) − min(KERNEL start); `PASS1/SCALER/PASS2` from `TRISC_1` (math);
   `WAIT_C0` from `TRISC_0` summed separately for waits that start inside `PASS1` and inside `PASS2`;
   `READ_BLOCK` from `NCRISC`, `WRITE_ALL` from `BRISC`.
4. Median over the 10 remaining runs; cycles → µs.

### Reading "read µs" correctly (a trap)

`read µs` = sum of `READ_BLOCK` durations, and a block starts with `reserve_back`, which blocks when
compute has not freed the ring yet. In **resident** mode `c_0` fits the whole tensor, the reader never
waits, so it is the *pure* time to read the tensor once (f32 128 tiles, block 16: **9.4 µs**, 512 KB,
56 GB/s). In **stream** mode the same number is 148 µs, but that is almost all `reserve_back` waiting
for compute; it is not DRAM time. The comparison that matters is not read-vs-read but
*is compute ever blocked on the reader* (`WAIT_C0`).

## Results

`results/summary.md` has all 24 configurations; a selection (µs per run, median of 10):

| in | bits | tiles | block | stream | resident | gain | `WAIT_C0` p1 / p2 (stream) | `WAIT_C0` p1 / p2 (resident) | pure read of tensor (resident) |
|---|---|---|---|---|---|---|---|---|---|
| f32 | 32 | 128 | 16 | 190.8 | 190.0 | +0.4 % | 1.9 / 1.3 | 1.8 / 1.2 | 9.4 |
| f32 | 32 | 128 | 4 | 191.4 | 189.5 | +1.0 % | 1.5 / 1.6 | 1.5 / 1.2 | 19.0 |
| f32 | 8 | 128 | 4 | 83.1 | 81.6 | +1.7 % | 1.5 / 1.7 | 1.4 / 1.3 | 19.0 |
| bf16 | 8 | 128 | 4 | 67.8 | 67.6 | +0.2 % | **13.3** / 1.7 | 3.7 / 1.2 | 17.1 |
| bf16 | 8 | 128 | 16 | 68.1 | 68.0 | +0.1 % | 1.5 / 1.3 | 1.4 / 1.2 | 7.4 |
| f32 | 32 | 128 | 128 | 195.9 | 195.3 | +0.3 % | 7.2 / 1.2 | 7.2 / 1.2 | 6.6 |

Breakdown of one row (f32, 32 bit, 128 tiles, block 16): kernel 190.8 µs = `PASS1` 32.3 µs +
`SCALER` 0.13 µs + `PASS2` 157.5 µs (+ ~1 µs of start/stop); the writer's `WRITE_ALL` is 157.8 µs.

### What I conclude from it

1. **Compute is the bottleneck; the reader is far ahead.** Reading the tensor once takes 9.4 µs
   (56 GB/s) while pass 1 consumes it in 32 µs (16 GB/s) and pass 2 in 157 µs (3.3 GB/s). The second
   DRAM read overlaps compute and is never waited on.
2. **`WAIT_C0` is ~1 – 2 µs in total, in both modes**, and it is the same before and after. It is the
   unavoidable first-block latency (reader issues the first read, compute waits for it), not
   throughput starvation. Resident cannot remove that because the first block must still be read.
3. **The gain is 0 – 1.7 %**, equal to the *difference in `WAIT_C0` plus noise* and about the size of
   a fraction of one block's latency. Largest where `PASS2` is cheap (8-bit f32) and the block is small.
4. **Where reader pressure does show up**: bf16, 8-bit, block 4, pass 1: `WAIT_C0` 13.3 µs in stream
   mode, 3.7 µs resident (tiles are 2 KB, a block is only 8 KB, and the 2-block ring leaves the reader
   little slack). Resident helps there because the ring is deep, but it only buys ~0.2 % of the
   total, since pass 1 is a small part of the kernel. A larger reader block (16) removes the effect
   in stream mode too, with no L1-fit requirement.
5. **Block size matters more than residency.** With one block = the whole tensor (block 128) both modes
   lose ~5 µs (`WAIT_C0` p1 = 7.2 µs: compute cannot start until the whole tensor is read). The real op
   did that for small tensors when this bench was written (it used the largest block that fit L1, capped at the tile count; since replaced by `noc_read_tile_batch_size` / `cb_in_size_factor`), so for
   tensors that fit, a block of ~8–16 tiles is the cheaper win than a resident mode.

### Caveats (what this does *not* show)

- **One core.** The op is single-core, so DRAM sees one reader (56 GB/s peak at best, far below the
  chip). If the op is later split across many cores, all reading at once, DRAM bandwidth (shared) could
  become the limit and the second read would matter. Re-run this bench with several cores before
  concluding anything for a multi-core op.
- Compute cost is data-independent (fixed SFPU work per tile), so random input is representative; only
  `k` changes.
- 128 tiles is the largest size covered (marker buffer / L1 limits). Larger tensors do not fit in L1 anyway.
- `Int32` output only (no `Int8`); `bits = 16` is not run (it takes the same path as 32).
- Host-side time (`host_ms_per_run` in `checks.txt`) is printed for reference only, do not use it.

## Extending

Add a zone: wrap the code in `{ DeviceZoneScopedN("NAME"); ... }`, rebuild nothing on the host (kernels
are JIT-compiled), add the pairing logic to `per_run()` in `analyze.py`. Change the matrix: edit the loops
in `run_all.sh`; the file name `<mode>_<in>_<bits>b_<tiles>t_<block>blk.csv` is what `analyze.py` parses.
