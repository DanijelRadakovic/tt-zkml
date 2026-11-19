# Device cycles per tile (one core, device profiler)

Median of 10 runs of the LOOP zone divided by tiles * reps; 1 tile = 1024 elements, clock 1350 MHz. `full` = copy + quantize + pack of blocks of 3 tiles, `sfpu` = only the quantize calls. The unpack, math and pack columns are the three TRISC threads; the slowest of them bounds the loop. `vs twos_lib` is the change of the slowest thread.

## input fp32, full, 48 tiles

| variant | unpack | math | pack | slowest (cycles) | slowest (ns) | vs twos_lib |
|---|---|---|---|---|---|---|
| sm_int8pack | 538.3 | 539.1 | 539.1 | 539.1 | 399.3 | -0.0% |
| sm_uint8pack | 632.8 | 633.6 | 633.6 | 633.6 | 469.3 | +17.5% |
| twos_lib | 538.3 | 539.1 | 539.1 | 539.1 | 399.3 | +0.0% |
| twos_own | 538.3 | 539.1 | 539.1 | 539.1 | 399.3 | -0.0% |

## input fp32, full, 192 tiles

| variant | unpack | math | pack | slowest (cycles) | slowest (ns) | vs twos_lib |
|---|---|---|---|---|---|---|
| sm_int8pack | 538.3 | 540.6 | 540.6 | 540.6 | 400.5 | -0.0% |
| sm_uint8pack | 632.8 | 635.1 | 635.1 | 635.1 | 470.5 | +17.5% |
| twos_lib | 538.3 | 540.6 | 540.6 | 540.6 | 400.5 | +0.0% |
| twos_own | 538.3 | 540.6 | 540.6 | 540.6 | 400.5 | -0.0% |

## input fp32, sfpu, 48 tiles

| variant | unpack | math | pack | slowest (cycles) | slowest (ns) | vs twos_lib |
|---|---|---|---|---|---|---|
| sm_int8pack | 0.0 | 369.7 | 368.9 | 369.7 | 273.8 | +0.0% |
| sm_uint8pack | 0.0 | 465.7 | 464.7 | 465.7 | 344.9 | +26.0% |
| twos_lib | 0.0 | 369.7 | 368.9 | 369.7 | 273.8 | +0.0% |
| twos_own | 0.0 | 369.7 | 368.9 | 369.7 | 273.8 | -0.0% |

## input fp32, sfpu, 192 tiles

| variant | unpack | math | pack | slowest (cycles) | slowest (ns) | vs twos_lib |
|---|---|---|---|---|---|---|
| sm_int8pack | 0.0 | 369.7 | 368.9 | 369.7 | 273.8 | +0.0% |
| sm_uint8pack | 0.0 | 465.7 | 464.7 | 465.7 | 344.9 | +26.0% |
| twos_lib | 0.0 | 369.7 | 368.9 | 369.7 | 273.8 | +0.0% |
| twos_own | 0.0 | 369.7 | 368.9 | 369.7 | 273.8 | +0.0% |

## input bf16, full, 48 tiles

| variant | unpack | math | pack | slowest (cycles) | slowest (ns) | vs twos_lib |
|---|---|---|---|---|---|---|
| sm_uint8pack | 538.4 | 539.0 | 539.0 | 539.0 | 399.3 | +21.6% |
| twos_lib | 442.8 | 443.4 | 443.4 | 443.4 | 328.4 | +0.0% |
| twos_own | 442.9 | 443.4 | 443.4 | 443.4 | 328.4 | -0.0% |

## input bf16, full, 192 tiles

| variant | unpack | math | pack | slowest (cycles) | slowest (ns) | vs twos_lib |
|---|---|---|---|---|---|---|
| sm_uint8pack | 538.5 | 539.9 | 539.9 | 539.9 | 399.9 | +21.6% |
| twos_lib | 442.6 | 444.0 | 444.0 | 444.0 | 328.9 | +0.0% |
| twos_own | 442.6 | 444.0 | 444.0 | 444.0 | 328.9 | -0.0% |

## input bf16, sfpu, 48 tiles

| variant | unpack | math | pack | slowest (cycles) | slowest (ns) | vs twos_lib |
|---|---|---|---|---|---|---|
| sm_uint8pack | 0.0 | 465.7 | 464.7 | 465.7 | 344.9 | +26.0% |
| twos_lib | 0.0 | 369.7 | 368.9 | 369.7 | 273.8 | +0.0% |
| twos_own | 0.0 | 369.7 | 368.9 | 369.7 | 273.8 | +0.0% |

## input bf16, sfpu, 192 tiles

| variant | unpack | math | pack | slowest (cycles) | slowest (ns) | vs twos_lib |
|---|---|---|---|---|---|---|
| sm_uint8pack | 0.0 | 465.7 | 464.7 | 465.7 | 344.9 | +26.0% |
| twos_lib | 0.0 | 369.7 | 368.9 | 369.7 | 273.8 | +0.0% |
| twos_own | 0.0 | 369.7 | 368.9 | 369.7 | 273.8 | +0.0% |
