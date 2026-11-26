#### 8 bits

| tiles | CPU 1 thread | CPU 16 threads | device | fastest | device vs CPU 1 thread | CPU + remainder and argmax: 1 thread / 16 threads | copy M31 to host |
|---|---|---|---|---|---|---|---|
| 16 | 0.012 ms (0.75 us/tile) | 0.016 ms (1.01 us/tile) | 0.022 ms (1.37 us/tile) | CPU 1 thread | 1.8x slower | 0.019 / 0.026 ms | 0.025 ms |
| 64 | 0.081 ms (1.26 us/tile) | 0.083 ms (1.29 us/tile) | 0.067 ms (1.04 us/tile) | device | 1.2x faster | 0.106 / 0.128 ms | 0.078 ms |
| 256 | 0.351 ms (1.37 us/tile) | 0.146 ms (0.57 us/tile) | 0.247 ms (0.96 us/tile) | CPU 16 threads | 1.4x faster | 0.448 / 0.237 ms | 0.220 ms |
| 1024 | 1.399 ms (1.37 us/tile) | 0.529 ms (0.52 us/tile) | 0.967 ms (0.94 us/tile) | CPU 16 threads | 1.4x faster | 1.841 / 0.805 ms | 1.981 ms |
| 4096 | 7.426 ms (1.81 us/tile) | 2.355 ms (0.57 us/tile) | 3.846 ms (0.94 us/tile) | CPU 16 threads | 1.9x faster | 10.339 / 3.995 ms | 7.974 ms |

#### 16 bits

| tiles | CPU 1 thread | CPU 16 threads | device | fastest | device vs CPU 1 thread | CPU + remainder and argmax: 1 thread / 16 threads | copy M31 to host |
|---|---|---|---|---|---|---|---|
| 16 | 0.012 ms (0.75 us/tile) | 0.016 ms (0.97 us/tile) | 0.034 ms (2.11 us/tile) | CPU 1 thread | 2.8x slower | 0.013 / 0.017 ms | 0.024 ms |
| 64 | 0.048 ms (0.75 us/tile) | 0.046 ms (0.72 us/tile) | 0.114 ms (1.78 us/tile) | CPU 16 threads | 2.4x slower | 0.053 / 0.054 ms | 0.081 ms |
| 256 | 0.201 ms (0.79 us/tile) | 0.053 ms (0.21 us/tile) | 0.436 ms (1.70 us/tile) | CPU 16 threads | 2.2x slower | 0.222 / 0.084 ms | 0.223 ms |
| 1024 | 0.924 ms (0.90 us/tile) | 0.279 ms (0.27 us/tile) | 1.722 ms (1.68 us/tile) | CPU 16 threads | 1.9x slower | 1.067 / 0.512 ms | 0.857 ms |
| 4096 | 7.519 ms (1.84 us/tile) | 2.331 ms (0.57 us/tile) | 6.867 ms (1.68 us/tile) | CPU 16 threads | 1.1x faster | 10.398 / 3.982 ms | 8.094 ms |

#### 32 bits

| tiles | CPU 1 thread | CPU 16 threads | device | fastest | device vs CPU 1 thread | CPU + remainder and argmax: 1 thread / 16 threads | copy M31 to host |
|---|---|---|---|---|---|---|---|
| 16 | 0.012 ms (0.76 us/tile) | 0.017 ms (1.04 us/tile) | 0.033 ms (2.06 us/tile) | CPU 1 thread | 2.7x slower | 0.013 / 0.021 ms | 0.023 ms |
| 64 | 0.049 ms (0.76 us/tile) | 0.045 ms (0.70 us/tile) | 0.111 ms (1.74 us/tile) | CPU 16 threads | 2.3x slower | 0.053 / 0.046 ms | 0.077 ms |
| 256 | 0.204 ms (0.80 us/tile) | 0.054 ms (0.21 us/tile) | 0.423 ms (1.65 us/tile) | CPU 16 threads | 2.1x slower | 0.225 / 0.105 ms | 0.213 ms |
| 1024 | 0.916 ms (0.89 us/tile) | 0.277 ms (0.27 us/tile) | 1.672 ms (1.63 us/tile) | CPU 16 threads | 1.8x slower | 1.070 / 0.510 ms | 0.860 ms |
| 4096 | 7.472 ms (1.82 us/tile) | 2.310 ms (0.56 us/tile) | 6.666 ms (1.63 us/tile) | CPU 16 threads | 1.1x faster | 10.275 / 3.992 ms | 7.928 ms |
