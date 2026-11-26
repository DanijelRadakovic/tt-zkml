
### x = 32 x 64, 64 x 64 layers (network time in us; speedup of the device network against the CPU)

| L | MACs | CPU 1 thr | CPU 6 thr | CPU 12 thr | device net fp32 | device net bf16 | bf16 layers loop | bf16 vs CPU 1 | bf16 vs best CPU | fused vs layer loop |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.1 M | 17 | 32 | 28 | 32 | 50 | 30 | 0.3x | 0.3x | 0.6x |
| 2 | 0.3 M | 29 | 58 | 51 | 64 | 68 | 54 | 0.4x | 0.4x | 0.8x |
| 4 | 0.5 M | 56 | 104 | 98 | 73 | 75 | 95 | 0.7x | 0.7x | 1.3x |
| 6 | 0.8 M | 82 | 159 | 143 | 97 | 106 | 137 | 0.8x | 0.8x | 1.3x |

### x = 32 x 128, 128 x 128 layers (network time in us; speedup of the device network against the CPU)

| L | MACs | CPU 1 thr | CPU 6 thr | CPU 12 thr | device net fp32 | device net bf16 | bf16 layers loop | bf16 vs CPU 1 | bf16 vs best CPU | fused vs layer loop |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.5 M | 48 | 52 | 50 | 39 | 36 | 35 | 1.3x | 1.3x | 1.0x |
| 2 | 1.0 M | 89 | 105 | 99 | 71 | 51 | 77 | 1.7x | 1.7x | 1.5x |
| 4 | 2.1 M | 176 | 201 | 174 | 94 | 83 | 100 | 2.1x | 2.1x | 1.2x |
| 6 | 3.1 M | 262 | 293 | 327 | 132 | 131 | 195 | 2.0x | 2.0x | 1.5x |

### x = 32 x 256, 256 x 256 layers (network time in us; speedup of the device network against the CPU)

| L | MACs | CPU 1 thr | CPU 6 thr | CPU 12 thr | device net fp32 | device net bf16 | bf16 layers loop | bf16 vs CPU 1 | bf16 vs best CPU | fused vs layer loop |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.1 M | 171 | 74 | 86 | 69 | 50 | 49 | 3.4x | 1.5x | 1.0x |
| 2 | 4.2 M | 334 | 134 | 168 | 102 | 86 | 73 | 3.9x | 1.6x | 0.8x |
| 4 | 8.4 M | 666 | 251 | 323 | 186 | 140 | 124 | 4.8x | 1.8x | 0.9x |
| 6 | 12.6 M | 984 | 377 | 474 | 280 | 209 | 168 | 4.7x | 1.8x | 0.8x |

### x = 32 x 512, 512 x 512 layers (network time in us; speedup of the device network against the CPU)

| L | MACs | CPU 1 thr | CPU 6 thr | CPU 12 thr | device net fp32 | device net bf16 | bf16 layers loop | bf16 vs CPU 1 | bf16 vs best CPU | fused vs layer loop |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8.4 M | 630 | 189 | 171 | n/a | 105 | 103 | 6.0x | 1.6x | 1.0x |
| 2 | 16.8 M | 1247 | 367 | 338 | n/a | 190 | 179 | 6.6x | 1.8x | 0.9x |
| 4 | 33.6 M | 2465 | 729 | 671 | n/a | 365 | 334 | 6.7x | 1.8x | 0.9x |
| 6 | 50.3 M | 3710 | 1097 | 985 | n/a | 520 | 488 | 7.1x | 1.9x | 0.9x |

### x = 128 x 64, 64 x 64 layers (network time in us; speedup of the device network against the CPU)

| L | MACs | CPU 1 thr | CPU 6 thr | CPU 12 thr | device net fp32 | device net bf16 | bf16 layers loop | bf16 vs CPU 1 | bf16 vs best CPU | fused vs layer loop |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.5 M | 46 | 41 | 36 | 49 | 37 | 45 | 1.2x | 1.0x | 1.2x |
| 2 | 1.0 M | 81 | 73 | 66 | 56 | 54 | 67 | 1.5x | 1.2x | 1.2x |
| 4 | 2.1 M | 153 | 137 | 125 | 88 | 86 | 102 | 1.8x | 1.4x | 1.2x |
| 6 | 3.1 M | 224 | 204 | 187 | 119 | 115 | 141 | 2.0x | 1.6x | 1.2x |

### x = 128 x 128, 128 x 128 layers (network time in us; speedup of the device network against the CPU)

| L | MACs | CPU 1 thr | CPU 6 thr | CPU 12 thr | device net fp32 | device net bf16 | bf16 layers loop | bf16 vs CPU 1 | bf16 vs best CPU | fused vs layer loop |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.1 M | 123 | 77 | 69 | 54 | 48 | 56 | 2.5x | 1.4x | 1.1x |
| 2 | 4.2 M | 230 | 148 | 136 | 80 | 72 | 68 | 3.2x | 1.9x | 0.9x |
| 4 | 8.4 M | 445 | 286 | 236 | 133 | 121 | 110 | 3.7x | 1.9x | 0.9x |
| 6 | 12.6 M | 646 | 410 | 429 | 193 | 181 | 155 | 3.6x | 2.3x | 0.9x |

### x = 128 x 256, 256 x 256 layers (network time in us; speedup of the device network against the CPU)

| L | MACs | CPU 1 thr | CPU 6 thr | CPU 12 thr | device net fp32 | device net bf16 | bf16 layers loop | bf16 vs CPU 1 | bf16 vs best CPU | fused vs layer loop |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8.4 M | 392 | 136 | 142 | 103 | 81 | 78 | 4.8x | 1.7x | 1.0x |
| 2 | 16.8 M | 745 | 262 | 275 | 161 | 131 | 128 | 5.7x | 2.0x | 1.0x |
| 4 | 33.6 M | 1462 | 523 | 526 | 288 | 256 | 229 | 5.7x | 2.0x | 0.9x |
| 6 | 50.3 M | 2169 | 756 | 784 | 418 | 352 | 331 | 6.2x | 2.1x | 0.9x |

### x = 128 x 512, 512 x 512 layers (network time in us; speedup of the device network against the CPU)

| L | MACs | CPU 1 thr | CPU 6 thr | CPU 12 thr | device net fp32 | device net bf16 | bf16 layers loop | bf16 vs CPU 1 | bf16 vs best CPU | fused vs layer loop |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 33.6 M | 1358 | 420 | 352 | n/a | 213 | 180 | 6.4x | 1.7x | 0.8x |
| 2 | 67.1 M | 2833 | 817 | 688 | n/a | 329 | 322 | 8.6x | 2.1x | 1.0x |
| 4 | 134.2 M | 5222 | 1608 | 1341 | n/a | n/a | 618 | n/a | n/a | n/a |
| 6 | 201.3 M | 7863 | 2411 | 2018 | n/a | n/a | 910 | n/a | n/a | n/a |
