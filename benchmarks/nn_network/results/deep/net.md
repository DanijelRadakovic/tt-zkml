
### x = 32 x 64, 64 x 64 layers (network time in us; speedup of the device network against the CPU)

| L | MACs | CPU 1 thr | CPU 6 thr | CPU 12 thr | device net fp32 | device net bf16 | bf16 layers loop | bf16 vs CPU 1 | bf16 vs best CPU | fused vs layer loop |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.3 M | 29 | 62 | 47 | 49 | 51 | 57 | 0.6x | 0.6x | 1.1x |
| 8 | 1.0 M | 108 | 223 | 167 | 131 | 130 | 192 | 0.8x | 0.8x | 1.5x |
| 16 | 2.1 M | 211 | 443 | 313 | 231 | 231 | 380 | 0.9x | 0.9x | 1.6x |
| 32 | 4.2 M | 427 | 849 | 620 | 431 | 444 | 740 | 1.0x | 1.0x | 1.7x |

### x = 32 x 128, 128 x 128 layers (network time in us; speedup of the device network against the CPU)

| L | MACs | CPU 1 thr | CPU 6 thr | CPU 12 thr | device net fp32 | device net bf16 | bf16 layers loop | bf16 vs CPU 1 | bf16 vs best CPU | fused vs layer loop |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.0 M | 89 | 108 | 104 | 59 | 71 | 60 | 1.2x | 1.2x | 0.8x |
| 8 | 4.2 M | 340 | 366 | 367 | 168 | 149 | 189 | 2.3x | 2.3x | 1.3x |
| 16 | 8.4 M | 701 | 644 | 702 | 310 | 270 | 358 | 2.6x | 2.4x | 1.3x |
| 32 | 16.8 M | 1352 | 1382 | 1346 | 589 | 508 | 709 | 2.7x | 2.6x | 1.4x |

### x = 32 x 256, 256 x 256 layers (network time in us; speedup of the device network against the CPU)

| L | MACs | CPU 1 thr | CPU 6 thr | CPU 12 thr | device net fp32 | device net bf16 | bf16 layers loop | bf16 vs CPU 1 | bf16 vs best CPU | fused vs layer loop |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4.2 M | 334 | 119 | 138 | 103 | 82 | 98 | 4.1x | 1.4x | 1.2x |
| 8 | 16.8 M | 1306 | 390 | 544 | 340 | 262 | 217 | 5.0x | 1.5x | 0.8x |
| 16 | 33.6 M | 2655 | 662 | 1027 | 650 | 503 | 405 | 5.3x | 1.3x | 0.8x |
| 32 | 67.1 M | 5249 | 1418 | 2143 | 1275 | 947 | 785 | 5.5x | 1.5x | 0.8x |

### x = 32 x 512, 512 x 512 layers (network time in us; speedup of the device network against the CPU)

| L | MACs | CPU 1 thr | CPU 6 thr | CPU 12 thr | device net fp32 | device net bf16 | bf16 layers loop | bf16 vs CPU 1 | bf16 vs best CPU | fused vs layer loop |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 16.8 M | 1236 | 232 | 244 | n/a | 198 | 183 | 6.2x | 1.2x | 0.9x |
| 8 | 67.1 M | 5047 | 943 | 940 | n/a | 687 | 637 | 7.3x | 1.4x | 0.9x |
| 16 | 134.2 M | 9896 | 1916 | 1921 | n/a | 1340 | 1246 | 7.4x | 1.4x | 0.9x |
| 32 | 268.4 M | 19804 | 3920 | 3791 | n/a | 2748 | 2469 | 7.2x | 1.4x | 0.9x |

### x = 128 x 64, 64 x 64 layers (network time in us; speedup of the device network against the CPU)

| L | MACs | CPU 1 thr | CPU 6 thr | CPU 12 thr | device net fp32 | device net bf16 | bf16 layers loop | bf16 vs CPU 1 | bf16 vs best CPU | fused vs layer loop |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.0 M | 80 | 69 | 57 | 58 | 71 | 62 | 1.1x | 0.8x | 0.9x |
| 8 | 4.2 M | 294 | 262 | 215 | 155 | 148 | 274 | 2.0x | 1.4x | 1.8x |
| 16 | 8.4 M | 582 | 467 | 370 | 280 | 267 | 373 | 2.2x | 1.4x | 1.4x |
| 32 | 16.8 M | 1147 | 905 | 772 | 529 | 509 | 729 | 2.3x | 1.5x | 1.4x |

### x = 128 x 128, 128 x 128 layers (network time in us; speedup of the device network against the CPU)

| L | MACs | CPU 1 thr | CPU 6 thr | CPU 12 thr | device net fp32 | device net bf16 | bf16 layers loop | bf16 vs CPU 1 | bf16 vs best CPU | fused vs layer loop |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4.2 M | 226 | 128 | 126 | 81 | 75 | 91 | 3.0x | 1.7x | 1.2x |
| 8 | 16.8 M | 848 | 446 | 434 | 242 | 225 | 200 | 3.8x | 1.9x | 0.9x |
| 16 | 33.6 M | 1682 | 788 | 857 | 449 | 403 | 374 | 4.2x | 2.0x | 0.9x |
| 32 | 67.1 M | 3417 | 1760 | 1756 | 860 | 797 | 719 | 4.3x | 2.2x | 0.9x |

### x = 128 x 256, 256 x 256 layers (network time in us; speedup of the device network against the CPU)

| L | MACs | CPU 1 thr | CPU 6 thr | CPU 12 thr | device net fp32 | device net bf16 | bf16 layers loop | bf16 vs CPU 1 | bf16 vs best CPU | fused vs layer loop |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 16.8 M | 756 | 185 | 205 | 158 | 145 | 128 | 5.2x | 1.3x | 0.9x |
| 8 | 67.1 M | 2892 | 660 | 801 | 539 | 440 | 428 | 6.6x | 1.5x | 1.0x |
| 16 | 134.2 M | 5696 | 1227 | 1643 | 1037 | 837 | 819 | 6.8x | 1.5x | 1.0x |
| 32 | 268.4 M | 11389 | 2536 | 3169 | 2021 | 1664 | 1609 | 6.8x | 1.5x | 1.0x |

### x = 128 x 512, 512 x 512 layers (network time in us; speedup of the device network against the CPU)

| L | MACs | CPU 1 thr | CPU 6 thr | CPU 12 thr | device net fp32 | device net bf16 | bf16 layers loop | bf16 vs CPU 1 | bf16 vs best CPU | fused vs layer loop |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 67.1 M | 2695 | 471 | 461 | n/a | 333 | 329 | 8.1x | 1.4x | 1.0x |
| 8 | 268.4 M | 10425 | 1930 | 1862 | n/a | 1174 | 1241 | 8.9x | 1.6x | 1.1x |
| 16 | 536.9 M | 20761 | 3774 | 3730 | n/a | 2403 | 2410 | 8.6x | 1.6x | 1.0x |
| 32 | 1073.7 M | 41192 | 7725 | 7393 | n/a | 4743 | 4750 | 8.7x | 1.6x | 1.0x |
