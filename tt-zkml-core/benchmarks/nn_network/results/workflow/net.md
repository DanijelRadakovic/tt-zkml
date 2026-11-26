
### x = 32 x 64, 64 x 64 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | dev bf16 q | layer loop | dev bf16 vs best CPU | dev bf16 q vs best VNNI | fused q vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.3 M | 28 | 64 | 48 | 20 | 20 | 22 | 31 | 29 | 42 | 58 | 1.0x | 0.5x | 1.4x |
| 8 | 1.0 M | 106 | 215 | 172 | 71 | 60 | 74 | 53 | 47 | 43 | 194 | 2.3x | 1.4x | 4.5x |
| 16 | 2.1 M | 240 | 419 | 309 | 141 | 112 | 130 | 78 | 65 | 58 | 370 | 3.7x | 1.9x | 6.3x |
| 32 | 4.2 M | 433 | 829 | 609 | 288 | 200 | 234 | 130 | 112 | 92 | 734 | 3.9x | 2.2x | 7.9x |

### x = 32 x 128, 128 x 128 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | dev bf16 q | layer loop | dev bf16 vs best CPU | dev bf16 q vs best VNNI | fused q vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.0 M | 89 | 101 | 109 | 60 | 33 | 28 | 42 | 45 | 32 | 68 | 2.0x | 0.9x | 2.1x |
| 8 | 4.2 M | 347 | 392 | 385 | 232 | 93 | 95 | 93 | 74 | 49 | 192 | 4.7x | 1.9x | 3.9x |
| 16 | 8.4 M | 699 | 714 | 649 | 471 | 156 | 176 | 162 | 124 | 72 | 364 | 5.2x | 2.2x | 5.1x |
| 32 | 16.8 M | 1399 | 1431 | 1493 | 920 | 316 | 345 | 298 | 219 | 128 | 719 | 6.4x | 2.5x | 5.6x |

### x = 32 x 256, 256 x 256 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | dev bf16 q | layer loop | dev bf16 vs best CPU | dev bf16 q vs best VNNI | fused q vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4.2 M | 332 | 102 | 142 | 217 | 58 | 53 | 90 | 70 | 42 | 96 | 1.5x | 1.2x | 2.3x |
| 8 | 16.8 M | 1307 | 375 | 533 | 821 | 177 | 173 | 257 | 178 | 86 | 216 | 2.1x | 2.0x | 2.5x |
| 16 | 33.6 M | 2580 | 738 | 1078 | 1612 | 363 | 342 | 484 | 333 | 147 | 405 | 2.2x | 2.3x | 2.8x |
| 32 | 67.1 M | 5212 | 1576 | 2228 | 3178 | 710 | 668 | 976 | 631 | 274 | 1043 | 2.5x | 2.4x | 3.8x |

### x = 32 x 512, 512 x 512 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | dev bf16 q | layer loop | dev bf16 vs best CPU | dev bf16 q vs best VNNI | fused q vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 16.8 M | 1274 | 242 | 236 | 776 | 145 | 122 | n/a | 168 | 86 | 184 | 1.4x | 1.4x | 2.1x |
| 8 | 67.1 M | 4920 | 980 | 966 | 3111 | 571 | 493 | n/a | 574 | 252 | 637 | 1.7x | 2.0x | 2.5x |
| 16 | 134.2 M | 9977 | 1893 | 1925 | 6241 | 1146 | 982 | n/a | 1149 | 472 | 1245 | 1.6x | 2.1x | 2.6x |
| 32 | 268.4 M | 19670 | 3844 | 3795 | 12464 | 2319 | 1945 | n/a | 2231 | 916 | 2496 | 1.7x | 2.1x | 2.7x |

### x = 128 x 64, 64 x 64 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | dev bf16 q | layer loop | dev bf16 vs best CPU | dev bf16 q vs best VNNI | fused q vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.0 M | 81 | 69 | 55 | 46 | 28 | 25 | 38 | 36 | 36 | 64 | 1.5x | 0.7x | 1.8x |
| 8 | 4.2 M | 296 | 256 | 205 | 160 | 78 | 88 | 72 | 66 | 61 | 203 | 3.1x | 1.3x | 3.3x |
| 16 | 8.4 M | 577 | 496 | 366 | 315 | 148 | 141 | 112 | 105 | 96 | 375 | 3.5x | 1.5x | 3.9x |
| 32 | 16.8 M | 1160 | 921 | 725 | 612 | 241 | 280 | 209 | 182 | 163 | 735 | 4.0x | 1.5x | 4.5x |

### x = 128 x 128, 128 x 128 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | dev bf16 q | layer loop | dev bf16 vs best CPU | dev bf16 q vs best VNNI | fused q vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4.2 M | 230 | 126 | 123 | 117 | 38 | 39 | 59 | 53 | 47 | 72 | 2.3x | 0.8x | 1.5x |
| 8 | 16.8 M | 847 | 474 | 406 | 424 | 128 | 113 | 146 | 124 | 100 | 205 | 3.3x | 1.1x | 2.1x |
| 16 | 33.6 M | 1722 | 893 | 751 | 824 | 213 | 227 | 257 | 219 | 168 | 378 | 3.4x | 1.3x | 2.2x |
| 32 | 67.1 M | 3395 | 1736 | 1687 | 1614 | 422 | 451 | 482 | 398 | 308 | 723 | 4.2x | 1.4x | 2.3x |

### x = 128 x 256, 256 x 256 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | dev bf16 q | layer loop | dev bf16 vs best CPU | dev bf16 q vs best VNNI | fused q vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 16.8 M | 743 | 162 | 197 | 345 | 84 | 76 | 130 | 106 | 79 | 127 | 1.5x | 1.0x | 1.6x |
| 8 | 67.1 M | 2955 | 610 | 769 | 1299 | 264 | 246 | 394 | 317 | 216 | 425 | 1.9x | 1.1x | 2.0x |
| 16 | 134.2 M | 5804 | 1265 | 1603 | 2540 | 499 | 489 | 749 | 590 | 393 | 815 | 2.1x | 1.2x | 2.1x |
| 32 | 268.4 M | 11515 | 2639 | 3239 | 4970 | 1019 | 954 | 1453 | 1152 | 753 | 1602 | 2.3x | 1.3x | 2.1x |

### x = 128 x 512, 512 x 512 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | dev bf16 q | layer loop | dev bf16 vs best CPU | dev bf16 q vs best VNNI | fused q vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 67.1 M | 2800 | 469 | 466 | 1112 | 200 | 176 | n/a | 287 | 187 | 328 | 1.6x | 0.9x | 1.7x |
| 8 | 268.4 M | 10352 | 1850 | 1837 | 4250 | 785 | 684 | n/a | 1022 | 608 | 1207 | 1.8x | 1.1x | 2.0x |
| 16 | 536.9 M | 20568 | 3812 | 3706 | 8505 | 1562 | 1367 | n/a | 1948 | 1165 | 2403 | 1.9x | 1.2x | 2.1x |
| 32 | 1073.7 M | 41104 | 7659 | 7441 | 17031 | 3122 | 2698 | n/a | 3877 | 2345 | 4735 | 1.9x | 1.2x | 2.0x |
