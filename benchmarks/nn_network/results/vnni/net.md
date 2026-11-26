
### x = 32 x 64, 64 x 64 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | layer loop | dev bf16 vs best CPU | dev bf16 vs best VNNI | fused vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.3 M | 29 | 60 | 47 | 20 | 20 | 22 | 56 | 50 | 60 | 0.6x | 0.4x | 1.2x |
| 8 | 1.0 M | 106 | 223 | 186 | 73 | 60 | 74 | 131 | 130 | 273 | 0.8x | 0.5x | 2.1x |
| 16 | 2.1 M | 215 | 435 | 315 | 144 | 113 | 134 | 230 | 230 | 370 | 0.9x | 0.5x | 1.6x |
| 32 | 4.2 M | 426 | 792 | 616 | 292 | 195 | 229 | 426 | 430 | 725 | 1.0x | 0.5x | 1.7x |

### x = 32 x 128, 128 x 128 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | layer loop | dev bf16 vs best CPU | dev bf16 vs best VNNI | fused vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.0 M | 88 | 105 | 100 | 60 | 32 | 29 | 59 | 76 | 60 | 1.2x | 0.4x | 0.8x |
| 8 | 4.2 M | 359 | 380 | 371 | 239 | 92 | 96 | 168 | 147 | 192 | 2.4x | 0.6x | 1.3x |
| 16 | 8.4 M | 702 | 674 | 711 | 466 | 148 | 172 | 307 | 268 | 360 | 2.5x | 0.6x | 1.3x |
| 32 | 16.8 M | 1373 | 1460 | 1352 | 898 | 293 | 327 | 589 | 507 | 713 | 2.7x | 0.6x | 1.4x |

### x = 32 x 256, 256 x 256 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | layer loop | dev bf16 vs best CPU | dev bf16 vs best VNNI | fused vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4.2 M | 332 | 115 | 146 | 212 | 58 | 52 | 103 | 82 | 75 | 1.4x | 0.6x | 0.9x |
| 8 | 16.8 M | 1312 | 395 | 533 | 813 | 179 | 170 | 342 | 283 | 215 | 1.4x | 0.6x | 0.8x |
| 16 | 33.6 M | 2598 | 695 | 1010 | 1648 | 363 | 344 | 667 | 492 | 406 | 1.4x | 0.7x | 0.8x |
| 32 | 67.1 M | 5265 | 1460 | 2179 | 3277 | 733 | 668 | 1354 | 946 | 784 | 1.5x | 0.7x | 0.8x |

### x = 32 x 512, 512 x 512 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | layer loop | dev bf16 vs best CPU | dev bf16 vs best VNNI | fused vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 16.8 M | 1266 | 234 | 242 | 784 | 145 | 121 | n/a | 194 | 183 | 1.2x | 0.6x | 0.9x |
| 8 | 67.1 M | 4953 | 954 | 950 | 3062 | 569 | 492 | n/a | 694 | 642 | 1.4x | 0.7x | 0.9x |
| 16 | 134.2 M | 10139 | 1962 | 1899 | 6197 | 1738 | 974 | n/a | 1338 | 1277 | 1.4x | 0.7x | 1.0x |
| 32 | 268.4 M | 20063 | 3967 | 3793 | 12336 | 2355 | 1962 | n/a | 2783 | 2490 | 1.4x | 0.7x | 0.9x |

### x = 128 x 64, 64 x 64 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | layer loop | dev bf16 vs best CPU | dev bf16 vs best VNNI | fused vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.0 M | 80 | 70 | 59 | 46 | 28 | 26 | 59 | 71 | 70 | 0.8x | 0.4x | 1.0x |
| 8 | 4.2 M | 298 | 262 | 201 | 156 | 77 | 87 | 154 | 147 | 276 | 1.4x | 0.5x | 1.9x |
| 16 | 8.4 M | 581 | 465 | 373 | 315 | 144 | 142 | 280 | 266 | 370 | 1.4x | 0.5x | 1.4x |
| 32 | 16.8 M | 1148 | 924 | 738 | 614 | 245 | 288 | 524 | 503 | 733 | 1.5x | 0.5x | 1.5x |

### x = 128 x 128, 128 x 128 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | layer loop | dev bf16 vs best CPU | dev bf16 vs best VNNI | fused vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4.2 M | 226 | 128 | 128 | 118 | 38 | 39 | 82 | 74 | 72 | 1.7x | 0.5x | 1.0x |
| 8 | 16.8 M | 856 | 435 | 454 | 433 | 131 | 116 | 241 | 216 | 200 | 2.0x | 0.5x | 0.9x |
| 16 | 33.6 M | 1709 | 822 | 872 | 825 | 213 | 226 | 444 | 405 | 374 | 2.0x | 0.5x | 0.9x |
| 32 | 67.1 M | 3348 | 1801 | 1649 | 1646 | 421 | 444 | 854 | 772 | 718 | 2.1x | 0.5x | 0.9x |

### x = 128 x 256, 256 x 256 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | layer loop | dev bf16 vs best CPU | dev bf16 vs best VNNI | fused vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 16.8 M | 741 | 173 | 208 | 356 | 85 | 65 | 164 | 134 | 132 | 1.3x | 0.5x | 1.0x |
| 8 | 67.1 M | 2915 | 650 | 803 | 1269 | 257 | 247 | 548 | 453 | 424 | 1.4x | 0.5x | 0.9x |
| 16 | 134.2 M | 5672 | 1296 | 1585 | 2503 | 513 | 487 | 1072 | 848 | 848 | 1.5x | 0.6x | 1.0x |
| 32 | 268.4 M | 11776 | 2518 | 3134 | 5101 | 1024 | 964 | 2060 | 1647 | 1608 | 1.5x | 0.6x | 1.0x |

### x = 128 x 512, 512 x 512 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | layer loop | dev bf16 vs best CPU | dev bf16 vs best VNNI | fused vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 67.1 M | 2691 | 481 | 490 | 1134 | 203 | 183 | n/a | 332 | 328 | 1.5x | 0.6x | 1.0x |
| 8 | 268.4 M | 10377 | 1892 | 1865 | 4258 | 780 | 698 | n/a | 1175 | 1239 | 1.6x | 0.6x | 1.1x |
| 16 | 536.9 M | 20484 | 3818 | 3784 | 8529 | 1567 | 1374 | n/a | 2444 | 2400 | 1.5x | 0.6x | 1.0x |
| 32 | 1073.7 M | 41325 | 7589 | 7455 | 17294 | 3186 | 2745 | n/a | 4746 | 4752 | 1.6x | 0.6x | 1.0x |
