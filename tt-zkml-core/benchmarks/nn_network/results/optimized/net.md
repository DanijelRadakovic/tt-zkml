
### x = 32 x 64, 64 x 64 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | dev bf16 q | layer loop | dev bf16 vs best CPU | dev bf16 q vs best VNNI | fused q vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.3 M | 29 | 58 | 54 | 20 | 20 | 23 | 31 | 29 | 41 | 58 | 1.0x | 0.5x | 1.4x |
| 8 | 1.0 M | 107 | 224 | 170 | 72 | 58 | 73 | 54 | 58 | 51 | 194 | 1.9x | 1.1x | 3.8x |
| 16 | 2.1 M | 214 | 441 | 334 | 145 | 112 | 122 | 76 | 75 | 59 | 375 | 2.8x | 1.9x | 6.4x |
| 32 | 4.2 M | 428 | 828 | 656 | 297 | 199 | 231 | 130 | 105 | 91 | 725 | 4.1x | 2.2x | 8.0x |

### x = 32 x 128, 128 x 128 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | dev bf16 q | layer loop | dev bf16 vs best CPU | dev bf16 q vs best VNNI | fused q vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.0 M | 88 | 102 | 101 | 60 | 33 | 28 | 42 | 37 | 34 | 60 | 2.4x | 0.9x | 1.8x |
| 8 | 4.2 M | 347 | 371 | 357 | 231 | 92 | 98 | 94 | 73 | 49 | 195 | 4.7x | 1.9x | 4.0x |
| 16 | 8.4 M | 674 | 717 | 573 | 464 | 156 | 174 | 162 | 122 | 72 | 364 | 4.7x | 2.2x | 5.1x |
| 32 | 16.8 M | 1348 | 1428 | 1337 | 919 | 296 | 342 | 299 | 222 | 126 | 724 | 6.0x | 2.4x | 5.7x |

### x = 32 x 256, 256 x 256 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | dev bf16 q | layer loop | dev bf16 vs best CPU | dev bf16 q vs best VNNI | fused q vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4.2 M | 336 | 103 | 144 | 212 | 58 | 53 | 84 | 63 | 42 | 75 | 1.6x | 1.3x | 1.8x |
| 8 | 16.8 M | 1300 | 357 | 533 | 836 | 179 | 175 | 255 | 182 | 87 | 217 | 2.0x | 2.0x | 2.5x |
| 16 | 33.6 M | 2661 | 764 | 1096 | 1618 | 359 | 343 | 484 | 333 | 152 | 405 | 2.3x | 2.3x | 2.7x |
| 32 | 67.1 M | 5319 | 1585 | 2096 | 3239 | 714 | 675 | 937 | 633 | 272 | 788 | 2.5x | 2.5x | 2.9x |

### x = 32 x 512, 512 x 512 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | dev bf16 q | layer loop | dev bf16 vs best CPU | dev bf16 q vs best VNNI | fused q vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 16.8 M | 1275 | 244 | 239 | 783 | 151 | 127 | n/a | 168 | 83 | 184 | 1.4x | 1.5x | 2.2x |
| 8 | 67.1 M | 4936 | 938 | 981 | 3145 | 578 | 490 | n/a | 573 | 254 | 635 | 1.6x | 1.9x | 2.5x |
| 16 | 134.2 M | 10088 | 1923 | 1883 | 6205 | 1153 | 968 | n/a | 1116 | 470 | 1245 | 1.7x | 2.1x | 2.7x |
| 32 | 268.4 M | 19981 | 3879 | 3806 | 12316 | 2377 | 1947 | n/a | 2200 | 949 | 2487 | 1.7x | 2.1x | 2.6x |

### x = 128 x 64, 64 x 64 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | dev bf16 q | layer loop | dev bf16 vs best CPU | dev bf16 q vs best VNNI | fused q vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.0 M | 80 | 69 | 56 | 46 | 26 | 26 | 38 | 36 | 43 | 61 | 1.6x | 0.6x | 1.4x |
| 8 | 4.2 M | 305 | 256 | 202 | 158 | 76 | 85 | 73 | 65 | 73 | 198 | 3.1x | 1.0x | 2.7x |
| 16 | 8.4 M | 582 | 496 | 371 | 313 | 149 | 161 | 117 | 103 | 95 | 375 | 3.6x | 1.6x | 3.9x |
| 32 | 16.8 M | 1183 | 903 | 775 | 615 | 242 | 279 | 210 | 182 | 170 | 729 | 4.3x | 1.4x | 4.3x |

### x = 128 x 128, 128 x 128 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | dev bf16 q | layer loop | dev bf16 vs best CPU | dev bf16 q vs best VNNI | fused q vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4.2 M | 225 | 129 | 120 | 116 | 38 | 39 | 59 | 58 | 47 | 74 | 2.1x | 0.8x | 1.6x |
| 8 | 16.8 M | 858 | 451 | 464 | 420 | 125 | 132 | 146 | 130 | 100 | 204 | 3.5x | 1.3x | 2.1x |
| 16 | 33.6 M | 1688 | 898 | 806 | 840 | 219 | 226 | 257 | 218 | 166 | 378 | 3.7x | 1.3x | 2.3x |
| 32 | 67.1 M | 3426 | 1739 | 1812 | 1614 | 426 | 445 | 476 | 400 | 305 | 716 | 4.3x | 1.4x | 2.4x |

### x = 128 x 256, 256 x 256 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | dev bf16 q | layer loop | dev bf16 vs best CPU | dev bf16 q vs best VNNI | fused q vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 16.8 M | 741 | 164 | 203 | 349 | 85 | 78 | 131 | 104 | 78 | 130 | 1.6x | 1.0x | 1.7x |
| 8 | 67.1 M | 2872 | 605 | 761 | 1278 | 258 | 252 | 393 | 315 | 217 | 423 | 1.9x | 1.2x | 1.9x |
| 16 | 134.2 M | 5751 | 1233 | 1576 | 2514 | 504 | 486 | 745 | 592 | 392 | 815 | 2.1x | 1.2x | 2.1x |
| 32 | 268.4 M | 11539 | 2612 | 3214 | 5079 | 1022 | 968 | 1488 | 1144 | 752 | 1600 | 2.3x | 1.3x | 2.1x |

### x = 128 x 512, 512 x 512 layers (network time in us; CPU = plain C++, VNNI = AVX-512 VNNI matmul, 1/6/12 threads)

| L | MACs | CPU 1 | CPU 6 | CPU 12 | VNNI 1 | VNNI 6 | VNNI 12 | dev fp32 | dev bf16 | dev bf16 q | layer loop | dev bf16 vs best CPU | dev bf16 q vs best VNNI | fused q vs loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 67.1 M | 2604 | 470 | 463 | 1121 | 205 | 182 | n/a | 285 | 185 | 328 | 1.6x | 1.0x | 1.8x |
| 8 | 268.4 M | 10339 | 1920 | 1853 | 4386 | 783 | 685 | n/a | 1021 | 608 | 1198 | 1.8x | 1.1x | 2.0x |
| 16 | 536.9 M | 20693 | 3810 | 3712 | 8662 | 1563 | 1378 | n/a | 1972 | 1164 | 2401 | 1.9x | 1.2x | 2.1x |
| 32 | 1073.7 M | 42377 | 7659 | 7670 | 17050 | 3144 | 2765 | n/a | 3878 | 2311 | 4683 | 2.0x | 1.2x | 2.0x |
