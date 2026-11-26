
### x = 32 x 64, 64 x 64 layers

Forward pass and load (us). `pre-quant` forward passes do not quantize the weights; `load` quantizes them once.

| L | CPU per call | device per call | CPU pre-quant forward | device pre-quant forward | CPU load | device load | pre-quant forward: device vs CPU |
|---|---|---|---|---|---|---|---|
| 2 | 20 | 29 | 8 | 42 | 11 | 35 | 0.20x |
| 8 | 60 | 47 | 19 | 43 | 36 | 106 | 0.45x |
| 16 | 112 | 65 | 35 | 58 | 62 | 194 | 0.60x |
| 32 | 200 | 112 | 64 | 92 | 123 | 376 | 0.70x |

The whole workflow: load once, then N forward passes (total us; lower is better).

| L | N | device per call | device pre-quant | CPU per call | CPU pre-quant | best |
|---|---|---|---|---|---|---|
| 2 | 1 | 29 | 77 | 20 | 19 | CPU pre-quant |
| 2 | 10 | 289 | 454 | 199 | 93 | CPU pre-quant |
| 2 | 100 | 2890 | 4225 | 1990 | 831 | CPU pre-quant |
| 8 | 1 | 47 | 150 | 60 | 55 | device per call |
| 8 | 10 | 466 | 538 | 599 | 229 | CPU pre-quant |
| 8 | 100 | 4660 | 4416 | 5990 | 1966 | CPU pre-quant |
| 16 | 1 | 65 | 252 | 112 | 97 | device per call |
| 16 | 10 | 653 | 777 | 1122 | 413 | CPU pre-quant |
| 16 | 100 | 6530 | 6024 | 11.2 ms | 3572 | CPU pre-quant |
| 32 | 1 | 112 | 468 | 200 | 188 | device per call |
| 32 | 10 | 1117 | 1300 | 2000 | 768 | CPU pre-quant |
| 32 | 100 | 11.2 ms | 9616 | 20.0 ms | 6573 | CPU pre-quant |

### x = 32 x 128, 128 x 128 layers

Forward pass and load (us). `pre-quant` forward passes do not quantize the weights; `load` quantizes them once.

| L | CPU per call | device per call | CPU pre-quant forward | device pre-quant forward | CPU load | device load | pre-quant forward: device vs CPU |
|---|---|---|---|---|---|---|---|
| 2 | 28 | 45 | 10 | 32 | 17 | 52 | 0.31x |
| 8 | 93 | 74 | 24 | 49 | 54 | 107 | 0.48x |
| 16 | 156 | 124 | 44 | 72 | 108 | 197 | 0.61x |
| 32 | 316 | 219 | 88 | 128 | 220 | 373 | 0.69x |

The whole workflow: load once, then N forward passes (total us; lower is better).

| L | N | device per call | device pre-quant | CPU per call | CPU pre-quant | best |
|---|---|---|---|---|---|---|
| 2 | 1 | 45 | 84 | 28 | 27 | CPU pre-quant |
| 2 | 10 | 448 | 374 | 281 | 116 | CPU pre-quant |
| 2 | 100 | 4480 | 3272 | 2810 | 1007 | CPU pre-quant |
| 8 | 1 | 74 | 156 | 93 | 78 | device per call |
| 8 | 10 | 745 | 599 | 930 | 292 | CPU pre-quant |
| 8 | 100 | 7450 | 5027 | 9300 | 2434 | CPU pre-quant |
| 16 | 1 | 124 | 269 | 156 | 153 | device per call |
| 16 | 10 | 1245 | 917 | 1565 | 550 | CPU pre-quant |
| 16 | 100 | 12.4 ms | 7397 | 15.7 ms | 4528 | CPU pre-quant |
| 32 | 1 | 219 | 501 | 316 | 308 | device per call |
| 32 | 10 | 2193 | 1654 | 3156 | 1100 | CPU pre-quant |
| 32 | 100 | 21.9 ms | 13.2 ms | 31.6 ms | 9030 | CPU pre-quant |

### x = 32 x 256, 256 x 256 layers

Forward pass and load (us). `pre-quant` forward passes do not quantize the weights; `load` quantizes them once.

| L | CPU per call | device per call | CPU pre-quant forward | device pre-quant forward | CPU load | device load | pre-quant forward: device vs CPU |
|---|---|---|---|---|---|---|---|
| 2 | 53 | 70 | 12 | 42 | 38 | 75 | 0.29x |
| 8 | 173 | 178 | 38 | 86 | 149 | 254 | 0.44x |
| 16 | 342 | 333 | 78 | 147 | 294 | 484 | 0.53x |
| 32 | 668 | 631 | 152 | 274 | 580 | 978 | 0.55x |

The whole workflow: load once, then N forward passes (total us; lower is better).

| L | N | device per call | device pre-quant | CPU per call | CPU pre-quant | best |
|---|---|---|---|---|---|---|
| 2 | 1 | 70 | 117 | 53 | 51 | CPU pre-quant |
| 2 | 10 | 698 | 499 | 530 | 170 | CPU pre-quant |
| 2 | 100 | 6980 | 4315 | 5300 | 1297 | CPU pre-quant |
| 8 | 1 | 178 | 341 | 173 | 192 | CPU per call |
| 8 | 10 | 1777 | 1120 | 1727 | 537 | CPU pre-quant |
| 8 | 100 | 17.8 ms | 8904 | 17.3 ms | 3984 | CPU pre-quant |
| 16 | 1 | 333 | 630 | 342 | 376 | device per call |
| 16 | 10 | 3330 | 1950 | 3417 | 1089 | CPU pre-quant |
| 16 | 100 | 33.3 ms | 15.1 ms | 34.2 ms | 8145 | CPU pre-quant |
| 32 | 1 | 631 | 1252 | 668 | 742 | device per call |
| 32 | 10 | 6313 | 3722 | 6681 | 2138 | CPU pre-quant |
| 32 | 100 | 63.1 ms | 28.4 ms | 66.8 ms | 15.8 ms | CPU pre-quant |

### x = 32 x 512, 512 x 512 layers

Forward pass and load (us). `pre-quant` forward passes do not quantize the weights; `load` quantizes them once.

| L | CPU per call | device per call | CPU pre-quant forward | device pre-quant forward | CPU load | device load | pre-quant forward: device vs CPU |
|---|---|---|---|---|---|---|---|
| 2 | 122 | 168 | 24 | 86 | 125 | 223 | 0.28x |
| 8 | 493 | 574 | 86 | 252 | 468 | 837 | 0.34x |
| 16 | 982 | 1149 | 170 | 472 | 918 | 1656 | 0.36x |
| 32 | 1945 | 2231 | 333 | 916 | 1897 | 3377 | 0.36x |

The whole workflow: load once, then N forward passes (total us; lower is better).

| L | N | device per call | device pre-quant | CPU per call | CPU pre-quant | best |
|---|---|---|---|---|---|---|
| 2 | 1 | 168 | 309 | 122 | 151 | CPU per call |
| 2 | 10 | 1680 | 1085 | 1219 | 378 | CPU pre-quant |
| 2 | 100 | 16.8 ms | 8843 | 12.2 ms | 2573 | CPU pre-quant |
| 8 | 1 | 574 | 1089 | 493 | 556 | CPU per call |
| 8 | 10 | 5745 | 3360 | 4927 | 1345 | CPU pre-quant |
| 8 | 100 | 57.5 ms | 26.1 ms | 49.3 ms | 9091 | CPU pre-quant |
| 16 | 1 | 1149 | 2128 | 982 | 1090 | CPU per call |
| 16 | 10 | 11.5 ms | 6376 | 9825 | 2645 | CPU pre-quant |
| 16 | 100 | 114.9 ms | 48.9 ms | 98.2 ms | 18.0 ms | CPU pre-quant |
| 32 | 1 | 2231 | 4292 | 1945 | 2234 | CPU per call |
| 32 | 10 | 22.3 ms | 12.5 ms | 19.5 ms | 5264 | CPU pre-quant |
| 32 | 100 | 223.1 ms | 95.0 ms | 194.5 ms | 35.5 ms | CPU pre-quant |

### x = 128 x 64, 64 x 64 layers

Forward pass and load (us). `pre-quant` forward passes do not quantize the weights; `load` quantizes them once.

| L | CPU per call | device per call | CPU pre-quant forward | device pre-quant forward | CPU load | device load | pre-quant forward: device vs CPU |
|---|---|---|---|---|---|---|---|
| 2 | 25 | 36 | 14 | 36 | 9 | 35 | 0.38x |
| 8 | 78 | 66 | 32 | 61 | 36 | 105 | 0.53x |
| 16 | 141 | 105 | 64 | 96 | 64 | 192 | 0.67x |
| 32 | 241 | 182 | 121 | 163 | 126 | 369 | 0.74x |

The whole workflow: load once, then N forward passes (total us; lower is better).

| L | N | device per call | device pre-quant | CPU per call | CPU pre-quant | best |
|---|---|---|---|---|---|---|
| 2 | 1 | 36 | 71 | 25 | 22 | CPU pre-quant |
| 2 | 10 | 356 | 393 | 254 | 146 | CPU pre-quant |
| 2 | 100 | 3560 | 3615 | 2540 | 1379 | CPU pre-quant |
| 8 | 1 | 66 | 166 | 78 | 68 | device per call |
| 8 | 10 | 655 | 717 | 785 | 358 | CPU pre-quant |
| 8 | 100 | 6550 | 6225 | 7850 | 3265 | CPU pre-quant |
| 16 | 1 | 105 | 288 | 141 | 129 | device per call |
| 16 | 10 | 1050 | 1149 | 1411 | 708 | CPU pre-quant |
| 16 | 100 | 10.5 ms | 9762 | 14.1 ms | 6504 | CPU pre-quant |
| 32 | 1 | 182 | 532 | 241 | 246 | device per call |
| 32 | 10 | 1823 | 2003 | 2410 | 1332 | CPU pre-quant |
| 32 | 100 | 18.2 ms | 16.7 ms | 24.1 ms | 12.2 ms | CPU pre-quant |

### x = 128 x 128, 128 x 128 layers

Forward pass and load (us). `pre-quant` forward passes do not quantize the weights; `load` quantizes them once.

| L | CPU per call | device per call | CPU pre-quant forward | device pre-quant forward | CPU load | device load | pre-quant forward: device vs CPU |
|---|---|---|---|---|---|---|---|
| 2 | 38 | 53 | 22 | 47 | 18 | 52 | 0.46x |
| 8 | 113 | 124 | 56 | 100 | 54 | 111 | 0.56x |
| 16 | 213 | 219 | 109 | 168 | 114 | 197 | 0.65x |
| 32 | 422 | 398 | 216 | 308 | 222 | 371 | 0.70x |

The whole workflow: load once, then N forward passes (total us; lower is better).

| L | N | device per call | device pre-quant | CPU per call | CPU pre-quant | best |
|---|---|---|---|---|---|---|
| 2 | 1 | 53 | 99 | 38 | 39 | CPU per call |
| 2 | 10 | 533 | 521 | 381 | 233 | CPU pre-quant |
| 2 | 100 | 5330 | 4742 | 3810 | 2168 | CPU pre-quant |
| 8 | 1 | 124 | 210 | 113 | 110 | CPU pre-quant |
| 8 | 10 | 1245 | 1109 | 1128 | 614 | CPU pre-quant |
| 8 | 100 | 12.4 ms | 10.1 ms | 11.3 ms | 5654 | CPU pre-quant |
| 16 | 1 | 219 | 365 | 213 | 223 | CPU per call |
| 16 | 10 | 2188 | 1880 | 2129 | 1204 | CPU pre-quant |
| 16 | 100 | 21.9 ms | 17.0 ms | 21.3 ms | 11.0 ms | CPU pre-quant |
| 32 | 1 | 398 | 679 | 422 | 437 | device per call |
| 32 | 10 | 3981 | 3452 | 4218 | 2378 | CPU pre-quant |
| 32 | 100 | 39.8 ms | 31.2 ms | 42.2 ms | 21.8 ms | CPU pre-quant |

### x = 128 x 256, 256 x 256 layers

Forward pass and load (us). `pre-quant` forward passes do not quantize the weights; `load` quantizes them once.

| L | CPU per call | device per call | CPU pre-quant forward | device pre-quant forward | CPU load | device load | pre-quant forward: device vs CPU |
|---|---|---|---|---|---|---|---|
| 2 | 76 | 106 | 35 | 79 | 37 | 76 | 0.44x |
| 8 | 246 | 317 | 117 | 216 | 148 | 256 | 0.54x |
| 16 | 489 | 590 | 227 | 393 | 294 | 480 | 0.58x |
| 32 | 954 | 1152 | 456 | 753 | 580 | 947 | 0.61x |

The whole workflow: load once, then N forward passes (total us; lower is better).

| L | N | device per call | device pre-quant | CPU per call | CPU pre-quant | best |
|---|---|---|---|---|---|---|
| 2 | 1 | 106 | 155 | 76 | 72 | CPU pre-quant |
| 2 | 10 | 1063 | 870 | 762 | 383 | CPU pre-quant |
| 2 | 100 | 10.6 ms | 8016 | 7620 | 3497 | CPU pre-quant |
| 8 | 1 | 317 | 472 | 246 | 265 | CPU per call |
| 8 | 10 | 3169 | 2413 | 2460 | 1320 | CPU pre-quant |
| 8 | 100 | 31.7 ms | 21.8 ms | 24.6 ms | 11.9 ms | CPU pre-quant |
| 16 | 1 | 590 | 873 | 489 | 526 | CPU per call |
| 16 | 10 | 5901 | 4412 | 4893 | 2583 | CPU pre-quant |
| 16 | 100 | 59.0 ms | 39.8 ms | 48.9 ms | 23.0 ms | CPU pre-quant |
| 32 | 1 | 1152 | 1700 | 954 | 1036 | CPU per call |
| 32 | 10 | 11.5 ms | 8477 | 9538 | 5136 | CPU pre-quant |
| 32 | 100 | 115.2 ms | 76.2 ms | 95.4 ms | 46.1 ms | CPU pre-quant |

### x = 128 x 512, 512 x 512 layers

Forward pass and load (us). `pre-quant` forward passes do not quantize the weights; `load` quantizes them once.

| L | CPU per call | device per call | CPU pre-quant forward | device pre-quant forward | CPU load | device load | pre-quant forward: device vs CPU |
|---|---|---|---|---|---|---|---|
| 2 | 176 | 287 | 80 | 187 | 122 | 223 | 0.42x |
| 8 | 684 | 1022 | 287 | 608 | 471 | 835 | 0.47x |
| 16 | 1367 | 1948 | 565 | 1165 | 918 | 1692 | 0.49x |
| 32 | 2698 | 3877 | 1125 | 2345 | 1860 | 3375 | 0.48x |

The whole workflow: load once, then N forward passes (total us; lower is better).

| L | N | device per call | device pre-quant | CPU per call | CPU pre-quant | best |
|---|---|---|---|---|---|---|
| 2 | 1 | 287 | 411 | 176 | 202 | CPU per call |
| 2 | 10 | 2874 | 2097 | 1755 | 917 | CPU pre-quant |
| 2 | 100 | 28.7 ms | 19.0 ms | 17.6 ms | 8072 | CPU pre-quant |
| 8 | 1 | 1022 | 1444 | 684 | 758 | CPU per call |
| 8 | 10 | 10.2 ms | 6919 | 6843 | 3342 | CPU pre-quant |
| 8 | 100 | 102.2 ms | 61.7 ms | 68.4 ms | 29.2 ms | CPU pre-quant |
| 16 | 1 | 1948 | 2857 | 1367 | 1483 | CPU per call |
| 16 | 10 | 19.5 ms | 13.3 ms | 13.7 ms | 6570 | CPU pre-quant |
| 16 | 100 | 194.8 ms | 118.2 ms | 136.7 ms | 57.4 ms | CPU pre-quant |
| 32 | 1 | 3877 | 5721 | 2698 | 2985 | CPU per call |
| 32 | 10 | 38.8 ms | 26.8 ms | 27.0 ms | 13.1 ms | CPU pre-quant |
| 32 | 100 | 387.7 ms | 237.9 ms | 269.8 ms | 114.4 ms | CPU pre-quant |
