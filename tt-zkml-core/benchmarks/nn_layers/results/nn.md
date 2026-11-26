
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
