// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Matmul of resident tiles on the FPU: C (MT x NT tiles) = A (MT x KT) * B (KT x NT), accumulated in DST, repeated
// `reps` times and packed after every repetition. Whether it is an int8 or a bf16 matmul is decided only by the data
// formats of c_0 and c_1 (Int8 turns on ALU_ACC_CTRL_INT8_math_enabled, Float16_b does not); the code is the same.
//
// Compile args: MT, NT, MODE (0 = matmul_tiles, 1 = matmul_block).
// Runtime args: reps, KT.
//
// CBs: c_0 = A tiles, c_1 = B tiles, c_16 = MT*NT output tiles (Int32 for int8, Float32 for bf16).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t reps = get_arg_val<uint32_t>(0);
    const uint32_t kt = get_arg_val<uint32_t>(1);
    constexpr uint32_t mt = get_compile_time_arg_val(0);
    constexpr uint32_t nt = get_compile_time_arg_val(1);
    constexpr uint32_t mode = get_compile_time_arg_val(2);
    constexpr uint32_t C0 = tt::CBIndex::c_0, C1 = tt::CBIndex::c_1, C16 = tt::CBIndex::c_16;
    CircularBuffer cb_a(C0), cb_b(C1), cb_out(C16);

    compute_kernel_hw_startup<SrcOrder::Reverse>(C0, C1, C16);
    if constexpr (mode == 0) {
        matmul_init(C0, C1);
    } else {
        matmul_block_init(C0, C1, 0, nt, mt, kt);
    }
    cb_a.wait_front(mt * kt);
    cb_b.wait_front(kt * nt);
    cb_out.reserve_back(mt * nt);

    {
        DeviceZoneScopedN("MATMUL");
        for (uint32_t r = 0; r < reps; r++) {
            tile_regs_acquire();
            if constexpr (mode == 0) {
                for (uint32_t k = 0; k < kt; k++) {
                    for (uint32_t m = 0; m < mt; m++) {
                        for (uint32_t n = 0; n < nt; n++) {
                            matmul_tiles(C0, C1, m * kt + k, k * nt + n, m * nt + n);
                        }
                    }
                }
            } else {
                // One call per step of the inner dimension; the block is MT x NT output tiles. A tiles of step k
                // are at k, k + kt, ... (row-major MT x KT), B tiles at k * nt ... (row-major KT x NT).
                for (uint32_t k = 0; k < kt; k++) {
                    matmul_block(C0, C1, k, k * nt, 0, 0, nt, mt, kt);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < mt * nt; i++) {
                pack_tile<true>(i, C16, i);
            }
            tile_regs_release();
        }
    }
    cb_out.push_back(mt * nt);
}
