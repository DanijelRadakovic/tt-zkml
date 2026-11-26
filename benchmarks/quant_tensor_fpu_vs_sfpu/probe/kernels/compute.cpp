// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"
#include "pot_scale_sfpu.hpp"

constexpr uint32_t C0 = tt::CBIndex::c_0, C1 = tt::CBIndex::c_1, C16 = tt::CBIndex::c_16, C2 = tt::CBIndex::c_2;

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);  // resident tiles in c_0 (multiple of 4)
    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t reps = get_compile_time_arg_val(1);
    CircularBuffer cb_in(C0), cb_sc(C1), cb_out(C16), cb_ones(C2);
    compute_kernel_hw_startup(C0, C16);

    // c_1 = tile of 4.0f (Float32), c_2 = tile of 1.0f (bf16)
    cb_sc.reserve_back(1);
    tile_regs_acquire(); fill_tile_init(); fill_tile(0, 4.0f); tile_regs_commit(); tile_regs_wait();
    pack_reconfig_data_format(C1); pack_tile(0, C1); tile_regs_release(); cb_sc.push_back(1);
    cb_ones.reserve_back(1);
    tile_regs_acquire(); fill_tile_init(); fill_tile(0, mode == 1 ? -1.0f : 1.0f); tile_regs_commit(); tile_regs_wait();
    pack_reconfig_data_format(C2); pack_tile(0, C2); tile_regs_release(); cb_ones.push_back(1);
    cb_sc.wait_front(1); cb_ones.wait_front(1);
    cb_in.wait_front(num_tiles);
    cb_out.reserve_back(1);

    tile_regs_acquire();
    {
        DeviceZoneScopedN("LOOP");
        if constexpr (mode == 100) {            // current pass 1: copy_tile + SFPU absmax, blocks of 4
            MATH(zkml::sfpu::absmax_init());
            copy_init(C0);
            for (uint32_t r = 0; r < reps; r++)
                for (uint32_t b = 0; b < num_tiles; b += 4) {
                    for (uint32_t i = 0; i < 4; i++) copy_tile(C0, b + i, i);
                    for (uint32_t i = 0; i < 4; i++) MATH(zkml::sfpu::absmax_tile(i));
                }
        } else if constexpr (mode == 105) {     // only copy_tile (A2D), blocks of 4
            copy_init(C0);
            for (uint32_t r = 0; r < reps; r++)
                for (uint32_t b = 0; b < num_tiles; b += 4)
                    for (uint32_t i = 0; i < 4; i++) copy_tile(C0, b + i, i);
        } else if constexpr (mode == 106) {     // only the SFPU absmax on garbage DST, blocks of 4
            MATH(zkml::sfpu::absmax_init());
            for (uint32_t r = 0; r < reps; r++)
                for (uint32_t b = 0; b < num_tiles; b += 4)
                    for (uint32_t i = 0; i < 4; i++) MATH(zkml::sfpu::absmax_tile(i));
        } else if constexpr (mode == 101 || mode == 0 || mode == 1) {     // FPU GMPOOL, column max, accumulate in DST 0 (0/1: scaler +1/-1)
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(C0, C2, C16);
            for (uint32_t r = 0; r < reps; r++)
                for (uint32_t t = 0; t < num_tiles; t++)
                    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(C0, C2, t, 0, 0);
        } else if constexpr (mode == 102 || mode == 2) {     // FPU GMPOOL, scalar max
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(C0, C2, C16);
            for (uint32_t r = 0; r < reps; r++)
                for (uint32_t t = 0; t < num_tiles; t++)
                    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(C0, C2, t, 0, 0);
        }
    }
    tile_regs_commit(); tile_regs_wait();
    pack_reconfig_data_format(C16); pack_tile(0, C16); tile_regs_release();
    cb_out.push_back(1);
    // reduce_uninit resets the packer's edge masks, so it comes after the pack of the reduce result.
    if constexpr (mode == 101 || mode == 102 || mode <= 2) reduce_uninit();

    if constexpr (mode == 103 || mode == 104) {  // pass-2 multiply: FPU mul_tiles vs copy + SFPU mul, blocks of 3
        {
            DeviceZoneScopedN("LOOP2");
            if constexpr (mode == 103) mul_init(C0, C1);
            else { copy_init(C0); mul_binary_tile_init(); }
            for (uint32_t r = 0; r < reps; r++)
                for (uint32_t b = 0; b < num_tiles; b += 3) {
                    const uint32_t n = (num_tiles - b) < 3 ? (num_tiles - b) : 3;
                    tile_regs_acquire();
                    if constexpr (mode == 103) {
                        for (uint32_t i = 0; i < n; i++) mul_tiles(C0, C1, b + i, 0, i);
                    } else {
                        for (uint32_t i = 0; i < n; i++) copy_tile(C0, b + i, i);
                        reconfig_data_format_srca(C0, C1);
                        copy_init(C1);
                        copy_tile(C1, 0, 3);
                        for (uint32_t i = 0; i < n; i++) mul_binary_tile(i, 3, i);
                        reconfig_data_format_srca(C1, C0);
                        copy_init(C0);
                    }
                    tile_regs_commit(); tile_regs_wait(); tile_regs_release();
                }
        }
    }
}
