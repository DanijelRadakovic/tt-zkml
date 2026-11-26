// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Pass 2 of the 8-bit SFPU quantizer, repeated: for blocks of 3 tiles, copy the input tiles and the fp32 scaler tile
// into DST, quantize each tile to int8 (the variant under test), pack the block. The same sequence as the SFPU branch of
// pot_quantize_per_tensor's quantize.cpp (DST_TILES = 4: 3 data tiles + the scaler in slot 3, fp32 DST).
//
// Compile args: VARIANT (variants.hpp), MODE (0 = copy + quantize + pack, 1 = only the quantize calls on whatever DST holds:
// the SFPU cost on its own). Runtime args: reps, tiles (a multiple of 3).
// CBs: c_0 = input tiles (Float32, or Float16_b), c_1 = scaler tile (Float32), c_16 = output tiles (Int8 or UInt8), `tiles` pages.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/quantization.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"
#include "variants.hpp"

constexpr uint32_t C0 = tt::CBIndex::c_0, C1 = tt::CBIndex::c_1, C16 = tt::CBIndex::c_16;
constexpr uint32_t BLOCK = 3, DST_SCALE = 3;

template <uint32_t V>
ALWI void quant_one(uint32_t i) {
    if constexpr (V == zkml_bench::TWOS_LIB) {
        quant_int8_tile(i, DST_SCALE, i);
    } else {
        MATH((::ckernel::_sfpu_binary_check_<DST_SYNC_MODE>(i, DST_SCALE, i, VectorMode::RC)));
        MATH((_llk_math_eltwise_binary_sfpu_params_(zkml_bench::quant_tile_fn<V>, i, DST_SCALE, i, VectorMode::RC)));
    }
}

void kernel_main() {
    const uint32_t reps = get_arg_val<uint32_t>(0);
    const uint32_t tiles = get_arg_val<uint32_t>(1);
    constexpr uint32_t variant = get_compile_time_arg_val(0);
    constexpr uint32_t mode = get_compile_time_arg_val(1);
    CircularBuffer cb_in(C0), cb_scale(C1), cb_out(C16);

    compute_kernel_hw_startup(C0, C16);
    cb_in.wait_front(tiles);
    cb_scale.wait_front(1);
    cb_out.reserve_back(tiles);

    // Records the replay body and loads the LREG constants.
    if constexpr (variant == zkml_bench::TWOS_LIB) {
        quant_int8_tile_init(0);
    } else {
        MATH((::ckernel::llk_math_eltwise_binary_sfpu_init<::SfpuType::quant_int32>(zkml_bench::quant_init<variant>, 0u)));
    }

    {
        DeviceZoneScopedN("LOOP");
        for (uint32_t r = 0; r < reps; r++) {
            for (uint32_t b = 0; b < tiles; b += BLOCK) {
                tile_regs_acquire();
                if constexpr (mode == 0) {
                    copy_init(C0);
                    for (uint32_t i = 0; i < BLOCK; i++) {
                        copy_tile(C0, b + i, i);
                    }
                    reconfig_data_format_srca(C0, C1);
                    copy_init(C1);
                    copy_tile(C1, 0, DST_SCALE);
                }
                for (uint32_t i = 0; i < BLOCK; i++) {
                    quant_one<variant>(i);
                }
                if constexpr (mode == 0) {
                    reconfig_data_format_srca(C1, C0);  // back to the input format for the next block
                }
                tile_regs_commit();
                tile_regs_wait();
                if constexpr (mode == 0) {
                    for (uint32_t i = 0; i < BLOCK; i++) {
                        pack_tile<true>(i, C16, b + i);
                    }
                }
                tile_regs_release();
            }
        }
    }
    cb_out.push_back(tiles);
}
