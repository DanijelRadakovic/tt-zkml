// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/quantization.h"
#include "api/dataflow/circular_buffer.h"
#include "common/kernel_debug.hpp"

// clang-format off
/**
 * @brief Compute kernel for per-tensor PoT quantization on the SFPU:
 *        `q = round_half_even(x * 2^k + zero_point)`, clamped to the int8 range.
 *
 * `quant_tile(a, b, out)` computes `a * b + zero_point`. DST slot 1 therefore holds the reciprocal
 * scale `2^k`, not the scale `2^-k`. With a power-of-two value the multiply is exact in fp32. The
 * quantization is symmetric, so `zero_point = 0`.
 *
 * The scale tile in `c_1` is waited on once and copied into DST for every input tile, because DST is
 * released after each tile. The input may be Float16_b while the scale is always Float32, so the SrcA
 * data format is reconfigured between the two copies.
 *
 * | Circular buffer | Direction | Format                          | Contents                     |
 * |-----------------|-----------|---------------------------------|------------------------------|
 * | `c_0`           | consumes  | Float32 or Float16_b            | Input tiles                  |
 * | `c_1`           | consumes  | Float32                         | Scale tile (`2^k`), held     |
 * | `c_16`          | produces  | Int32, or UInt8 for Int8 output | Quantized tiles              |
 *
 * **Compile-time args**
 *
 * | Index | Name          | Values | Description                                                         |
 * |-------|---------------|--------|---------------------------------------------------------------------|
 * | 0     | `output_int8` | 0      | Int32 output (values in int8 range), two's complement               |
 * |       |               | 1      | Int8 output, two's complement bytes packed through the UInt8 packer |
 *
 * **Runtime args**
 *
 * | Index | Name        | Type     | Description                |
 * |-------|-------------|----------|----------------------------|
 * | 0     | `num_tiles` | uint32_t | Number of tiles to process |
 *
 * @pre  The program enables `fp32_dest_acc_en`. Float32 CBs use `UnpackToDestFp32`.
 * @note Rounding is round-half-to-even. A ZK circuit that checks this result must use the same rule.
 * @see ../dataflow/reader.cpp, ../dataflow/writer.cpp
 */
// clang-format on
void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_scale_id = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_id = tt::CBIndex::c_16;
    constexpr uint32_t zero_point_bits = 0;  // fp32 bits of 0.0f: symmetric quantization
    constexpr bool output_int8 = get_compile_time_arg_val(0) == 1;

    // DST slots
    constexpr uint32_t dst_x = 0;
    constexpr uint32_t dst_scale = 1;

    CircularBuffer cb_in(cb_in_id);
    CircularBuffer cb_scale(cb_scale_id);
    CircularBuffer cb_out(cb_out_id);

    compute_kernel_hw_startup(cb_in_id, cb_out_id);
    if constexpr (output_int8) {
        quant_int8_tile_init(zero_point_bits);
    } else {
        quant_tile_init(zero_point_bits);
    }

    cb_scale.wait_front(1);
    DPRINT_UNPACK("Scaler\n");
    UNPACK(zkml::debug::print_tile_rows(cb_scale_id, 0));

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_in.wait_front(1);
        DPRINT_UNPACK("Tile={}\n", t);
        UNPACK(zkml::debug::print_tile_rows(cb_in_id, 0));
        cb_out.reserve_back(1);

        tile_regs_acquire();
        copy_init(cb_in_id);
        copy_tile(cb_in_id, 0, dst_x);
        // The input may be Float16_b while the scale is always Float32.
        reconfig_data_format_srca(cb_in_id, cb_scale_id);
        copy_init(cb_scale_id);
        copy_tile(cb_scale_id, 0, dst_scale);
        if constexpr (output_int8) {
            quant_int8_tile(dst_x, dst_scale, dst_x);
            // DST holds each result as its two's-complement byte: negative q prints as 256 + q (-32 -> 224).
            DPRINT_MATH("Contents of DST 0 after quant_int8_tile() execution:\n");
            zkml::debug::print_dest_tile_rows(dst_x, DataFormat::Int32);
            DPRINT_MATH("Contents of DST 1 after quant_int8_tile() execution:\n");
            zkml::debug::print_dest_tile_rows(dst_scale, DataFormat::Float32);
        } else {
            quant_tile(dst_x, dst_scale, dst_x);
            DPRINT_MATH("Contents of DST 0 after quant_tile() execution:\n");
            zkml::debug::print_dest_tile_rows(dst_x, DataFormat::Int32);
            DPRINT_MATH("Contents of DST 1 after quant_tile() execution:\n");
            zkml::debug::print_dest_tile_rows(dst_scale, DataFormat::Float32);
        }
        reconfig_data_format_srca(cb_scale_id, cb_in_id);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(dst_x, cb_out_id);
        tile_regs_release();
        DPRINT_PACK("Tile={}\n", t);
        PACK(zkml::debug::print_tile_rows(cb_out_id, 0));
        cb_out.push_back(1);
        cb_in.pop_front(1);
    }

    cb_scale.pop_front(1);
}
