// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/quantization.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
#include "common/kernel_debug.hpp"
#include "pot_scale_sfpu.hpp"

// clang-format off
/**
 * @brief Compute kernel for per-tensor PoT quantization with a device-computed scaler:
 *        `k = max{k : absmax(x) * 2^k <= Q}`, `q = clamp(round_half_even(x * 2^k), -Q, Q)`,
 *        with `Q = 2^(bit_width - 1) - 1`.
 *
 * It reads the input twice.
 *
 * Pass 1, absmax: each input tile is copied into DST and folded into a running per-lane absmax
 * that stays in an SFPU register (see pot_scale_sfpu.hpp). After the last tile the 32 lanes are
 * reduced and turned into the scaler `2^k`, still in the SFPU register.
 *
 * The scaler is then broadcast into a DST tile and packed to `c_1` (reused by pass 2) and to
 * `c_17` (returned to the host).
 *
 * Pass 2, quantize: each input tile and the scaler tile are copied into DST and quantized.
 * - 8 bits: `quant_tile` / `quant_int8_tile`. Their clamp is [-128, 127], not [-127, 127], but
 *   `|x * 2^k| <= 127` by the choice of k, so -128 is never produced.
 * - 16 and 32 bits: `quant_tile` clamps to the int8 range and its RNE trick only holds for
 *   `|v| < 2^22`, so these widths compose SFPU ops instead: multiply by `2^k` (exact), round half
 *   to even, clamp to [-Q, Q] and convert fp32 -> int32 (exact, the value is already an integer).
 *
 * | Circular buffer | Direction          | Format                          | Contents                       |
 * |-----------------|--------------------|---------------------------------|--------------------------------|
 * | `c_0`           | consumes           | Float32 or Float16_b            | Input tiles, twice             |
 * | `c_1`           | produces, consumes | Float32                         | Scaler tile (`2^k`), held      |
 * | `c_16`          | produces           | Int32, or UInt8 for Int8 output | Quantized tiles                |
 * | `c_17`          | produces           | Float32                         | Scaler tile for the host       |
 *
 * **Compile-time args**
 *
 * | Index | Name          | Values     | Description                                                         |
 * |-------|---------------|------------|---------------------------------------------------------------------|
 * | 0     | `output_int8` | 0          | Int32 output, two's complement                                      |
 * |       |               | 1          | Int8 output, two's complement bytes packed through the UInt8 packer |
 * | 1     | `bit_width`   | 8, 16, 32  | Quantized width; Int8 output requires 8                             |
 *
 * **Runtime args**
 *
 * | Index | Name        | Type     | Description                       |
 * |-------|-------------|----------|-----------------------------------|
 * | 0     | `num_tiles` | uint32_t | Number of tiles in the input      |
 *
 * @pre  The program enables `fp32_dest_acc_en`. Float32 CBs use `UnpackToDestFp32`.
 * @pre  The input is finite. An Inf or NaN absmax gives a meaningless scaler.
 * @note Rounding is round-half-to-even. A ZK circuit that checks this result must use the same rule.
 * @see ../dataflow/reader.cpp, ../dataflow/writer.cpp, pot_scale_sfpu.hpp
 */
// clang-format on
void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr bool output_int8 = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t bit_width = get_compile_time_arg_val(1);
    static_assert(bit_width == 8 || bit_width == 16 || bit_width == 32, "bit_width must be 8, 16 or 32");
    static_assert(!output_int8 || bit_width == 8, "Int8 output requires bit_width 8");

    constexpr uint32_t cb_in_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_scale_id = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_id = tt::CBIndex::c_16;
    constexpr uint32_t cb_scale_out_id = tt::CBIndex::c_17;
    constexpr uint32_t zero_point_bits = 0;  // fp32 bits of 0.0f: symmetric quantization

    // fp32 bit patterns of -Q and Q for the 16/32-bit clamp. 2^31 - 1 rounds to 2^31 in fp32; the
    // fp32 -> int32 typecast saturates 2^31 to INT32_MAX, and |x * 2^k| <= Q keeps it unreachable.
    constexpr float q_max = static_cast<float>((1ull << (bit_width - 1)) - 1);
    constexpr uint32_t q_max_bits = __builtin_bit_cast(uint32_t, q_max);
    constexpr uint32_t q_min_bits = __builtin_bit_cast(uint32_t, -q_max);

    // DST slots
    constexpr uint32_t dst_x = 0;
    constexpr uint32_t dst_scale = 1;

    CircularBuffer cb_in(cb_in_id);
    CircularBuffer cb_scale(cb_scale_id);
    CircularBuffer cb_out(cb_out_id);
    CircularBuffer cb_scale_out(cb_scale_out_id);

    compute_kernel_hw_startup(cb_in_id, cb_out_id);

    // ---- Pass 1: absmax over the whole tensor, kept in an SFPU register ----
    // The whole pass and the scaler below share one DST section: nothing is packed until the
    // scaler, so one MATH -> PACK handshake replaces one per input tile. Every copy_tile overwrites
    // dst_x; the SFPU stall in the next absmax_tile orders it after the copy.
    MATH(zkml::sfpu::absmax_init());
    copy_init(cb_in_id);
    cb_scale.reserve_back(1);
    cb_scale_out.reserve_back(1);
    tile_regs_acquire();
    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_in.wait_front(1);
        copy_tile(cb_in_id, 0, dst_x);
        MATH(zkml::sfpu::absmax_tile(dst_x));
        cb_in.pop_front(1);
    }

    // ---- Scaler: SFPU register -> DST -> c_1 (for pass 2) and c_17 (for the host) ----
    // TODO(perf): quant_tile() and mul_binary_tile() read the scale from DST, so the scaler that is
    // already in an SFPU register is broadcast into a DST tile, packed to c_1 and copied back into
    // DST for every input tile. A quant variant that takes the scale from an LREG would remove the
    // broadcast, the pack and the per-tile copy.
    MATH(zkml::sfpu::absmax_to_scaler<bit_width>(dst_scale));
    DPRINT_MATH("Scaler in DST {}:\n", dst_scale);
    zkml::debug::print_dest_tile_rows(dst_scale, DataFormat::Float32, 0, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_scale_id);
    pack_tile(dst_scale, cb_scale_id);
    pack_tile(dst_scale, cb_scale_out_id);
    tile_regs_release();
    cb_scale.push_back(1);
    cb_scale_out.push_back(1);
    pack_reconfig_data_format(cb_out_id);

    // ---- Pass 2: quantize ----
    // TODO(perf): pass 2 reads the whole input from DRAM a second time. Keeping the tiles in L1
    // (when they fit) or fusing the absmax into a previous op would remove the re-read.
    if constexpr (bit_width == 8) {
        // Records the replay body and loads LREG constants, so it must come after pass 1.
        if constexpr (output_int8) {
            quant_int8_tile_init(zero_point_bits);
        } else {
            quant_tile_init(zero_point_bits);
        }
    }

    cb_scale.wait_front(1);
    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_in.wait_front(1);
        DPRINT_UNPACK("Tile={}\n", t);
        UNPACK(zkml::debug::print_tile_rows(cb_in_id, 0));
        cb_out.reserve_back(1);

        tile_regs_acquire();
        copy_init(cb_in_id);
        copy_tile(cb_in_id, 0, dst_x);
        // The input may be Float16_b while the scaler is always Float32.
        reconfig_data_format_srca(cb_in_id, cb_scale_id);
        copy_init(cb_scale_id);
        copy_tile(cb_scale_id, 0, dst_scale);
        if constexpr (bit_width == 8) {
            if constexpr (output_int8) {
                // DST holds each result as its two's-complement byte: negative q prints as 256 + q.
                quant_int8_tile(dst_x, dst_scale, dst_x);
            } else {
                quant_tile(dst_x, dst_scale, dst_x);
            }
        } else {
            // TODO(perf): four SFPU passes with an init each. A fused 16/32-bit quant SFPU function
            // (multiply, RNE, clamp, convert in one sweep over DST) would replace them.
            mul_binary_tile_init();
            mul_binary_tile(dst_x, dst_scale, dst_x);  // x * 2^k, exact for a normal 2^k
            rounding_op_tile_init();
            round_tile(dst_x, 0);  // round half to even
            clamp_tile_init();
            clamp_tile(dst_x, q_min_bits, q_max_bits);
            typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Int32)>();
            typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Int32)>(dst_x);
        }
        DPRINT_MATH("Contents of DST {} after quantization:\n", dst_x);
        zkml::debug::print_dest_tile_rows(dst_x, DataFormat::Int32);
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
