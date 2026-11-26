// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"
#include "device/kernels/compute/layer_stages.hpp"

// clang-format off
/**
 * @brief One quantized layer on one core: quantize x and W, matmul on the FPU, activation, requantize.
 *
 *     x (fp32) --quantize--> q_x (int8)  \\
 *                                          matmul (FPU, int32) -> relu -> requantize -> out (int8)
 *     W (fp32) --quantize--> q_w (int8)  /
 *
 * - Quantize (twice, the same code): pass 1 finds the absmax and the scaler `2^k` of the tensor, kept in LReg7 and packed to
 *   its scaler CB; pass 2 multiplies by it from LReg7, rounds half to even and packs the sign-magnitude byte
 *   (`quant_sm8_lm_tile`: SFPLOADMACRO schedules the multiply-add, the subtract and the store, the SFPCAST to sign-magnitude is issued
 *   between two macros) through the Int8 packer, which writes sign + 7 magnitude bits: the FPU's Int8 format. For bf16 input the
 *   buffers are UInt8 and the packer is configured for Int8 by hand (`pack_reconfig_int8_view`).
 * - Matmul: `q_x` and `q_w` are Int8 CBs. Reconfigured to Int8, the FPU multiplies at HiFi4 into an int32 accumulator in DST, exact.
 *   Output blocks of up to `DST_TILES` tiles: the K loop, then an SFPU sweep (layer_sfpu.hpp) that folds the integer absmax of the
 *   activated accumulator into LReg7 (the tile is untouched), then the block is packed as Int32 to `c_6`: the raw sign-magnitude bits.
 *   The packer drains block i while the FPU computes block i + 1. `c_6` is the raw accumulator, before the activation: the
 *   writer returns it as the `acc` tensor, which the host needs for the execution trace.
 * - Requantize: the absmax gives the scaler `2^k'` (`k'` = the largest with `absmax * 2^k' <= 127`, as the quantizer does), and
 *   the integer requantizer (round half to even, shift by k') runs over `c_6`. No fp32 anywhere.
 *
 * | Circular buffer | Format                  | Contents                                                 |
 * |-----------------|-------------------------|----------------------------------------------------------|
 * | `c_0`, `c_1`    | Float32                 | x tiles (Mt x Kt), W tiles (Kt x Nt), resident           |
 * | `c_2`, `c_3`    | Int8                    | q_x, q_w bytes, resident (Int8 packer in, FPU out)       |
 * | `c_6`           | Int32                   | raw accumulator (Mt x Nt), resident; the `acc` output    |
 * | `c_16`          | UInt8                   | out tiles, a ring                                        |
 * | `c_17`..`c_19`  | Float32                 | scaler tiles of x, W and the requantization              |
 *
 * **Compile-time args**: `[0]` Mt, `[1]` Kt, `[2]` Nt, `[3]` out ring pages, `[4]` activation (0 none, 1 ReLU), `[5]` 1 for fp32
 * input, 0 for bf16.
 * @pre  The program enables `fp32_dest_acc_en`, HiFi4. Float32 CBs use `UnpackToDestFp32`. Kt <= 2080.
 * @note Between `absmax_to_scaler` and the last quantize sweep of a tensor nothing may use LReg7 (quant_sfpu.hpp).
 */
// clang-format on

using namespace zkml::stages;

constexpr uint32_t c_x = tt::CBIndex::c_0, c_w = tt::CBIndex::c_1;
constexpr uint32_t c_qx = tt::CBIndex::c_2,
                   c_qw = tt::CBIndex::c_3;  // Int8 (fp32 input) or UInt8 (bf16 input): the FPU reads both as Int8
constexpr uint32_t c_acc = tt::CBIndex::c_6;
constexpr uint32_t c_out = tt::CBIndex::c_16;
constexpr uint32_t c_scale_x = tt::CBIndex::c_17, c_scale_w = tt::CBIndex::c_18, c_scale_out = tt::CBIndex::c_19;

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t out_pages = get_compile_time_arg_val(3);
    constexpr bool relu = get_compile_time_arg_val(4) == 1;
    constexpr bool fp32_in = get_compile_time_arg_val(5) == 1;
    static_assert(Kt <= 2080, "K must be at most 66560: K * 127 * 127 must stay below 2^30 (31 magnitude bits)");
    constexpr bool int8_view = !fp32_in;  // bf16: UInt8 buffers, Int8 views by hand

    compute_kernel_hw_startup(c_x, c_out);

    // ---- quantize x and W ----
    {
        DeviceZoneScopedN("QUANT_X");
        scale_pass<c_x, c_scale_x>(Mt * Kt);
        quant_pass<c_x, c_qx, false, false, int8_view>(Mt * Kt, Mt * Kt);
    }
    {
        DeviceZoneScopedN("QUANT_W");
        scale_pass<c_w, c_scale_w>(Kt * Nt);
        quant_pass<c_w, c_qw, false, false, int8_view>(Kt * Nt, Kt * Nt);
    }

    // ---- matmul (FPU, int8 -> int32); SFPU: integer absmax of the activated accumulator; pack int32 ----
    {
        DeviceZoneScopedN("MATMUL");
        matmul_stage<c_qx, c_qw, c_acc, relu, int8_view, Mt, Kt, Nt>();
    }

    // ---- requantize: the scaler of the activated accumulator, then pass 2 of the quantizer over the accumulator ----
    {
        DeviceZoneScopedN("REQUANT");
        pack_requant_scaler<c_acc, c_scale_out>();
        CircularBuffer(c_scale_out).push_back(1);
        quant_pass<c_acc, c_out, true, relu>(Mt * Nt, out_pages);
    }
}
