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
#include "device/operations/network_execution/device/network_layout.hpp"

// clang-format off
/**
 * @brief A network of quantized layers on one core: one compute kernel that loops over the layers.
 *
 *     x --quantize--> act_0 --matmul--> acc_0 --activation--> requantize --> act_1 --matmul--> acc_1 --> ... --> out
 *     w_l --quantize--> q_w (per layer)
 *
 * Only the input of layer 0 and the weights are quantized from fp32 / bf16. The requantized output of layer l *is* the quantized
 * input of layer l + 1: the requantization sweep packs it straight into `act_{l+1}`, so between the layers the activations stay in
 * L1 as int8 and are never dequantized. The last layer packs into the output ring instead.
 *
 * Per layer (all stages are shared with execute_layer, see layer_stages.hpp):
 *   1. (layer 0) quantize x into act_0
 *   2. quantize the weights of the layer: pass 1 (scaler) and pass 2 (bytes into q_w), then pop `c_1` so the reader can read the
 *      next layer's weights while this layer computes
 *   3. matmul act_l x q_w into the accumulator; pop q_w
 *   4. the scaler of the requantization, then the requantization sweep into act_{l+1} (or the output ring); only then is the
 *      scaler pushed, which tells the writer that the accumulator has been read
 *
 * `c_1` (weights), `c_2` (q_w) and `c_3` (accumulator) are reused for every layer and padded to their capacity so the pointers
 * are back at the start (see network_layout.hpp). The activation buffers are one per layer and never reused.
 *
 * **Compile-time args**: `[0]` L, `[1]` Mt, `[2]` fp32 input (1) or bf16 (0), `[3]` pages of the output ring, `[4]` `CAP_W`, `[5]`
 * `CAP_QW`, `[6]` `CAP_ACC` (tiles), then per layer `Kt, Nt, relu`.
 * @pre  The program enables `fp32_dest_acc_en`, HiFi4. The input buffers use `UnpackToDestFp32` for fp32. Kt <= 2080.
 */
// clang-format on

using namespace zkml::stages;
namespace net = zkml::network;

constexpr uint32_t L = get_compile_time_arg_val(0);
constexpr uint32_t Mt = get_compile_time_arg_val(1);
constexpr bool fp32_in = get_compile_time_arg_val(2) == 1;
constexpr uint32_t out_pages = get_compile_time_arg_val(3);
constexpr uint32_t CAP_W = get_compile_time_arg_val(4);
constexpr uint32_t CAP_QW = get_compile_time_arg_val(5);
constexpr uint32_t CAP_ACC = get_compile_time_arg_val(6);
constexpr uint32_t layer_args = 7;
static_assert(L >= 1 && L <= net::max_layers, "1 to max_layers layers");

template <uint32_t l>
constexpr uint32_t Kt_of() {
    return get_compile_time_arg_val(layer_args + 3 * l);
}
template <uint32_t l>
constexpr uint32_t Nt_of() {
    return get_compile_time_arg_val(layer_args + 3 * l + 1);
}
template <uint32_t l>
constexpr bool relu_of() {
    return get_compile_time_arg_val(layer_args + 3 * l + 2) == 1;
}

// Pushes `pad` tiles of padding (no data) to cb, so that a buffer reused for the next layer is back at its start.
template <uint32_t cb>
ALWI void push_padding(uint32_t pad) {
    if (pad != 0) {
        CircularBuffer c(cb);
        c.reserve_back(pad);
        c.push_back(pad);
    }
}

template <uint32_t l>
ALWI void run_layer() {
    constexpr uint32_t Kt = Kt_of<l>(), Nt = Nt_of<l>();
    constexpr bool relu = relu_of<l>();
    constexpr bool last = l + 1 == L;
    static_assert(
        Kt <= net::max_k_tiles, "K must be at most 66560: K * 127 * 127 must stay below 2^30 (31 magnitude bits)");
    constexpr uint32_t cb_in = net::act_cb(l);

    // 1. layer 0: quantize x into act_0
    if constexpr (l == 0) {
        scale_pass<net::c_x, net::c_scale_x>(Mt * Kt);
        quant_pass<net::c_x, cb_in, false, false, true>(Mt * Kt, Mt * Kt);
    }

    // 2. quantize the weights of this layer into q_w, then free the weight buffer for the reader
    scale_pass<net::c_w, net::c_scale_w>(Kt * Nt);
    quant_pass<net::c_w, net::c_qw, false, false, true>(Kt * Nt, Kt * Nt);
    push_padding<net::c_qw>(CAP_QW - Kt * Nt);
    CircularBuffer(net::c_w).pop_front(CAP_W);

    // 3. matmul into the accumulator; the activation sweep folds the integer absmax into LReg7
    matmul_stage<cb_in, net::c_qw, net::c_acc, relu, true, Mt, Kt, Nt>();
    push_padding<net::c_acc>(CAP_ACC - Mt * Nt);
    CircularBuffer(net::c_qw).pop_front(CAP_QW);

    // 4. requantize: the scaler, then the sweep into the next layer's activation buffer (or the output ring)
    pack_requant_scaler<net::c_acc, net::c_scale_out>();
    if constexpr (last) {
        quant_pass<net::c_acc, net::c_out, true, relu>(Mt * Nt, out_pages);
    } else {
        quant_pass<net::c_acc, net::act_cb(l + 1), true, relu>(Mt * Nt, Mt * Nt);
    }
    CircularBuffer(net::c_scale_out).push_back(1);  // the accumulator has been read: the writer may write and pop it
}

template <uint32_t l>
struct LayerLoop {
    static ALWI void run() {
        run_layer<l>();
        if constexpr (l + 1 < L) {
            LayerLoop<l + 1>::run();
        }
    }
};

void kernel_main() {
    compute_kernel_hw_startup(net::c_x, net::c_out);
    LayerLoop<0>::run();
}
