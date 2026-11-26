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
 * @brief A network of quantized layers on one core: one compute kernel with one copy of the layer code, run in a loop over the layers.
 *
 *     x --quantize--> act_a --matmul--> acc_0 --activation--> requantize --> act_b --matmul--> acc_1 --> ... --> out
 *     w_l --quantize--> q_w (per layer)
 *
 * Only the input of layer 0 and the weights are quantized from fp32 / bf16. The requantized output of layer l *is* the quantized
 * input of layer l + 1: the requantization sweep packs it straight into the other activation buffer, so between the layers the
 * activations stay in L1 as int8 and are never dequantized. The last layer packs into the output ring instead.
 *
 * The two activation buffers are used alternately (ping-pong): layer l reads `act_a` (l even) or `act_b` (l odd) and writes the other.
 * The layers are run-time values (`Kt`, `Nt`, `relu` per layer, runtime args), the circular buffers of a layer are run-time arguments
 * of the stage functions, so the code does not grow with the number of layers.
 *
 * Per layer (all stages are shared with execute_layer, see layer_stages.hpp):
 *   0. (before layer 0) quantize x into act_a
 *   1. quantize the weights of the layer: pass 1 (scaler) and pass 2 (bytes into q_w), then pop `c_1` so the reader can read the
 *      next layer's weights while this layer computes
 *   2. matmul act_in x q_w into the accumulator; pop q_w and the input activation buffer
 *   3. the scaler of the requantization, then the requantization sweep into the other activation buffer (or the output ring); only
 *      then is the scaler pushed, which tells the writer that the accumulator has been read
 *
 * `c_1` (weights), `c_2` (q_w), `c_3` (accumulator) and the two activation buffers are reused for every layer and padded to their
 * capacity (every push and pop is the full capacity), so the pointers are back at the start (see network_layout.hpp).
 *
 * **Compile-time args**: `[0]` Mt, `[1]` fp32 input (1) or bf16 (0), `[2]` pages of the output ring, `[3]` `CAP_W`, `[4]` `CAP_QW`, `[5]`
 * `CAP_ACC`, `[6]` `CAP_ACT` (tiles), `[7]` 1 with pre-quantized weights (step 1 is skipped: the reader fills `c_2`). **Runtime args**: `[0]` number of layers, then per layer `Kt, Nt, relu`.
 * @pre  The program enables `fp32_dest_acc_en`, HiFi4. The input buffers use `UnpackToDestFp32` for fp32. Kt <= 2080.
 */
// clang-format on

using namespace zkml::stages;
namespace net = zkml::network;

constexpr uint32_t Mt = get_compile_time_arg_val(0);
constexpr bool fp32_in = get_compile_time_arg_val(1) == 1;
constexpr uint32_t out_pages = get_compile_time_arg_val(2);
constexpr uint32_t CAP_W = get_compile_time_arg_val(3);
constexpr uint32_t CAP_QW = get_compile_time_arg_val(4);
constexpr uint32_t CAP_ACC = get_compile_time_arg_val(5);
constexpr uint32_t CAP_ACT = get_compile_time_arg_val(6);
constexpr bool PRE_QUANTIZED = get_compile_time_arg_val(7) == 1;  // the reader delivers the quantized weights into c_2

// Pushes `pad` tiles of padding (no data) to cb, so that a buffer reused for the next layer is back at its start.
ALWI void push_padding(uint32_t cb, uint32_t pad) {
    if (pad != 0) {
        CircularBuffer c(cb);
        c.reserve_back(pad);
        c.push_back(pad);
    }
}

// One layer: Kt x Nt tiles of weights, the input activations in `cb_in` (Mt x Kt tiles), the output into `cb_out` (the next layer's
// activation buffer, or the output ring when `last`).
ALWI void run_layer(uint32_t cb_in, uint32_t cb_next, uint32_t Kt, uint32_t Nt, bool relu, bool last) {
    if constexpr (!PRE_QUANTIZED) {
    #ifdef PROFILE_KERNEL
        {  // profiling only: how long compute waits for the first tile of this layer's weights (the reader is behind)
            DeviceZoneScopedN("WAIT_W");
            CircularBuffer(net::c_w).wait_front(1);
        }
    #endif
        // 1. quantize the weights of this layer into q_w, then free the weight buffer for the reader
        {
            DeviceZoneScopedN("QUANT_W");
            {
                DeviceZoneScopedN("QW_SCALE");
                scale_pass_rt(net::c_w, net::c_scale_w, Kt * Nt);
            }
            {
                DeviceZoneScopedN("QW_QUANT");
                quant_pass_rt<false, false, true>(net::c_w, net::c_qw, Kt * Nt, Kt * Nt);
            }
            push_padding(net::c_qw, CAP_QW - Kt * Nt);
            CircularBuffer(net::c_w).pop_front(CAP_W);
        }
    }
#ifdef PROFILE_KERNEL
    {  // profiling only: how long compute waits for the writer to free the accumulator of the previous layer
        DeviceZoneScopedN("WAIT_ACC");
        CircularBuffer(net::c_acc).reserve_back(1);
    }
#endif
    // 2. matmul into the accumulator; the activation sweep folds the integer absmax into LReg7
    {
        DeviceZoneScopedN("MATMUL");
        if (relu) {
            matmul_stage_rt<true, true>(cb_in, net::c_qw, net::c_acc, Mt, Kt, Nt);
        } else {
            matmul_stage_rt<false, true>(cb_in, net::c_qw, net::c_acc, Mt, Kt, Nt);
        }
        push_padding(net::c_acc, CAP_ACC - Mt * Nt);
        CircularBuffer(net::c_qw).pop_front(CAP_QW);
        CircularBuffer(cb_in).pop_front(CAP_ACT);
    }

    // 3. requantize: the scaler, then the sweep into the next layer's activation buffer (or the output ring)
    {
        DeviceZoneScopedN("REQUANT");
        const uint32_t cb_out = last ? net::c_out : cb_next;
        const uint32_t pages = last ? out_pages : CAP_ACT;
        if (Kt <= net::max_float_requant_kt) {
            // exact in fp32 (K * 127 * 127 <= 2^24): the pipelined float requantizer; the packer applies the ReLU
            pack_requant_scaler_rt<true>(net::c_acc, net::c_scale_out);
            if (relu) {
                quant_pass_rt<true, true, true, true>(net::c_acc, cb_out, Mt * Nt, pages);
            } else {
                quant_pass_rt<true, false, true, true>(net::c_acc, cb_out, Mt * Nt, pages);
            }
        } else {
            pack_requant_scaler_rt(net::c_acc, net::c_scale_out);
            if (relu) {
                quant_pass_rt<true, true>(net::c_acc, cb_out, Mt * Nt, pages);
            } else {
                quant_pass_rt<true, false>(net::c_acc, cb_out, Mt * Nt, pages);
            }
        }
        if (!last) {
            push_padding(cb_next, CAP_ACT - Mt * Nt);
        }
    }
    CircularBuffer(net::c_scale_out).push_back(1);  // the accumulator has been read: the writer may write and pop it
}

void kernel_main() {
    const uint32_t L = get_arg_val<uint32_t>(0);
    const uint32_t Kt0 = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(net::c_x, net::c_out);

    // layer 0's input: quantize x into act_a
    {
        DeviceZoneScopedN("QUANT_X");
        scale_pass_rt(net::c_x, net::c_scale_x, Mt * Kt0);
        quant_pass_rt<false, false, true>(net::c_x, net::c_act_a, Mt * Kt0, CAP_ACT);
        push_padding(net::c_act_a, CAP_ACT - Mt * Kt0);
    }
    for (uint32_t l = 0; l < L; l++) {
        const uint32_t Kt = get_arg_val<uint32_t>(1 + 3 * l);
        const uint32_t Nt = get_arg_val<uint32_t>(2 + 3 * l);
        const bool relu = get_arg_val<uint32_t>(3 + 3 * l) != 0;
        const bool odd = (l & 1) != 0;
        run_layer(odd ? net::c_act_b : net::c_act_a, odd ? net::c_act_a : net::c_act_b, Kt, Nt, relu, l + 1 == L);
    }
}
