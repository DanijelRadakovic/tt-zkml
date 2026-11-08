// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// clang-format off
/**
 * @brief NoC / DRAM microbenchmark kernel; the host picks what to measure with runtime arg 0.
 *
 * Every mode writes a 64-word result page (`c_1`) to page `res_page` of the DRAM results buffer.
 * Words 0 and 1 are this core's virtual NoC coordinates `my_x`, `my_y`. Times are AICLK cycles
 * from `get_timestamp()`. Latencies are stored ×16 to keep one fractional hex digit.
 *
 * | Mode | Measures                         | a0..a4                                              | Result words                                     |
 * |------|----------------------------------|-----------------------------------------------------|--------------------------------------------------|
 * | 0    | DRAM bank latency                | in_addr, iters, bytes                               | 2+b: cycles×16 for bank b; 16+b: bank's NoC xy; 30: num banks |
 * | 1    | core-to-core round trip          | target x, target y, target L1 addr, iters, bytes    | 2: read+barrier ×16; 3: write+ack ×16            |
 * | 2    | DRAM bandwidth                   | in_addr, first page, num pages, batch, page stride  | 2: elapsed; 3/4: start timestamp lo/hi           |
 * | 3    | one-way latency, sender          | target x, target y                                  | —                                                |
 * | 4    | one-way latency, receiver        | —                                                   | 2: arrival − send timestamp                      |
 *
 * **Compile-time args:** `TensorAccessorArgs` of the input buffer (4 KB pages), then of the results buffer.
 *
 * **Runtime args:** 0 = mode, 1 = results buffer address, 2 = result page, 3..7 = a0..a4.
 *
 * **CBs:** `c_0` 64 KB landing area for reads (holds up to 16 pages in flight), `c_1` the 256 B result page.
 *
 * @pre Modes 3 and 4 run in the same program on two cores, so `c_0` is at the same L1 address on both.
 */
// clang-format on
void kernel_main() {
    const uint32_t mode = get_arg_val<uint32_t>(0);
    const uint32_t res_addr = get_arg_val<uint32_t>(1);
    const uint32_t res_page = get_arg_val<uint32_t>(2);
    const uint32_t a0 = get_arg_val<uint32_t>(3);
    const uint32_t a1 = get_arg_val<uint32_t>(4);
    const uint32_t a2 = get_arg_val<uint32_t>(5);
    const uint32_t a3 = get_arg_val<uint32_t>(6);
    const uint32_t a4 = get_arg_val<uint32_t>(7);

    constexpr uint32_t page_bytes = 4096;
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto res_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto res = TensorAccessor(res_args, res_addr, 256);

    const uint32_t scratch = get_write_ptr(0);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
    for (uint32_t i = 0; i < 64; i++) {
        out[i] = 0;
    }
    out[0] = my_x[noc_index];
    out[1] = my_y[noc_index];

    if (mode == 0) {
        // DRAM bank latency: one request in flight. Pages b, b+N, ... all live in bank b.
        const auto in = TensorAccessor(in_args, a0, page_bytes);
        const uint32_t iters = a1, bytes = a2;
        for (uint32_t b = 0; b < NUM_DRAM_BANKS; b++) {
            noc_async_read(in.get_noc_addr(b), scratch, bytes);  // warm-up
            noc_async_read_barrier();
            const uint64_t t0 = get_timestamp();
            for (uint32_t i = 0; i < iters; i++) {
                noc_async_read(in.get_noc_addr(b + NUM_DRAM_BANKS * (i & 15)), scratch, bytes);
                noc_async_read_barrier();
            }
            const uint64_t t1 = get_timestamp();
            out[2 + b] = static_cast<uint32_t>(((t1 - t0) * 16) / iters);
            out[16 + b] = dram_bank_to_noc_xy[noc_index][b];
        }
        out[30] = NUM_DRAM_BANKS;
    } else if (mode == 1) {
        // Core to core: read + barrier, then write + barrier (waits for the ack), one at a time.
        const uint64_t dst = get_noc_addr(a0, a1, a2);
        const uint32_t iters = a3, bytes = a4;
        noc_async_read(dst, scratch, bytes);  // warm-up
        noc_async_read_barrier();
        uint64_t t0 = get_timestamp();
        for (uint32_t i = 0; i < iters; i++) {
            noc_async_read(dst, scratch, bytes);
            noc_async_read_barrier();
        }
        uint64_t t1 = get_timestamp();
        out[2] = static_cast<uint32_t>(((t1 - t0) * 16) / iters);
        t0 = get_timestamp();
        for (uint32_t i = 0; i < iters; i++) {
            noc_async_write(scratch, dst, bytes);
            noc_async_write_barrier();
        }
        t1 = get_timestamp();
        out[3] = static_cast<uint32_t>(((t1 - t0) * 16) / iters);
    } else if (mode == 2) {
        // Bandwidth: `batch` reads in flight per barrier, page += stride.
        const auto in = TensorAccessor(in_args, a0, page_bytes);
        const uint32_t first = a1, n = a2, batch = a3, stride = a4;
        const uint64_t t0 = get_timestamp();
        uint32_t page = first;
        for (uint32_t done = 0; done < n; done += batch) {
            for (uint32_t i = 0; i < batch; i++) {
                noc_async_read(in.get_noc_addr(page), scratch + i * page_bytes, page_bytes);
                page += stride;
            }
            noc_async_read_barrier();
        }
        const uint64_t t1 = get_timestamp();
        out[2] = static_cast<uint32_t>(t1 - t0);
        out[3] = static_cast<uint32_t>(t0);
        out[4] = static_cast<uint32_t>(t0 >> 32);
    } else if (mode == 3) {
        // One-way sender: wait for the receiver to arm, then write {timestamp, marker} into its c_0.
        volatile tt_l1_ptr uint32_t* msg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + page_bytes);
        const uint64_t start = get_timestamp();
        while (get_timestamp() - start < 50000) {
        }
        const uint64_t t = get_timestamp();
        msg[0] = static_cast<uint32_t>(t);
        msg[1] = static_cast<uint32_t>(t >> 32);
        msg[2] = 0xABCD;
        msg[3] = 0;
        noc_async_write(scratch + page_bytes, get_noc_addr(a0, a1, scratch), 16);
        noc_async_write_barrier();
    } else if (mode == 4) {
        // One-way receiver: spin until the marker lands, then record arrival - send time.
        // Relies on the wall clocks of all cores agreeing.
        volatile tt_l1_ptr uint32_t* msg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        msg[0] = msg[1] = msg[2] = msg[3] = 0;
        while (msg[2] != 0xABCD) {
        }
        const uint64_t t1 = get_timestamp();
        const uint64_t t0 = msg[0] | (static_cast<uint64_t>(msg[1]) << 32);
        out[2] = static_cast<uint32_t>(t1 - t0);
    }

    noc_async_write(reinterpret_cast<uint32_t>(out), res.get_noc_addr(res_page), 256);
    noc_async_write_barrier();
}
