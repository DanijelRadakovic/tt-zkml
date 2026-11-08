// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Reader for the L1-fit benchmark. Reads the input tiles from DRAM in blocks of BLOCK_SIZE tiles
// (BLOCK_SIZE reads, one barrier, one push) and repeats the whole tensor NUM_PASSES times:
//   NUM_PASSES = 2  streaming: the tensor crosses the NoC twice (what the real op does today)
//   NUM_PASSES = 1  resident:  the tensor crosses the NoC once; compute re-reads it from L1
// Every block is wrapped in a "READ_BLOCK" profiler zone, so the zone durations in the device log
// are the time the reader needs per block (issue reads + wait for all of them + push).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_PASSES = get_compile_time_arg_val(1);
    constexpr auto in_args = TensorAccessorArgs<2>();
    const auto in = TensorAccessor(in_args, in_addr);

    Noc noc;
    CircularBuffer cb_in(tt::CBIndex::c_0);
    const uint32_t tile_bytes = cb_in.get_tile_size();

    const uint32_t total_reads = NUM_PASSES * num_tiles;
    uint32_t page = 0;
    for (uint32_t done = 0; done < total_reads; done += BLOCK_SIZE) {
        const uint32_t remaining = total_reads - done;
        const uint32_t n = remaining < BLOCK_SIZE ? remaining : BLOCK_SIZE;
        {
            DeviceZoneScopedN("READ_BLOCK");
            cb_in.reserve_back(n);  // blocks here when compute has not freed the ring yet
            for (uint32_t i = 0; i < n; i++) {
                noc.async_read(in, cb_in, tile_bytes, {.page_id = page}, {.offset_bytes = i * tile_bytes});
                if (++page == num_tiles) {
                    page = 0;
                }
            }
            noc.async_read_barrier();
            cb_in.push_back(n);
        }
    }
}
