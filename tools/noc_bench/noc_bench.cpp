// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// NoC / DRAM microbenchmark: DRAM bank latency, core-to-core latency, one-way latency and DRAM
// bandwidth, measured with one data-movement kernel (kernels/bench.cpp) on chosen cores of an
// otherwise idle chip. The results are the chip-level numbers in docs/typst/dataflow.typ.
//
// Usage:   .build/default/zkml_noc_bench [all|bank|c2c|oneway|bw]      (default: all)
//
// Output is one line per measurement, easy to grep and parse:
//   grid <w> x <h> clock_mhz <f> dram_banks <n>
//   map logical <x> <y> virtual <vx> <vy>
//   bank noc<n> bytes <b> core <x> <y> xy <vx> <vy> lat <cycles per bank...> endpoints <x-y per bank...>
//   c2c noc<n> bytes <b> src <x> <y> dst <x> <y> dstxy .. srcxy .. read <cyc> write <cyc>
//   oneway noc<n> src <x> <y> dst <x> <y> dstxy <vx> <vy> cyc <cycles>
//   bw mode <interleaved|one-bank> batch <B> cores <n> span_cyc .. max_cyc .. avg_cyc .. agg_GBps .. percore_GBps ..
// Logical coordinates are host CoreCoords; xy/dstxy/endpoints are the virtual NoC coordinates the
// kernel sees. Cycles are AICLK cycles (get_timestamp()).
//
// TT_METAL_RUNTIME_ROOT defaults to the TT_METAL_HOME used at configure time; set it to override.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "common/kernel_path.hpp"

using namespace tt::tt_metal;

namespace {

constexpr uint32_t PAGE_BYTES = 4096;   // input page = one fp32 tile
constexpr uint32_t IN_PAGES = 65536;    // 256 MB interleaved input buffer
constexpr uint32_t RES_WORDS = 64;      // result words per job (one 256 B page)
constexpr uint32_t RES_PAGES = 256;     // max jobs per program

// Kernel modes (runtime arg 0), see kernels/bench.cpp
enum Mode : uint32_t { BANK_LATENCY = 0, CORE_TO_CORE = 1, BANDWIDTH = 2, ONEWAY_SEND = 3, ONEWAY_RECV = 4 };

struct Bench {
    std::shared_ptr<distributed::MeshDevice> dev;
    std::shared_ptr<distributed::MeshBuffer> in_buf;
    std::shared_ptr<distributed::MeshBuffer> res_buf;
    std::string kernel = zkml::common::kernel_path(__FILE__, "kernels/bench.cpp");
};

struct Job {
    CoreCoord core;
    std::vector<uint32_t> args;  // mode, then up to 5 mode args; res_addr and res_page are inserted
};

// Runs one program with the bench kernel on every job core; returns RES_WORDS words per job.
std::vector<std::vector<uint32_t>> run(Bench& b, const std::vector<Job>& jobs, int risc) {
    Program program = CreateProgram();
    std::set<CoreRange> ranges;
    for (const auto& j : jobs) {
        ranges.insert(CoreRange(j.core));
    }
    const CoreRangeSet cores(ranges);
    // c_0: 64 KB landing area for reads (up to 16 tiles in flight); c_1: the 256 B result page
    CreateCircularBuffer(
        program, cores, CircularBufferConfig(16 * PAGE_BYTES, {{0, tt::DataFormat::Float32}}).set_page_size(0, PAGE_BYTES));
    CreateCircularBuffer(
        program, cores, CircularBufferConfig(RES_WORDS * 4, {{1, tt::DataFormat::Float32}}).set_page_size(1, RES_WORDS * 4));

    std::vector<uint32_t> ct_args;
    TensorAccessorArgs(*b.in_buf->get_backing_buffer()).append_to(ct_args);
    TensorAccessorArgs(*b.res_buf->get_backing_buffer()).append_to(ct_args);
    const auto kernel = CreateKernel(
        program,
        b.kernel,
        cores,
        DataMovementConfig{
            .processor = risc == 0 ? DataMovementProcessor::RISCV_0 : DataMovementProcessor::RISCV_1,
            .noc = risc == 0 ? NOC::RISCV_0_default : NOC::RISCV_1_default,
            .compile_args = ct_args});
    for (size_t i = 0; i < jobs.size(); i++) {
        std::vector<uint32_t> a = jobs[i].args;
        a.insert(a.begin() + 1, {static_cast<uint32_t>(b.res_buf->address()), static_cast<uint32_t>(i)});
        a.resize(8, 0);
        SetRuntimeArgs(program, kernel, jobs[i].core, a);
    }

    auto& cq = b.dev->mesh_command_queue();
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(b.dev->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> raw;
    distributed::EnqueueReadMeshBuffer(cq, raw, b.res_buf, true);
    std::vector<std::vector<uint32_t>> out;
    for (size_t i = 0; i < jobs.size(); i++) {
        out.emplace_back(raw.begin() + i * RES_WORDS, raw.begin() + (i + 1) * RES_WORDS);
    }
    return out;
}

std::vector<CoreCoord> all_cores(const CoreCoord& grid) {
    std::vector<CoreCoord> cores;
    for (uint32_t y = 0; y < grid.y; y++) {
        for (uint32_t x = 0; x < grid.x; x++) {
            cores.push_back({x, y});
        }
    }
    return cores;
}

// Every core in turn reads 64 B / 4 KB from each DRAM bank, one request in flight, on both NoCs.
void bank_latency(Bench& b, const CoreCoord& grid) {
    const auto in_addr = static_cast<uint32_t>(b.in_buf->address());
    for (int risc = 0; risc < 2; risc++) {
        for (uint32_t bytes : {64u, 4096u}) {
            for (const auto& core : all_cores(grid)) {
                const auto o = run(b, {{core, {BANK_LATENCY, in_addr, 64u, bytes}}}, risc)[0];
                const uint32_t banks = o[30];
                fmt::print("bank noc{} bytes {} core {} {} xy {} {} lat", risc, bytes, core.x, core.y, o[0], o[1]);
                for (uint32_t k = 0; k < banks; k++) {
                    fmt::print(" {:.1f}", o[2 + k] / 16.0);
                }
                fmt::print(" endpoints");
                for (uint32_t k = 0; k < banks; k++) {
                    fmt::print(" {}-{}", o[16 + k] & 63, o[16 + k] >> 6);  // NOC_XY_ENCODING: y << 6 | x
                }
                fmt::print("\n");
            }
        }
    }
}

// Round trip (read + barrier, write + ack) from a corner and a middle core to every other core.
void core_to_core(Bench& b, const CoreCoord& grid) {
    // An address in the free L1 of the (idle) target core
    const auto target_l1 =
        static_cast<uint32_t>(b.dev->allocator()->get_base_allocator_addr(HalMemType::L1) + 128 * 1024);
    const std::vector<CoreCoord> sources = {{0, 0}, {grid.x / 2, grid.y / 2}};
    for (int risc = 0; risc < 2; risc++) {
        for (const auto& src : sources) {
            for (uint32_t bytes : {64u, 4096u}) {
                for (const auto& dst : all_cores(grid)) {
                    if (dst == src) {
                        continue;
                    }
                    const auto v = b.dev->worker_core_from_logical_core(dst);
                    const auto o = run(
                        b,
                        {{src, {CORE_TO_CORE, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), target_l1, 64u, bytes}}},
                        risc)[0];
                    fmt::print(
                        "c2c noc{} bytes {} src {} {} dst {} {} dstxy {} {} srcxy {} {} read {:.1f} write {:.1f}\n",
                        risc, bytes, src.x, src.y, dst.x, dst.y, v.x, v.y, o[0], o[1], o[2] / 16.0, o[3] / 16.0);
                }
            }
        }
    }
}

// One-way latency: the source writes its timestamp into the target's L1, the target subtracts.
void oneway(Bench& b, const CoreCoord& grid) {
    const std::vector<CoreCoord> sources = {{0, 0}, {grid.x / 2, grid.y / 2}};
    for (int risc = 0; risc < 2; risc++) {
        for (const auto& src : sources) {
            for (const auto& dst : all_cores(grid)) {
                if (dst == src) {
                    continue;
                }
                const auto v = b.dev->worker_core_from_logical_core(dst);
                uint32_t best = ~0u;
                for (int rep = 0; rep < 3; rep++) {
                    const auto r = run(
                        b,
                        {{src, {ONEWAY_SEND, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y)}},
                         {dst, {ONEWAY_RECV}}},
                        risc);
                    best = std::min(best, r[1][2]);
                }
                fmt::print(
                    "oneway noc{} src {} {} dst {} {} dstxy {} {} cyc {}\n", risc, src.x, src.y, dst.x, dst.y, v.x, v.y,
                    best);
            }
        }
    }
}

// 1..all cores each read 256 tiles with B reads per barrier: interleaved pages, or only bank 0.
void bandwidth(Bench& b, const CoreCoord& grid) {
    const auto cores = all_cores(grid);
    const auto in_addr = static_cast<uint32_t>(b.in_buf->address());
    const uint32_t num_banks = b.dev->num_dram_channels();
    const double mhz = b.dev->get_clock_rate_mhz();
    constexpr uint32_t n = 256;  // tiles per core
    for (bool one_bank : {false, true}) {
        for (uint32_t batch : {1u, 4u, 16u}) {
            for (uint32_t ncores : {1u, 2u, 4u, 8u, 16u, 32u, 64u, static_cast<uint32_t>(cores.size())}) {
                if (ncores > cores.size()) {
                    continue;
                }
                std::vector<Job> jobs;
                for (uint32_t c = 0; c < ncores; c++) {
                    const CoreCoord core = cores[(c * cores.size()) / ncores];  // spread over the grid
                    const uint32_t first = one_bank ? 0 : (c * n) % (IN_PAGES - n);
                    const uint32_t stride = one_bank ? num_banks : 1;  // every num_banks'th page is bank 0
                    jobs.push_back({core, {BANDWIDTH, in_addr, first, n, batch, stride}});
                }
                const auto r = run(b, jobs, 0);
                uint64_t t_first = ~0ull, t_last = 0, max_dur = 0, sum_dur = 0;
                for (const auto& o : r) {
                    const uint64_t t0 = o[3] | (static_cast<uint64_t>(o[4]) << 32);
                    t_first = std::min(t_first, t0);
                    t_last = std::max(t_last, t0 + o[2]);
                    max_dur = std::max<uint64_t>(max_dur, o[2]);
                    sum_dur += o[2];
                }
                const uint64_t span = t_last - t_first;
                const double avg_dur = static_cast<double>(sum_dur) / ncores;
                const double bytes_per_core = static_cast<double>(n) * PAGE_BYTES;
                fmt::print(
                    "bw mode {} batch {} cores {} span_cyc {} max_cyc {} avg_cyc {:.0f} agg_GBps {:.1f} percore_GBps {:.2f}\n",
                    one_bank ? "one-bank" : "interleaved", batch, ncores, span, max_dur, avg_dur,
                    ncores * bytes_per_core / (span / mhz * 1e-6) / 1e9, bytes_per_core / (avg_dur / mhz * 1e-6) / 1e9);
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string what = argc > 1 ? argv[1] : "all";
    if (what != "all" && what != "bank" && what != "c2c" && what != "oneway" && what != "bw") {
        fmt::print(stderr, "usage: {} [all|bank|c2c|oneway|bw]\n", argv[0]);
        return 2;
    }

    // TT_METAL_RUNTIME_ROOT, if set, takes precedence over this.
    SetRootDir(ZKML_TT_METAL_HOME);

    Bench b;
    b.dev = distributed::MeshDevice::create_unit_mesh(0);
    const distributed::DeviceLocalBufferConfig in_cfg{.page_size = PAGE_BYTES, .buffer_type = BufferType::DRAM};
    const distributed::DeviceLocalBufferConfig res_cfg{.page_size = RES_WORDS * 4, .buffer_type = BufferType::DRAM};
    b.in_buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = static_cast<uint64_t>(IN_PAGES) * PAGE_BYTES}, in_cfg, b.dev.get());
    b.res_buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = RES_PAGES * RES_WORDS * 4}, res_cfg, b.dev.get());

    const auto grid = b.dev->compute_with_storage_grid_size();
    fmt::print(
        "grid {} x {} clock_mhz {} dram_banks {}\n", grid.x, grid.y, b.dev->get_clock_rate_mhz(),
        b.dev->num_dram_channels());
    for (const auto& c : all_cores(grid)) {
        const auto v = b.dev->worker_core_from_logical_core(c);
        fmt::print("map logical {} {} virtual {} {}\n", c.x, c.y, v.x, v.y);
    }

    if (what == "all" || what == "bank") {
        bank_latency(b, grid);
    }
    if (what == "all" || what == "c2c") {
        core_to_core(b, grid);
    }
    if (what == "all" || what == "oneway") {
        oneway(b, grid);
    }
    if (what == "all" || what == "bw") {
        bandwidth(b, grid);
    }

    // Buffers must be freed before the device closes
    b.in_buf.reset();
    b.res_buf.reset();
    b.dev->close();
    return 0;
}
