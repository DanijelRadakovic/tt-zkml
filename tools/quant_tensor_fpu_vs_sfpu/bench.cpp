// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Benchmark: pot_quantize_per_tensor with the SFPU multiply (current) against the FPU multiply (new).
//
// Both variants run the REAL kernels of the op (reader, compute, writer under
// src/operations/pot_quantization/per_tensor/device/kernels/), with the same CBs the program factory
// builds. The only difference is the compile-time arg `use_fpu` of the compute kernel (and what it implies:
// scaler unpack mode, c_16 size). Every run is checked against the host reference.
//
// Usage:
//   zkml_quant_fpu_vs_sfpu profile <bf16|f32> <sfpu|fpu> <bits 8|16|32> <int32|int8> <tiles> <iters>
//   zkml_quant_fpu_vs_sfpu wall    <bf16|f32> <sfpu|fpu> <bits>         <int32|int8> <tiles> <iters>
//   zkml_quant_fpu_vs_sfpu power   <bf16|f32> <sfpu|fpu> <bits>         <int32|int8> <tiles> <cores> <seconds>
//
//   profile  one core, <iters> runs back to back. Run it with TT_METAL_DEVICE_PROFILER=1: the kernels carry
//            DeviceZoneScopedN zones (PASS1, SCALER, PASS2) and analyze.py reads the device log. Cycles.
//   wall     one core, no profiler. Host wall time of one run (enqueue + finish) and of a pipelined stream.
//   power    the same program on <cores> cores at once for <seconds>, while the board power sensor
//            (hwmon, W) is sampled. One core draws far less than the sensor resolution (1 W), so many cores
//            run together; the reported energy is per core-run.
//
// The line printed at the end starts with the mode name and is `key value` pairs, one run per line.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <fmt/core.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "common/kernel_path.hpp"
#include "common/pot_reference.hpp"

using namespace tt::tt_metal;
using Clock = std::chrono::steady_clock;

namespace {

constexpr uint32_t dst_tiles = 4;  // PotQuantizePerTensorOperation::dst_tiles
constexpr uint32_t reader_batch = 16;
constexpr uint32_t in_cb_factor = 2;

struct Params {
    bool bf16;
    bool fpu;
    uint32_t bits;
    bool out_int8;
    uint32_t tiles;
};

// The hwmon device of the Blackhole card: board power in microwatts.
std::string find_power_file() {
    namespace fs = std::filesystem;
    for (const auto& e : fs::directory_iterator("/sys/class/hwmon")) {
        std::ifstream name(e.path() / "name");
        std::string n;
        if (name >> n && n == "blackhole") {
            return (e.path() / "power1_input").string();
        }
    }
    return {};
}

double read_power_watts(const std::string& file) {
    std::ifstream f(file);
    uint64_t uw = 0;
    f >> uw;
    return static_cast<double>(uw) / 1e6;
}

struct Stat {
    double mean = 0, stdev = 0, min = 0, max = 0;
    size_t n = 0;
};

Stat stat_of(const std::vector<double>& v) {
    Stat s;
    s.n = v.size();
    if (v.empty()) {
        return s;
    }
    double sum = 0;
    s.min = s.max = v[0];
    for (double x : v) {
        sum += x;
        s.min = std::min(s.min, x);
        s.max = std::max(s.max, x);
    }
    s.mean = sum / v.size();
    double sq = 0;
    for (double x : v) {
        sq += (x - s.mean) * (x - s.mean);
    }
    s.stdev = std::sqrt(sq / v.size());
    return s;
}

// Samples the power sensor in a thread until stop() is called.
class PowerSampler {
public:
    explicit PowerSampler(std::string file) : file_(std::move(file)) {}
    void start() {
        running_ = true;
        samples_.clear();
        thread_ = std::thread([this] {
            while (running_) {
                samples_.push_back(read_power_watts(file_));
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
    std::vector<double> stop() {
        running_ = false;
        thread_.join();
        return samples_;
    }

private:
    std::string file_;
    std::atomic<bool> running_{false};
    std::vector<double> samples_;
    std::thread thread_;
};

// Finite values in [-8, 8), exact in bf16 so the reference sees what the device sees.
std::vector<float> make_input(const Params& p) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    std::vector<float> x(static_cast<size_t>(p.tiles) * 1024);
    for (auto& v : x) {
        v = dist(rng);
        if (p.bf16) {
            uint32_t u;
            std::memcpy(&u, &v, 4);
            u &= 0xFFFF0000u;
            std::memcpy(&v, &u, 4);
        }
    }
    return x;
}

std::vector<uint32_t> pack_input(const Params& p, const std::vector<float>& x) {
    std::vector<uint32_t> words;
    if (p.bf16) {  // two bf16 per word: element 2i in the low half, 2i + 1 in the high half
        words.resize(x.size() / 2);
        for (size_t i = 0; i < words.size(); i++) {
            uint32_t a, b;
            std::memcpy(&a, &x[2 * i], 4);
            std::memcpy(&b, &x[2 * i + 1], 4);
            words[i] = (a >> 16) | (b & 0xFFFF0000u);
        }
    } else {
        words.resize(x.size());
        std::memcpy(words.data(), x.data(), x.size() * 4);
    }
    return words;
}

struct Setup {
    std::shared_ptr<distributed::MeshDevice> dev;
    std::vector<CoreCoord> cores;
    std::vector<std::shared_ptr<distributed::MeshBuffer>> in_buf, out_buf, scale_buf;
    distributed::MeshWorkload workload;
    uint32_t out_tile_bytes = 0;
    std::vector<float> x;  // host copy of the input, for the reference check
};

// The program the factory builds, on `n_cores` cores (one program, a private buffer set per core).
Setup build(const Params& p, uint32_t n_cores) {
    Setup s;
    s.dev = distributed::MeshDevice::create_unit_mesh(0);
    auto* dev = s.dev.get();
    const auto grid = dev->compute_with_storage_grid_size();
    for (uint32_t i = 0; i < n_cores; i++) {
        s.cores.push_back(CoreCoord{i % grid.x, i / grid.x});
    }

    const tt::DataFormat in_format = p.bf16 ? tt::DataFormat::Float16_b : tt::DataFormat::Float32;
    const tt::DataFormat out_format = p.out_int8 ? tt::DataFormat::UInt8 : tt::DataFormat::Int32;
    const tt::DataFormat scale_format = tt::DataFormat::Float32;
    const uint32_t in_tile_bytes = tt::tile_size(in_format);
    s.out_tile_bytes = tt::tile_size(out_format);
    const uint32_t scale_tile_bytes = tt::tile_size(scale_format);

    auto make_buffer = [&](uint32_t pages, uint32_t page_bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = static_cast<uint64_t>(pages) * page_bytes},
            distributed::DeviceLocalBufferConfig{.page_size = page_bytes, .buffer_type = BufferType::DRAM},
            dev);
    };

    s.x = make_input(p);
    const auto words = pack_input(p, s.x);
    for (uint32_t i = 0; i < n_cores; i++) {
        s.in_buf.push_back(make_buffer(p.tiles, in_tile_bytes));
        s.out_buf.push_back(make_buffer(p.tiles, s.out_tile_bytes));
        s.scale_buf.push_back(make_buffer(1, scale_tile_bytes));
        distributed::EnqueueWriteMeshBuffer(dev->mesh_command_queue(), s.in_buf.back(), words, true);
    }

    Program program = CreateProgram();
    std::set<CoreRange> ranges;
    for (const auto& c : s.cores) {
        ranges.insert(CoreRange(c));
    }
    const CoreRangeSet core_set(ranges);

    const uint32_t batch_tiles = std::min(reader_batch, p.tiles);
    const uint32_t in_cb_pages = batch_tiles * in_cb_factor;
    const uint32_t block_tiles = p.fpu ? dst_tiles : dst_tiles - 1;  // see the program factory
    const uint32_t out_cb_pages = 2 * block_tiles;
    auto cb = [&](uint32_t index, uint32_t pages, uint32_t page_bytes, tt::DataFormat fmt) {
        CreateCircularBuffer(
            program, core_set, CircularBufferConfig(pages * page_bytes, {{index, fmt}}).set_page_size(index, page_bytes));
    };
    cb(tt::CBIndex::c_0, in_cb_pages, in_tile_bytes, in_format);
    cb(tt::CBIndex::c_1, 1, scale_tile_bytes, scale_format);
    cb(tt::CBIndex::c_16, out_cb_pages, s.out_tile_bytes, out_format);
    cb(tt::CBIndex::c_17, 1, scale_tile_bytes, scale_format);

    const std::string op_dir = "../../src/operations/pot_quantization/per_tensor/device/kernels/";
    std::vector<uint32_t> reader_ct = {batch_tiles};
    TensorAccessorArgs(*s.in_buf[0]->get_backing_buffer()).append_to(reader_ct);
    const auto reader = CreateKernel(
        program,
        zkml::common::kernel_path(__FILE__, (op_dir + "dataflow/reader.cpp").c_str()),
        core_set,
        ReaderDataMovementConfig(reader_ct));

    std::vector<uint32_t> writer_ct;
    TensorAccessorArgs(*s.out_buf[0]->get_backing_buffer()).append_to(writer_ct);
    TensorAccessorArgs(*s.scale_buf[0]->get_backing_buffer()).append_to(writer_ct);
    const auto writer = CreateKernel(
        program,
        zkml::common::kernel_path(__FILE__, (op_dir + "dataflow/writer.cpp").c_str()),
        core_set,
        WriterDataMovementConfig(writer_ct));

    std::vector<UnpackToDestMode> unpack(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (!p.bf16) {
        unpack[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    }
    unpack[tt::CBIndex::c_1] = p.fpu ? UnpackToDestMode::Default : UnpackToDestMode::UnpackToDestFp32;
    const auto compute = CreateKernel(
        program,
        zkml::common::kernel_path(__FILE__, (op_dir + "compute/quantize.cpp").c_str()),
        core_set,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = std::move(unpack),
            .math_approx_mode = false,
            // output_int8, bit_width, c_0 ring size in tiles, DST tiles per section, FPU multiply
            .compile_args = {p.out_int8 ? 1u : 0u, p.bits, in_cb_pages, dst_tiles, p.fpu ? 1u : 0u},
            .compiler_include_paths = {zkml::common::kernel_include_dir()}});

    for (uint32_t i = 0; i < n_cores; i++) {
        const auto& c = s.cores[i];
        SetRuntimeArgs(program, reader, c, {static_cast<uint32_t>(s.in_buf[i]->address()), p.tiles});
        SetRuntimeArgs(
            program,
            writer,
            c,
            {static_cast<uint32_t>(s.out_buf[i]->address()), static_cast<uint32_t>(s.scale_buf[i]->address()), p.tiles});
        SetRuntimeArgs(program, compute, c, {p.tiles});
    }
    s.workload.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(program));
    return s;
}

// Compares core `i` against the host reference. Returns the number of wrong values (scaler counts as one).
size_t check(Setup& s, const Params& p, uint32_t i, float* scaler_out, uint64_t* checksum_out) {
    auto& cq = s.dev->mesh_command_queue();
    std::vector<uint32_t> out, scale;
    distributed::EnqueueReadMeshBuffer(cq, out, s.out_buf[i], true);
    distributed::EnqueueReadMeshBuffer(cq, scale, s.scale_buf[i], true);
    float got_scaler;
    std::memcpy(&got_scaler, &scale[0], 4);

    const int32_t k = zkml::reference::pot_k(zkml::reference::absmax(s.x), p.bits);
    const auto ref = zkml::reference::pot_quantize(s.x, k, p.bits);
    size_t bad = got_scaler == std::ldexp(1.0f, k) ? 0 : 1;
    uint64_t checksum = 1469598103934665603ull;  // FNV-1a over the output words
    for (uint32_t w : out) {
        checksum = (checksum ^ w) * 1099511628211ull;
    }
    for (size_t e = 0; e < s.x.size(); e++) {
        int32_t q;
        if (p.out_int8) {
            const auto* bytes = reinterpret_cast<const int8_t*>(out.data());
            q = bytes[e];
        } else {
            q = static_cast<int32_t>(out[e]);
        }
        bad += q != ref[e];
    }
    *scaler_out = got_scaler;
    *checksum_out = checksum;
    return bad;
}

void run_once(Setup& s) {
    auto& cq = s.dev->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, s.workload, false);
    distributed::Finish(cq);
}

std::string config_string(const Params& p) {
    return fmt::format(
        "in {} unit {} bits {} out {} tiles {}",
        p.bf16 ? "bf16" : "f32",
        p.fpu ? "fpu" : "sfpu",
        p.bits,
        p.out_int8 ? "int8" : "int32",
        p.tiles);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 8) {
        fmt::print(
            stderr,
            "usage: {0} profile|wall <bf16|f32> <sfpu|fpu> <bits> <int32|int8> <tiles> <iters>\n"
            "       {0} power        <bf16|f32> <sfpu|fpu> <bits> <int32|int8> <tiles> <cores> <seconds>\n",
            argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    Params p{
        .bf16 = std::string(argv[2]) == "bf16",
        .fpu = std::string(argv[3]) == "fpu",
        .bits = static_cast<uint32_t>(std::stoul(argv[4])),
        .out_int8 = std::string(argv[5]) == "int8",
        .tiles = static_cast<uint32_t>(std::stoul(argv[6]))};
    if (p.fpu && !p.bf16) {
        fmt::print(stderr, "the FPU variant needs a bf16 input\n");
        return 2;
    }
    SetRootDir(ZKML_TT_METAL_HOME);

    if (mode == "profile" || mode == "wall") {
        const uint32_t iters = std::stoul(argv[7]);
        Setup s = build(p, 1);
        // Run 0 compiles the kernels and warms the caches; analyze.py drops it too.
        run_once(s);
        std::vector<double> lat_us;
        for (uint32_t i = 1; i < iters; i++) {
            const auto t0 = Clock::now();
            run_once(s);
            lat_us.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
        }
        double stream_us = 0;
        if (mode == "wall") {
            // Pipelined: enqueue all runs, then wait once.
            auto& cq = s.dev->mesh_command_queue();
            const auto t0 = Clock::now();
            for (uint32_t i = 0; i < iters; i++) {
                distributed::EnqueueMeshWorkload(cq, s.workload, false);
            }
            distributed::Finish(cq);
            stream_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / iters;
        } else {
            ReadMeshDeviceProfilerResults(*s.dev);  // writes profile_log_device.csv
        }
        float scaler;
        uint64_t checksum;
        const size_t bad = check(s, p, 0, &scaler, &checksum);
        std::sort(lat_us.begin(), lat_us.end());
        fmt::print(
            "{} {} | scaler {} checksum {:016x} wrong {} | host_latency_us_median {:.1f} min {:.1f} stream_us_per_run "
            "{:.1f}\n",
            mode,
            config_string(p),
            scaler,
            checksum,
            bad,
            lat_us.empty() ? 0.0 : lat_us[lat_us.size() / 2],
            lat_us.empty() ? 0.0 : lat_us.front(),
            stream_us);
        s.dev->close();
        return bad == 0 ? 0 : 1;
    }

    if (mode == "power") {
        if (argc < 9) {
            fmt::print(stderr, "power needs <cores> <seconds>\n");
            return 2;
        }
        const uint32_t n_cores = std::stoul(argv[7]);
        const double seconds = std::stod(argv[8]);
        const std::string power_file = find_power_file();
        if (power_file.empty()) {
            fmt::print(stderr, "no hwmon device named 'blackhole' under /sys/class/hwmon\n");
            return 2;
        }
        Setup s = build(p, n_cores);
        run_once(s);  // compile + warm up
        run_once(s);
        float scaler;
        uint64_t checksum;
        const size_t bad = check(s, p, 0, &scaler, &checksum);
        const size_t bad_last = check(s, p, n_cores - 1, &scaler, &checksum);

        // Idle: the device is open, firmware running, no workload. This is the baseline.
        PowerSampler sampler(power_file);
        sampler.start();
        std::this_thread::sleep_for(std::chrono::seconds(6));
        const auto idle = stat_of(sampler.stop());

        // Load: back-to-back runs for `seconds`, 8 enqueued at a time.
        auto& cq = s.dev->mesh_command_queue();
        sampler.start();
        const auto t0 = Clock::now();
        uint64_t runs = 0;
        while (std::chrono::duration<double>(Clock::now() - t0).count() < seconds) {
            for (int i = 0; i < 8; i++) {
                distributed::EnqueueMeshWorkload(cq, s.workload, false);
            }
            distributed::Finish(cq);
            runs += 8;
        }
        const double wall_s = std::chrono::duration<double>(Clock::now() - t0).count();
        const auto load = stat_of(sampler.stop());

        const double us_per_run = wall_s / runs * 1e6;
        const double e_total_per_core_run = load.mean * (wall_s / runs) / n_cores;
        const double e_dyn_per_core_run = (load.mean - idle.mean) * (wall_s / runs) / n_cores;
        fmt::print(
            "power {} cores {} seconds {:.1f} | wrong {} {} | runs {} us_per_run {:.2f} | idle_W mean {:.3f} sd {:.3f} "
            "n {} | load_W mean {:.3f} sd {:.3f} n {} | delta_W {:.3f} | energy_total_mJ_per_core_run {:.4f} "
            "energy_dynamic_mJ_per_core_run {:.4f}\n",
            config_string(p),
            n_cores,
            wall_s,
            bad,
            bad_last,
            runs,
            us_per_run,
            idle.mean,
            idle.stdev,
            idle.n,
            load.mean,
            load.stdev,
            load.n,
            load.mean - idle.mean,
            e_total_per_core_run * 1e3,
            e_dyn_per_core_run * 1e3);
        s.dev->close();
        return (bad + bad_last) == 0 ? 0 : 1;
    }

    fmt::print(stderr, "unknown mode {}\n", mode);
    return 2;
}
