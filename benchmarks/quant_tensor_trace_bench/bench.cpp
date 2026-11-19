// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Wall-clock benchmark of the quantizer and pot_quant_m31 on the device:
//   quantize    pot_quantize_per_tensor(QuantRange::M31): q and the scaler
//   to_m31      pot_quant_m31(q)
//   readback    copying the M31 tensor to the host (the cost of NOT keeping the data on the device)
// for bf16 input on one core, program cache warm, input already on the device. The CPU numbers are in
// tools/stwo-cpu-simd (the same shapes: 32 x 32*tiles).
//
// Usage: zkml_quant_trace_bench [--tiles 16,64,256,1024,4096] [--csv path]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

#include "operations/pot_quantization/per_tensor/pot_quantize_m31/pot_quant_m31.hpp"
#include "operations/pot_quantization/per_tensor/pot_quantize/pot_quantize_per_tensor.hpp"

using namespace ttnn;
using tt::tt_metal::DataType;
using zkml::prim::BitWidth;
using zkml::prim::QuantRange;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Time per call of `f` (it returns a tensor, kept alive until the end). n calls back to back, then one wait for the
// device queue (Finish), so the time is the throughput of the op including its host-side dispatch.
template <typename F>
double per_call_ms(MeshDevice* dev, F f, double budget_ms = 400.0) {
    auto sync = [&] { tt::tt_metal::distributed::Finish(dev->mesh_command_queue()); };
    (void)f();  // warm up: compile and cache
    sync();
    const auto t0 = Clock::now();
    (void)f();
    sync();
    const double one = std::max(ms_since(t0), 0.01);
    const int n = static_cast<int>(std::clamp(budget_ms / one, 3.0, 200.0));
    std::vector<double> runs;
    for (int rep = 0; rep < 5; rep++) {
        const auto t = Clock::now();
        std::optional<Tensor> last;
        for (int i = 0; i < n; i++) {
            last = f();
        }
        sync();
        runs.push_back(ms_since(t) / n);
    }
    return median(runs);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<uint32_t> tiles = {16, 64, 256, 1024, 4096};
    std::string csv_path;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--tiles") && i + 1 < argc) {
            tiles.clear();
            for (char* s = std::strtok(argv[++i], ","); s; s = std::strtok(nullptr, ",")) {
                tiles.push_back(static_cast<uint32_t>(std::atoi(s)));
            }
        } else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) {
            csv_path = argv[++i];
        }
    }

    tt::tt_metal::SetRootDir(ZKML_TT_METAL_HOME);
    auto dev = open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);

    std::string csv = "bits,tiles,elements,op,ms,melem_per_s,ns_per_tile\n";
    std::printf("| bits | tiles | quantize ms | to_m31 ms | quantize+m31 ms | readback of m31 ms |\n");
    std::printf("|---|---|---|---|---|---|\n");
    for (uint32_t bits : {8u, 16u, 32u}) {
        const BitWidth width(static_cast<BitWidth::Value>(bits));
        for (uint32_t t : tiles) {
            const uint32_t H = 32, W = 32 * t;
            std::vector<float> x(static_cast<size_t>(H) * W);
            uint64_t s = 0x9E3779B97F4A7C15ull;
            for (auto& v : x) {  // xorshift64*, uniform in [-8, 8), bf16-exact
                s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
                const float u = static_cast<float>((s * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.0f;
                uint32_t b;
                const float f = u * 16.0f - 8.0f;
                std::memcpy(&b, &f, 4);
                b = (b + 0x8000u) & 0xFFFF0000u;
                std::memcpy(&v, &b, 4);
            }
            tt::tt_metal::TensorSpec spec(
                Shape({H, W}), tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
            Tensor in = Tensor::from_vector(x, spec, dev.get());
            const DataType out_dt = DataType::INT32;

            auto quantize = [&] {
                return zkml::prim::pot_quantize_per_tensor(in, width, out_dt, std::nullopt, 16, 2, QuantRange::M31);
            };
            // The scaler is returned only so the lambda gives a tensor; the queue is waited for as a whole.
            const double q_ms = per_call_ms(dev.get(), [&] { return std::get<1>(quantize()); });
            auto quant = quantize();
            const double m_ms = per_call_ms(dev.get(), [&] { return zkml::prim::pot_quant_m31(std::get<0>(quant)); });
            auto m31 = zkml::prim::pot_quant_m31(std::get<0>(quant));
            std::vector<double> rb;
            for (int i = 0; i < 5; i++) {
                const auto t0 = Clock::now();
                const auto host = m31.to_vector<uint32_t>();
                rb.push_back(ms_since(t0));
                if (host.size() != x.size()) {
                    return 1;
                }
            }
            const double rb_ms = median(rb);

            const double n = static_cast<double>(x.size());
            auto row = [&](const char* op, double ms) {
                csv += std::to_string(bits) + "," + std::to_string(t) + "," + std::to_string(x.size()) + "," + op + "," +
                       std::to_string(ms) + "," + std::to_string(n / (ms * 1e3)) + "," + std::to_string(ms * 1e6 / t) + "\n";
            };
            row("quantize", q_ms);
            row("to_m31", m_ms);
            row("readback_m31", rb_ms);
            std::printf("| %u | %u | %.3f | %.3f | %.3f | %.3f |\n", bits, t, q_ms, m_ms, q_ms + m_ms, rb_ms);
            std::fflush(stdout);
        }
    }
    if (!csv_path.empty()) {
        FILE* f = std::fopen(csv_path.c_str(), "w");
        std::fputs(csv.c_str(), f);
        std::fclose(f);
        std::printf("\nwrote %s\n", csv_path.c_str());
    }
    dev->close();
    return 0;
}
