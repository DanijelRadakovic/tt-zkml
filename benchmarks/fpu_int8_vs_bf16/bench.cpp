// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// int8 against bf16 matmul on the Tensix FPU, for quantized values q in [-127, 127] (see README.md).
//
// One core, one output block of MT x NT = 2 x 2 tiles (64 x 64 outputs) with the whole K dimension (KT tiles of 32)
// accumulated in DST. A and B are copied to L1 once and the block is recomputed `reps` times, so the number is the
// throughput of the matrix engine plus unpacker for operands already in L1, with the pack of the result included.
//
//   int8   A, B as Int8 (sign-magnitude, 1 byte)  -> ALU_ACC_CTRL_INT8_math_enabled, Int32 accumulator in DST
//   bf16   A, B as Float16_b holding the integers -> floating-point multiply, FP32 accumulator in DST
//
// For every (path, math fidelity, mode, K, data pattern) the benchmark also checks the result against an exact int64
// matmul on the host. Timing is wall clock around the workload with two repetition counts; the difference removes the
// launch, the DRAM -> L1 copy and the write-back.
//
// Usage: zkml_fpu_int8_vs_bf16_bench [--kt 1,4,16,32,64,128] [--fidelity lofi,hifi2,hifi3,hifi4] [--mode tiles,block]
//                                    [--csv path]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "common/kernel_path.hpp"

using namespace tt::tt_metal;
using Clock = std::chrono::steady_clock;

namespace {

constexpr uint32_t MT = 2, NT = 2;

enum class Path { Int8, Bf16 };

// --twos: write the Int8 operands as two's complement bytes, which is what pot_quantize_per_tensor emits for INT8
// output, instead of the sign-magnitude bytes the unpacker expects.
bool g_twos_complement_int8 = false;

struct Fidelity {
    const char* name;
    MathFidelity value;
};

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Matrix (rows x cols, row-major, values in [-127, 127]) -> tiles in the L1 layout of `path`: tiles row-major, inside a
// tile the 4 faces of 16 x 16 (top-left, top-right, bottom-left, bottom-right), each face row-major.
std::vector<uint32_t> tilize(const std::vector<int32_t>& m, uint32_t rows, uint32_t cols, Path path) {
    const uint32_t rt = rows / 32, ct = cols / 32;
    const size_t elem = path == Path::Int8 ? 1 : 2;
    std::vector<uint8_t> bytes(static_cast<size_t>(rows) * cols * elem);
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            const size_t tile = static_cast<size_t>(r / 32) * ct + c / 32;
            const size_t face = ((r % 32) / 16) * 2 + (c % 32) / 16;
            const size_t index = tile * 1024 + face * 256 + (r % 16) * 16 + (c % 16);
            const int32_t v = m[static_cast<size_t>(r) * cols + c];
            if (path == Path::Int8) {
                if (g_twos_complement_int8) {
                    bytes[index] = static_cast<uint8_t>(static_cast<int8_t>(v));
                } else {
                    bytes[index] =
                        v < 0 ? static_cast<uint8_t>(0x80 | (-v)) : static_cast<uint8_t>(v);  // sign-magnitude
                }
            } else {
                const float f = static_cast<float>(v);
                uint32_t bits;
                std::memcpy(&bits, &f, 4);
                const uint16_t b16 = static_cast<uint16_t>(bits >> 16);  // bf16 is exact for |v| <= 256
                std::memcpy(&bytes[index * 2], &b16, 2);
            }
        }
    }
    (void)rt;
    std::vector<uint32_t> words((bytes.size() + 3) / 4);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    return words;
}

// Output tiles (MT x NT, 4-byte elements) -> row-major matrix of raw 32-bit words.
std::vector<uint32_t> untilize_words(const std::vector<uint32_t>& tiles, uint32_t rows, uint32_t cols) {
    const uint32_t ct = cols / 32;
    std::vector<uint32_t> out(static_cast<size_t>(rows) * cols);
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            const size_t tile = static_cast<size_t>(r / 32) * ct + c / 32;
            const size_t face = ((r % 32) / 16) * 2 + (c % 32) / 16;
            out[static_cast<size_t>(r) * cols + c] = tiles[tile * 1024 + face * 256 + (r % 16) * 16 + (c % 16)];
        }
    }
    return out;
}

struct Check {
    uint64_t mismatches = 0;
    int64_t max_abs_error = 0;
    const char* encoding = "";
};

// Compares the device result with the exact matmul. The int8 path writes Int32 tiles: tried as two's complement and as
// sign-magnitude, the better one is reported. The bf16 path writes Float32.
Check compare(const std::vector<uint32_t>& words, const std::vector<int64_t>& exact, Path path) {
    auto run = [&](auto decode, const char* name) {
        Check c;
        c.encoding = name;
        for (size_t i = 0; i < exact.size(); i++) {
            const double got = decode(words[i]);
            const int64_t err = std::llround(std::fabs(got - static_cast<double>(exact[i])));
            if (err != 0) {
                c.mismatches++;
                c.max_abs_error = std::max(c.max_abs_error, err);
            }
        }
        return c;
    };
    if (path == Path::Bf16) {
        return run(
            [](uint32_t w) {
                float f;
                std::memcpy(&f, &w, 4);
                return static_cast<double>(f);
            },
            "float32");
    }
    Check twos = run([](uint32_t w) { return static_cast<double>(static_cast<int32_t>(w)); }, "two's complement");
    Check sm = run(
        [](uint32_t w) {
            const int64_t mag = w & 0x7FFFFFFFu;
            return static_cast<double>((w >> 31) ? -mag : mag);
        },
        "sign-magnitude");
    return twos.mismatches <= sm.mismatches ? twos : sm;
}

struct Variant {
    Program program;
    KernelHandle reader, writer, compute;
    std::shared_ptr<distributed::MeshBuffer> a, b, out;
};

}  // namespace

int main(int argc, char** argv) {
    std::vector<uint32_t> kts = {1, 4, 16, 32, 64, 128};
    std::vector<std::string> fidelities = {"lofi", "hifi2", "hifi3", "hifi4"};
    std::vector<std::string> modes = {"tiles", "block"};
    std::string csv_path;
    bool verbose = false;
    auto split = [](char* s) {
        std::vector<std::string> v;
        for (char* t = std::strtok(s, ","); t; t = std::strtok(nullptr, ",")) {
            v.push_back(t);
        }
        return v;
    };
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--kt") && i + 1 < argc) {
            kts.clear();
            for (auto& s : split(argv[++i])) {
                kts.push_back(std::stoul(s));
            }
        } else if (!std::strcmp(argv[i], "--fidelity") && i + 1 < argc) {
            fidelities = split(argv[++i]);
        } else if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) {
            modes = split(argv[++i]);
        } else if (!std::strcmp(argv[i], "--twos")) {
            g_twos_complement_int8 = true;
        } else if (!std::strcmp(argv[i], "--verbose")) {
            verbose = true;
        } else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) {
            csv_path = argv[++i];
        }
    }
    auto fidelity_of = [](const std::string& s) {
        if (s == "lofi") {
            return MathFidelity::LoFi;
        }
        if (s == "hifi2") {
            return MathFidelity::HiFi2;
        }
        if (s == "hifi3") {
            return MathFidelity::HiFi3;
        }
        return MathFidelity::HiFi4;
    };

    SetRootDir(ZKML_TT_METAL_HOME);
    auto dev = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = dev->mesh_command_queue();
    const CoreCoord core{0, 0};

    std::string csv =
        "path,fidelity,mode,K,pattern,block_ns,ns_per_tile_matmul,gmac_per_s,mismatches,max_abs_error,out_encoding\n";
    std::printf(
        "| path | fidelity | mode | K | pattern | ns / 2x2 block | ns / tile matmul | GMAC/s | wrong outputs (of 4096) "
        "| max error |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n");

    for (const std::string& mode_name : modes) {
        for (const std::string& fid_name : fidelities) {
            for (Path path : {Path::Int8, Path::Bf16}) {
                for (uint32_t kt : kts) {
                    const uint32_t K = 32 * kt, M = 32 * MT, N = 32 * NT;
                    const tt::DataFormat in_fmt = path == Path::Int8 ? tt::DataFormat::Int8 : tt::DataFormat::Float16_b;
                    const tt::DataFormat out_fmt = path == Path::Int8 ? tt::DataFormat::Int32 : tt::DataFormat::Float32;
                    const uint32_t in_bytes = tt::tile_size(in_fmt), out_bytes = tt::tile_size(out_fmt);
                    const uint32_t tiles_a = MT * kt, tiles_b = kt * NT, tiles_out = MT * NT;

                    // The patterns: random q in [-127, 127] (what real data looks like); small: A and B in [-31, 31]
                    // (products below 2^10); medium: A and B in [-45, 45] (products below 2^11); narrow: A in [-31,
                    // 31], B in [-15, 15] (the range for which the int8 mode needs one fidelity phase); constant: all
                    // +127 (every product is 127 * 127).
                    for (const char* pattern : {"random", "medium", "small", "narrow", "constant"}) {
                        std::vector<int32_t> A(static_cast<size_t>(M) * K), B(static_cast<size_t>(K) * N);
                        std::mt19937 rng(1234 + kt);
                        const bool constant = !std::strcmp(pattern, "constant");
                        const int32_t lim_a = !std::strcmp(pattern, "small") || !std::strcmp(pattern, "narrow") ? 31
                                              : !std::strcmp(pattern, "medium")                                 ? 45
                                                                                                                : 127;
                        const int32_t lim_b = !std::strcmp(pattern, "small")    ? 31
                                              : !std::strcmp(pattern, "narrow") ? 15
                                              : !std::strcmp(pattern, "medium") ? 45
                                                                                : 127;
                        std::uniform_int_distribution<int32_t> da(-lim_a, lim_a), db(-lim_b, lim_b);
                        for (auto& v : A) {
                            v = constant ? 127 : da(rng);
                        }
                        for (auto& v : B) {
                            v = constant ? 127 : db(rng);
                        }
                        std::vector<int64_t> exact(static_cast<size_t>(M) * N, 0);
                        for (uint32_t i = 0; i < M; i++) {
                            for (uint32_t k = 0; k < K; k++) {
                                for (uint32_t j = 0; j < N; j++) {
                                    exact[static_cast<size_t>(i) * N + j] +=
                                        static_cast<int64_t>(A[static_cast<size_t>(i) * K + k]) *
                                        B[static_cast<size_t>(k) * N + j];
                                }
                            }
                        }

                        auto make_buffer = [&](uint32_t tiles, uint32_t page) {
                            return distributed::MeshBuffer::create(
                                distributed::ReplicatedBufferConfig{.size = uint64_t(tiles) * page},
                                distributed::DeviceLocalBufferConfig{
                                    .page_size = page, .buffer_type = BufferType::DRAM},
                                dev.get());
                        };
                        auto buf_a = make_buffer(tiles_a, in_bytes);
                        auto buf_b = make_buffer(tiles_b, in_bytes);
                        auto buf_out = make_buffer(tiles_out, out_bytes);
                        distributed::EnqueueWriteMeshBuffer(cq, buf_a, tilize(A, M, K, path), true);
                        distributed::EnqueueWriteMeshBuffer(cq, buf_b, tilize(B, K, N, path), true);

                        // One program per run (cheap: the binaries are cached by the JIT; the CBs differ with K).
                        auto run = [&](uint32_t reps) {
                            Program program = CreateProgram();
                            auto cb = [&](uint32_t index, uint32_t tiles, uint32_t bytes, tt::DataFormat f) {
                                CreateCircularBuffer(
                                    program,
                                    core,
                                    CircularBufferConfig(tiles * bytes, {{index, f}}).set_page_size(index, bytes));
                            };
                            cb(0, tiles_a, in_bytes, in_fmt);
                            cb(1, tiles_b, in_bytes, in_fmt);
                            cb(16, tiles_out, out_bytes, out_fmt);
                            std::vector<uint32_t> rct;
                            TensorAccessorArgs(*buf_a->get_backing_buffer()).append_to(rct);
                            TensorAccessorArgs(*buf_b->get_backing_buffer()).append_to(rct);
                            auto reader = CreateKernel(
                                program,
                                zkml::common::kernel_path(__FILE__, "kernels/reader.cpp"),
                                core,
                                ReaderDataMovementConfig(rct));
                            std::vector<uint32_t> wct;
                            TensorAccessorArgs(*buf_out->get_backing_buffer()).append_to(wct);
                            auto writer = CreateKernel(
                                program,
                                zkml::common::kernel_path(__FILE__, "kernels/writer.cpp"),
                                core,
                                WriterDataMovementConfig(wct));
                            auto compute = CreateKernel(
                                program,
                                zkml::common::kernel_path(__FILE__, "kernels/compute.cpp"),
                                core,
                                ComputeConfig{
                                    .math_fidelity = fidelity_of(fid_name),
                                    .fp32_dest_acc_en = true,
                                    .compile_args = {MT, NT, mode_name == "tiles" ? 0u : 1u}});
                            SetRuntimeArgs(
                                program,
                                reader,
                                core,
                                {uint32_t(buf_a->address()), uint32_t(buf_b->address()), tiles_a, tiles_b});
                            SetRuntimeArgs(program, writer, core, {uint32_t(buf_out->address()), tiles_out});
                            SetRuntimeArgs(program, compute, core, {reps, kt});
                            distributed::MeshWorkload wl;
                            wl.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(program));
                            const auto t0 = Clock::now();
                            distributed::EnqueueMeshWorkload(cq, wl, false);
                            distributed::Finish(cq);
                            return ms_since(t0);
                        };

                        // Correctness: one repetition.
                        run(1);
                        std::vector<uint32_t> raw;
                        distributed::EnqueueReadMeshBuffer(cq, raw, buf_out, true);
                        const auto out_words = untilize_words(raw, M, N);
                        const Check check = compare(out_words, exact, path);
                        if (verbose && check.mismatches) {
                            std::printf("  first outputs (device raw word, exact):");
                            for (int i = 0; i < 6; i++) {
                                if (path == Path::Bf16) {
                                    float f;
                                    std::memcpy(&f, &out_words[i], 4);
                                    std::printf(" (%.1f, %lld)", f, (long long)exact[i]);
                                } else {
                                    std::printf(" (0x%08x, %lld)", out_words[i], (long long)exact[i]);
                                }
                            }
                            std::printf("\n");
                        }

                        // Timing (random pattern only: the data does not change the speed).
                        double block_ns = 0, tile_ns = 0, gmacs = 0;
                        if (!std::strcmp(pattern, "random")) {
                            // reps so that one run takes about 20 ms
                            uint32_t reps = 64;
                            for (int i = 0; i < 3; i++) {
                                const double ms = run(reps);
                                reps =
                                    static_cast<uint32_t>(std::clamp(reps * (20.0 / std::max(ms, 0.05)), 16.0, 4.0e6));
                            }
                            const uint32_t r1 = reps, r2 = 4 * reps;
                            std::vector<double> diffs;
                            for (int i = 0; i < 5; i++) {
                                const double t1 = run(r1), t2 = run(r2);
                                diffs.push_back((t2 - t1) * 1e6 / (r2 - r1));
                            }
                            block_ns = median(diffs);
                            tile_ns = block_ns / (MT * NT * kt);
                            gmacs = (double(M) * N * K) / block_ns;  // MAC per ns = GMAC/s
                        }
                        std::printf(
                            "| %s | %s | %s | %u | %s | %.0f | %.1f | %.0f | %llu | %lld |\n",
                            path == Path::Int8 ? "int8" : "bf16",
                            fid_name.c_str(),
                            mode_name.c_str(),
                            K,
                            pattern,
                            block_ns,
                            tile_ns,
                            gmacs,
                            static_cast<unsigned long long>(check.mismatches),
                            static_cast<long long>(check.max_abs_error));
                        std::fflush(stdout);
                        csv += std::string(path == Path::Int8 ? "int8" : "bf16") + "," + fid_name + "," + mode_name +
                               "," + std::to_string(K) + "," + pattern + "," + std::to_string(block_ns) + "," +
                               std::to_string(tile_ns) + "," + std::to_string(gmacs) + "," +
                               std::to_string(check.mismatches) + "," + std::to_string(check.max_abs_error) + "," +
                               check.encoding + "\n";
                    }
                }
            }
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
