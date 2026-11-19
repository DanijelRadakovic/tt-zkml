// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Two's complement against sign-magnitude int8 output of the SFPU quantizer: correctness, device cycles and time (see
// README.md). The kernel (kernels/compute.cpp) is pass 2 of the 8-bit SFPU quantizer on one core, for four encodings of the
// output byte (kernels/variants.hpp):
//
//   twos_lib        tt-metal's quant_int8_tile: two's complement byte, UInt8 packer (what pot_quantize_per_tensor does)
//   twos_own        the same kernel as a copy in this directory (checks that the copy adds nothing)
//   sm_int8pack     SFPCAST to sign-magnitude, Int8 packer
//   sm_uint8pack    sign-magnitude byte built with abs/shift/and/or, UInt8 packer
//
// Usage:
//   zkml_sign_mag_fpu_compat_bench check   [tiles] [variant]                every variant against the host answer
// All commands take --input fp32|bf16|fp16 (default fp32) and --extra-cb <format> (adds an unused CB c_5).
//   zkml_sign_mag_fpu_compat_bench profile <variant> <full|sfpu> <tiles> <reps> [runs]   run `runs` times; run with
//                                                                  TT_METAL_DEVICE_PROFILER=1 and analyze.py (cycles)
//   zkml_sign_mag_fpu_compat_bench wall    <variant> <full|sfpu> <tiles> <reps>          host wall time per tile
// `full` = copy + quantize + pack, `sfpu` = only the quantize calls. tiles is a multiple of 3.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

struct Variant {
    const char* name;
    uint32_t id;
    tt::DataFormat out_format;
    bool sign_magnitude;
};

const Variant kVariants[] = {
    {"twos_lib", 0, tt::DataFormat::UInt8, false},
    {"twos_own", 1, tt::DataFormat::UInt8, false},
    {"sm_int8pack", 2, tt::DataFormat::Int8, true},
    {"sm_uint8pack", 3, tt::DataFormat::UInt8, true},
};

const Variant& find_variant(const std::string& name) {
    for (const auto& v : kVariants) {
        if (name == v.name) return v;
    }
    std::fprintf(stderr, "unknown variant %s\n", name.c_str());
    std::exit(2);
}

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr float kScale = 32.0f;  // 2^5

// Input CB format: fp32 (default), bf16 or fp16. The JIT refuses an Int8 CB next to a Float16_b CB ("All input data-formats
// must have the same exponent format": Int8 counts as an A-exponent format), so sm_int8pack builds only with fp32 input.
// --extra-cb <format>: one more, unused CB (c_5) of this format next to the real ones, to ask the JIT which formats may
// share a program with an Int8 CB.
tt::DataFormat g_extra_cb = tt::DataFormat::Invalid;
tt::DataFormat parse_format(const std::string& n) {
    if (n == "fp32") return tt::DataFormat::Float32;
    if (n == "bf16") return tt::DataFormat::Float16_b;
    if (n == "fp16") return tt::DataFormat::Float16;
    if (n == "tf32") return tt::DataFormat::Tf32;
    if (n == "bfp8_b") return tt::DataFormat::Bfp8_b;
    if (n == "bfp4_b") return tt::DataFormat::Bfp4_b;
    if (n == "bfp8_a") return tt::DataFormat::Bfp8;
    if (n == "int8") return tt::DataFormat::Int8;
    if (n == "uint8") return tt::DataFormat::UInt8;
    if (n == "int16") return tt::DataFormat::Int16;
    if (n == "uint16") return tt::DataFormat::UInt16;
    if (n == "int32") return tt::DataFormat::Int32;
    if (n == "uint32") return tt::DataFormat::UInt32;
    std::fprintf(stderr, "unknown format %s\n", n.c_str());
    std::exit(2);
}

enum class Input { Fp32, Bf16, Fp16 };
Input g_input = Input::Fp32;
tt::DataFormat in_format() {
    return g_input == Input::Bf16 ? tt::DataFormat::Float16_b : g_input == Input::Fp16 ? tt::DataFormat::Float16 : tt::DataFormat::Float32;
}

// Input element g: x = m / 64 with m in [-256, 256] (exact in bf16), so x * 32 = m / 2 covers every half-integer from -128
// to 128: ties (rounded to even), the whole int8 range, and values beyond +-127 that the clamp has to handle.
int32_t m_of(size_t g) { return static_cast<int32_t>((g * 37) % 513) - 256; }

uint8_t expected_byte(int32_t m, bool sign_magnitude) {
    const double scaled = m / 2.0;  // x * 32
    int64_t q = static_cast<int64_t>(std::nearbyint(scaled));  // ties to even
    q = std::clamp<int64_t>(q, sign_magnitude ? -127 : -128, 127);
    if (sign_magnitude) {
        return q < 0 ? static_cast<uint8_t>(0x80 | (-q)) : static_cast<uint8_t>(q);
    }
    return static_cast<uint8_t>(static_cast<int8_t>(q));
}

struct Setup {
    std::shared_ptr<distributed::MeshDevice> dev;
    std::shared_ptr<distributed::MeshBuffer> in, scale, out;
    uint32_t tiles;
};

Setup make_setup(uint32_t tiles) {
    Setup s;
    s.tiles = tiles;
    s.dev = distributed::MeshDevice::create_unit_mesh(0);
    auto make = [&](uint32_t pages, uint32_t page_bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = uint64_t(pages) * page_bytes},
            distributed::DeviceLocalBufferConfig{.page_size = page_bytes, .buffer_type = BufferType::DRAM},
            s.dev.get());
    };
    const uint32_t in_bytes = tt::tile_size(in_format());
    s.in = make(tiles, in_bytes);
    s.scale = make(1, tt::tile_size(tt::DataFormat::Float32));
    s.out = make(tiles, tt::tile_size(tt::DataFormat::UInt8));
    // Input: fp32, or bf16 / fp16 (two per word, element 2i in the low half).
    auto fp32_bits = [&](size_t g) {
        const float f = static_cast<float>(m_of(g)) / 64.0f;
        uint32_t b;
        std::memcpy(&b, &f, 4);
        return b;
    };
    // fp32 -> fp16 for the values used here (m / 64, at most 9 significant bits: exact, normal or zero).
    auto fp16_bits = [&](size_t g) -> uint32_t {
        const uint32_t b = fp32_bits(g);
        const uint32_t sign = (b >> 16) & 0x8000u;
        if ((b & 0x7FFFFFFFu) == 0) return sign;
        const int exp = static_cast<int>((b >> 23) & 0xFF) - 127 + 15;
        return sign | (static_cast<uint32_t>(exp) << 10) | ((b >> 13) & 0x3FFu);
    };
    std::vector<uint32_t> words(g_input == Input::Fp32 ? static_cast<size_t>(tiles) * 1024 : static_cast<size_t>(tiles) * 512);
    for (size_t w = 0; w < words.size(); w++) {
        if (g_input == Input::Fp32) {
            words[w] = fp32_bits(w);
        } else if (g_input == Input::Bf16) {
            words[w] = (fp32_bits(2 * w) >> 16) | (fp32_bits(2 * w + 1) & 0xFFFF0000u);
        } else {
            words[w] = fp16_bits(2 * w) | (fp16_bits(2 * w + 1) << 16);
        }
    }
    distributed::EnqueueWriteMeshBuffer(s.dev->mesh_command_queue(), s.in, words, true);
    std::vector<uint32_t> sc(1024, 0);
    const float scale = kScale;
    for (auto& w : sc) std::memcpy(&w, &scale, 4);
    distributed::EnqueueWriteMeshBuffer(s.dev->mesh_command_queue(), s.scale, sc, true);
    return s;
}

// Runs the kernel once; returns the host wall time in ms (enqueue + finish).
double run(Setup& s, const Variant& v, uint32_t mode, uint32_t reps) {
    Program program = CreateProgram();
    const CoreCoord core{0, 0};
    auto cb = [&](uint32_t index, uint32_t tiles, uint32_t bytes, tt::DataFormat f) {
        CreateCircularBuffer(program, core, CircularBufferConfig(tiles * bytes, {{index, f}}).set_page_size(index, bytes));
    };
    cb(0, s.tiles, tt::tile_size(in_format()), in_format());
    cb(1, 1, tt::tile_size(tt::DataFormat::Float32), tt::DataFormat::Float32);
    cb(16, s.tiles, tt::tile_size(v.out_format), v.out_format);
    if (g_extra_cb != tt::DataFormat::Invalid) cb(5, 1, tt::tile_size(g_extra_cb), g_extra_cb);
    std::vector<uint32_t> rct;
    TensorAccessorArgs(*s.in->get_backing_buffer()).append_to(rct);
    TensorAccessorArgs(*s.scale->get_backing_buffer()).append_to(rct);
    auto reader = CreateKernel(program, zkml::common::kernel_path(__FILE__, "kernels/reader.cpp"), core, ReaderDataMovementConfig(rct));
    std::vector<uint32_t> wct;
    TensorAccessorArgs(*s.out->get_backing_buffer()).append_to(wct);
    auto writer = CreateKernel(program, zkml::common::kernel_path(__FILE__, "kernels/writer.cpp"), core, WriterDataMovementConfig(wct));
    // The scaler tile goes to DST at full precision, like in the op (SFPU path).
    std::vector<UnpackToDestMode> unpack_to_dest(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest[1] = UnpackToDestMode::UnpackToDestFp32;
    if (g_input == Input::Fp32) unpack_to_dest[0] = UnpackToDestMode::UnpackToDestFp32;  // as the op does for Float32 input
    auto compute = CreateKernel(
        program, zkml::common::kernel_path(__FILE__, "kernels/compute.cpp"), core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest,
            .compile_args = {v.id, mode},
            .compiler_include_paths = {zkml::common::kernel_path(__FILE__, "kernels")}});
    SetRuntimeArgs(program, reader, core, {uint32_t(s.in->address()), uint32_t(s.scale->address()), s.tiles});
    SetRuntimeArgs(program, writer, core, {uint32_t(s.out->address()), s.tiles});
    SetRuntimeArgs(program, compute, core, {reps, s.tiles});
    auto& cq = s.dev->mesh_command_queue();
    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(s.dev->shape()), std::move(program));
    const auto t0 = Clock::now();
    distributed::EnqueueMeshWorkload(cq, wl, false);
    distributed::Finish(cq);
    return ms_since(t0);
}

int do_check(uint32_t tiles, const std::string& only) {
    Setup s = make_setup(tiles);
    std::printf("| variant | packer | wrong bytes of %u | first wrong (element: got, expected) |\n|---|---|---|---|\n", tiles * 1024);
    int failures = 0;
    for (const auto& v : kVariants) {
        if (!only.empty() && only != v.name) continue;
        run(s, v, 0, 1);
        std::vector<uint32_t> raw;
        distributed::EnqueueReadMeshBuffer(s.dev->mesh_command_queue(), raw, s.out, true);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(raw.data());
        size_t wrong = 0;
        std::string first;
        std::set<int32_t> wrong_m;
        for (size_t g = 0; g < static_cast<size_t>(tiles) * 1024; g++) {
            const uint8_t want = expected_byte(m_of(g), v.sign_magnitude);
            if (bytes[g] != want) {
                if (wrong < 3) {
                    char buf[80];
                    std::snprintf(buf, sizeof buf, "%zu (m=%d): 0x%02x, 0x%02x; ", g, m_of(g), bytes[g], want);
                    first += buf;
                }
                wrong++;
                wrong_m.insert(m_of(g));
            }
        }
        failures += wrong != 0;
        if (wrong) {
            first += " wrong m values:";
            int shown = 0;
            for (int32_t m : wrong_m) {
                if (shown++ < 12) first += " " + std::to_string(m);
            }
            if (wrong_m.size() > 12) first += " ... (" + std::to_string(wrong_m.size()) + " values)";
        }
        std::printf("| %s | %s | %zu | %s |\n", v.name, v.out_format == tt::DataFormat::Int8 ? "Int8" : "UInt8", wrong, first.c_str());
        std::fflush(stdout);
    }
    s.dev->close();
    return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: see the header of bench.cpp\n");
        return 2;
    }
    SetRootDir(ZKML_TT_METAL_HOME);
    // --input fp32|bf16|fp16 can stand anywhere
    std::vector<char*> args;
    for (int i = 0; i < argc; i++) {
        if (!std::strcmp(argv[i], "--input") && i + 1 < argc) {
            const std::string f = argv[++i];
            g_input = f == "bf16" ? Input::Bf16 : f == "fp16" ? Input::Fp16 : Input::Fp32;
        } else if (!std::strcmp(argv[i], "--extra-cb") && i + 1 < argc) {
            g_extra_cb = parse_format(argv[++i]);
        } else {
            args.push_back(argv[i]);
        }
    }
    argc = static_cast<int>(args.size());
    argv = args.data();
    const std::string cmd = argv[1];
    if (cmd == "check") {
        return do_check(argc > 2 ? std::stoul(argv[2]) : 48, argc > 3 ? argv[3] : "");
    }
    if (argc < 6) {
        std::fprintf(stderr, "usage: see the header of bench.cpp\n");
        return 2;
    }
    const Variant& v = find_variant(argv[2]);
    const uint32_t mode = std::string(argv[3]) == "sfpu" ? 1 : 0;
    const uint32_t tiles = std::stoul(argv[4]);
    const uint32_t reps = std::stoul(argv[5]);
    Setup s = make_setup(tiles);
    if (cmd == "profile") {
        const int runs = argc > 6 ? std::atoi(argv[6]) : 11;
        for (int i = 0; i < runs; i++) run(s, v, mode, reps);
        ReadMeshDeviceProfilerResults(*s.dev);
        std::printf("profile %s %s tiles %u reps %u runs %d\n", v.name, argv[3], tiles, reps, runs);
    } else if (cmd == "wall") {
        run(s, v, mode, reps);  // warm-up: JIT
        std::vector<double> per_tile_ns;
        for (int i = 0; i < 7; i++) {
            const double t1 = run(s, v, mode, reps), t2 = run(s, v, mode, 4 * reps);
            per_tile_ns.push_back((t2 - t1) * 1e6 / (3.0 * reps * tiles));
        }
        std::sort(per_tile_ns.begin(), per_tile_ns.end());
        std::printf("wall %s %s tiles %u reps %u: %.2f ns per tile (median of 7; min %.2f max %.2f)\n", v.name, argv[3], tiles, reps,
                    per_tile_ns[3], per_tile_ns.front(), per_tile_ns.back());
    } else {
        std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
        return 2;
    }
    s.dev->close();
    return 0;
}
