// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// On-device test for zkml::prim::pot_quantize_per_tensor against a CPU reference:
//   Q = 2^(bits - 1) - 1,  k = max{k <= 127 : absmax(x) * 2^k <= Q},
//   q = clamp(round_half_even(x * 2^k), -Q, Q)
// Checks both outputs: the scaler 2^k the device derived from the data, and q.
//
// Covers:
//   - FLOAT32/BFLOAT16 input, bit widths 8/16/32, INT32 output (all widths) and INT8 (8 bits)
//   - data at several magnitudes, so k is positive, negative, and large enough that
//     |x * 2^k| >= 2^23 (32 bits), plus ties (RNE) on the grid values
//   - absmax set by a positive or a negative element
//   - absmax exactly Q (k = 0) and just above it (k = -1)
//   - an all-zero tensor (k = 127, q = 0)
//   - the absmax in every one of the 1024 positions of a tile, and in every tile of a multi-tile
//     tensor, which exercises the SFPU per-lane accumulation and the 32-lane reduction
//   - odd and large tile counts (1..13 tiles, then 257/1001/2049 tiles, far beyond the c_0 reader
//     block), so the compute kernel's DST chunks (3 and 4 tiles) are cut at the end of the tensor
//     and at the ends of the c_0 and c_16 rings, with the rings wrapping many times
//   - the reader knobs noc_read_tile_batch_size / cb_in_size_factor: prime tile counts (the last
//     batch of the read stream is short), a batch of 1, a batch larger than the tensor, a batch that
//     divides the tensor (66 tiles = 6 x 11) and one that does not, and the argument checks
//     (batch >= 1, factor >= 2, c_0 must fit in L1)
//   - the program cache: a second pass over every case must hit cached programs
//
// The L1 pre-check these ops use is tested on its own in tests/cpp/common/test_l1_check.cpp.
//
// One test per group below (--gtest_filter=PotQuantizePerTensor.TileCountSweeps, ...). Each case reports
// its own failure with a label; passing cases are printed only for the matrix and the large tile counts.
//
// TT_METAL_RUNTIME_ROOT defaults to the TT_METAL_HOME used at configure time; set it to override.

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "common/pot_reference.hpp"
#include "operations/pot_quantization/per_tensor/pot_quantize_per_tensor.hpp"

using namespace ttnn;
using tt::tt_metal::DataType;
using zkml::prim::BitWidth;

namespace {

const char* name_of(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32: return "f32";
        case DataType::BFLOAT16: return "bf16";
        case DataType::INT32: return "int32";
        case DataType::INT8: return "int8";
        default: return "?";
    }
}

struct Result {
    bool ok;
    int32_t k;
    std::vector<int32_t> q;
};

// Runs one case. `verbose` prints a line even when the case passes.
Result run(
    MeshDevice* dev,
    const std::string& label,
    uint32_t H,
    uint32_t W,
    const std::vector<float>& data,
    DataType in_dt,
    DataType out_dt,
    BitWidth width,
    bool verbose = true,
    uint32_t batch = 16,
    uint32_t factor = 2) {
    tt::tt_metal::TensorSpec spec(
        Shape({H, W}), tt::tt_metal::TensorLayout(in_dt, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
    Tensor in = Tensor::from_vector(data, spec, dev);
    // The reference uses the values the device actually sees (BFLOAT16 rounds the input).
    const std::vector<float> x = in.to_vector<float>();

    auto [out, scaler] = zkml::prim::pot_quantize_per_tensor(in, width, out_dt, std::nullopt, batch, factor);

    std::vector<int32_t> q;
    if (out_dt == DataType::INT8) {
        auto v = out.to_vector<int8_t>();
        q.assign(v.begin(), v.end());
    } else {
        q = out.to_vector<int32_t>();
    }
    const float got_scaler = scaler.to_vector<float>().at(0);

    const int32_t k = zkml::reference::pot_k(zkml::reference::absmax(x), width.bits());
    const float ref_scaler = std::ldexp(1.0f, k);
    const std::vector<int32_t> ref = zkml::reference::pot_quantize(x, k, width.bits());

    bool ok = true;
    if (got_scaler != ref_scaler) {
        std::printf(
            "  [%s] scaler mismatch: got %g (k=%g) expected %g (k=%d), absmax=%g\n",
            label.c_str(),
            got_scaler,
            std::log2(got_scaler),
            ref_scaler,
            k,
            zkml::reference::absmax(x));
        ok = false;
    }
    size_t bad = 0;
    for (size_t i = 0; i < x.size(); i++) {
        if (q[i] != ref[i]) {
            if (bad < 5) {
                std::printf("  [%s] mismatch i=%zu x=%.9g got=%d ref=%d\n", label.c_str(), i, x[i], q[i], ref[i]);
            }
            bad++;
        }
    }
    ok = ok && bad == 0;
    EXPECT_TRUE(ok) << label << " (" << name_of(in_dt) << " -> " << name_of(out_dt) << ", " << width.bits() << " bits)";
    if (verbose || !ok) {
        std::printf(
            "%-4s %-32s in=%-4s out=%-5s bits=%-2u k=%-4d %zu/%zu mismatches\n",
            ok ? "ok" : "FAIL",
            label.c_str(),
            name_of(in_dt),
            name_of(out_dt),
            width.bits(),
            k,
            bad,
            x.size());
    }
    return {ok, k, std::move(q)};
}

// bf16-exact grid in [-8, 8) with ties at small k; FLOAT32 adds a non-grid offset.
std::vector<float> grid(uint32_t n, DataType in_dt, float multiplier) {
    std::vector<float> x(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = static_cast<float>(static_cast<int>(i % 512) - 256) / 32.0f;
        if (in_dt == DataType::FLOAT32) {
            x[i] += static_cast<float>(i % 7) * 1e-3f;
        }
        x[i] *= multiplier;
    }
    return x;
}

// Small background with one large element at `pos`.
std::vector<float> spike(uint32_t n, uint32_t pos, float value) {
    std::vector<float> x(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 1e-3f;
    }
    x[pos] = value;
    return x;
}

struct Config {
    DataType in_dt;
    DataType out_dt;
    BitWidth width;
};

std::vector<Config> all_configs() {
    return std::vector<Config>{
        {DataType::FLOAT32, DataType::INT32, BitWidth::Bits8},
        {DataType::FLOAT32, DataType::INT8, BitWidth::Bits8},
        {DataType::FLOAT32, DataType::INT32, BitWidth::Bits16},
        {DataType::FLOAT32, DataType::INT32, BitWidth::Bits32},
        {DataType::BFLOAT16, DataType::INT32, BitWidth::Bits8},
        {DataType::BFLOAT16, DataType::INT8, BitWidth::Bits8},
        {DataType::BFLOAT16, DataType::INT32, BitWidth::Bits16},
        {DataType::BFLOAT16, DataType::INT32, BitWidth::Bits32},
    };
}

// Every case except the position sweeps. Returns the quantized outputs, in order, so a second
// pass can check that the cached programs give the same results.
void run_matrix(MeshDevice* dev, std::vector<std::vector<int32_t>>& outputs) {
    constexpr uint32_t H = 64, W = 96;
    auto record = [&](Result r) { outputs.push_back(std::move(r.q)); };

    for (const auto& c : all_configs()) {
        // 1: absmax 8 from a negative element. -1: absmax 8 from a positive one.
        // 16: absmax 128, above Q for 8 bits. 1e-3: large k. 3e4: negative k.
        for (float m : {1.0f, -1.0f, 16.0f, 1e-3f, 3e4f}) {
            char label[64];
            std::snprintf(label, sizeof label, "grid x %g", m);
            record(run(dev, label, H, W, grid(H * W, c.in_dt, m), c.in_dt, c.out_dt, c.width));
        }

        // absmax exactly Q gives k = 0; Q + 0.5 needs k = -1 (the device's k - 1 correction).
        // BFLOAT16 rounds 32767 up to 32768, so its 16-bit "Q" case already lands on k = -1.
        // Q = 2^31 - 1 is not an fp32, so the 32-bit case uses the fp32 just below it; no fp32 lies
        // between 2^31 - 1 and 2^31, so the 32-bit correction never fires and has no Q + 0.5 case.
        const float q = c.width == BitWidth::Bits32 ? 2147483520.0f : static_cast<float>(zkml::reference::pot_qmax(c.width.bits()));
        record(run(dev, "absmax = Q", H, W, spike(H * W, 1234, q), c.in_dt, c.out_dt, c.width));
        if (c.width != BitWidth::Bits32) {
            record(run(dev, "absmax = Q + 0.5", H, W, spike(H * W, 1234, q + 0.5f), c.in_dt, c.out_dt, c.width));
        }

        record(run(dev, "all zeros", H, W, std::vector<float>(H * W, 0.0f), c.in_dt, c.out_dt, c.width));
    }
}

// The absmax at every position of one tile, then in every tile of a multi-tile tensor. The
// background alone would give a much larger k, so a lane the reduction missed shows up as a
// scaler mismatch.
void run_position_sweeps(MeshDevice* dev) {
    size_t cases = 0;

    constexpr uint32_t T = 32;
    for (uint32_t pos = 0; pos < T * T; pos++) {
        const float value = (pos % 2) ? -5.0f : 5.0f;
        char label[64];
        std::snprintf(label, sizeof label, "1-tile spike at %u", pos);
        run(dev, label, T, T, spike(T * T, pos, value), DataType::FLOAT32, DataType::INT32, BitWidth::Bits8, false);
        cases++;
    }

    // 3 x 2 tiles. Row-major positions of the first and last element of each tile.
    constexpr uint32_t H = 96, W = 64;
    for (uint32_t tr = 0; tr < H / T; tr++) {
        for (uint32_t tc = 0; tc < W / T; tc++) {
            for (uint32_t corner : {0u, T * T - 1}) {
                const uint32_t r = tr * T + corner / T;
                const uint32_t col = tc * T + corner % T;
                char label[64];
                std::snprintf(label, sizeof label, "6-tile spike at (%u, %u)", r, col);
                run(dev, label, H, W, spike(H * W, r * W + col, -5.0f), DataType::FLOAT32, DataType::INT32, BitWidth::Bits8, false);
                cases++;
            }
        }
    }
    std::printf("position sweeps: %zu cases\n", cases);
}

// Tile-dependent data (every tile differs, so a chunk that is dropped, repeated or written to the
// wrong page shows up) with the absmax in the very last element, which sits in the tail chunk.
std::vector<float> tiled(uint32_t n_tiles, DataType in_dt, float sign) {
    std::vector<float> x(static_cast<size_t>(n_tiles) * 1024);
    for (size_t i = 0; i < x.size(); i++) {
        const int v = static_cast<int>((i * 7 + (i / 1024) * 13) % 512) - 256;  // bf16-exact after / 32
        x[i] = static_cast<float>(v) / 32.0f;
        if (in_dt == DataType::FLOAT32) {
            x[i] += static_cast<float>(i % 7) * 1e-3f;
        }
    }
    x.back() = 9.0f * sign;
    return x;
}

// Odd and large tile counts for every config. The tensor is 1 tile high and n tiles wide.
void run_tile_count_sweeps(MeshDevice* dev) {
    size_t cases = 0;
    for (uint32_t n : {1u, 2u, 3u, 5u, 7u, 9u, 11u, 13u, 257u, 1001u, 2049u}) {
        for (const auto& c : all_configs()) {
            char label[64];
            std::snprintf(label, sizeof label, "%u tiles", n);
            const bool large = n >= 257;
            run(dev, label, 32, 32 * n, tiled(n, c.in_dt, (n % 4 == 1) ? 1.0f : -1.0f), c.in_dt, c.out_dt, c.width,
                large);
            cases++;
        }
    }
    std::printf("tile count sweeps: %zu cases\n", cases);
}

// The reader batch / ring-depth knobs. Tile counts include primes (the stream of 2 * n reads is
// then never a whole number of batches, so the last batch is short) and 66 = 6 x 11 with batch 11.
void run_batch_sweeps(MeshDevice* dev) {
    size_t cases = 0;
    struct Knobs {
        uint32_t batch;
        uint32_t factor;
    };
    const std::vector<Knobs> knobs = {{1, 2}, {2, 2}, {3, 3}, {7, 2}, {11, 3}, {16, 2}, {64, 2}, {5, 4}};
    const std::vector<Config> configs = {
        {DataType::FLOAT32, DataType::INT32, BitWidth::Bits8},
        {DataType::FLOAT32, DataType::INT32, BitWidth::Bits32},
        {DataType::BFLOAT16, DataType::INT8, BitWidth::Bits8},
    };
    for (uint32_t n : {1u, 2u, 3u, 7u, 11u, 13u, 66u, 67u, 101u}) {
        for (const auto& k : knobs) {
            for (const auto& c : configs) {
                char label[64];
                std::snprintf(label, sizeof label, "%u tiles batch %u x %u", n, k.batch, k.factor);
                run(dev, label, 32, 32 * n, tiled(n, c.in_dt, (n % 2) ? 1.0f : -1.0f), c.in_dt, c.out_dt, c.width, false,
                    k.batch, k.factor);
                cases++;
            }
        }
    }
    std::printf("batch sweeps: %zu cases\n", cases);
}

// Invalid knobs must be rejected, not run.
void run_rejections(MeshDevice* dev) {
    struct Bad {
        const char* what;
        uint32_t batch;
        uint32_t factor;
    };
    const std::vector<Bad> bad = {
        {"batch 0", 0, 2},
        {"factor 1", 8, 1},
        {"factor 0", 8, 0},
        {"c_0 larger than L1", 1000, 4},  // 4000 fp32 tiles = 16 MB
    };
    for (const auto& b : bad) {
        EXPECT_ANY_THROW(run(dev, b.what, 32, 32 * 2000, tiled(2000, DataType::FLOAT32, 1.0f), DataType::FLOAT32,
                             DataType::INT32, BitWidth::Bits8, false, b.batch, b.factor))
            << b.what;
    }
}

class PotQuantizePerTensor : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // TT_METAL_RUNTIME_ROOT, if set, takes precedence over this.
        tt::tt_metal::SetRootDir(ZKML_TT_METAL_HOME);
        device_ = open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);
    }
    static void TearDownTestSuite() {
        device_->close();
        device_.reset();
    }
    static MeshDevice* dev() { return device_.get(); }

    static inline std::shared_ptr<MeshDevice> device_;
};

// Pass 0 builds each program (cache miss -> create()). Pass 1 repeats every case with new
// tensor addresses (cache hit -> override_runtime_arguments()) and must add no cache entry.
TEST_F(PotQuantizePerTensor, MatrixAndProgramCache) {
    std::vector<std::vector<int32_t>> outputs[2];
    size_t entries[2] = {0, 0};
    for (int pass = 0; pass < 2; pass++) {
        std::printf("pass %d (%s)\n", pass, pass == 0 ? "program cache miss" : "program cache hit");
        run_matrix(dev(), outputs[pass]);
        entries[pass] = dev()->num_program_cache_entries();
        std::printf("program cache entries: %zu\n", entries[pass]);
    }
    EXPECT_EQ(entries[1], entries[0]) << "the second pass added program cache entries";
    EXPECT_EQ(outputs[0], outputs[1]) << "the cached programs gave different results";
}

// The op has no custom program hash, so ttnn's default hash must cover every attribute: changing any
// one of them must create a new program, and repeating a call must reuse it. If an attribute were
// missing from the key, the changed call would silently run the old program.
TEST_F(PotQuantizePerTensor, ProgramCacheKeyCoversEveryAttribute) {
    constexpr uint32_t n = 6;
    const auto data = tiled(n, DataType::BFLOAT16, 1.0f);
    auto call = [&](DataType out_dt, BitWidth width, uint32_t batch, uint32_t factor) {
        run(dev(), "cache key", 32, 32 * n, data, DataType::BFLOAT16, out_dt, width, false, batch, factor);
        return dev()->num_program_cache_entries();
    };
    const size_t base = call(DataType::INT32, BitWidth::Bits8, 2, 2);

    EXPECT_EQ(call(DataType::INT32, BitWidth::Bits8, 2, 2), base) << "same attributes must hit the cache";
    EXPECT_EQ(call(DataType::INT32, BitWidth::Bits8, 3, 2), base + 1) << "noc_read_tile_batch_size";
    EXPECT_EQ(call(DataType::INT32, BitWidth::Bits8, 2, 3), base + 2) << "cb_in_size_factor";
    EXPECT_EQ(call(DataType::INT32, BitWidth::Bits16, 2, 2), base + 3) << "bit_width";
    EXPECT_EQ(call(DataType::INT8, BitWidth::Bits8, 2, 2), base + 4) << "output_dtype";
    EXPECT_EQ(call(DataType::INT32, BitWidth::Bits8, 3, 2), base + 4) << "an earlier variant must still hit";
}

TEST_F(PotQuantizePerTensor, PositionSweeps) { run_position_sweeps(dev()); }
TEST_F(PotQuantizePerTensor, TileCountSweeps) { run_tile_count_sweeps(dev()); }
TEST_F(PotQuantizePerTensor, BatchSweeps) { run_batch_sweeps(dev()); }
TEST_F(PotQuantizePerTensor, RejectsInvalidReaderKnobs) { run_rejections(dev()); }

}  // namespace
