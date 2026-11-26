// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// On-device test for zkml::prim::pot_quant_m31: q -> q mod (2^31 - 1), as UINT32 in [0, p).
//
// Covers:
//   - INT32 input with the edge values of every width (0, +-1, +-Q for Q = 127, 32767, 2^30 - 1) and a long random
//     tensor, with tile counts around the compute block (1..9, then 257, 1001): the residues equal the reference, are
//     canonical (< p), map injectively and read back to the same signed value
//   - INT8 input (the two's-complement bytes the quantizer writes)
//   - the chain pot_quantize_per_tensor(QuantRange::M31) -> pot_quant_m31 for 8, 16 and 32 bits
//   - why the M31 range exists: at 32 bits with QuantRange::Full, q = 2^31 - 1 maps to p, not a canonical residue
//   - the program cache: the second call of every case hits
//   - agreement with the Rust CPU/SIMD implementation (tools/stwo-cpu-simd): `CrossCheckWithRust` reads the files its
//     `dump` binary wrote (set ZKML_RUST_REF_DIR; skipped otherwise) and compares q, k and the M31 residues (for
//     QuantRange::M31), for every width, both ranges and both input dtypes
//
// TT_METAL_RUNTIME_ROOT defaults to the TT_METAL_HOME used at configure time; set it to override.

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>
#include "common/pot_reference.hpp"
#include "operations/pot_quantization/per_tensor/pot_quantize_m31/pot_quant_m31.hpp"
#include "operations/pot_quantization/per_tensor/pot_quantize/pot_quantize_per_tensor.hpp"

using namespace ttnn;
using tt::tt_metal::DataType;
using zkml::prim::BitWidth;

namespace {

class PotQuantM31 : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
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

// Checks the device result against the reference and the field properties. Returns false on any mismatch.
bool check_residues(const char* label, const std::vector<int32_t>& q, const Tensor& m31) {
    EXPECT_EQ(m31.dtype(), DataType::UINT32);
    const std::vector<uint32_t> got = m31.to_vector<uint32_t>();
    if (got.size() != q.size()) {
        std::printf("  [%s] %zu residues for %zu values\n", label, got.size(), q.size());
        return false;
    }
    size_t bad = 0;
    std::unordered_map<uint32_t, int32_t> seen;  // residue -> the value that produced it
    for (size_t i = 0; i < q.size(); i++) {
        const uint32_t want = zkml::reference::pot_to_m31(q[i]);
        const bool canonical = got[i] < zkml::prim::m31_modulus;
        const bool readback = canonical && zkml::reference::pot_from_m31(got[i]) == q[i];
        const auto [it, fresh] = seen.emplace(got[i], q[i]);
        const bool injective = fresh || it->second == q[i];  // two different q must not share a residue
        if (got[i] != want || !canonical || !readback || !injective) {
            if (bad < 5) {
                std::printf("  [%s] i=%zu q=%d got=%u want=%u\n", label, i, q[i], got[i], want);
            }
            bad++;
        }
    }
    if (bad) {
        std::printf("  [%s] %zu/%zu wrong\n", label, bad, q.size());
    }
    return bad == 0;
}

Tensor to_device(const std::vector<int32_t>& q, uint32_t H, uint32_t W, DataType dt, MeshDevice* dev) {
    tt::tt_metal::TensorSpec spec(
        Shape({H, W}), tt::tt_metal::TensorLayout(dt, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
    if (dt == DataType::INT8) {
        std::vector<int8_t> v(q.begin(), q.end());
        return Tensor::from_vector(v, spec, dev);
    }
    return Tensor::from_vector(q, spec, dev);
}

}  // namespace

TEST_F(PotQuantM31, Int32EdgeValuesAndTileCounts) {
    for (uint32_t tiles : {1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 257u, 1001u}) {
        const uint32_t H = 32, W = 32 * tiles;
        std::vector<int32_t> q(H * W);
        std::mt19937 gen(tiles);
        for (size_t i = 0; i < q.size(); i++) {
            // Rotate through the extremes of every width, then random values of any magnitude up to 2^30 - 1.
            static const int32_t edge[] = {0, 1, -1, 127, -127, 32767, -32767, 1073741823, -1073741823, 1073741822,
                                           -1073741822, 536870912, -536870912, 65536, -65536, 2, -2};
            q[i] = (i % 3 == 0) ? edge[(i / 3) % (sizeof edge / sizeof edge[0])]
                                : static_cast<int32_t>(gen() % 2147483647u) / 2 * ((gen() & 1) ? 1 : -1);
        }
        char label[32];
        std::snprintf(label, sizeof label, "int32 %u tiles", tiles);
        for (int repeat = 0; repeat < 2; repeat++) {  // the second call hits the program cache
            Tensor in = to_device(q, H, W, DataType::INT32, dev());
            EXPECT_TRUE(check_residues(label, q, zkml::prim::pot_quant_m31(in))) << label;
        }
    }
}

TEST_F(PotQuantM31, Int8) {
    for (uint32_t tiles : {1u, 3u, 5u, 9u}) {
        const uint32_t H = 32, W = 32 * tiles;
        std::vector<int32_t> q(H * W);
        for (size_t i = 0; i < q.size(); i++) {
            q[i] = static_cast<int32_t>(i % 255) - 127;  // -127 .. 127, every value
        }
        char label[32];
        std::snprintf(label, sizeof label, "int8 %u tiles", tiles);
        Tensor in = to_device(q, H, W, DataType::INT8, dev());
        EXPECT_TRUE(check_residues(label, q, zkml::prim::pot_quant_m31(in))) << label;
    }
}

TEST_F(PotQuantM31, AfterQuantizer) {
    constexpr uint32_t H = 64, W = 96;
    std::vector<float> x(H * W);
    std::mt19937 gen(7);
    std::normal_distribution<float> dist(0.0f, 3.0f);
    for (auto& v : x) {
        v = dist(gen);
    }
    struct Case {
        BitWidth width;
        DataType out_dt;
    };
    for (const Case c : {Case{BitWidth::Bits8, DataType::INT8}, Case{BitWidth::Bits8, DataType::INT32},
                         Case{BitWidth::Bits16, DataType::INT32}, Case{BitWidth::Bits32, DataType::INT32}}) {
        {
            tt::tt_metal::TensorSpec spec(
                Shape({H, W}),
                tt::tt_metal::TensorLayout(DataType::FLOAT32, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
            Tensor in = Tensor::from_vector(x, spec, dev());
            auto [quantized, scaler] = zkml::prim::pot_quantize_per_tensor(
                in, c.width, c.out_dt, std::nullopt, 16, 2, zkml::prim::QuantRange::M31);
            std::vector<int32_t> q;
            if (c.out_dt == DataType::INT8) {
                auto v = quantized.to_vector<int8_t>();
                q.assign(v.begin(), v.end());
            } else {
                q = quantized.to_vector<int32_t>();
            }
            char label[48];
            std::snprintf(label, sizeof label, "quantizer %u bits %s", c.width.bits(),
                          c.out_dt == DataType::INT8 ? "int8" : "int32");
            EXPECT_TRUE(check_residues(label, q, zkml::prim::pot_quant_m31(quantized))) << label;
        }
    }
}

// With QuantRange::Full the largest 32-bit values do not fit M31: q = 2^31 - 128 is above (p - 1) / 2 and, read back as
// a signed value, would be taken for a negative one. QuantRange::M31 keeps every q readable.
TEST_F(PotQuantM31, FullRangeDoesNotFitM31) {
    constexpr uint32_t H = 32, W = 32;
    std::vector<float> x(H * W, 0.5f);
    x[3] = 2147483520.0f;  // the largest fp32 below 2^31
    tt::tt_metal::TensorSpec spec(
        Shape({H, W}), tt::tt_metal::TensorLayout(DataType::FLOAT32, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
    Tensor in = Tensor::from_vector(x, spec, dev());
    for (auto range : {zkml::prim::QuantRange::Full, zkml::prim::QuantRange::M31}) {
        auto result = zkml::prim::pot_quantize_per_tensor(
            in, BitWidth::Bits32, DataType::INT32, std::nullopt, 16, 2, range);
        const auto q = std::get<0>(result).to_vector<int32_t>();
        const auto m = zkml::prim::pot_quant_m31(std::get<0>(result)).to_vector<uint32_t>();
        const bool all_read_back = [&] {
            for (size_t i = 0; i < q.size(); i++) {
                if (zkml::reference::pot_from_m31(m[i]) != q[i]) {
                    return false;
                }
            }
            return true;
        }();
        if (range == zkml::prim::QuantRange::Full) {
            EXPECT_EQ(q[3], 2147483520);
            EXPECT_FALSE(all_read_back) << "the largest q has to be misread as a negative number";
        } else {
            EXPECT_LE(q[3], (1 << 30) - 1);
            EXPECT_TRUE(all_read_back);
        }
    }
}

// The reference cases written by `cargo run --release --bin dump -- <dir>` in tools/stwo-cpu-simd.
TEST_F(PotQuantM31, CrossCheckWithRust) {
    const char* dir_env = std::getenv("ZKML_RUST_REF_DIR");
    if (!dir_env) {
        GTEST_SKIP() << "set ZKML_RUST_REF_DIR to the output directory of tools/stwo-cpu-simd's `dump`";
    }
    const std::string dir = dir_env;
    auto load = [&](const std::string& name, auto zero) {
        std::ifstream f(dir + "/" + name, std::ios::binary);
        EXPECT_TRUE(f.good()) << name;
        using T = decltype(zero);
        std::vector<T> v;
        T e;
        while (f.read(reinterpret_cast<char*>(&e), sizeof e)) {
            v.push_back(e);
        }
        return v;
    };
    for (uint32_t bits : {8u, 16u, 32u}) {
      for (const auto range : {zkml::prim::QuantRange::Full, zkml::prim::QuantRange::M31}) {
        const std::string b = "b" + std::to_string(bits) + (range == zkml::prim::QuantRange::Full ? "_full" : "_m31");
        std::ifstream meta(dir + "/" + b + ".meta");
        uint32_t H = 0, W = 0, file_bits = 0, argmax = 0;  // argmax is in the file, the device no longer returns it
        int32_t k = 0;
        ASSERT_TRUE(static_cast<bool>(meta >> H >> W >> file_bits >> k >> argmax)) << b;
        ASSERT_EQ(file_bits, bits);
        const auto x = load(b + ".x.f32", 0.0f);
        const auto want_q = load(b + ".q.i32", int32_t{0});
        const auto want_m31 = load(b + ".m31.u32", uint32_t{0});
        ASSERT_EQ(x.size(), size_t{H} * W);

        struct Case {
            DataType in_dt;
            DataType out_dt;
        };
        std::vector<Case> cases = {{DataType::FLOAT32, DataType::INT32}, {DataType::BFLOAT16, DataType::INT32}};
        if (bits == 8) {
            cases.push_back({DataType::BFLOAT16, DataType::INT8});
        }
        for (const auto& c : cases) {
            tt::tt_metal::TensorSpec spec(
                Shape({H, W}),
                tt::tt_metal::TensorLayout(c.in_dt, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
            Tensor in = Tensor::from_vector(x, spec, dev());  // the values are bf16-exact: both dtypes see the same x
            auto [quantized, scaler_tensor] = zkml::prim::pot_quantize_per_tensor(
                in, BitWidth(static_cast<BitWidth::Value>(bits)), c.out_dt, std::nullopt, 16, 2, range);
            std::vector<int32_t> q;
            if (c.out_dt == DataType::INT8) {
                auto v = quantized.to_vector<int8_t>();
                q.assign(v.begin(), v.end());
            } else {
                q = quantized.to_vector<int32_t>();
            }
            const float scaler = scaler_tensor.to_vector<float>().at(0);
            const auto m31 = zkml::prim::pot_quant_m31(quantized).to_vector<uint32_t>();  // injective for M31 only
            const std::string label = b + (c.in_dt == DataType::FLOAT32 ? " f32" : " bf16") +
                                      (c.out_dt == DataType::INT8 ? " int8" : " int32");
            EXPECT_EQ(scaler, std::ldexp(1.0f, k)) << label;
            EXPECT_EQ(q, want_q) << label;
            if (range == zkml::prim::QuantRange::M31) {
                EXPECT_EQ(m31, want_m31) << label;
            }
        }
      }
    }
}
