// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Tests for zkml::common::check_if_cbs_can_fit_in_l1 (src/common/l1_check.hpp), on its own, with no
// op involved. Needs a Tenstorrent device.
//
// The check promises to accept exactly the circular-buffer (CB) layouts tt-metal accepts when the
// program is first enqueued. The tests that matter therefore compare the two directly: build a program
// of no-op kernels and the given CBs, find with the check the largest last CB it accepts, and require
// that tt-metal runs the program at that size and refuses it one page more. Layouts with unaligned CB
// sizes make the alignment padding part of that boundary, and an L1-resident buffer moves the upper
// limit down.
//
// TT_METAL_RUNTIME_ROOT defaults to the TT_METAL_HOME used at configure time; set it to override.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>

#include "common/kernel_path.hpp"
#include "common/l1_check.hpp"

namespace {

using namespace tt::tt_metal;
using zkml::common::CbSpec;
using zkml::common::check_if_cbs_can_fit_in_l1;

// The boundary search varies the size of the last CB (c_31) in 16 B steps. tt-metal allows at most 65535
// pages per CB, i.e. about 1 MB at 16 B, so a filler CB (c_30) takes the rest of the space first and
// leaves less than that free.
constexpr uint32_t var_page_bytes = 16;
constexpr uint32_t max_cb_pages = 65535;
constexpr CbSpec filler_cb = {30, 600, 1024};  // 600 KB

uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

class L1CheckTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        SetRootDir(ZKML_TT_METAL_HOME);  // TT_METAL_RUNTIME_ROOT, if set, takes precedence
        device_ = distributed::MeshDevice::create_unit_mesh(0);
    }
    static void TearDownTestSuite() {
        device_->close();
        device_.reset();
    }

    static distributed::MeshDevice* dev() { return device_.get(); }
    static uint64_t alignment() { return dev()->allocator()->get_alignment(BufferType::DRAM); }

    // True if tt-metal accepts a program with these CBs (and a no-op kernel): it validates the CB
    // region when the program is first enqueued, so that is where a layout that does not fit throws.
    static bool tt_metal_accepts(const std::vector<CbSpec>& cbs) {
        Program program{};
        const CoreCoord core{0, 0};
        for (const auto& cb : cbs) {
            CreateCircularBuffer(
                program,
                core,
                CircularBufferConfig(cb.num_pages * cb.page_bytes, {{cb.index, tt::DataFormat::UInt8}})
                    .set_page_size(cb.index, cb.page_bytes));
        }
        CreateKernel(
            program,
            zkml::common::kernel_path(__FILE__, "kernels/noop.cpp"),
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(dev()->shape()), std::move(program));
        auto& cq = dev()->mesh_command_queue();
        try {
            distributed::EnqueueMeshWorkload(cq, workload, false);
            distributed::Finish(cq);
        } catch (const std::exception&) {
            return false;
        }
        return true;
    }

    static bool check_accepts(const std::vector<CbSpec>& cbs) {
        try {
            check_if_cbs_can_fit_in_l1(dev(), cbs);
        } catch (const std::exception&) {
            return false;
        }
        return true;
    }

    // `fixed` CBs, the filler, then c_31 of `pages` x 16 B. The layout the boundary search varies.
    static std::vector<CbSpec> with_last(std::vector<CbSpec> fixed, uint32_t pages) {
        fixed.push_back(filler_cb);
        fixed.push_back({31, pages, var_page_bytes});
        return fixed;
    }

    // Finds the largest last CB the check accepts and requires tt-metal to agree on both sides:
    // it runs at that size and refuses one page more.
    static void expect_same_boundary_as_tt_metal(const std::vector<CbSpec>& fixed) {
        uint32_t lo = 0;  // accepted (the fixed CBs alone must fit)
        uint32_t hi = max_cb_pages;  // rejected
        ASSERT_TRUE(check_accepts(with_last(fixed, lo)));
        ASSERT_FALSE(check_accepts(with_last(fixed, hi))) << "the filler leaves more than 1 MB free";
        while (hi - lo > 1) {
            const uint32_t mid = lo + (hi - lo) / 2;
            (check_accepts(with_last(fixed, mid)) ? lo : hi) = mid;
        }
        SCOPED_TRACE(fmt::format("largest accepted last CB: {} pages x {} B", lo, var_page_bytes));
        EXPECT_TRUE(tt_metal_accepts(with_last(fixed, lo))) << "the check accepts what tt-metal rejects";
        EXPECT_FALSE(tt_metal_accepts(with_last(fixed, lo + 1))) << "the check rejects what tt-metal accepts";
    }

    static inline std::shared_ptr<distributed::MeshDevice> device_;
};

// ---- the check on its own ---------------------------------------------------------------------

TEST_F(L1CheckTest, AcceptsNoCbs) { EXPECT_NO_THROW(check_if_cbs_can_fit_in_l1(dev(), {})); }

TEST_F(L1CheckTest, AcceptsSmallCbs) {
    const std::vector<CbSpec> cbs = {{0, 8, 2048}, {1, 1, 4096}, {16, 6, 4096}};
    EXPECT_NO_THROW(check_if_cbs_can_fit_in_l1(dev(), cbs));
}

TEST_F(L1CheckTest, RejectsCbLargerThanL1) {
    const std::vector<CbSpec> cbs = {{0, 4000, 4096}};  // 16 MB
    EXPECT_THROW(check_if_cbs_can_fit_in_l1(dev(), cbs), std::exception);
}

TEST_F(L1CheckTest, RejectsCbsThatFitOnlyOneByOne) {
    // Each CB is below the free space, their sum is not.
    const uint64_t free_bytes = dev()->l1_size_per_core() / 2 + var_page_bytes;
    const uint32_t pages = static_cast<uint32_t>(free_bytes / var_page_bytes);
    EXPECT_NO_THROW(check_if_cbs_can_fit_in_l1(dev(), std::vector<CbSpec>{{0, pages, var_page_bytes}}));
    EXPECT_THROW(
        check_if_cbs_can_fit_in_l1(dev(), std::vector<CbSpec>{{0, pages, var_page_bytes}, {1, pages, var_page_bytes}}),
        std::exception);
}

TEST_F(L1CheckTest, ErrorNamesEveryCbWithItsOffsetAndTheHint) {
    const std::vector<CbSpec> cbs = {{0, 2, 2048}, {16, 4000, 4096}};
    try {
        check_if_cbs_can_fit_in_l1(dev(), cbs, "Lower the batch.");
        FAIL() << "expected an exception";
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("c_0: 2 x 2048 B = 4096 B at +0"), std::string::npos) << msg;
        EXPECT_NE(msg.find("c_16: 4000 x 4096 B = 16384000 B at +4096"), std::string::npos) << msg;
        EXPECT_NE(msg.find("Lower the batch."), std::string::npos) << msg;
    }
}

TEST_F(L1CheckTest, ErrorShowsAlignmentPaddingBetweenCbs) {
    // c_0 is 100 B, so c_1 starts at the next multiple of the alignment, not at 100.
    const std::vector<CbSpec> cbs = {{0, 1, 100}, {1, 1, 64}, {2, 4000, 4096}};
    try {
        check_if_cbs_can_fit_in_l1(dev(), cbs);
        FAIL() << "expected an exception";
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        const uint64_t c1_start = align_up(100, alignment());
        const uint64_t c2_start = align_up(c1_start + 64, alignment());
        EXPECT_NE(msg.find(fmt::format("c_1: 1 x 64 B = 64 B at +{}", c1_start)), std::string::npos) << msg;
        EXPECT_NE(msg.find(fmt::format("at +{}", c2_start)), std::string::npos) << msg;
    }
}

// ---- the check against tt-metal ---------------------------------------------------------------

TEST_F(L1CheckTest, MatchesTtMetalOnOneCb) { expect_same_boundary_as_tt_metal({}); }

TEST_F(L1CheckTest, MatchesTtMetalWithAlignedCbsBefore) {
    expect_same_boundary_as_tt_metal({{0, 8, 2048}, {1, 1, 4096}});
}

// A CB size that is not a multiple of the alignment leaves padding before the next CB. If the check
// ignored it, it would accept a layout tt-metal rejects, by up to one alignment unit.
TEST_F(L1CheckTest, MatchesTtMetalWithUnalignedCbsBefore) {
    ASSERT_GT(alignment(), 1u);
    expect_same_boundary_as_tt_metal({{0, 1, 100}});
    expect_same_boundary_as_tt_metal({{0, 3, 50}, {1, 1, 16}});
    expect_same_boundary_as_tt_metal({{0, 1, 4}, {1, 1, 20}, {2, 1, 36}});
}

// An L1-resident buffer takes space from the top of L1, which lowers the limit the CBs must stay below.
TEST_F(L1CheckTest, MatchesTtMetalWithAnL1BufferAllocated) {
    const uint32_t num_banks = dev()->allocator()->get_num_banks(BufferType::L1);
    const uint64_t page_bytes = 200 * 1024;
    auto l1_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = page_bytes * num_banks},
        distributed::DeviceLocalBufferConfig{.page_size = page_bytes, .buffer_type = BufferType::L1},
        dev());

    // The buffer must really have moved the limit; otherwise this test proves nothing new.
    ASSERT_LT(*dev()->lowest_occupied_compute_l1_address(), dev()->l1_size_per_core());
    expect_same_boundary_as_tt_metal({{0, 1, 100}});
}

}  // namespace
