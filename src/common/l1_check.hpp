// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include <fmt/format.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>

namespace zkml::common {

/// One circular buffer of a program: its index (c_<index>), page count and page size.
struct CbSpec {
    uint32_t index;
    uint32_t num_pages;
    uint32_t page_bytes;

    uint64_t bytes() const { return static_cast<uint64_t>(num_pages) * page_bytes; }
};

/**
 * @brief Throws if the circular buffers of a program cannot fit in the L1 of a Tensix core.
 *
 * It repeats the placement tt-metal does when the program is first enqueued
 * (`ProgramImpl::allocate_circular_buffers` + `validate_circular_buffer_region`):
 * - CBs are placed upwards from the allocator base (`get_base_allocator_addr(L1)`), in the order they
 *   are created. Everything below the base (firmware, mailboxes, kernel binaries, runtime args, CB
 *   configs) is already excluded.
 * - Each CB starts at the next multiple of the allocator's DRAM alignment (`get_alignment(DRAM)`,
 *   64 B on Blackhole), so a CB whose size is not a multiple of it leaves padding before the next one.
 * - The last CB must end at or below the lowest live L1 buffer (`lowest_occupied_compute_l1_address`,
 *   which covers L1-resident tensors) and within the L1 size.
 * `create_cb` / `CreateCircularBuffer` check nothing themselves; tt-metal fails at enqueue, with an
 * error that does not say which of the op's parameters to change. This fails early, names every CB,
 * and accepts exactly what tt-metal accepts for a program whose only L1 users are these CBs.
 *
 * Call it from a program factory's `create`, where the op's input and output tensors are already
 * allocated; never from the program hash, which ttnn computes before the outputs exist.
 *
 * @param device Device the program will run on.
 * @param cbs    Every CB the program creates, in the order the program creates them.
 * @param hint   Optional text appended to the error, e.g. the op's parameters that set the sizes.
 */
inline void check_if_cbs_can_fit_in_l1(
    tt::tt_metal::IDevice* device, std::span<const CbSpec> cbs, std::string_view hint = {}) {
    const auto& allocator = device->allocator();
    const uint64_t alignment = allocator->get_alignment(tt::tt_metal::BufferType::DRAM);
    const uint64_t l1_base = allocator->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint64_t l1_limit = std::min<uint64_t>(
        device->lowest_occupied_compute_l1_address().value_or(device->l1_size_per_core()),
        l1_base + allocator->get_bank_size(tt::tt_metal::BufferType::L1));
    const uint64_t l1_free = l1_limit > l1_base ? l1_limit - l1_base : 0;

    uint64_t end = l1_base;  // end of the last placed CB
    std::string breakdown;
    for (const auto& cb : cbs) {
        const uint64_t start = (end + alignment - 1) / alignment * alignment;
        end = start + cb.bytes();
        breakdown += fmt::format(
            "{}c_{}: {} x {} B = {} B at +{}", breakdown.empty() ? "" : ", ", cb.index, cb.num_pages, cb.page_bytes,
            cb.bytes(), start - l1_base);
    }
    const uint64_t needed = end - l1_base;  // includes the alignment padding
    TT_FATAL(
        end <= l1_limit,
        "The circular buffers need {} B of L1 (with {} B alignment) but {} B are free ({}).{}{}",
        needed,
        alignment,
        l1_free,
        breakdown,
        hint.empty() ? "" : " ",
        hint);
}

}  // namespace zkml::common
