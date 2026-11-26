// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Plain integer logic that decides how many tiles a compute kernel may handle in one block.
// It has no device calls, so a compute kernel and a host test can both include it.
//
// A compute kernel moves tiles through the DST in blocks of up to `max_block` tiles. A block is
// read from an input CB and written to an output CB. `copy_tile(cb, i, ...)` and `pack_tile(..., cb)`
// address page `pointer + i` without wrapping, so a block must lie inside each ring. When the
// input and the output ring have different sizes, they wrap at different tiles, and the block is
// cut at whichever ring end, or the end of the tensor, comes first.
//
// The kernel keeps every CB call (wait_front, reserve_back, push_back, pop_front) itself, and calls
// commit(n) with the same n. See the "Two CBs of different sizes" section of docs/typst/dataflow.typ.

#pragma once

#include <cstdint>

namespace zkml::common {

/// The position of one CB ring: the index of the next page. It moves like the hardware pointer.
class RingCursor {
public:
    constexpr explicit RingCursor(uint32_t pages) : pages_(pages) {}

    /// Pages before the end of the ring: the longest run that does not wrap.
    constexpr uint32_t room() const { return pages_ - pos_; }
    constexpr uint32_t pos() const { return pos_; }
    constexpr uint32_t pages() const { return pages_; }

    /// Moves forward by `n` pages. `n` never crosses the ring end, see DstBlockMediator::next().
    constexpr void advance(uint32_t n) {
        pos_ += n;
        if (pos_ == pages_) {
            pos_ = 0;
        }
    }

private:
    uint32_t pages_;
    uint32_t pos_ = 0;
};

/// Sizes the blocks of one pass over `num_tiles` tiles, for an input ring and an optional output ring.
/// The mediator advances the cursors it is given, so a cursor can be shared between passes.
class DstBlockMediator {
public:
    /// A pass that only reads (no output ring).
    constexpr DstBlockMediator(uint32_t max_block, uint32_t num_tiles, RingCursor& in) :
        max_block_(max_block), left_(num_tiles), in_(&in) {}

    /// A pass that reads from one ring and writes to another.
    constexpr DstBlockMediator(uint32_t max_block, uint32_t num_tiles, RingCursor& in, RingCursor& out) :
        max_block_(max_block), left_(num_tiles), in_(&in), out_(&out) {}

    /// Tiles in the next block: at most `max_block`, and neither past the end of the tensor nor
    /// across the end of a ring. Returns 0 when the pass is done. It changes nothing.
    constexpr uint32_t next() const {
        uint32_t n = max_block_ < left_ ? max_block_ : left_;
        n = n < in_->room() ? n : in_->room();
        if (out_ != nullptr) {
            n = n < out_->room() ? n : out_->room();
        }
        return n;
    }

    /// Call after the block of `n` tiles (from next()) has been handled and its CB calls are done.
    constexpr void commit(uint32_t n) {
        in_->advance(n);
        if (out_ != nullptr) {
            out_->advance(n);
        }
        left_ -= n;
        done_ += n;
    }

    /// Tiles committed so far.
    constexpr uint32_t done() const { return done_; }

private:
    uint32_t max_block_;
    uint32_t left_;
    uint32_t done_ = 0;
    RingCursor* in_;
    RingCursor* out_ = nullptr;
};

}  // namespace zkml::common
