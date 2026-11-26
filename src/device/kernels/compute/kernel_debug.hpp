// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// Debug helpers for device kernels (compiled by tt-metal at runtime).

#pragma once

#include <cstdint>

#include "api/debug/dprint.h"
#if defined(COMPILE_FOR_TRISC) && defined(DEBUG_PRINT_ENABLED) && !defined(FORCE_DPRINT_OFF)
#include "api/debug/dprint_tensix.h"
#endif

namespace zkml::debug {

#if defined(COMPILE_FOR_TRISC) && defined(DEBUG_PRINT_ENABLED) && !defined(FORCE_DPRINT_OFF)

// clang-format off
/**
 * @brief Prints rows `[row_begin, row_end)` of a tile in a CB in logical (row-major) order, with all
 *        32 columns of each row on one line.
 *
 * tt-metal's `print_full_tile` cuts every row of a 4-byte format (Float32, Int32) after 16 values,
 * because one `TileSlice` holds at most 64 bytes, and by default prints rows in memory (face) order.
 * This helper prints each row as two untilized 16-column slices instead, so every format shows all 32
 * values per row:
 *
 * | Thread | CB pointer read                    | Use for          |
 * |--------|------------------------------------|------------------|
 * | UNPACK | read pointer (front of the CB)     | input CBs        |
 * | PACK   | write pointer (back of the CB)     | output CBs       |
 * | MATH   | none: `TileSlice` reports an error | do not call      |
 *
 * Wrap the call in `UNPACK(...)` or `PACK(...)` so only that thread prints.
 *
 * **Examples** 
 * 
 * ```
 * UNPACK(print_tile_rows(cb_in_id, 0));       // prints all 32 rows (0-31)
 * UNPACK(print_tile_rows(cb_in_id, 0, 0, 4)); // prints 4 rows (0,1,2,3)
 * UNPACK(print_tile_rows(cb_in_id, 0, 16, 32)); // prints 16 rows (16-31)
 * UNPACK(print_tile_rows(cb_in_id, 0, 5, 6));   // prints only row 5
 * ```
 *
 * @param cb_id     CB index (`tt::CBIndex::c_*`), not the `CircularBuffer` object.
 * @param tile_id   Tile offset from the CB pointer the thread reads (0 = the current tile).
 * @param row_begin First row to print, in `[0, 32)`.
 * @param row_end   One past the last row to print, in `(row_begin, 32]`.
 *
 * @pre  UNPACK: the tile has arrived (`wait_front` returned). 
 *       PACK: `pack_tile` has been called and the tile not yet pushed (`push_back`).
 * @note Compiles to an empty function unless DPRINT is enabled (`TT_METAL_DPRINT_CORES` set), so calls
 *       do not need `#if defined(DEBUG_PRINT_ENABLED)` guards.
 * @note On PACK the helper first calls `ckernel::tensix_sync()`. `pack_tile` only issues packer
 *       instructions, so without the wait the print can show the CB slot's previous (stale) contents.
 * @note Two `DPRINT` calls per row are deliberate: one call with both slices (about 80 bytes each)
 *       comes close to the 204-byte per-thread DPRINT buffer.
 * @note No range check: `row_end` above 32 reads past the tile.
 * @see  https://github.com/tenstorrent/tt-metal/blob/v0.81.0/tt_metal/hw/inc/api/debug/dprint_pages.h#L124
 */
// clang-format on
inline void print_tile_rows(uint32_t cb_id, uint32_t tile_id, uint8_t row_begin = 0, uint8_t row_end = 32) {
    // pack_tile() only issues packer instructions; wait until the packer has written L1, or the
    // print shows stale data. No effect on the other threads.
    PACK(ckernel::tensix_sync());
    constexpr uint8_t half_width = 16;
    for (uint8_t r = row_begin; r < row_end; ++r) {
        const uint8_t r_next = static_cast<uint8_t>(r + 1);
        DPRINT(
            "{}: {}",
            r,
            TileSlice(
                cb_id,
                tile_id,
                SliceRange{.h0 = r, .h1 = r_next, .hs = 1, .w0 = 0, .w1 = half_width, .ws = 1},
                /*endl_rows=*/false,
                /*print_untilized=*/true));
        DPRINT(
            " {}\n",
            TileSlice(
                cb_id,
                tile_id,
                SliceRange{.h0 = r, .h1 = r_next, .hs = 1, .w0 = half_width, .w1 = 2 * half_width, .ws = 1},
                /*endl_rows=*/false,
                /*print_untilized=*/true));
    }
}

namespace detail {

// Prints one logical tile row from a 32-bit DST (one value per 32-bit word), 16 words per face row.
// Blackhole only: reads DST through the direct DST window, as tt-metal's Blackhole printers do.
inline void print_dest_row_32bit(uint8_t r, uint32_t left_row, DataFormat format) {
#if defined(ARCH_BLACKHOLE)
    constexpr uint32_t face_width = 16;
    const volatile uint32_t* dest = reinterpret_cast<const volatile uint32_t*>(0xFFBD8000);
    const uint32_t right_row = left_row + NUM_ROWS_PER_FACE;
    uint32_t left[face_width + 1];  // +1: room for the array type, as in tt-metal's printers
    uint32_t right[face_width + 1];
    for (uint32_t i = 0; i < face_width; ++i) {
        left[i] = dest[left_row * face_width + i];
        right[i] = dest[right_row * face_width + i];
    }
    DPRINT("{}: {} ", r, dp_typed_array_t<face_width>(static_cast<uint16_t>(format), left));
    DPRINT("{}\n", dp_typed_array_t<face_width>(static_cast<uint16_t>(format), right));
#else
    DPRINT("print_dest_tile_rows: 32-bit DST is only supported on Blackhole\n");
#endif
}

// Prints one logical tile row from a 16-bit DST: each face row is 16 values packed into 8 words, read
// through the debug bus with tt-metal's address remapping (get_dest_row_id), as dprint_tensix_dest_reg does.
inline void print_dest_row_16bit(uint8_t r, uint32_t left_row, DataFormat format) {
    constexpr uint32_t words_per_face_row = 8;
    uint32_t left[words_per_face_row + 1];  // +1: room for the array type, as in tt-metal's printers
    uint32_t right[words_per_face_row + 1];
    dbg_read_dest_acc_row(get_dest_row_id(left_row, /*is_float32=*/false), left);
    dbg_read_dest_acc_row(get_dest_row_id(left_row + NUM_ROWS_PER_FACE, /*is_float32=*/false), right);
    DPRINT("{}: {} ", r, dp_typed_array_t<words_per_face_row>(static_cast<uint16_t>(format), left));
    DPRINT("{}\n", dp_typed_array_t<words_per_face_row>(static_cast<uint16_t>(format), right));
}

}  // namespace detail

// clang-format off
/**
 * @brief Prints rows `[row_begin, row_end)` of a tile in the DST register in logical (row-major)
 *        order, all 32 columns per row, printing the values as `format`.
 *
 * tt-metal's `dprint_tensix_dest_reg` takes the number format from the hardware config. With a
 * 32-bit DST (`fp32_dest_acc_en`) that is always Float32, so integer results show up as `-nan`
 * (negative values) and tiny denormals like `6e-45` (the integer 4). It also prints face by face and
 * always reads the first DST half. This helper prints the same DST contents in the format you pass,
 * in logical rows, from the DST half in use.
 *
 * The format must match how DST is configured (`fp32_dest_acc_en` in the program's `ComputeConfig`):
 *
 * | DST mode                              | Formats                                           | Supported on        |
 * |---------------------------------------|---------------------------------------------------|---------------------|
 * | 32-bit (`fp32_dest_acc_en = true`)    | `Float32`, `Int32`, `UInt32`                      | Blackhole           |
 * | 16-bit (`fp32_dest_acc_en = false`)   | `Float16_b`, `Float16`, `UInt16`, `Int8`, `UInt8` | Wormhole, Blackhole |
 *
 * Any other combination prints an error line instead of values.
 *
 * DST is split into two halves that `tile_regs_acquire()` alternates between. The function reads the half
 * the math thread is currently using (`ckernel::dest_offset_id`), so `dst_tile` is the same index the
 * compute API uses.
 *
 * `row_begin` and `row_end` select the rows of the DST tile to print, as a half-open range:
 * they are **logical** rows 0-31 of the tile, not DST register rows. DST stores a tile as
 * four 16x16 faces (64 register rows of 16 values), so the helper assembles each
 * logical row `r` from two register rows: row `r % 16` of face `2 * (r / 16)` (columns 0-15) and the
 * same row of the next face (columns 16-31):
 *
 * | Logical row `r` | Left half (cols 0-15)     | Right half (cols 16-31)   |
 * |-----------------|---------------------------|---------------------------|
 * | 0               | face 0, row 0             | face 1, row 0             |
 * | 5               | face 0, row 5             | face 1, row 5             |
 * | 16              | face 2, row 0             | face 3, row 0             |
 * | 31              | face 2, row 15            | face 3, row 15            |
 *
 * **Examples**
 *
 * ```
 * print_dest_tile_rows(dst_x, DataFormat::Int32);               // 32-bit DST: all 32 rows (0-31) as Int32
 * print_dest_tile_rows(dst_x, DataFormat::Int32, 0, 4);         // 32-bit DST: 4 rows (0,1,2,3) as Int32
 * print_dest_tile_rows(dst_scale, DataFormat::Float32, 16, 32); // 32-bit DST: 16 rows (16-31) as Float32
 * print_dest_tile_rows(0, DataFormat::Float16_b, 5, 6);         // 16-bit DST: only row 5 as bfloat16
 * ```
 *
 * Call it between `tile_regs_acquire()` and `tile_regs_commit()`, **without** an `UNPACK`/`MATH`/`PACK` wrapper:
 * it pauses unpack and pack (`dbg_halt`) while the math thread reads DST.
 *
 * @param dst_tile  DST tile index.
 * @param format    How to print the values, see the table above.
 * @param row_begin First logical row of the tile to print, in `[0, 32)`.
 * @param row_end   One past the last logical row to print, in `(row_begin, 32]`. No range check: values
 *                  above 32 read the next DST tile.
 *
 * @note Compiles to an empty function unless DPRINT is enabled.
 * @see  https://github.com/tenstorrent/tt-metal/blob/v0.81.0/tt_metal/hw/inc/api/debug/dprint_tensix.h
 */
// clang-format on
inline void print_dest_tile_rows(uint32_t dst_tile, DataFormat format, uint8_t row_begin = 0, uint8_t row_end = 32) {
    dbg_halt();
    MATH({
        const bool dest_32bit = READ_HW_CFG_0_REG_FIELD(ALU_ACC_CTRL_Fp32_enabled);
        const bool format_32bit =
            format == DataFormat::Float32 || format == DataFormat::Int32 || format == DataFormat::UInt32;
        const bool format_16bit = format == DataFormat::Float16_b || format == DataFormat::Float16 ||
                                  format == DataFormat::UInt16 || format == DataFormat::Int8 ||
                                  format == DataFormat::UInt8;
        if (dest_32bit ? !format_32bit : !format_16bit) {
            // Reading DST at the wrong width prints plausible-looking garbage, so refuse instead.
            DPRINT(
                "print_dest_tile_rows: format {} does not match the {}-bit DST\n",
                static_cast<uint32_t>(format),
                dest_32bit ? 32 : 16);
        } else {
            // tile_regs_acquire() alternates between the two DST halves. The direct 32-bit window sees the
            // whole register, so start at the half math is using. Debug-bus reads (16-bit path) are already
            // relative to that half (checked on the device), so they need no offset.
            const uint32_t half_base = dest_32bit && ckernel::dest_offset_id != 0 ? BIT32_DEST_REGISTER_HALF_SIZE : 0;
            DPRINT("DST tile {}\n", dst_tile);
            for (uint8_t r = row_begin; r < row_end; ++r) {
                // Register row of the left face (columns 0-15); the right face follows NUM_ROWS_PER_FACE later.
                const uint32_t left_row = half_base + dst_tile * NUM_ROWS_PER_TILE +
                                          (r / NUM_ROWS_PER_FACE) * 2 * NUM_ROWS_PER_FACE + r % NUM_ROWS_PER_FACE;
                if (dest_32bit) {
                    detail::print_dest_row_32bit(r, left_row, format);
                } else {
                    detail::print_dest_row_16bit(r, left_row, format);
                }
            }
        }
    })
    dbg_unhalt();
}

#else
// No-op versions, used for non-compute kernels or when DPRINT is disabled.

inline void print_tile_rows(uint32_t, uint32_t, uint8_t = 0, uint8_t = 32) {}

// The format is a template parameter only because tt-metal's DataFormat may not be declared here
// (it comes with the compute headers); any argument, such as DataFormat::Int32, is accepted and ignored.
template <typename Format>
inline void print_dest_tile_rows(uint32_t, Format, uint8_t = 0, uint8_t = 32) {}

#endif

}  // namespace zkml::debug
