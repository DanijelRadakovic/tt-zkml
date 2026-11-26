/* SPDX-FileCopyrightText: © 2026 Danijel Radakovic
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @file tt_zkml.h
 * @brief The C ABI of tt-zkml-core: the interface the Rust crate (tt-zkml-sys) and the Python package bind to.
 *
 * Rules of this interface:
 *  - plain C types only: no C++ types, no exceptions cross it; every function returns a `ttzk_status`, and `ttzk_last_error()` describes
 *    the last failure of the calling thread;
 *  - objects are opaque handles with explicit create / destroy functions (none yet);
 *  - `TTZK_ABI_VERSION` changes when an existing function changes; new functions do not change it.
 *
 * Objects: a `ttzk_device` (an open device), a `ttzk_model` (a chain of quantized layers on a device, with its parameters) and a `ttzk_trace`
 * (the result of a forward pass copied to the host). Create with `ttzk_*_open/create/forward`, release with the matching `*_close/destroy`.
 *
 * Threads: tt-metal's device context is thread-affine. Open a device, create models on it, run them and destroy them all on ONE thread, and
 * destroy them before the process exits (a trace is plain host memory and may be used anywhere).
 */
#ifndef TT_ZKML_H
#define TT_ZKML_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define TTZK_API __declspec(dllexport)
#else
#define TTZK_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Version of this ABI (not of the library). */
#define TTZK_ABI_VERSION 1u

typedef enum ttzk_status {
    TTZK_OK = 0,
    TTZK_ERR_INVALID_ARGUMENT = 1, /**< a pointer is null or a value is out of range */
    TTZK_ERR_DEVICE = 2,           /**< the Tenstorrent runtime failed or no device is available */
    TTZK_ERR_INTERNAL = 3          /**< any other exception inside the library */
} ttzk_status;

/** The ABI version the library was built with: compare with `TTZK_ABI_VERSION` of the header you compiled against. */
TTZK_API uint32_t ttzk_abi_version(void);

/** The library version, "major.minor.patch". The string is static. */
TTZK_API const char* ttzk_version(void);

/** A description of the last error of the calling thread ("" if none). The string stays valid until the next call on this thread. */
TTZK_API const char* ttzk_last_error(void);

/**
 * Number of Tenstorrent devices on the PCIe bus. It does not open a device.
 *
 * Note for the functions that will: tt-metal's device context is thread-affine (its chip lock is a mutex owned by the thread that created the
 * context, and it is released when the context is destroyed). A device handle must therefore be created, used and destroyed on one thread, and
 * destroyed before the process exits.
 */
TTZK_API ttzk_status ttzk_num_devices(uint32_t* out_count);

typedef struct ttzk_device ttzk_device;
typedef struct ttzk_model ttzk_model;
typedef struct ttzk_trace ttzk_trace;

typedef enum ttzk_dtype {
    TTZK_FLOAT32 = 0,
    TTZK_BFLOAT16 = 1 /**< the input and the weights are stored as bf16 on the device (rounded to nearest even) */
} ttzk_dtype;

/** A fully connected layer `y = x @ W` with `W` of shape `[in_features, out_features]`. Sizes are multiples of 32. */
typedef struct ttzk_layer {
    uint32_t in_features;
    uint32_t out_features;
    uint32_t relu; /**< 1: ReLU on the accumulator (before the requantization), 0: none */
} ttzk_layer;

/** Opens device `id` (0 for the first). */
TTZK_API ttzk_status ttzk_device_open(uint32_t id, ttzk_device** out_device);
/** Closes the device; `NULL` is allowed. Destroy the models of the device first. */
TTZK_API void ttzk_device_close(ttzk_device* device);

/** A model of `n_layers` layers; they must chain (`in_features` = `out_features` of the previous). Parameters are loaded separately. */
TTZK_API ttzk_status ttzk_model_create(ttzk_device* device, const ttzk_layer* layers, size_t n_layers, ttzk_dtype dtype, ttzk_model** out_model);
/** `NULL` is allowed. */
TTZK_API void ttzk_model_destroy(ttzk_model* model);

/**
 * Copies the weights to the device: the `W` of every layer, row-major `[in_features, out_features]`, one after the other in layer order.
 * `count` is the number of floats and must be the sum of `in_features * out_features`.
 */
TTZK_API ttzk_status ttzk_model_load_parameters(ttzk_model* model, const float* weights, size_t count);

/**
 * Runs the forward pass on `x` (`[batch, in_features]` row-major, `count = batch * in_features` floats, `batch` a multiple of 32) and returns
 * the execution trace. The device runs the whole network as one program.
 */
TTZK_API ttzk_status ttzk_model_forward(ttzk_model* model, const float* x, size_t count, uint32_t batch, ttzk_trace** out_trace);

/** What a trace is about. */
typedef struct ttzk_trace_info {
    uint32_t batch;
    uint32_t in_features;
    uint32_t layers;
    float scaler_x; /**< 2^kx, the quantization scaler of the input */
} ttzk_trace_info;

/** The values of one layer of a trace. The pointer stays valid until the trace is destroyed. */
typedef struct ttzk_trace_layer {
    uint32_t out_features;
    float scaler_w;           /**< 2^kw, the quantization scaler of the weights */
    float scaler_out;         /**< 2^k of the requantization; its output is the quantized input of the next layer */
    const uint32_t* acc;      /**< `[batch, out_features]` row-major: the raw sign-magnitude bits of the int32 accumulator (bit 31 = sign) */
    size_t acc_len;
} ttzk_trace_layer;

TTZK_API ttzk_status ttzk_trace_get_info(const ttzk_trace* trace, ttzk_trace_info* out_info);
TTZK_API ttzk_status ttzk_trace_get_layer(const ttzk_trace* trace, uint32_t layer, ttzk_trace_layer* out_layer);
/** The output of the last layer: `[batch, out_features of the last layer]` sign-magnitude int8 bytes (bit 7 = sign). */
TTZK_API ttzk_status ttzk_trace_get_output(const ttzk_trace* trace, const uint8_t** out_data, size_t* out_len);
/** `NULL` is allowed. */
TTZK_API void ttzk_trace_destroy(ttzk_trace* trace);

/**
 * The power-of-two quantization exponent: the largest `k` with `absmax * 2^k <= Q`, `Q = 2^(bits - 1) - 1`, at most 127
 * (the same definition the device computes and the host reference uses). `bits` is 8, 16 or 32.
 */
TTZK_API ttzk_status ttzk_pot_k(float absmax, uint32_t bits, int32_t* out_k);

#ifdef __cplusplus
}
#endif

#endif /* TT_ZKML_H */
