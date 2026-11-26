// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// The C ABI (include/tt_zkml/tt_zkml.h) on top of the public C++ API of tt-zkml-core (zkml::Device, zkml::Model, zkml::Trace).
// Every function catches all exceptions: nothing C++ crosses the boundary.

#include "tt_zkml/tt_zkml.h"

#include <cmath>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

#include "api/tt_zkml.hpp"

#ifndef TTZK_VERSION_STRING
#define TTZK_VERSION_STRING "0.1.0"
#endif

struct ttzk_device {
    std::unique_ptr<zkml::Device> device;
};
struct ttzk_model {
    zkml::Model model;
};
struct ttzk_trace {
    zkml::Trace trace;
};

namespace {

thread_local std::string g_last_error;

ttzk_status fail(ttzk_status status, std::string message) {
    g_last_error = std::move(message);
    return status;
}

// Runs `f`, turns its exceptions into a status and a message.
template <typename F>
ttzk_status guarded(F&& f, ttzk_status on_runtime_error = TTZK_ERR_INTERNAL) {
    try {
        g_last_error.clear();
        return f();
    } catch (const std::invalid_argument& e) {
        return fail(TTZK_ERR_INVALID_ARGUMENT, e.what());
    } catch (const std::logic_error& e) {
        return fail(TTZK_ERR_INVALID_ARGUMENT, e.what());
    } catch (const std::exception& e) {
        return fail(on_runtime_error, e.what());
    } catch (...) {
        return fail(TTZK_ERR_INTERNAL, "unknown exception");
    }
}

ttzk_status null_arg(const char* name) { return fail(TTZK_ERR_INVALID_ARGUMENT, std::string(name) + " is null"); }

}  // namespace

extern "C" {

uint32_t ttzk_abi_version(void) { return TTZK_ABI_VERSION; }

const char* ttzk_version(void) { return TTZK_VERSION_STRING; }

const char* ttzk_last_error(void) { return g_last_error.c_str(); }

ttzk_status ttzk_num_devices(uint32_t* out_count) {
    if (out_count == nullptr) {
        return null_arg("out_count");
    }
    return guarded(
        [&] {
            // Counts the PCIe devices without creating the tt-metal context (see the note on threads in the header).
            *out_count = zkml::Device::count();
            return TTZK_OK;
        },
        TTZK_ERR_DEVICE);
}

ttzk_status ttzk_device_open(uint32_t id, ttzk_device** out_device) {
    if (out_device == nullptr) {
        return null_arg("out_device");
    }
    *out_device = nullptr;
    return guarded(
        [&] {
            auto d = std::make_unique<ttzk_device>();
            d->device = std::make_unique<zkml::Device>(id);
            *out_device = d.release();
            return TTZK_OK;
        },
        TTZK_ERR_DEVICE);
}

void ttzk_device_close(ttzk_device* device) {
    try {
        delete device;
    } catch (const std::exception& e) {
        g_last_error = e.what();
    } catch (...) {
        g_last_error = "unknown exception while closing the device";
    }
}

ttzk_status ttzk_model_create(ttzk_device* device, const ttzk_layer* layers, size_t n_layers, ttzk_dtype dtype, ttzk_model** out_model) {
    if (out_model == nullptr) {
        return null_arg("out_model");
    }
    *out_model = nullptr;
    if (device == nullptr) {
        return null_arg("device");
    }
    if (layers == nullptr || n_layers == 0) {
        return fail(TTZK_ERR_INVALID_ARGUMENT, "layers is null or empty");
    }
    if (dtype != TTZK_FLOAT32 && dtype != TTZK_BFLOAT16) {
        return fail(TTZK_ERR_INVALID_ARGUMENT, "unknown dtype");
    }
    return guarded([&] {
        std::vector<zkml::LayerSpec> specs;
        for (size_t l = 0; l < n_layers; l++) {
            specs.push_back({layers[l].in_features, layers[l].out_features, layers[l].relu != 0});
        }
        auto m = std::unique_ptr<ttzk_model>(
            new ttzk_model{zkml::Model(*device->device, std::move(specs), dtype == TTZK_FLOAT32 ? zkml::DataType::Float32 : zkml::DataType::BFloat16)});
        *out_model = m.release();
        return TTZK_OK;
    });
}

void ttzk_model_destroy(ttzk_model* model) { delete model; }

ttzk_status ttzk_model_load_parameters(ttzk_model* model, const float* weights, size_t count) {
    if (model == nullptr) {
        return null_arg("model");
    }
    if (weights == nullptr) {
        return null_arg("weights");
    }
    return guarded([&] {
        size_t expected = 0;
        for (const auto& l : model->model.layers()) {
            expected += static_cast<size_t>(l.in_features) * l.out_features;
        }
        if (count != expected) {
            throw std::invalid_argument("load_parameters: expected " + std::to_string(expected) + " floats, got " + std::to_string(count));
        }
        std::vector<std::vector<float>> per_layer;
        size_t off = 0;
        for (const auto& l : model->model.layers()) {
            const size_t n = static_cast<size_t>(l.in_features) * l.out_features;
            per_layer.emplace_back(weights + off, weights + off + n);
            off += n;
        }
        model->model.load_parameters(per_layer);
        return TTZK_OK;
    });
}

ttzk_status ttzk_model_forward(ttzk_model* model, const float* x, size_t count, uint32_t batch, ttzk_trace** out_trace) {
    if (out_trace == nullptr) {
        return null_arg("out_trace");
    }
    *out_trace = nullptr;
    if (model == nullptr) {
        return null_arg("model");
    }
    if (x == nullptr) {
        return null_arg("x");
    }
    return guarded([&] {
        auto t = std::make_unique<ttzk_trace>();
        t->trace = model->model.forward(std::vector<float>(x, x + count), batch);
        *out_trace = t.release();
        return TTZK_OK;
    });
}

ttzk_status ttzk_trace_get_info(const ttzk_trace* trace, ttzk_trace_info* out_info) {
    if (trace == nullptr) {
        return null_arg("trace");
    }
    if (out_info == nullptr) {
        return null_arg("out_info");
    }
    out_info->batch = trace->trace.batch;
    out_info->in_features = trace->trace.in_features;
    out_info->layers = trace->trace.layers();
    out_info->scaler_x = trace->trace.scaler_x;
    return TTZK_OK;
}

ttzk_status ttzk_trace_get_layer(const ttzk_trace* trace, uint32_t layer, ttzk_trace_layer* out_layer) {
    if (trace == nullptr) {
        return null_arg("trace");
    }
    if (out_layer == nullptr) {
        return null_arg("out_layer");
    }
    const auto& t = trace->trace;
    if (layer >= t.layers()) {
        return fail(TTZK_ERR_INVALID_ARGUMENT, "layer " + std::to_string(layer) + " out of range (" + std::to_string(t.layers()) + " layers)");
    }
    out_layer->out_features = t.widths[layer];
    out_layer->scaler_w = t.scaler_w[layer];
    out_layer->scaler_out = t.scaler_out[layer];
    out_layer->acc = t.acc[layer].data();
    out_layer->acc_len = t.acc[layer].size();
    return TTZK_OK;
}

ttzk_status ttzk_trace_get_output(const ttzk_trace* trace, const uint8_t** out_data, size_t* out_len) {
    if (trace == nullptr) {
        return null_arg("trace");
    }
    if (out_data == nullptr || out_len == nullptr) {
        return null_arg("out_data / out_len");
    }
    *out_data = trace->trace.output.data();
    *out_len = trace->trace.output.size();
    return TTZK_OK;
}

void ttzk_trace_destroy(ttzk_trace* trace) { delete trace; }

ttzk_status ttzk_pot_k(float absmax, uint32_t bits, int32_t* out_k) {
    if (out_k == nullptr) {
        return null_arg("out_k");
    }
    if (bits != 8 && bits != 16 && bits != 32) {
        return fail(TTZK_ERR_INVALID_ARGUMENT, "bits must be 8, 16 or 32");
    }
    if (!std::isfinite(absmax) || absmax < 0.0f) {
        return fail(TTZK_ERR_INVALID_ARGUMENT, "absmax must be finite and not negative");
    }
    return guarded([&] {
        const double q = std::ldexp(1.0, static_cast<int>(bits) - 1) - 1.0;
        int32_t k = 127;  // an all-zero input gets the largest exponent
        while (std::ldexp(static_cast<double>(absmax), k) > q) {
            k--;
        }
        *out_k = k;
        return TTZK_OK;
    });
}

}  // extern "C"
