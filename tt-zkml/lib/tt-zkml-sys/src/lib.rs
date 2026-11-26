//! Raw bindings to the C ABI of tt-zkml (`abi/include/tt_zkml/tt_zkml.h`).
//!
//! Hand-written for now (the surface is small); generate them with `bindgen` once the header grows. Keep `TTZK_ABI_VERSION` in step with the
//! header: `tt-zkml` checks it at start-up.
#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_void};

/// The ABI version these declarations were written against.
pub const TTZK_ABI_VERSION: u32 = 1;

pub type ttzk_status = i32;
pub const TTZK_OK: ttzk_status = 0;
pub const TTZK_ERR_INVALID_ARGUMENT: ttzk_status = 1;
pub const TTZK_ERR_DEVICE: ttzk_status = 2;
pub const TTZK_ERR_INTERNAL: ttzk_status = 3;

pub type ttzk_dtype = i32;
pub const TTZK_FLOAT32: ttzk_dtype = 0;
pub const TTZK_BFLOAT16: ttzk_dtype = 1;

/// Opaque handles.
#[repr(C)]
pub struct ttzk_device {
    _private: [u8; 0],
}
#[repr(C)]
pub struct ttzk_model {
    _private: [u8; 0],
}
#[repr(C)]
pub struct ttzk_trace {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ttzk_layer {
    pub in_features: u32,
    pub out_features: u32,
    pub relu: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ttzk_trace_info {
    pub batch: u32,
    pub in_features: u32,
    pub layers: u32,
    pub scaler_x: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ttzk_trace_layer {
    pub out_features: u32,
    pub scaler_w: f32,
    pub scaler_out: f32,
    pub acc: *const u32,
    pub acc_len: usize,
}

extern "C" {
    pub fn ttzk_abi_version() -> u32;
    pub fn ttzk_version() -> *const c_char;
    pub fn ttzk_last_error() -> *const c_char;
    pub fn ttzk_num_devices(out_count: *mut u32) -> ttzk_status;
    pub fn ttzk_pot_k(absmax: f32, bits: u32, out_k: *mut i32) -> ttzk_status;

    pub fn ttzk_device_open(id: u32, out_device: *mut *mut ttzk_device) -> ttzk_status;
    pub fn ttzk_device_close(device: *mut ttzk_device);
    pub fn ttzk_model_create(
        device: *mut ttzk_device,
        layers: *const ttzk_layer,
        n_layers: usize,
        dtype: ttzk_dtype,
        out_model: *mut *mut ttzk_model,
    ) -> ttzk_status;
    pub fn ttzk_model_destroy(model: *mut ttzk_model);
    pub fn ttzk_model_load_parameters(model: *mut ttzk_model, weights: *const f32, count: usize) -> ttzk_status;
    pub fn ttzk_model_forward(
        model: *mut ttzk_model,
        x: *const f32,
        count: usize,
        batch: u32,
        out_trace: *mut *mut ttzk_trace,
    ) -> ttzk_status;
    pub fn ttzk_trace_get_info(trace: *const ttzk_trace, out_info: *mut ttzk_trace_info) -> ttzk_status;
    pub fn ttzk_trace_get_layer(trace: *const ttzk_trace, layer: u32, out_layer: *mut ttzk_trace_layer) -> ttzk_status;
    pub fn ttzk_trace_get_output(trace: *const ttzk_trace, out_data: *mut *const u8, out_len: *mut usize) -> ttzk_status;
    pub fn ttzk_trace_destroy(trace: *mut ttzk_trace);
}

// keep the unused import quiet when the declarations above change
const _: Option<*const c_void> = None;
