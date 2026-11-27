//! Safe wrappers over the C ABI of tt-zkml (the `abi` project of the repository).
//!
//! The device context of tt-metal is thread-affine: open the [`Device`], create and run the [`Model`]s and drop them all on ONE thread, before the
//! process exits. The types here are therefore `!Send` and `!Sync`, and a [`Model`] borrows its device so it cannot outlive it.

use std::ffi::CStr;
use std::fmt;
use std::marker::PhantomData;
use std::os::raw::c_char;
use std::ptr;

use tt_zkml_sys as sys;

/// A call into the library failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub status: i32,
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (status {})", self.message, self.status)
    }
}

impl std::error::Error for Error {}

type Result<T> = std::result::Result<T, Error>;

fn cstr(p: *const c_char) -> String {
    // SAFETY: the library returns NUL-terminated strings; they are copied at once.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

fn check(status: sys::ttzk_status) -> Result<()> {
    if status == sys::TTZK_OK {
        Ok(())
    } else {
        // SAFETY: no arguments; the string is valid until the next call on this thread.
        Err(Error { status, message: cstr(unsafe { sys::ttzk_last_error() }) })
    }
}

/// The ABI version of the loaded library.
pub fn abi_version() -> u32 {
    // SAFETY: no arguments.
    unsafe { sys::ttzk_abi_version() }
}

/// Fails if the library was built against another ABI than these bindings.
pub fn check_abi() -> Result<()> {
    let found = abi_version();
    if found == sys::TTZK_ABI_VERSION {
        Ok(())
    } else {
        Err(Error { status: sys::TTZK_ERR_INTERNAL, message: format!("ABI mismatch: the library has {found}, these bindings expect {}", sys::TTZK_ABI_VERSION) })
    }
}

/// The version of the library ("major.minor.patch").
pub fn version() -> String {
    // SAFETY: a static NUL-terminated string.
    cstr(unsafe { sys::ttzk_version() })
}

/// Number of Tenstorrent devices on the PCIe bus (does not open one).
pub fn num_devices() -> Result<u32> {
    let mut n = 0u32;
    // SAFETY: `n` is a valid out pointer.
    check(unsafe { sys::ttzk_num_devices(&mut n) })?;
    Ok(n)
}

/// The power-of-two quantization exponent: the largest `k` with `absmax * 2^k <= 2^(bits - 1) - 1` (at most 127).
pub fn pot_k(absmax: f32, bits: u32) -> Result<i32> {
    let mut k = 0i32;
    // SAFETY: `k` is a valid out pointer.
    check(unsafe { sys::ttzk_pot_k(absmax, bits, &mut k) })?;
    Ok(k)
}

/// The type of the floats on the device (the arithmetic is int8 either way).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Dtype {
    F32,
    Bf16,
}

impl Dtype {
    fn raw(self) -> sys::ttzk_dtype {
        match self {
            Dtype::F32 => sys::TTZK_FLOAT32,
            Dtype::Bf16 => sys::TTZK_BFLOAT16,
        }
    }
}

/// A fully connected layer `y = x @ W`, `W` of shape `[in_features, out_features]`, with an optional ReLU on the accumulator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Layer {
    pub in_features: u32,
    pub out_features: u32,
    pub relu: bool,
}

/// An open device.
pub struct Device {
    raw: *mut sys::ttzk_device,
    _not_send: PhantomData<*const ()>,
}

impl Device {
    /// Opens device `id` (0 for the first).
    pub fn open(id: u32) -> Result<Device> {
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` is a valid out pointer.
        check(unsafe { sys::ttzk_device_open(id, &mut raw) })?;
        Ok(Device { raw, _not_send: PhantomData })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: `raw` came from ttzk_device_open and is closed once; its models borrow it, so they are gone.
        unsafe { sys::ttzk_device_close(self.raw) }
    }
}

/// A chain of quantized layers on a device.
pub struct Model<'d> {
    raw: *mut sys::ttzk_model,
    layers: Vec<Layer>,
    _device: PhantomData<&'d Device>,
}

impl<'d> Model<'d> {
    /// The layers must chain (`in_features` of a layer is `out_features` of the previous one); all sizes are multiples of 32.
    pub fn new(device: &'d Device, layers: &[Layer], dtype: Dtype) -> Result<Model<'d>> {
        let raw_layers: Vec<sys::ttzk_layer> =
            layers.iter().map(|l| sys::ttzk_layer { in_features: l.in_features, out_features: l.out_features, relu: l.relu as u32 }).collect();
        let mut raw = ptr::null_mut();
        // SAFETY: the pointers are valid for the call; the library copies the layer descriptions.
        check(unsafe { sys::ttzk_model_create(device.raw, raw_layers.as_ptr(), raw_layers.len(), dtype.raw(), &mut raw) })?;
        Ok(Model { raw, layers: layers.to_vec(), _device: PhantomData })
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Copies the weights to the device: `weights[l]` is the `W` of layer `l`, `[in_features, out_features]` row-major.
    pub fn load_parameters(&mut self, weights: &[Vec<f32>]) -> Result<()> {
        let flat: Vec<f32> = weights.iter().flatten().copied().collect();
        // SAFETY: `flat` is valid for `flat.len()` floats; the library copies it.
        check(unsafe { sys::ttzk_model_load_parameters(self.raw, flat.as_ptr(), flat.len()) })
    }

    /// Runs the forward pass on `x` (`[batch, in_features]` row-major, `batch` a multiple of 32) and returns the execution trace.
    pub fn forward(&self, x: &[f32], batch: u32) -> Result<Trace> {
        let mut t = ptr::null_mut();
        // SAFETY: `x` is valid for `x.len()` floats; `t` is a valid out pointer.
        check(unsafe { sys::ttzk_model_forward(self.raw, x.as_ptr(), x.len(), batch, &mut t) })?;
        let trace = read_trace(t);
        // SAFETY: the trace was created above and is released once, after it was copied.
        unsafe { sys::ttzk_trace_destroy(t) };
        trace
    }
}

impl Drop for Model<'_> {
    fn drop(&mut self) {
        // SAFETY: `raw` came from ttzk_model_create and is destroyed once.
        unsafe { sys::ttzk_model_destroy(self.raw) }
    }
}

/// The values of one layer of a [`Trace`].
#[derive(Clone, Debug, PartialEq)]
pub struct LayerTrace {
    pub out_features: u32,
    /// `2^kw`, the quantization scaler of the weights.
    pub scaler_w: f32,
    /// `2^k` of the requantization; its output is the quantized input of the next layer.
    pub scaler_out: f32,
    /// `[batch, out_features]` row-major: the raw sign-magnitude bits of the int32 accumulator (bit 31 = sign).
    pub acc: Vec<u32>,
}

impl LayerTrace {
    /// The accumulator as signed integers.
    pub fn acc_values(&self) -> Vec<i32> {
        self.acc.iter().map(|&a| if a >> 31 != 0 { -((a & 0x7FFF_FFFF) as i32) } else { a as i32 }).collect()
    }
}

/// The execution trace of one forward pass, copied out of the library.
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    pub batch: u32,
    pub in_features: u32,
    /// `2^kx`, the quantization scaler of the input.
    pub scaler_x: f32,
    pub layers: Vec<LayerTrace>,
    /// `[batch, out_features of the last layer]`: sign-magnitude int8 bytes (bit 7 = sign).
    pub output: Vec<u8>,
}

impl Trace {
    /// The output as signed integers.
    pub fn output_values(&self) -> Vec<i32> {
        self.output.iter().map(|&b| if b & 0x80 != 0 { -((b & 0x7F) as i32) } else { b as i32 }).collect()
    }

    /// The output as floats, approximately what the unquantized network computes: the int8 output divided by the product of all the scalers
    /// (`scaler_x`, and `scaler_w` and `scaler_out` of every layer; all powers of two, so the scaling is exact). The quantization error stays in it.
    pub fn dequantized_output(&self) -> Vec<f32> {
        let scale: f64 = self.layers.iter().fold(self.scaler_x as f64, |s, l| s * l.scaler_w as f64 * l.scaler_out as f64);
        self.output_values().iter().map(|&v| (v as f64 / scale) as f32).collect()
    }
}

fn read_trace(t: *const sys::ttzk_trace) -> Result<Trace> {
    let mut info = sys::ttzk_trace_info { batch: 0, in_features: 0, layers: 0, scaler_x: 0.0 };
    // SAFETY: `t` is a live trace, `info` a valid out pointer.
    check(unsafe { sys::ttzk_trace_get_info(t, &mut info) })?;
    let mut layers = Vec::new();
    for l in 0..info.layers {
        let mut v = sys::ttzk_trace_layer { out_features: 0, scaler_w: 0.0, scaler_out: 0.0, acc: ptr::null(), acc_len: 0 };
        // SAFETY: as above.
        check(unsafe { sys::ttzk_trace_get_layer(t, l, &mut v) })?;
        // SAFETY: `acc` points to `acc_len` values that live as long as the trace; they are copied.
        let acc = unsafe { std::slice::from_raw_parts(v.acc, v.acc_len) }.to_vec();
        layers.push(LayerTrace { out_features: v.out_features, scaler_w: v.scaler_w, scaler_out: v.scaler_out, acc });
    }
    let mut data = ptr::null();
    let mut len = 0usize;
    // SAFETY: as above.
    check(unsafe { sys::ttzk_trace_get_output(t, &mut data, &mut len) })?;
    // SAFETY: `data` points to `len` bytes that live as long as the trace; they are copied.
    let output = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
    Ok(Trace { batch: info.batch, in_features: info.in_features, scaler_x: info.scaler_x, layers, output })
}
