# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""Python bindings of tt-zkml: a thin layer over the C ABI (abi/include/tt_zkml/tt_zkml.h) with ctypes.

    import tt_zkml
    with tt_zkml.Device(0) as dev:
        model = tt_zkml.Model(dev, [tt_zkml.Layer(64, 128, relu=True), tt_zkml.Layer(128, 32)], dtype="bf16")
        model.load_parameters([w0, w1])           # flat sequences of floats, [in, out] row-major
        trace = model.forward(x, batch=32)         # x: [batch, in] row-major
        trace.output, trace.acc(0), trace.scaler_x ...

The shared library is found through the environment variable TT_ZKML_ABI_LIB (the file) or, in a development checkout, in abi/.build/default/lib. Its
dependencies (the tt-metal libraries) must be loadable: put <TT_METAL_HOME>/build/lib on LD_LIBRARY_PATH if the library has no RPATH to it.

Threads: the device context of tt-metal is thread-affine: open the Device, create and run the Models and close everything on ONE thread, before the
process exits (use the `with` blocks).
"""
import ctypes
import os
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ABI_VERSION = 1

TTZK_OK, TTZK_ERR_INVALID_ARGUMENT, TTZK_ERR_DEVICE, TTZK_ERR_INTERNAL = 0, 1, 2, 3
_DTYPES = {"f32": 0, "float32": 0, "bf16": 1, "bfloat16": 1}


class TtZkmlError(RuntimeError):
    """A call into the library failed; the message is `ttzk_last_error()`."""

    def __init__(self, status, message):
        super().__init__(f"{message} (status {status})")
        self.status = status


class _Layer(ctypes.Structure):
    _fields_ = [("in_features", ctypes.c_uint32), ("out_features", ctypes.c_uint32), ("relu", ctypes.c_uint32)]


class _TraceInfo(ctypes.Structure):
    _fields_ = [("batch", ctypes.c_uint32), ("in_features", ctypes.c_uint32), ("layers", ctypes.c_uint32), ("scaler_x", ctypes.c_float)]


class _TraceLayer(ctypes.Structure):
    _fields_ = [
        ("out_features", ctypes.c_uint32),
        ("scaler_w", ctypes.c_float),
        ("scaler_out", ctypes.c_float),
        ("acc", ctypes.POINTER(ctypes.c_uint32)),
        ("acc_len", ctypes.c_size_t),
    ]


def _find_library():
    env = os.environ.get("TT_ZKML_ABI_LIB")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    # <repo>/abi/python/tt_zkml/__init__.py: the abi project's own build, else the repository root build
    for base in (here.parents[2],):
        p = base / ".build" / "default" / "lib" / "libtt_zkml_abi.so"
        if p.exists():
            return p
    raise FileNotFoundError("libtt_zkml_abi.so not found: build abi/ (cmake --preset default && cmake --build .build/default) or set TT_ZKML_ABI_LIB")


def _load():
    lib = ctypes.CDLL(str(_find_library()))
    P = ctypes.POINTER
    lib.ttzk_abi_version.restype = ctypes.c_uint32
    lib.ttzk_version.restype = ctypes.c_char_p
    lib.ttzk_last_error.restype = ctypes.c_char_p
    lib.ttzk_num_devices.argtypes = [P(ctypes.c_uint32)]
    lib.ttzk_pot_k.argtypes = [ctypes.c_float, ctypes.c_uint32, P(ctypes.c_int32)]
    lib.ttzk_device_open.argtypes = [ctypes.c_uint32, P(ctypes.c_void_p)]
    lib.ttzk_device_close.argtypes = [ctypes.c_void_p]
    lib.ttzk_device_close.restype = None
    lib.ttzk_model_create.argtypes = [ctypes.c_void_p, P(_Layer), ctypes.c_size_t, ctypes.c_int, P(ctypes.c_void_p)]
    lib.ttzk_model_destroy.argtypes = [ctypes.c_void_p]
    lib.ttzk_model_destroy.restype = None
    lib.ttzk_model_load_parameters.argtypes = [ctypes.c_void_p, P(ctypes.c_float), ctypes.c_size_t]
    lib.ttzk_model_forward.argtypes = [ctypes.c_void_p, P(ctypes.c_float), ctypes.c_size_t, ctypes.c_uint32, P(ctypes.c_void_p)]
    lib.ttzk_trace_get_info.argtypes = [ctypes.c_void_p, P(_TraceInfo)]
    lib.ttzk_trace_get_layer.argtypes = [ctypes.c_void_p, ctypes.c_uint32, P(_TraceLayer)]
    lib.ttzk_trace_get_output.argtypes = [ctypes.c_void_p, P(P(ctypes.c_uint8)), P(ctypes.c_size_t)]
    lib.ttzk_trace_destroy.argtypes = [ctypes.c_void_p]
    lib.ttzk_trace_destroy.restype = None
    if lib.ttzk_abi_version() != ABI_VERSION:
        raise RuntimeError(f"ABI mismatch: the library has {lib.ttzk_abi_version()}, these bindings expect {ABI_VERSION}")
    return lib


_lib = _load()


def _check(status):
    if status != TTZK_OK:
        raise TtZkmlError(status, _lib.ttzk_last_error().decode())


def abi_version() -> int:
    return _lib.ttzk_abi_version()


def version() -> str:
    return _lib.ttzk_version().decode()


def num_devices() -> int:
    n = ctypes.c_uint32()
    _check(_lib.ttzk_num_devices(ctypes.byref(n)))
    return n.value


def pot_k(absmax: float, bits: int = 8) -> int:
    """The power-of-two quantization exponent: the largest k with absmax * 2^k <= 2^(bits - 1) - 1 (at most 127)."""
    k = ctypes.c_int32()
    _check(_lib.ttzk_pot_k(absmax, bits, ctypes.byref(k)))
    return k.value


def _floats(values: Sequence[float]) -> "array":
    return values if isinstance(values, array) and values.typecode == "f" else array("f", values)


def _ptr(a: "array"):
    addr, _ = a.buffer_info()
    return ctypes.cast(addr, ctypes.POINTER(ctypes.c_float))


@dataclass
class Layer:
    in_features: int
    out_features: int
    relu: bool = False


@dataclass
class LayerTrace:
    out_features: int
    scaler_w: float
    scaler_out: float
    acc: list  # raw sign-magnitude bits, [batch, out_features] row-major (bit 31 = sign)

    def acc_values(self) -> list:
        """The accumulator as signed integers."""
        return [-(a & 0x7FFFFFFF) if a >> 31 else a for a in self.acc]


@dataclass
class Trace:
    """The execution trace of one forward pass, copied out of the library."""

    batch: int
    in_features: int
    scaler_x: float
    layers: list  # of LayerTrace
    output: bytes  # [batch, out_features of the last layer], sign-magnitude int8 (bit 7 = sign)

    def acc(self, layer: int) -> list:
        return self.layers[layer].acc

    def output_values(self) -> list:
        """The output as signed integers."""
        return [-(b & 0x7F) if b & 0x80 else b for b in self.output]


class Device:
    """An open device. Use as a context manager."""

    def __init__(self, device_id: int = 0):
        self._h = ctypes.c_void_p()
        _check(_lib.ttzk_device_open(device_id, ctypes.byref(self._h)))

    def close(self):
        if self._h:
            _lib.ttzk_device_close(self._h)
            self._h = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


class Model:
    """A chain of quantized fully connected layers on a device."""

    def __init__(self, device: Device, layers: Sequence[Layer], dtype: str = "bf16"):
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
        self._layers = list(layers)
        arr = (_Layer * len(self._layers))(*[_Layer(l.in_features, l.out_features, 1 if l.relu else 0) for l in self._layers])
        self._h = ctypes.c_void_p()
        self._device = device  # keeps the device alive
        _check(_lib.ttzk_model_create(device._h, arr, len(self._layers), _DTYPES[dtype], ctypes.byref(self._h)))

    def load_parameters(self, weights: Sequence[Sequence[float]]):
        """`weights[l]`: the W of layer l, [in, out] row-major."""
        flat = array("f")
        for w in weights:
            flat.extend(_floats(w))
        _check(_lib.ttzk_model_load_parameters(self._h, _ptr(flat), len(flat)))

    def forward(self, x: Sequence[float], batch: int) -> Trace:
        xs = _floats(x)
        t = ctypes.c_void_p()
        _check(_lib.ttzk_model_forward(self._h, _ptr(xs), len(xs), batch, ctypes.byref(t)))
        try:
            info = _TraceInfo()
            _check(_lib.ttzk_trace_get_info(t, ctypes.byref(info)))
            layers = []
            for l in range(info.layers):
                v = _TraceLayer()
                _check(_lib.ttzk_trace_get_layer(t, l, ctypes.byref(v)))
                layers.append(LayerTrace(v.out_features, v.scaler_w, v.scaler_out, list(v.acc[: v.acc_len])))
            data = ctypes.POINTER(ctypes.c_uint8)()
            n = ctypes.c_size_t()
            _check(_lib.ttzk_trace_get_output(t, ctypes.byref(data), ctypes.byref(n)))
            return Trace(info.batch, info.in_features, info.scaler_x, layers, bytes(data[: n.value]))
        finally:
            _lib.ttzk_trace_destroy(t)

    def destroy(self):
        if self._h:
            _lib.ttzk_model_destroy(self._h)
            self._h = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()

    def __del__(self):
        self.destroy()
