# abi: the C ABI of tt-zkml and its bindings

`include/tt_zkml/tt_zkml.h` is the interface of the C++ core for other languages. Rules: plain C types, opaque handles (`ttzk_device`,
`ttzk_model`, `ttzk_trace`) with explicit create / destroy, every call returns a `ttzk_status` and `ttzk_last_error()` says what failed, no C++
exception crosses the boundary, `TTZK_ABI_VERSION` changes only when an existing function changes.

```
ttzk_device_open / ttzk_device_close
ttzk_model_create(device, layers, n, dtype) / ttzk_model_load_parameters(model, weights) / ttzk_model_forward(model, x, batch) -> ttzk_trace
ttzk_trace_get_info / ttzk_trace_get_layer / ttzk_trace_get_output / ttzk_trace_destroy
ttzk_version, ttzk_abi_version, ttzk_num_devices, ttzk_pot_k
```

- `src/tt_zkml.cpp` implements it on top of the C++ API (`zkml::Device`, `zkml::Model`, `zkml::Trace` of `tt-zkml-core/src/api`) and builds
  `libtt_zkml_abi.so` (CMake target `tt_zkml_abi`, output `.build/default/lib/`).
- `python/`: the Python package `tt_zkml` (ctypes). `python3 -m unittest discover -s tests` in that directory.
- The Rust bindings are the crate `tt-zkml-sys` of the Rust workspace (`tt-zkml/lib/tt-zkml-sys`), declared by hand from this header; the library
  checks `ttzk_abi_version()` at start-up.
- `tests/test_abi.cpp`: the C ABI, including a forward pass compared bit for bit with a host reference.

Threads: tt-metal's device context is thread-affine. Open the device, create models, run and destroy them on one thread, before the process exits.
