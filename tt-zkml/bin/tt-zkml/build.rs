// Puts the run-time search paths of the shared libraries (the C ABI library and the tt-metal libraries it needs) into the binary, so it runs without
// LD_LIBRARY_PATH. The directories come from tt-zkml-sys (see its build.rs): `links` metadata reaches the packages that depend on it directly, so
// this package depends on tt-zkml-sys as well.
use std::env;

fn main() {
    for var in ["DEP_TT_ZKML_ABI_LIB_DIR", "DEP_TT_ZKML_ABI_METAL_LIB_DIR"] {
        if let Ok(d) = env::var(var) {
            println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{d}");
        }
    }
}
