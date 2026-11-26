// Puts the run-time search paths of the shared libraries (the C ABI library and the tt-metal libraries it needs) into the tests of this package, so
// they run without LD_LIBRARY_PATH. The directories come from tt-zkml-sys (see its build.rs). A binary that uses this crate does the same in its own
// build.rs (see bin/tt-zkml/build.rs).
use std::env;

fn main() {
    for var in ["DEP_TT_ZKML_ABI_LIB_DIR", "DEP_TT_ZKML_ABI_METAL_LIB_DIR"] {
        if let Ok(d) = env::var(var) {
            println!("cargo:rustc-link-arg-tests={}", format!("-Wl,-rpath,{d}"));
        }
    }
}
