// Links the shared library of the C ABI (abi/, libtt_zkml_abi.so).
//
// Where it looks:
//   TT_ZKML_ABI_LIB_DIR    the directory with libtt_zkml_abi.so (default: <repo>/abi/.build/default/lib)
//   TT_METAL_HOME          for the RPATH of the tt-metal libraries the core library depends on (<TT_METAL_HOME>/build/lib)
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=TT_ZKML_ABI_LIB_DIR");
    println!("cargo:rerun-if-env-changed=TT_METAL_HOME");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // default: the abi project's own build, else the repository root build
    let lib_dir = env::var("TT_ZKML_ABI_LIB_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        ["../../../abi/.build/default/lib"]
            .iter()
            .map(|d| manifest.join(d))
            .find(|d| d.join("libtt_zkml_abi.so").exists())
            .unwrap_or_else(|| manifest.join("../../../abi/.build/default/lib"))
    });
    let lib_dir = lib_dir.canonicalize().unwrap_or(lib_dir);
    if !lib_dir.join("libtt_zkml_abi.so").exists() {
        panic!(
            "libtt_zkml_abi.so not found in {}: build the repository first \
             (cmake --preset default && cmake --build .build/default --target tt_zkml_abi, in abi/) or set TT_ZKML_ABI_LIB_DIR",
            lib_dir.display()
        );
    }
    println!("cargo:rerun-if-changed={}", lib_dir.join("libtt_zkml_abi.so").display());
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=tt_zkml_abi");
    // run without LD_LIBRARY_PATH: the core library, and the tt-metal libraries it needs. `rustc-link-arg` only applies to the artifacts of the package
    // that owns the build script, so the directories are passed on as metadata (DEP_TT_ZKML_ABI_LIB_DIR, DEP_TT_ZKML_CORE_METAL_LIB_DIR) to
    // the build script of every package that depends on this crate (see ../tt-zkml/build.rs and ../../bin/tt-zkml/build.rs).
    println!("cargo:lib_dir={}", lib_dir.display());
    if let Ok(home) = env::var("TT_METAL_HOME") {
        let metal = PathBuf::from(home).join("build/lib");
        println!("cargo:metal_lib_dir={}", metal.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath-link,{}", metal.display());
    }
}
