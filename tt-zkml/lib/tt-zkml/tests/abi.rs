// A test program without the libtest harness (`harness = false` in Cargo.toml): the device context of tt-metal is thread-affine (its chip lock is released at
// process exit on the main thread), and libtest runs every test on a thread of its own. Here everything runs on the main thread.
use tt_zkml::{device, Device, Dtype, Layer, Model};

fn main() {
    abi_and_version();
    pot_k_matches_the_definition();
    errors_carry_the_message();
    forward_on_the_device();
    println!("all tests passed");
}

fn abi_and_version() {
    device::check_abi().unwrap();
    assert_eq!(device::version().split('.').count(), 3);
}

fn pot_k_matches_the_definition() {
    assert_eq!(device::pot_k(1.0, 8).unwrap(), 6);
    assert_eq!(device::pot_k(127.0, 8).unwrap(), 0);
    assert_eq!(device::pot_k(128.0, 8).unwrap(), -1);
    assert_eq!(device::pot_k(0.0, 8).unwrap(), 127);
    assert_eq!(device::pot_k(1.0, 16).unwrap(), 14);
}

fn errors_carry_the_message() {
    let e = device::pot_k(1.0, 12).unwrap_err();
    assert_eq!(e.status, tt_zkml_sys::TTZK_ERR_INVALID_ARGUMENT);
    assert!(e.message.contains("bits"));
}

// A small LCG: no dependency for a test input.
fn lcg(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + ((s >> 40) as f32 / (1u64 << 24) as f32) * (hi - lo)
        })
        .collect()
}

// Everything that needs the device.
fn forward_on_the_device() {
    assert!(device::num_devices().unwrap() >= 1);
    let dev = Device::open(0).unwrap();
    let layers = [Layer { in_features: 64, out_features: 128, relu: true }, Layer { in_features: 128, out_features: 32, relu: false }];
    let mut model = Model::new(&dev, &layers, Dtype::Bf16).unwrap();

    // not loaded yet, a wrong batch, a wrong chain
    let x = lcg(1, 64 * 64, -2.0, 2.0);
    assert!(model.forward(&x, 64).is_err());
    assert!(Model::new(&dev, &[Layer { in_features: 64, out_features: 128, relu: false }, Layer { in_features: 64, out_features: 32, relu: false }], Dtype::Bf16).is_err());

    model.load_parameters(&[lcg(2, 64 * 128, -1.0, 1.0), lcg(3, 128 * 32, -1.0, 1.0)]).unwrap();
    assert!(model.forward(&x, 16).is_err());
    let t = model.forward(&x, 64).unwrap();
    assert_eq!(t.batch, 64);
    assert_eq!(t.in_features, 64);
    assert_eq!(t.layers.len(), 2);
    assert_eq!(t.layers[0].acc.len(), 64 * 128);
    assert_eq!(t.layers[1].acc.len(), 64 * 32);
    assert_eq!(t.output.len(), 64 * 32);
    assert!(t.scaler_x > 0.0 && t.layers.iter().all(|l| l.scaler_w > 0.0 && l.scaler_out > 0.0));
    assert!(t.output_values().iter().all(|v| v.abs() <= 127));
    // deterministic
    assert_eq!(model.forward(&x, 64).unwrap(), t);
}
