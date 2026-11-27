// Prints the columns of the worked examples of docs/zk-circuits.pdf, computed by the real trace generator.
//
//     cargo run --release --example circuit_examples
use tt_zkml::air::kind::{QCols, RqCols};
use tt_zkml::air::spec::{LayerSpec, Mode, activate, choose_ko};
use tt_zkml::air::trace::{quant_row, requant_row};
use tt_zkml::witness::{float_parts, quantize, shift_of};
use tt_zkml::device::Dtype;

fn signed(v: u32) -> i64 {
    if v > 1 << 30 { v as i64 - 2147483647 } else { v as i64 }
}

fn main() {
    println!("== tensor quantization: W = [127.0, 2.5, -3.5, 0.0], kw = 0 (127 * 2^0 <= 127)");
    let w = [127.0f32, 2.5, -3.5, 0.0];
    let kexp = 0;
    let c = QCols::new(kexp);
    let n = c.nz.as_ref().unwrap();
    let hit = w.iter().position(|&v| {
        let (_, m, ep) = float_parts(v);
        let sh = shift_of(ep, kexp) as i64;
        (1..=20).contains(&sh) && 2 * m as i64 > 127 * (1i64 << sh)
    });
    println!("hit element: {hit:?}");
    for (i, &v) in w.iter().enumerate() {
        let parts = float_parts(v);
        let q = quantize(v, 1.0, Dtype::F32);
        let row = quant_row(kexp, &c, i, parts, q, Some(i) == hit).unwrap();
        let g = |idx: usize| signed(row[idx].0);
        println!(
            "v={v}: s={} m={} e'={} raw={} | q={} qa={} sh={} p2={} c={} diff={} r={} e={} half={} g={} z={} h={} hs={} v1={:?} v2={:?} rn={:?} hv={:?}",
            parts.0, parts.1, parts.2, shift_of(parts.2, kexp), g(c.q), g(n.qa), g(n.sh), g(n.p2), g(n.c), g(n.diff), g(n.r), g(n.e), g(n.half), g(n.g), g(n.z), g(n.h), g(n.hs),
            n.v1.iter().map(|&i| g(i)).collect::<Vec<_>>(),
            n.v2.iter().map(|&i| g(i)).collect::<Vec<_>>(),
            n.rn.iter().map(|&i| g(i)).collect::<Vec<_>>(),
            n.hv.iter().map(|&i| g(i)).collect::<Vec<_>>(),
        );
    }
    println!("== input X = [0.3, -1.5, 0.02, 2.0]");
    let x = [0.3f32, -1.5, 0.02, 2.0];
    let kx = tt_zkml::air::spec::pot_exponent(2.0);
    for &v in &x {
        let (s, m, ep) = float_parts(v);
        println!("v={v}: s={s} m={m} e'={ep} raw={} q={}", shift_of(ep, kx), quantize(v, 2f32.powi(kx), Dtype::F32));
    }
    println!("kx = {kx}");

    println!("== requantization: acc = [1000, -2500, 300]");
    let acc = [1000i32, -2500, 300];
    let ko = choose_ko(&acc);
    for relu in [false, true] {
        let spec = LayerSpec { m: 1, k: 1, n: 3, relu, ko, kw: 0, kx: 0 };
        let rc = RqCols::new(&spec);
        println!("relu={relu}: ko={ko} mode={:?} T={}", spec.mode(), spec.hit_threshold());
        let hit = acc.iter().position(|a| a.unsigned_abs() >= spec.hit_threshold()).unwrap();
        for (i, &a) in acc.iter().enumerate() {
            let out = activate(a, &spec);
            let row = requant_row(&spec, &rc, i, a, out, if i == hit { a.signum() } else { 0 }).unwrap();
            let g = |idx: usize| signed(row[idx].0);
            let r = rc.right.as_ref().unwrap();
            let h = rc.hit.as_ref().unwrap();
            let relu_cols = rc.relu.as_ref().map(|q| (g(q.p), g(q.ap), q.neg_limbs.iter().map(|&i| g(i)).collect::<Vec<_>>()));
            println!(
                "acc={a}: out={} relu(p,ap,neg)={:?} e={} half={} v1={:?} v2={:?} u1={:?} u2={:?} hp={} hn={} hv={:?}",
                g(rc.out),
                relu_cols,
                g(r.e),
                g(r.half),
                r.v1_limbs.iter().map(|&i| g(i)).collect::<Vec<_>>(),
                r.v2_limbs.iter().map(|&i| g(i)).collect::<Vec<_>>(),
                r.u1_limbs.iter().map(|&i| g(i)).collect::<Vec<_>>(),
                r.u2_limbs.iter().map(|&i| g(i)).collect::<Vec<_>>(),
                g(h.hp),
                g(h.hn),
                h.limbs.iter().map(|&i| g(i)).collect::<Vec<_>>(),
            );
        }
    }
    println!("mode check: {:?}", Mode::Right(5));
}
