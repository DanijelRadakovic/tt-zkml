// Host-side reference for PoT vs. arbitrary-scale symmetric int8 quantization,
// modelling the integer data flow of an int8 FPU matmul on Tenstorrent:
//
//   float x, w --(PoT quant)--> int8 q_x, q_w --(FPU matmul, Int32 DEST)--> int32 acc
//              --(requant)--> int8 y
//
// Build: c++ -std=c++20 -O2 zkml/pot_quantazation.cpp -o pot_quantazation && ./pot_quantazation

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int32_t Q = 127;  // symmetric int8; also matches TT sign-magnitude int8 (no -128)

int32_t clamp_q(int64_t v) { return static_cast<int32_t>(std::clamp<int64_t>(v, -Q, Q)); }

float absmax(const std::vector<float>& v) {
    float m = 0.0f;
    for (float x : v) {
        m = std::max(m, std::fabs(x));
    }
    return m;
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

// Arbitrary-scale symmetric: s = absmax / Q, q = round(x / s).
struct SymQuant {
    float scale;
    std::vector<int32_t> q;
};

SymQuant quantize_sym(const std::vector<float>& x) {
    SymQuant r{absmax(x) / Q, {}};
    r.q.reserve(x.size());
    for (float v : x) {
        r.q.push_back(clamp_q(std::lround(v / r.scale)));
    }
    return r;
}

// PoT symmetric: scale = 2^-k (k = fractional bits), q = round(x * 2^k).
struct PotQuant {
    int k;
    std::vector<int32_t> q;
};

PotQuant quantize_pot_with_k(const std::vector<float>& x, int k) {
    PotQuant r{k, {}};
    r.q.reserve(x.size());
    const float mul = std::ldexp(1.0f, k);  // exact in fp32
    for (float v : x) {
        r.q.push_back(clamp_q(std::lround(v * mul)));
    }
    return r;
}

double quant_mse(const std::vector<float>& x, const PotQuant& p) {
    double e = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        const double d = x[i] - std::ldexp(static_cast<double>(p.q[i]), -p.k);
        e += d * d;
    }
    return e / x.size();
}

// k = floor(log2(Q / absmax)) never clips; k + 1 halves the step but clips outliers.
// Keep whichever gives the lower MSE.
PotQuant quantize_pot(const std::vector<float>& x) {
    const int k_noclip = static_cast<int>(std::floor(std::log2(Q / absmax(x))));
    PotQuant a = quantize_pot_with_k(x, k_noclip);
    PotQuant b = quantize_pot_with_k(x, k_noclip + 1);
    return quant_mse(x, a) <= quant_mse(x, b) ? a : b;
}

// ---------------------------------------------------------------------------
// Matmul: [M x K] * [K x N], int8 operands, exact int64 accumulation so we can
// see what an Int16 or Int32 accumulator would have to hold.
// ---------------------------------------------------------------------------

std::vector<int64_t> matmul_int(
    const std::vector<int32_t>& a, const std::vector<int32_t>& b, int M, int K, int N) {
    std::vector<int64_t> c(static_cast<size_t>(M) * N, 0);
    for (int m = 0; m < M; m++) {
        for (int kk = 0; kk < K; kk++) {
            const int64_t av = a[m * K + kk];
            for (int n = 0; n < N; n++) {
                c[m * N + n] += av * b[kk * N + n];
            }
        }
    }
    return c;
}

std::vector<double> matmul_float(const std::vector<float>& a, const std::vector<float>& b, int M, int K, int N) {
    std::vector<double> c(static_cast<size_t>(M) * N, 0.0);
    for (int m = 0; m < M; m++) {
        for (int kk = 0; kk < K; kk++) {
            for (int n = 0; n < N; n++) {
                c[m * N + n] += static_cast<double>(a[m * K + kk]) * b[kk * N + n];
            }
        }
    }
    return c;
}

// ---------------------------------------------------------------------------
// Requantization variants
// ---------------------------------------------------------------------------

// (1) PoT, pure integer: y = clamp((acc + 2^(n-1)) >> n). Round-half-up.
//     Same semantics as add_int32 + right_shift_tile<Int32> on device, and as the
//     circuit constraint  acc + 2^(n-1) = y * 2^n + r,  0 <= r < 2^n.
int32_t requant_pot_shift(int64_t acc, int n) {
    if (n <= 0) {
        return clamp_q(acc * (int64_t{1} << -n));
    }
    return clamp_q((acc + (int64_t{1} << (n - 1))) >> n);  // >> on signed is arithmetic (C++20 guaranteed)
}

// (2) What TT requant_tile does: fp32(acc) [RNE cast] * fp32 scale [RNE] -> round-half-even.
//     With a PoT scale the multiply is exact; the cast is inexact once |acc| > 2^24.
int32_t requant_tt_fp32(int64_t acc, float scale) {
    std::fesetround(FE_TONEAREST);
    const float a = static_cast<float>(acc);  // int -> fp32, RNE
    const float v = a * scale;                // fp32 multiply, RNE
    return clamp_q(static_cast<int64_t>(std::nearbyint(v)));
}

// (3) Arbitrary scale, integer-only (gemmlowp/TFLite style): M ~= M0 * 2^-shift, M0 in [2^30, 2^31).
struct FixedMul {
    int64_t m0;
    int shift;
};

FixedMul make_fixed_multiplier(double M) {
    int exp = 0;
    const double frac = std::frexp(M, &exp);  // M = frac * 2^exp, frac in [0.5, 1)
    int64_t m0 = std::llround(frac * (int64_t{1} << 31));
    if (m0 == (int64_t{1} << 31)) {
        m0 /= 2;
        exp++;
    }
    return {m0, 31 - exp};
}

int32_t requant_sym_fixed(int64_t acc, FixedMul f) {
    const int64_t prod = acc * f.m0;  // up to ~63 bits: this is the wide intermediate a circuit must range-check
    return clamp_q((prod + (int64_t{1} << (f.shift - 1))) >> f.shift);
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

double sqnr_db(const std::vector<double>& ref, const std::vector<double>& approx) {
    double sig = 0.0, err = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        sig += ref[i] * ref[i];
        err += (ref[i] - approx[i]) * (ref[i] - approx[i]);
    }
    return 10.0 * std::log10(sig / err);
}

int bits_needed(int64_t v) {  // signed two's-complement bits to hold v
    int b = 1;
    while (v >= (int64_t{1} << (b - 1)) || v < -(int64_t{1} << (b - 1))) {
        b++;
    }
    return b;
}

std::vector<float> gaussian(size_t n, float sigma, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> d(0.0f, sigma);
    std::vector<float> v(n);
    for (float& x : v) {
        x = d(gen);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Experiments
// ---------------------------------------------------------------------------

void experiment_layer(int K) {
    const int M = 32, N = 32;
    const auto x = gaussian(static_cast<size_t>(M) * K, 1.0f, 1);
    const auto w = gaussian(static_cast<size_t>(K) * N, 1.0f / std::sqrt(static_cast<float>(K)), 2);
    const auto ref = matmul_float(x, w, M, K, N);

    // Output range calibrated on the float reference (as PTQ would on a calibration set).
    float y_absmax = 0.0f;
    for (double v : ref) {
        y_absmax = std::max(y_absmax, static_cast<float>(std::fabs(v)));
    }

    // --- arbitrary-scale symmetric ---
    const SymQuant sx = quantize_sym(x), sw = quantize_sym(w);
    const double s_y = y_absmax / Q;
    const auto acc_s = matmul_int(sx.q, sw.q, M, K, N);
    const FixedMul fm = make_fixed_multiplier(static_cast<double>(sx.scale) * sw.scale / s_y);
    std::vector<double> y_sym(acc_s.size());
    for (size_t i = 0; i < acc_s.size(); i++) {
        y_sym[i] = requant_sym_fixed(acc_s[i], fm) * s_y;
    }

    // --- PoT ---
    const PotQuant px = quantize_pot(x), pw = quantize_pot(w);
    const int k_y = static_cast<int>(std::floor(std::log2(Q / y_absmax)));
    const int n = px.k + pw.k - k_y;
    const auto acc_p = matmul_int(px.q, pw.q, M, K, N);
    std::vector<double> y_pot(acc_p.size());
    int64_t acc_max = 0;
    size_t int16_overflow = 0, tt_mismatch = 0;
    for (size_t i = 0; i < acc_p.size(); i++) {
        const int32_t y = requant_pot_shift(acc_p[i], n);
        y_pot[i] = std::ldexp(static_cast<double>(y), -k_y);
        acc_max = std::max<int64_t>(acc_max, std::abs(acc_p[i]));
        int16_overflow += (acc_p[i] > INT16_MAX || acc_p[i] < INT16_MIN);
        tt_mismatch += (y != requant_tt_fp32(acc_p[i], std::ldexp(1.0f, -n)));
    }

    std::printf(
        "K=%5d | SQNR sym %5.2f dB  PoT %5.2f dB  (loss %4.2f dB = %4.2f bit) | k_x=%d k_w=%d k_y=%d n=%2d | "
        "max|acc| %2d bits, int16 overflow %4zu/%zu | shift vs fp32-requant mismatches %zu | fixed-mul shift %d\n",
        K,
        sqnr_db(ref, y_sym),
        sqnr_db(ref, y_pot),
        sqnr_db(ref, y_sym) - sqnr_db(ref, y_pot),
        (sqnr_db(ref, y_sym) - sqnr_db(ref, y_pot)) / 6.02,
        px.k,
        pw.k,
        k_y,
        n,
        bits_needed(acc_max),
        int16_overflow,
        acc_p.size(),
        tt_mismatch,
        fm.shift);
}

// Worst-case accumulator growth: |acc| <= K * 127 * 127.
void accumulator_bounds() {
    std::printf("\nWorst-case accumulator (all operands = 127):\n");
    for (int K : {1, 2, 3, 32, 1024, 1041, 4096, 16384, 133144}) {
        const int64_t a = static_cast<int64_t>(K) * Q * Q;
        std::printf(
            "  K=%6d  |acc|=%11lld  needs %2d bits  int16:%s  fp32-exact(<=2^24):%s  int32:%s\n",
            K,
            static_cast<long long>(a),
            bits_needed(a),
            a <= INT16_MAX ? "ok  " : "OVF ",
            a <= (int64_t{1} << 24) ? "yes" : "NO ",
            a <= INT32_MAX ? "ok" : "OVF");
    }
}

// Show where the fp32 requant path (TT requant_tile) and the integer shift disagree, n = 19.
//   - exact ties: RNE vs half-up (any |acc|)
//   - just off a tie above 2^24: the int->fp32 cast snaps acc onto the tie (ulp = 4 at 2^25)
void fp32_cast_divergence() {
    constexpr int n = 19;
    const float scale = std::ldexp(1.0f, -n);
    std::printf("\nfp32 requant vs integer shift, n = %d:\n", n);
    const int64_t base = int64_t{1} << 25;  // 64 * 2^19
    const int64_t half = int64_t{1} << (n - 1);
    const struct {
        int64_t acc;
        const char* what;
    } cases[] = {
        {half, "exact tie 0.5           "},
        {base + half, "exact tie 64.5          "},
        {base + half + 1, "64.5 + 2^-19 (above tie)"},
        {base + 3 * half - 1, "65.5 - 2^-19 (below tie)"},
    };
    for (const auto& c : cases) {
        std::printf(
            "  %s acc=%9lld float(acc)=%9.0f  shift(half-up)=%3d  tt_fp32(RNE)=%3d\n",
            c.what,
            static_cast<long long>(c.acc),
            static_cast<double>(static_cast<float>(c.acc)),
            requant_pot_shift(c.acc, n),
            requant_tt_fp32(c.acc, scale));
    }
}

}  // namespace

int main() {
    std::printf("One 32xK * Kx32 layer, Gaussian data, int8 symmetric (Q=127):\n");
    for (int K : {32, 256, 1024, 4096}) {
        experiment_layer(K);
    }
    accumulator_bounds();
    fp32_cast_divergence();
    return 0;
}
