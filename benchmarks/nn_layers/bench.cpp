// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// A network of identical quantized layers on the CPU (1 thread, N threads) and on the device (see README.md).
//
// One layer is what zkml::prim::execute_layer computes: x [M, K] and w [K, N] (fp32) are quantized per tensor to 8 bits
// with a power-of-two scaler (Q = 127, round half to even), multiplied exactly in int32, ReLU is applied on the
// accumulator and the accumulator is requantized per tensor to 8 bits. The CPU implementation here is bit-exact with
// the device op (`check`). A network is L such layers, each with its own weights; all layers use the same input x (the
// device op takes fp32 / bf16 tensors, so layer outputs cannot be chained on the device without a host round trip; the
// work per layer is identical).
//
//   zkml_nn_bench cpu    <threads> <M> <K> <N> <L> <reps>      time of L layers on the CPU
//   zkml_nn_bench device <fp32|bf16> <M> <K> <N> <L> <reps>    time of L execute_layer calls on the device (program
//   cached) zkml_nn_bench check  <M> <K> <N>                           CPU against device, every output byte and scaler

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

#include "device/operations/layer_execution/execute_layer.hpp"

using Clock = std::chrono::steady_clock;

namespace {

// ----------------------------------------------------------------------------------------------------------------------------
// A small persistent thread pool: `run(fn)` executes fn(thread_index) on every thread (the calling thread is index 0)
// and returns when all are done. Workers spin (yielding) between jobs, so a dispatch costs about a microsecond.
class Pool {
public:
    explicit Pool(unsigned n) : n_(n) {
        for (unsigned t = 1; t < n_; t++) {
            threads_.emplace_back([this, t] {
                uint64_t seen = 0;
                while (true) {
                    uint64_t g;
                    while ((g = gen_.load(std::memory_order_acquire)) == seen) {
                        if (stop_.load(std::memory_order_relaxed)) {
                            return;
                        }
                        std::this_thread::yield();
                    }
                    seen = g;
                    (*job_)(t);
                    done_.fetch_add(1, std::memory_order_release);
                }
            });
        }
    }
    ~Pool() {
        stop_ = true;
        for (auto& t : threads_) {
            t.join();
        }
    }
    unsigned size() const { return n_; }
    void run(const std::function<void(unsigned)>& fn) {
        if (n_ == 1) {
            fn(0);
            return;
        }
        job_ = &fn;
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        fn(0);
        while (done_.load(std::memory_order_acquire) != n_ - 1) {
            std::this_thread::yield();
        }
    }
    // Static split of [0, n) over the threads.
    void parallel_for(size_t n, const std::function<void(size_t, size_t)>& body) {
        run([&](unsigned t) {
            const size_t lo = n * t / n_, hi = n * (t + 1) / n_;
            if (lo < hi) {
                body(lo, hi);
            }
        });
    }

private:
    unsigned n_;
    std::vector<std::thread> threads_;
    const std::function<void(unsigned)>* job_ = nullptr;
    std::atomic<uint64_t> gen_{0};
    std::atomic<unsigned> done_{0};
    std::atomic<bool> stop_{false};
};

// ----------------------------------------------------------------------------------------------------------------------------
// The layer on the CPU.

constexpr int kQ = 127;

int32_t pot_k(float absmax) {
    int32_t k = 127;
    while (std::ldexp(static_cast<double>(absmax), k) > kQ) {
        k--;
    }
    return k;
}

struct Scratch {
    std::vector<float> partial;   // per-thread absmax
    std::vector<int8_t> qx, qwT;  // q_x [M, K], q_w transposed [N, K]
    std::vector<int32_t> acc;     // [M, N]
    std::vector<int32_t> part_i;  // per-thread integer absmax
};

struct LayerResult {
    float scaler_x, scaler_w, scaler_out;
    std::vector<uint8_t> out;  // sign-magnitude bytes
};

float absmax_parallel(Pool& pool, Scratch& s, const float* x, size_t n) {
    s.partial.assign(pool.size(), 0.0f);
    pool.run([&](unsigned t) {
        const size_t lo = n * t / pool.size(), hi = n * (t + 1) / pool.size();
        float m = 0.0f;
        for (size_t i = lo; i < hi; i++) {
            m = std::max(m, std::fabs(x[i]));
        }
        s.partial[t] = m;
    });
    return *std::max_element(s.partial.begin(), s.partial.end());
}

inline int8_t quant1(float v, float scale) {
    return static_cast<int8_t>(std::nearbyintf(v * scale));  // v * 2^k is exact; nearbyint = round half to even
}

// One dot product of 4 rows of wT against one row of x, int8 x int8 -> int32.
inline void dot4(
    const int8_t* x, const int8_t* w0, const int8_t* w1, const int8_t* w2, const int8_t* w3, size_t K, int32_t* out) {
    int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (size_t k = 0; k < K; k++) {
        const int32_t xv = x[k];
        s0 += xv * w0[k];
        s1 += xv * w1[k];
        s2 += xv * w2[k];
        s3 += xv * w3[k];
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

// out must have M * N entries. Allocation-free after the first call with the same sizes.
void run_layer(
    Pool& pool,
    Scratch& s,
    const float* x,
    const float* w,
    size_t M,
    size_t K,
    size_t N,
    bool relu,
    uint8_t* out,
    float* scalers) {
    s.qx.resize(M * K);
    s.qwT.resize(N * K);
    s.acc.resize(M * N);

    // quantize x
    const float ax = absmax_parallel(pool, s, x, M * K);
    const int32_t kx = pot_k(ax);
    const float sx = std::ldexp(1.0f, kx);
    pool.parallel_for(M * K, [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; i++) {
            s.qx[i] = quant1(x[i], sx);
        }
    });
    // quantize w, stored transposed so that the matmul reads contiguous rows
    const float aw = absmax_parallel(pool, s, w, K * N);
    const int32_t kw = pot_k(aw);
    const float sw = std::ldexp(1.0f, kw);
    pool.parallel_for(K, [&](size_t lo, size_t hi) {
        for (size_t k = lo; k < hi; k++) {
            for (size_t n = 0; n < N; n++) {
                s.qwT[n * K + k] = quant1(w[k * N + n], sw);
            }
        }
    });

    // matmul (int32, exact), ReLU on the accumulator, integer absmax of the activated accumulator
    s.part_i.assign(pool.size(), 0);
    pool.run([&](unsigned t) {
        const size_t lo = M * t / pool.size(), hi = M * (t + 1) / pool.size();
        int32_t m = 0;
        for (size_t i = lo; i < hi; i++) {
            int32_t* row = &s.acc[i * N];
            size_t j = 0;
            for (; j + 4 <= N; j += 4) {
                dot4(
                    &s.qx[i * K],
                    &s.qwT[j * K],
                    &s.qwT[(j + 1) * K],
                    &s.qwT[(j + 2) * K],
                    &s.qwT[(j + 3) * K],
                    K,
                    row + j);
            }
            for (; j < N; j++) {
                int32_t a = 0;
                for (size_t k = 0; k < K; k++) {
                    a += static_cast<int32_t>(s.qx[i * K + k]) * s.qwT[j * K + k];
                }
                row[j] = a;
            }
            for (size_t j2 = 0; j2 < N; j2++) {
                const int32_t v = relu ? std::max(row[j2], 0) : std::abs(row[j2]);
                m = std::max(m, v);
            }
        }
        s.part_i[t] = m;
    });
    int32_t A = *std::max_element(s.part_i.begin(), s.part_i.end());

    // requantize: k = the largest with A * 2^k <= 127, round half to even, all in integers
    int32_t ko = 127;
    if (A != 0) {
        int t = 0;
        while ((static_cast<int64_t>(A) >> t) != 0) {
            t++;
        }
        ko = 7 - t;
        if (ko < 0 && static_cast<int64_t>(A) > (int64_t{127} << -ko)) {
            ko--;
        }
    }
    pool.parallel_for(M * N, [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; i++) {
            const int32_t a = s.acc[i];
            const int64_t mag = a < 0 ? -static_cast<int64_t>(a) : a;
            int64_t q;
            if (ko >= 0) {
                q = ko > 40 ? 0 : mag << ko;
            } else {
                const int sh = -ko;
                q = (mag + (int64_t{1} << (sh - 1)) - 1 + ((mag >> sh) & 1)) >> sh;
            }
            uint8_t byte = static_cast<uint8_t>(q);
            if (relu && a < 0) {
                byte = 0;
            } else if (a < 0 && q != 0) {
                byte = static_cast<uint8_t>(0x80 | q);
            }
            out[i] = byte;
        }
    });
    scalers[0] = sx;
    scalers[1] = sw;
    scalers[2] = std::ldexp(1.0f, ko);
}

std::vector<float> uniform(size_t n, uint32_t seed, float lo, float hi) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& e : v) {
        e = d(rng);
    }
    return v;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ----------------------------------------------------------------------------------------------------------------------------

int run_cpu(unsigned threads, size_t M, size_t K, size_t N, int L, int reps) {
    Pool pool(threads);
    const auto x = uniform(M * K, 7, -2.f, 2.f);
    std::vector<std::vector<float>> ws;
    for (int l = 0; l < L; l++) {
        ws.push_back(uniform(K * N, 100 + l, -2.f, 2.f));
    }
    Scratch s;
    std::vector<uint8_t> out(M * N);
    float sc[3];
    auto network = [&] {
        for (int l = 0; l < L; l++) {
            run_layer(pool, s, x.data(), ws[l].data(), M, K, N, true, out.data(), sc);
        }
    };
    network();
    network();
    std::vector<double> runs;
    for (int r = 0; r < reps; r++) {
        const auto t0 = Clock::now();
        network();
        runs.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
    }
    const double nn = median(runs);
    std::printf(
        "nn cpu threads=%u M=%zu K=%zu N=%zu L=%d: layer_us=%.1f nn_us=%.1f (median of %d)\n",
        threads,
        M,
        K,
        N,
        L,
        nn / L,
        nn,
        reps);
    return 0;
}

using tt::tt_metal::DataType;
using ttnn::Tensor;

struct DeviceCtx {
    std::shared_ptr<ttnn::MeshDevice> dev;
    Tensor tx;
    std::vector<Tensor> tw;
};

DeviceCtx make_device(const std::string& dt, size_t M, size_t K, size_t N, int L) {
    DeviceCtx c;
    tt::tt_metal::SetRootDir(ZKML_TT_METAL_HOME);
    c.dev = ttnn::open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);
    const DataType type = dt == "bf16" ? DataType::BFLOAT16 : DataType::FLOAT32;
    auto spec = [&](size_t h, size_t wd) {
        return tt::tt_metal::TensorSpec(
            ttnn::Shape({static_cast<uint32_t>(h), static_cast<uint32_t>(wd)}),
            tt::tt_metal::TensorLayout(type, tt::tt_metal::PageConfig(ttnn::Layout::TILE), ttnn::MemoryConfig{}));
    };
    c.tx = Tensor::from_vector(uniform(M * K, 7, -2.f, 2.f), spec(M, K), c.dev.get());
    for (int l = 0; l < L; l++) {
        c.tw.push_back(Tensor::from_vector(uniform(K * N, 100 + l, -2.f, 2.f), spec(K, N), c.dev.get()));
    }
    return c;
}

int run_device(const std::string& dt, size_t M, size_t K, size_t N, int L, int reps) {
    DeviceCtx c = make_device(dt, M, K, N, L);
    auto sync = [&] { tt::tt_metal::distributed::Finish(c.dev->mesh_command_queue()); };
    auto network = [&] {
        for (int l = 0; l < L; l++) {
            (void)zkml::prim::execute_layer(c.tx, c.tw[l], zkml::prim::Activation::ReLU);
        }
        sync();
    };
    network();  // compile
    network();
    std::vector<double> runs;
    for (int r = 0; r < reps; r++) {
        const auto t0 = Clock::now();
        network();
        runs.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
    }
    const double nn = median(runs);
    std::printf(
        "nn device %s M=%zu K=%zu N=%zu L=%d: layer_us=%.1f nn_us=%.1f (median of %d)\n",
        dt.c_str(),
        M,
        K,
        N,
        L,
        nn / L,
        nn,
        reps);
    c.dev->close();
    return 0;
}

int run_check(size_t M, size_t K, size_t N) {
    DeviceCtx c = make_device("fp32", M, K, N, 1);
    auto [out, acc, sx, sw, so] = zkml::prim::execute_layer(c.tx, c.tw[0], zkml::prim::Activation::ReLU);
    const auto got = out.to_vector<uint8_t>();
    const float gsx = sx.to_vector<float>().at(0), gsw = sw.to_vector<float>().at(0), gso = so.to_vector<float>().at(0);
    const auto x = uniform(M * K, 7, -2.f, 2.f);
    const auto w = uniform(K * N, 100, -2.f, 2.f);
    Pool pool(4);
    Scratch s;
    std::vector<uint8_t> want(M * N);
    float sc[3];
    run_layer(pool, s, x.data(), w.data(), M, K, N, true, want.data(), sc);
    size_t bad = 0;
    // to_vector of a TILE tensor returns the logical row-major values
    for (size_t i = 0; i < want.size(); i++) {
        bad += got[i] != want[i];
    }
    std::printf(
        "check M=%zu K=%zu N=%zu: scalers cpu %g %g %g device %g %g %g, %zu / %zu out bytes differ\n",
        M,
        K,
        N,
        sc[0],
        sc[1],
        sc[2],
        gsx,
        gsw,
        gso,
        bad,
        want.size());
    c.dev->close();
    return (bad || sc[0] != gsx || sc[1] != gsw || sc[2] != gso) ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: see the top of bench.cpp\n");
        return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "cpu" && argc >= 8) {
        return run_cpu(
            std::stoul(argv[2]),
            std::stoul(argv[3]),
            std::stoul(argv[4]),
            std::stoul(argv[5]),
            std::atoi(argv[6]),
            std::atoi(argv[7]));
    }
    if (cmd == "device" && argc >= 8) {
        return run_device(
            argv[2],
            std::stoul(argv[3]),
            std::stoul(argv[4]),
            std::stoul(argv[5]),
            std::atoi(argv[6]),
            std::atoi(argv[7]));
    }
    if (cmd == "check" && argc >= 5) {
        return run_check(std::stoul(argv[2]), std::stoul(argv[3]), std::stoul(argv[4]));
    }
    std::fprintf(stderr, "usage: see the top of bench.cpp\n");
    return 2;
}
