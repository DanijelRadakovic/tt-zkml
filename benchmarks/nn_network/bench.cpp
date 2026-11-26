// SPDX-FileCopyrightText: © 2026 Danijel Radakovic
//
// SPDX-License-Identifier: MIT

// A chained quantized network (L layers, ReLU between, none after the last) on the CPU and on the device (see README.md).
//
// The network is what zkml::prim::execute_network computes: x [M, K0] and every weight w_l [K_l, N_l] (fp32 / bf16) are quantized per
// tensor to 8 bits with a power-of-two scaler (Q = 127, round half to even); each layer is an exact int8 matmul into int32, the
// activation on the accumulator, and the per-tensor requantization to 8 bits, whose output is the quantized input of the next layer.
// Only x and the weights are quantized from floats. The CPU implementation is bit-exact with the device op (`check`).
//
//   zkml_nn_network_bench cpu     <threads> <M> <width> <L> <reps>     the chained network on the CPU
//   zkml_nn_network_bench device  <fp32|bf16> <M> <width> <L> <reps>   one execute_network call (program cached)
//   zkml_nn_network_bench layers  <fp32|bf16> <M> <width> <L> <reps>   L execute_layer calls (not chained), for comparison
//   zkml_nn_network_bench hostprof <fp32|bf16> <M> <width> <L> <reps>  where the HOST time of one call goes (no device time): the
//                                                                    pieces of ttnn::device_operation::launch, for both ops
//   zkml_nn_network_bench check   <M> <width> <L>                      CPU against device: every output byte, accumulator and scaler
// All layers are `width` x `width` (x is M x width).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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
#include "device/operations/layer_execution/device/execute_layer_device_operation.hpp"
#include "device/operations/network_execution/device/execute_network_device_operation.hpp"
#include "device/operations/network_execution/execute_network.hpp"
#include <ttnn/device_operation.hpp>
#include <ttnn/mesh_device_operation_adapter.hpp>

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
// The chained network on the CPU.

constexpr int kQ = 127;

int32_t pot_k(float absmax) {
    int32_t k = 127;
    while (std::ldexp(static_cast<double>(absmax), k) > kQ) {
        k--;
    }
    return k;
}

inline int8_t quant1(float v, float scale) {
    return static_cast<int8_t>(std::nearbyintf(v * scale));  // v * 2^k is exact; nearbyint = round half to even
}

struct Scratch {
    std::vector<float> partial;
    std::vector<int32_t> part_i;
    std::vector<int8_t> q_in, q_next, qwT;  // activations [M, K], [M, N]; q_w transposed [N, K]
    std::vector<int32_t> acc;
};

struct NetResult {
    float scaler_x;
    std::vector<float> scaler_w, scaler_out;
    std::vector<std::vector<int32_t>> acc;  // only with keep_acc
    std::vector<uint8_t> out;               // sign-magnitude bytes of the last layer
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

// The whole network: x [M, W], L weights [W, W]; relu after every layer but the last.
void run_network(
    Pool& pool,
    Scratch& s,
    const float* x,
    const std::vector<std::vector<float>>& ws,
    size_t M,
    size_t W,
    NetResult& r,
    bool keep_acc) {
    const size_t L = ws.size();
    s.q_in.resize(M * W);
    s.q_next.resize(M * W);
    s.qwT.resize(W * W);
    s.acc.resize(M * W);
    r.scaler_w.assign(L, 0.f);
    r.scaler_out.assign(L, 0.f);
    r.acc.clear();
    r.out.resize(M * W);

    // the only quantization of the input
    const int32_t kx = pot_k(absmax_parallel(pool, s, x, M * W));
    const float sx = std::ldexp(1.0f, kx);
    r.scaler_x = sx;
    pool.parallel_for(M * W, [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; i++) {
            s.q_in[i] = quant1(x[i], sx);
        }
    });

    for (size_t l = 0; l < L; l++) {
        const bool relu = l + 1 < L, last = l + 1 == L;
        const float* w = ws[l].data();
        const int32_t kw = pot_k(absmax_parallel(pool, s, w, W * W));
        const float sw = std::ldexp(1.0f, kw);
        r.scaler_w[l] = sw;
        pool.parallel_for(W, [&](size_t lo, size_t hi) {
            for (size_t k = lo; k < hi; k++) {
                for (size_t n = 0; n < W; n++) {
                    s.qwT[n * W + k] = quant1(w[k * W + n], sw);
                }
            }
        });

        s.part_i.assign(pool.size(), 0);
        pool.run([&](unsigned t) {
            const size_t lo = M * t / pool.size(), hi = M * (t + 1) / pool.size();
            int32_t m = 0;
            for (size_t i = lo; i < hi; i++) {
                int32_t* row = &s.acc[i * W];
                size_t j = 0;
                for (; j + 4 <= W; j += 4) {
                    dot4(&s.q_in[i * W], &s.qwT[j * W], &s.qwT[(j + 1) * W], &s.qwT[(j + 2) * W], &s.qwT[(j + 3) * W], W, row + j);
                }
                for (; j < W; j++) {
                    int32_t a = 0;
                    for (size_t k = 0; k < W; k++) {
                        a += static_cast<int32_t>(s.q_in[i * W + k]) * s.qwT[j * W + k];
                    }
                    row[j] = a;
                }
                for (size_t j2 = 0; j2 < W; j2++) {
                    m = std::max(m, relu ? std::max(row[j2], 0) : std::abs(row[j2]));
                }
            }
            s.part_i[t] = m;
        });
        const int32_t A = *std::max_element(s.part_i.begin(), s.part_i.end());
        if (keep_acc) {
            r.acc.push_back(s.acc);
        }

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
        r.scaler_out[l] = std::ldexp(1.0f, ko);
        // requantize: the output is the next layer's quantized input (signed int8), or the sign-magnitude bytes of the result
        pool.parallel_for(M * W, [&](size_t lo, size_t hi) {
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
                if (relu && a < 0) {
                    q = 0;
                }
                if (last) {
                    r.out[i] = static_cast<uint8_t>((a < 0 && q != 0) ? (0x80 | q) : q);
                } else {
                    s.q_next[i] = static_cast<int8_t>(a < 0 ? -q : q);
                }
            }
        });
        if (!last) {
            std::swap(s.q_in, s.q_next);
        }
    }
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

std::vector<std::vector<float>> make_ws(size_t W, int L) {
    std::vector<std::vector<float>> ws;
    for (int l = 0; l < L; l++) {
        ws.push_back(uniform(W * W, 100 + l, -2.f, 2.f));
    }
    return ws;
}

void report(const char* what, const std::string& cfg, size_t M, size_t W, int L, double us, int reps) {
    std::printf("net %s %sM=%zu W=%zu L=%d: net_us=%.1f layer_us=%.1f (median of %d)\n", what, cfg.c_str(), M, W, L, us, us / L, reps);
}

int run_cpu(unsigned threads, size_t M, size_t W, int L, int reps) {
    Pool pool(threads);
    const auto x = uniform(M * W, 7, -2.f, 2.f);
    const auto ws = make_ws(W, L);
    Scratch s;
    NetResult r;
    auto network = [&] { run_network(pool, s, x.data(), ws, M, W, r, false); };
    network();
    network();
    std::vector<double> runs;
    for (int i = 0; i < reps; i++) {
        const auto t0 = Clock::now();
        network();
        runs.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
    }
    report("cpu", "threads=" + std::to_string(threads) + " ", M, W, L, median(runs), reps);
    return 0;
}

using tt::tt_metal::DataType;
using ttnn::Tensor;

struct DeviceCtx {
    std::shared_ptr<ttnn::MeshDevice> dev;
    Tensor tx;
    std::vector<Tensor> tw;
};

DeviceCtx make_device(const std::string& dt, size_t M, size_t W, int L) {
    DeviceCtx c;
    tt::tt_metal::SetRootDir(ZKML_TT_METAL_HOME);
    c.dev = ttnn::open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);
    const DataType type = dt == "bf16" ? DataType::BFLOAT16 : DataType::FLOAT32;
    auto spec = [&](size_t h, size_t wd) {
        return tt::tt_metal::TensorSpec(
            ttnn::Shape({static_cast<uint32_t>(h), static_cast<uint32_t>(wd)}),
            tt::tt_metal::TensorLayout(type, tt::tt_metal::PageConfig(ttnn::Layout::TILE), ttnn::MemoryConfig{}));
    };
    c.tx = Tensor::from_vector(uniform(M * W, 7, -2.f, 2.f), spec(M, W), c.dev.get());
    const auto ws = make_ws(W, L);
    for (int l = 0; l < L; l++) {
        c.tw.push_back(Tensor::from_vector(ws[l], spec(W, W), c.dev.get()));
    }
    return c;
}

int run_device(bool chained, const std::string& dt, size_t M, size_t W, int L, int reps) {
    DeviceCtx c = make_device(dt, M, W, L);
    auto sync = [&] { tt::tt_metal::distributed::Finish(c.dev->mesh_command_queue()); };
    std::vector<zkml::prim::Activation> acts(L, zkml::prim::Activation::ReLU);
    acts.back() = zkml::prim::Activation::None;
    double enqueue_us = 0;  // host time until the last call has returned (before the device is waited for)
    auto network = [&] {
        const auto t0 = Clock::now();
        if (chained) {
            (void)zkml::prim::execute_network(c.tx, c.tw, acts);
        } else {
            for (int l = 0; l < L; l++) {
                (void)zkml::prim::execute_layer(c.tx, c.tw[l], acts[l]);
            }
        }
        enqueue_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
        sync();
    };
    network();  // compile
    network();
    std::vector<double> runs, enqueues;
    for (int i = 0; i < reps; i++) {
        const auto t0 = Clock::now();
        network();
        runs.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
        enqueues.push_back(enqueue_us);
    }
    std::printf("net host-enqueue %s %s M=%zu W=%zu L=%d: enqueue_us=%.1f\n", chained ? "device" : "layers", dt.c_str(), M, W, L, median(enqueues));
    report(chained ? "device" : "layers", dt + " ", M, W, L, median(runs), reps);
    tt::tt_metal::ReadMeshDeviceProfilerResults(*c.dev);  // with TT_METAL_DEVICE_PROFILER=1: writes profile_log_device.csv
    c.dev->close();
    return 0;
}


// ----------------------------------------------------------------------------------------------------------------------------
// Where the host time of one call goes. ttnn::device_operation::launch does, in this order: collect and validate the input
// tensors, create_output_tensors, place_output_tensors, look up the program cache (the hash and the exact canonical key of the
// attributes and the tensor arguments), then on a hit override_runtime_arguments and enqueue the workload. The first four pieces
// are public and are timed on their own here; "rest" is the total of a real call minus them: place_output_tensors, the cache
// lookup of the key, override_runtime_arguments, the enqueue and the bookkeeping of the call.

template <typename F>
double time_us(int reps, F&& f) {
    for (int i = 0; i < 5; i++) {
        f();
    }
    std::vector<double> v;
    for (int i = 0; i < reps; i++) {
        const auto t0 = Clock::now();
        f();
        v.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
    }
    return median(v);
}

template <typename Op, typename Launch>
void host_profile(
    const char* label, ttnn::MeshDevice* dev, const typename Op::operation_attributes_t& attrs, const typename Op::tensor_args_t& args, Launch&& launch, int reps) {
    using Adapter = ttnn::device_operation::MeshDeviceOperationAdapter<Op>;
    std::vector<std::reference_wrapper<const Tensor>> inputs;
    ttsl::reflection::visit_object_of_type<Tensor>([&inputs](const Tensor& t) { inputs.push_back(std::cref(t)); }, args);

    const double validate = time_us(reps, [&] { ttnn::device_operation::detail::validate_input_tensors<Op>(attrs, inputs); });
    const double outputs = time_us(reps, [&] { auto o = Op::create_output_tensors(attrs, args); (void)o; });
    const double hash = time_us(reps, [&] {
        volatile auto h = ttnn::device_operation::detail::compute_program_hash<Op>(attrs, args);
        (void)h;
    });
    const std::string name = std::string(ttsl::get_type_name<Op>());
    const double canonical = time_us(reps, [&] {
        auto k = Adapter::compute_mesh_workload_canonical_key(dev, name, attrs, args);
        (void)k;
    });
    // A real call: host time until it returns, the device is not waited for inside the timing (it is after each batch).
    auto sync = [&] { tt::tt_metal::distributed::Finish(dev->mesh_command_queue()); };
    for (int i = 0; i < 3; i++) {
        launch();
    }
    sync();
    std::vector<double> calls;
    for (int i = 0; i < reps; i++) {
        const auto t0 = Clock::now();
        launch();
        calls.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
        sync();
    }
    const double total = median(calls);
    const double rest = total - validate - outputs - hash - canonical;
    std::printf(
        "net hostprof %s: validate_us=%.1f create_outputs_us=%.1f program_hash_us=%.1f canonical_key_us=%.1f rest_us=%.1f total_call_us=%.1f\n",
        label, validate, outputs, hash, canonical, rest, total);
}

int run_hostprof(const std::string& dt, size_t M, size_t W, int L, int reps) {
    DeviceCtx c = make_device(dt, M, W, L);
    std::vector<zkml::prim::Activation> acts(L, zkml::prim::Activation::ReLU);
    acts.back() = zkml::prim::Activation::None;
    {
        using Op = zkml::operations::network_execution::ExecuteNetworkOperation;
        const Op::operation_attributes_t a{acts, c.tx.memory_config()};
        const Op::tensor_args_t t{c.tx, c.tw};
        host_profile<Op>(
            ("execute_network L=" + std::to_string(L) + " (" + std::to_string(2 + 3 * L) + " outputs)").c_str(),
            c.dev.get(), a, t, [&] { (void)zkml::prim::execute_network(c.tx, c.tw, acts); }, reps);
    }
    {
        using Op = zkml::operations::layer_execution::ExecuteLayerOperation;
        const Op::operation_attributes_t a{zkml::prim::Activation::ReLU, c.tx.memory_config()};
        const Op::tensor_args_t t{c.tx, c.tw[0]};
        host_profile<Op>(
            "execute_layer (5 outputs)", c.dev.get(), a, t, [&] { (void)zkml::prim::execute_layer(c.tx, c.tw[0], zkml::prim::Activation::ReLU); }, reps);
    }
    c.dev->close();
    return 0;
}

int run_check(size_t M, size_t W, int L) {
    DeviceCtx c = make_device("fp32", M, W, L);
    std::vector<zkml::prim::Activation> acts(L, zkml::prim::Activation::ReLU);
    acts.back() = zkml::prim::Activation::None;
    const auto got = zkml::prim::execute_network(c.tx, c.tw, acts);
    Pool pool(4);
    Scratch s;
    NetResult want;
    run_network(pool, s, uniform(M * W, 7, -2.f, 2.f).data(), make_ws(W, L), M, W, want, true);
    size_t bad = 0;
    const auto out = got.out.to_vector<uint8_t>();
    for (size_t i = 0; i < out.size(); i++) {
        bad += out[i] != want.out[i];
    }
    bad += got.scaler_x.to_vector<float>().at(0) != want.scaler_x;
    for (int l = 0; l < L; l++) {
        bad += got.scaler_w[l].to_vector<float>().at(0) != want.scaler_w[l];
        bad += got.scaler_out[l].to_vector<float>().at(0) != want.scaler_out[l];
        const auto acc = got.acc[l].to_vector<uint32_t>();
        for (size_t i = 0; i < acc.size(); i++) {
            const int32_t v = (acc[i] >> 31) ? -static_cast<int32_t>(acc[i] & 0x7FFFFFFFu) : static_cast<int32_t>(acc[i]);
            bad += v != want.acc[l][i];
        }
    }
    std::printf("check M=%zu W=%zu L=%d: %zu differences (output bytes, scalers, accumulators)\n", M, W, L, bad);
    c.dev->close();
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string cmd = argc > 1 ? argv[1] : "";
    if (cmd == "cpu" && argc >= 7) {
        return run_cpu(std::stoul(argv[2]), std::stoul(argv[3]), std::stoul(argv[4]), std::atoi(argv[5]), std::atoi(argv[6]));
    }
    if ((cmd == "device" || cmd == "layers") && argc >= 7) {
        return run_device(cmd == "device", argv[2], std::stoul(argv[3]), std::stoul(argv[4]), std::atoi(argv[5]), std::atoi(argv[6]));
    }
    if (cmd == "hostprof" && argc >= 7) {
        return run_hostprof(argv[2], std::stoul(argv[3]), std::stoul(argv[4]), std::atoi(argv[5]), std::atoi(argv[6]));
    }
    if (cmd == "check" && argc >= 5) {
        return run_check(std::stoul(argv[2]), std::stoul(argv[3]), std::atoi(argv[4]));
    }
    std::fprintf(stderr, "usage: see the top of bench.cpp\n");
    return 2;
}
