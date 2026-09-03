/*
 * libipt — C ABI implementation.
 *
 * Wraps the IPT inference core (IptClassifier) and translates C++/torch
 * exceptions into status codes + a thread-local error string. The core lives in
 * core/ — libipt owns it; consumers see only this C ABI (ipt.h).
 */
#include "ipt.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "ipt_classifier.h"   // core/
#include "leaky_integrator.h" // core/

struct ipt_classifier {
    IptClassifier   classifier;
    LeakyIntegrator integrator;
    double          smoothing_tau = 0.0;
    double          period_ms = 0.0;
    std::vector<std::string> class_names;

    ipt_classifier(std::string path, torch::DeviceType dev, double thr_db, int win_ms)
        : classifier(std::move(path), dev, thr_db, win_ms) {}
};

namespace {

thread_local std::string g_last_error = "";

// requested intra-op thread count (0 = torch default), applied at model load
std::atomic<int> g_num_threads{0};

torch::DeviceType to_device(ipt_device d) {
    switch (d) {
        case IPT_DEVICE_MPS:  return torch::kMPS;
        case IPT_DEVICE_CUDA: return torch::kCUDA;
        default:              return torch::kCPU;
    }
}

} // namespace

ipt_classifier* ipt_create(const char* model_path,
                           ipt_device device,
                           double energy_threshold_db,
                           int threshold_window_ms) {
    if (!model_path) {
        g_last_error = "ipt_create: model_path is null";
        return nullptr;
    }
    try {
        return new ipt_classifier(std::string(model_path),
                                  to_device(device),
                                  energy_threshold_db,
                                  threshold_window_ms);
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    }
}

void ipt_destroy(ipt_classifier* h) {
    delete h;
}

int ipt_initialize_model(ipt_classifier* h) {
    if (!h) { g_last_error = "ipt_initialize_model: handle is null"; return IPT_ERR_NULL; }
    try {
        h->classifier.initialize_model(g_num_threads.load());
        if (auto names = h->classifier.get_class_names()) {
            h->class_names = std::move(*names);
        }
        return IPT_OK;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return IPT_ERR_TORCH;
    }
}

void ipt_set_num_threads(int n) {
    g_num_threads.store(n < 0 ? 0 : n);
}

int ipt_get_num_threads(void) {
    return at::get_num_threads();
}

int ipt_init_buffers(ipt_classifier* h, int sample_rate, int input_vector_length) {
    if (!h) { g_last_error = "ipt_init_buffers: handle is null"; return IPT_ERR_NULL; }
    if (sample_rate <= 0 || input_vector_length <= 0) {
        g_last_error = "ipt_init_buffers: sample_rate and input_vector_length must be > 0";
        return IPT_ERR_INVALID_ARG;
    }
    try {
        h->classifier.initialize_buffers(sample_rate, input_vector_length);
        return IPT_OK;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return IPT_ERR_TORCH;
    }
}

int ipt_process(ipt_classifier* h,
                const double* samples, int num_samples,
                float* out_dist, int out_cap,
                double* out_latency_ms) {
    if (!h) { g_last_error = "ipt_process: handle is null"; return IPT_ERR_NULL; }
    if (!samples || num_samples < 0 || !out_dist || out_cap < 0) {
        g_last_error = "ipt_process: invalid argument";
        return IPT_ERR_INVALID_ARG;
    }
    try {
        std::vector<double> in(samples, samples + num_samples);
        auto result = h->classifier.process(std::move(in));
        if (!result) {
            return 0; // buffering or below threshold — not an error
        }

        // Return the raw (unsmoothed) distribution; callers apply temporal
        // smoothing explicitly via ipt_smooth() so they control the timestamp.
        const std::vector<float>& dist = result->distribution;

        const int n = static_cast<int>(dist.size());
        const int to_copy = n < out_cap ? n : out_cap;
        for (int i = 0; i < to_copy; ++i) {
            out_dist[i] = dist[static_cast<std::size_t>(i)];
        }
        if (out_latency_ms) {
            *out_latency_ms = result->inference_latency_ms;
        }
        return n;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return IPT_ERR_TORCH;
    }
}

void ipt_set_smoothing_tau(ipt_classifier* h, double tau_ms) {
    if (h) h->smoothing_tau = tau_ms < 0.0 ? 0.0 : tau_ms;
}

int ipt_smooth(ipt_classifier* h,
               const float* dist, int n,
               double frame_time_ms,
               float* out, int out_cap) {
    if (!h) { g_last_error = "ipt_smooth: handle is null"; return IPT_ERR_NULL; }
    if (!dist || n < 0 || !out || out_cap < 0) {
        g_last_error = "ipt_smooth: invalid argument";
        return IPT_ERR_INVALID_ARG;
    }

    h->integrator.set_tau(h->smoothing_tau);

    std::vector<float> in(dist, dist + n);
    std::vector<float> sm;
    if (frame_time_ms < 0.0) {
        // live, per-block processing — use the real-time clock
        sm = h->integrator.process(in);
    } else {
        // batched / offline — anchor smoothing to the supplied frame time so
        // results produced in a burst keep their true temporal spacing
        using clock = std::chrono::steady_clock;
        auto ts = clock::time_point{}
                + std::chrono::duration_cast<clock::duration>(
                      std::chrono::duration<double, std::milli>(frame_time_ms));
        sm = h->integrator.process(in, ts);
    }

    const int to_copy = static_cast<int>(sm.size()) < out_cap
                            ? static_cast<int>(sm.size()) : out_cap;
    for (int i = 0; i < to_copy; ++i) {
        out[i] = sm[static_cast<std::size_t>(i)];
    }
    return to_copy;
}

void ipt_set_energy_threshold(ipt_classifier* h, double threshold_db) {
    if (h) h->classifier.set_energy_threshold(threshold_db);
}

void ipt_set_threshold_window(ipt_classifier* h, int duration_ms) {
    if (h) h->classifier.set_threshold_window(duration_ms);
}

void ipt_set_period(ipt_classifier* h, double period_ms) {
    if (h) h->period_ms = period_ms < 0.0 ? 0.0 : period_ms;
}

int ipt_num_classes(ipt_classifier* h) {
    if (!h || h->class_names.empty()) return -1;
    return static_cast<int>(h->class_names.size());
}

int ipt_get_class_name(ipt_classifier* h, int index, char* out, int out_cap) {
    if (!h || !out || out_cap <= 0 || index < 0
        || index >= static_cast<int>(h->class_names.size())) {
        return -1;
    }
    const std::string& s = h->class_names[static_cast<std::size_t>(index)];
    const int n = static_cast<int>(s.size());
    const int to_copy = n < (out_cap - 1) ? n : (out_cap - 1);
    std::memcpy(out, s.data(), static_cast<std::size_t>(to_copy));
    out[to_copy] = '\0';
    return to_copy;
}

int ipt_acquire_window(ipt_classifier* h, const double* samples, int num_samples, float** out_window) {
    if (!h || !samples || num_samples < 0 || !out_window) {
        g_last_error = "ipt_acquire_window: invalid argument";
        return IPT_ERR_INVALID_ARG;
    }
    try {
        std::vector<double> in(samples, samples + num_samples);
        auto window = h->classifier.acquire_window(std::move(in));
        if (!window) {
            return 0; // no window ready
        }
        const int length = static_cast<int>(window->size());
        float* buf = new float[static_cast<size_t>(length)];
        std::memcpy(buf, window->data(), static_cast<size_t>(length) * sizeof(float));
        *out_window = buf;
        return length;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return IPT_ERR_TORCH;
    }
}

void ipt_free_window(float* window) {
    delete[] window;
}

int ipt_classify_batch(ipt_classifier* h, const float* const* windows, int num_windows, int window_length, float* out_dist, int out_cap, double* out_latency_ms) {
    if (!h || !windows || num_windows <= 0 || window_length <= 0 || !out_dist || out_cap < 0) {
        g_last_error = "ipt_classify_batch: invalid argument";
        return IPT_ERR_INVALID_ARG;
    }
    try {
        std::vector<std::vector<float>> windows_vec;
        windows_vec.reserve(static_cast<size_t>(num_windows));
        for (int i = 0; i < num_windows; i++) {
            windows_vec.emplace_back(windows[i], windows[i] + window_length);
        }
        auto results = h->classifier.classify(windows_vec);
        
        const int nclasses = static_cast<int>(results.empty() ? 0 : results.front().distribution.size());
        const int total_output = num_windows * nclasses;
        const int to_copy = std::min(total_output, out_cap);
        
        for (size_t i = 0; i < static_cast<size_t>(num_windows) && i < static_cast<size_t>(results.size()); i++) {
            const auto& dist = results[i].distribution;
            const int dist_size = static_cast<int>(dist.size());
            const int offset = i * nclasses;
            for (int j = 0; j < dist_size && offset + j < out_cap; j++) {
                out_dist[offset + j] = dist[static_cast<size_t>(j)];
            }
            if (out_latency_ms) {
                out_latency_ms[i] = results[i].inference_latency_ms;
            }
        }
        return static_cast<int>(results.size());
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return IPT_ERR_TORCH;
    }
}

const char* ipt_last_error(void) {
    return g_last_error.c_str();
}
