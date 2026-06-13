/*
 * libipt — C ABI implementation.
 *
 * Wraps the ipt_tilde inference core (IptClassifier) and translates C++/torch
 * exceptions into status codes + a thread-local error string. The core headers
 * are reused verbatim from the ipt_tilde repository (no duplication).
 */
#include "ipt.h"

#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "ipt_classifier.h"   // ipt_tilde/src
#include "leaky_integrator.h" // ipt_tilde/src

struct ipt_classifier {
    IptClassifier   classifier;
    LeakyIntegrator integrator;
    double          smoothing_tau = 0.0;
    std::vector<std::string> class_names;

    ipt_classifier(std::string path, torch::DeviceType dev, double thr_db, int win_ms)
        : classifier(std::move(path), dev, thr_db, win_ms) {}
};

namespace {

thread_local std::string g_last_error = "";

torch::DeviceType to_device(ipt_device d) {
    return d == IPT_DEVICE_MPS ? torch::kMPS : torch::kCPU;
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
        h->classifier.initialize_model();
        if (auto names = h->classifier.get_class_names()) {
            h->class_names = std::move(*names);
        }
        return IPT_OK;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return IPT_ERR_TORCH;
    }
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

        std::vector<float> dist = std::move(result->distribution);
        if (h->smoothing_tau > 0.0) {
            h->integrator.set_tau(h->smoothing_tau);
            dist = h->integrator.process(dist);
        }

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

void ipt_set_energy_threshold(ipt_classifier* h, double threshold_db) {
    if (h) h->classifier.set_energy_threshold(threshold_db);
}

void ipt_set_threshold_window(ipt_classifier* h, int duration_ms) {
    if (h) h->classifier.set_threshold_window(duration_ms);
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

const char* ipt_last_error(void) {
    return g_last_error.c_str();
}
